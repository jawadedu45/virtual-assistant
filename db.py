"""
db.py — persistent storage layer for the assistant.

Uses Turso via the `turso_serverless` package — a pure HTTP driver
purpose-built for stateless serverless environments like Vercel (no
local file involved at all, unlike the `libsql` package which expects
a local file that optionally syncs). This is what makes data survive
Vercel's ephemeral, no-persistent-disk serverless functions.

Requires two environment variables (set in .env locally, and in your
Vercel project's Environment Variables in production):
  TURSO_DATABASE_URL   e.g. libsql://your-db-yourname.aws-ap-south-1.turso.io
  TURSO_AUTH_TOKEN     the token generated in the Turso dashboard

Every table is scoped by business_id, so this can support multiple
businesses later without schema changes — for now everything uses
business_id = "default".
"""

import turso_serverless
import json
import os
import uuid
import logging
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DEFAULT_BUSINESS_ID = "default"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    """Yields a connection to the Turso database over HTTP. Raises a
    clear error the caller can catch if the database is unavailable
    (bad credentials, network issue) instead of crashing the whole app.
    Rows returned by conn.execute(...).fetchone()/.fetchall() behave
    like sqlite3.Row — support both row["col"] and dict(row)."""
    conn = None
    try:
        url = os.environ.get("TURSO_DATABASE_URL")
        token = os.environ.get("TURSO_AUTH_TOKEN")
        if not url or not token:
            raise RuntimeError(
                "TURSO_DATABASE_URL / TURSO_AUTH_TOKEN are not set. "
                "Add them to .env (local) or your host's environment variables (production)."
            )
        conn = turso_serverless.connect(url, auth_token=token)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def init_db():
    """Creates all tables if they don't exist yet, and ensures the
    default business row exists. Safe to call on every startup."""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS businesses (
            business_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS customers (
            user_id TEXT PRIMARY KEY,
            business_id TEXT NOT NULL REFERENCES businesses(business_id),
            name TEXT,
            email TEXT,
            phone TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            business_id TEXT NOT NULL REFERENCES businesses(business_id),
            user_id TEXT NOT NULL REFERENCES customers(user_id),
            title TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            is_deleted INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
            sender TEXT NOT NULL,        -- 'user' or 'assistant'
            message TEXT NOT NULL,
            message_type TEXT NOT NULL DEFAULT 'text',  -- 'text' | 'voice' | 'video'
            timestamp TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS customer_memory (
            user_id TEXT PRIMARY KEY REFERENCES customers(user_id),
            business_id TEXT NOT NULL REFERENCES businesses(business_id),
            preferences TEXT,       -- free-text notes (e.g. "prefers black, budget-conscious")
            important_info TEXT,    -- free-text notes (e.g. name, sizes, allergies)
            summary TEXT,           -- rolling summary of older conversation content
            last_interaction TEXT
        );

        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            business_id TEXT NOT NULL REFERENCES businesses(business_id),
            product_name TEXT NOT NULL,
            category TEXT,
            description TEXT,
            price REAL,
            currency TEXT DEFAULT 'PKR',
            color TEXT,
            size TEXT,
            stock INTEGER DEFAULT 0,
            image_url TEXT,
            video_url TEXT,
            keywords TEXT,          -- comma-separated, kept for simple matching
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, business_id);
        CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, timestamp);
        CREATE INDEX IF NOT EXISTS idx_products_business ON products(business_id);
        """)

        conn.execute(
            "INSERT OR IGNORE INTO businesses (business_id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (DEFAULT_BUSINESS_ID, "Khyber AI", _now(), _now())
        )
    logger.info("Database initialized.")


def migrate_products_from_json(json_path: str, business_id: str = DEFAULT_BUSINESS_ID):
    """One-time migration: if the products table is empty, load
    products.json into it so nothing is lost. Safe to call every
    startup — it's a no-op once products exist in the DB."""
    with get_conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS c FROM products WHERE business_id = ?", (business_id,)
        ).fetchone()["c"]
        if count > 0:
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                legacy_products = json.load(f)
        except FileNotFoundError:
            return

        for p in legacy_products:
            conn.execute(
                """INSERT INTO products
                   (product_id, business_id, product_name, category, description,
                    price, currency, color, size, stock, image_url, video_url,
                    keywords, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    business_id,
                    p.get("name", "Unnamed product"),
                    p.get("category"),
                    p.get("description"),
                    p.get("price"),
                    p.get("currency", "PKR"),
                    p.get("color"),
                    p.get("size"),
                    p.get("stock", 0),
                    p.get("image_url"),
                    p.get("video_file"),
                    ",".join(p.get("keywords", [])),
                    _now(),
                    _now(),
                )
            )
        logger.info(f"Migrated {len(legacy_products)} product(s) from {json_path}.")


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------

def get_or_create_customer(user_id: str, business_id: str = DEFAULT_BUSINESS_ID) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM customers WHERE user_id = ? AND business_id = ?",
            (user_id, business_id)
        ).fetchone()
        if row:
            return dict(row)
        conn.execute(
            "INSERT INTO customers (user_id, business_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, business_id, _now(), _now())
        )
        return {"user_id": user_id, "business_id": business_id}


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def create_conversation(user_id: str, business_id: str = DEFAULT_BUSINESS_ID, title: str = None) -> str:
    conversation_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO conversations (conversation_id, business_id, user_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (conversation_id, business_id, user_id, title, _now(), _now())
        )
    return conversation_id


def get_conversation(conversation_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ? AND is_deleted = 0",
            (conversation_id,)
        ).fetchone()
        return dict(row) if row else None


def list_conversations(user_id: str, business_id: str = DEFAULT_BUSINESS_ID,
                        limit: int = 20, offset: int = 0) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM conversations
               WHERE user_id = ? AND business_id = ? AND is_deleted = 0
               ORDER BY updated_at DESC LIMIT ? OFFSET ?""",
            (user_id, business_id, limit, offset)
        ).fetchall()
        return [dict(r) for r in rows]


def touch_conversation(conversation_id: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
            (_now(), conversation_id)
        )


def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """Soft delete — only succeeds if the conversation belongs to user_id."""
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE conversations SET is_deleted = 1, updated_at = ? WHERE conversation_id = ? AND user_id = ?",
            (_now(), conversation_id, user_id)
        )
        return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def save_message(conversation_id: str, sender: str, message: str, message_type: str = "text"):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO messages (conversation_id, sender, message, message_type, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (conversation_id, sender, message, message_type, _now())
        )
    touch_conversation(conversation_id)


def get_recent_messages(conversation_id: str, limit: int = 12) -> list[dict]:
    """Most recent N messages, oldest-first (ready to feed to the model)."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM (
                   SELECT * FROM messages WHERE conversation_id = ?
                   ORDER BY timestamp DESC LIMIT ?
               ) sub ORDER BY sub.timestamp ASC""",
            (conversation_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]


def get_messages_page(conversation_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM messages WHERE conversation_id = ?
               ORDER BY timestamp ASC LIMIT ? OFFSET ?""",
            (conversation_id, limit, offset)
        ).fetchall()
        return [dict(r) for r in rows]


def count_messages(conversation_id: str) -> int:
    with get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) AS c FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()["c"]


# ---------------------------------------------------------------------------
# Customer memory
# ---------------------------------------------------------------------------

def get_memory(user_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM customer_memory WHERE user_id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


def upsert_memory(user_id: str, business_id: str = DEFAULT_BUSINESS_ID,
                   preferences: str = None, important_info: str = None, summary: str = None):
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM customer_memory WHERE user_id = ?", (user_id,)
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE customer_memory SET
                   preferences = COALESCE(?, preferences),
                   important_info = COALESCE(?, important_info),
                   summary = COALESCE(?, summary),
                   last_interaction = ?
                   WHERE user_id = ?""",
                (preferences, important_info, summary, _now(), user_id)
            )
        else:
            conn.execute(
                """INSERT INTO customer_memory
                   (user_id, business_id, preferences, important_info, summary, last_interaction)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, business_id, preferences, important_info, summary, _now())
            )


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

def list_products(business_id: str = DEFAULT_BUSINESS_ID) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM products WHERE business_id = ? ORDER BY updated_at DESC",
            (business_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_product(product_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()
        return dict(row) if row else None


def search_products(business_id: str = DEFAULT_BUSINESS_ID, keyword: str = None,
                     category: str = None, max_price: float = None,
                     min_price: float = None, color: str = None,
                     in_stock_only: bool = True, limit: int = 5) -> list[dict]:
    query = "SELECT * FROM products WHERE business_id = ?"
    params = [business_id]

    if keyword:
        query += """ AND (
            LOWER(product_name) LIKE ? OR
            LOWER(description) LIKE ? OR
            LOWER(category) LIKE ? OR
            LOWER(keywords) LIKE ?
        )"""
        like = f"%{keyword.lower()}%"
        params += [like, like, like, like]
    if category:
        query += " AND LOWER(category) LIKE ?"
        params.append(f"%{category.lower()}%")
    if color:
        query += " AND LOWER(color) LIKE ?"
        params.append(f"%{color.lower()}%")
    if max_price is not None:
        query += " AND price <= ?"
        params.append(max_price)
    if min_price is not None:
        query += " AND price >= ?"
        params.append(min_price)
    if in_stock_only:
        query += " AND stock > 0"

    query += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def add_product(business_id: str = DEFAULT_BUSINESS_ID, **fields) -> str:
    product_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO products
               (product_id, business_id, product_name, category, description,
                price, currency, color, size, stock, image_url, video_url,
                keywords, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                product_id, business_id,
                fields.get("product_name", "Unnamed product"),
                fields.get("category"), fields.get("description"),
                fields.get("price"), fields.get("currency", "PKR"),
                fields.get("color"), fields.get("size"),
                fields.get("stock", 0), fields.get("image_url"),
                fields.get("video_url"), fields.get("keywords", ""),
                _now(), _now(),
            )
        )
    return product_id


def update_product(product_id: str, **fields) -> bool:
    if not fields:
        return False
    allowed = {"product_name", "category", "description", "price", "currency",
               "color", "size", "stock", "image_url", "video_url", "keywords"}
    updates = {k: v for k, v in fields.items() if k in allowed}
    if not updates:
        return False
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    params = list(updates.values()) + [_now(), product_id]
    with get_conn() as conn:
        cur = conn.execute(
            f"UPDATE products SET {set_clause}, updated_at = ? WHERE product_id = ?",
            params
        )
        return cur.rowcount > 0


def delete_product(product_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM products WHERE product_id = ?", (product_id,))
        return cur.rowcount > 0