from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
import os
import json
import logging
import base64
import wave
import io

import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://virtual-assistant-pi-seven.vercel.app",
        "http://localhost:8000",  # for local testing
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

with open("settings.json", "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

ASSISTANT_NAME = SETTINGS["assistant_name"]
GREETING = SETTINGS["greeting"]
FAQ = SETTINGS["faq"]

# --- Database setup ---------------------------------------------------
# Creates assistant.db (SQLite) on first run and migrates any existing
# products.json into it. Safe to call on every startup.
try:
  db.init_db()
  db.migrate_v2_sales_agent()
  db.migrate_products_from_json("products.json")
  DB_AVAILABLE = True
except Exception as e:
    logger.error(f"Database unavailable at startup: {e}")
    DB_AVAILABLE = False

BASE_SYSTEM_PROMPT = (
    f"You are {ASSISTANT_NAME}, built by Jawad — not by Google or any other company. "
    f"If asked who made you, who you work for, or what you are, always say you were built by Jawad. "
    f"Keep replies short, casual, and natural — like a normal conversation, not a formal AI response. "
    f"Detect the language the user writes in — English, Urdu, or Pashto — and always reply in that same language. "
    f"You can look up products in the store catalog using the search_products tool when the customer is "
    f"asking about items, prices, or availability. Only mention products that come back from that tool — "
    f"never invent products, prices, or stock levels."
)


def get_today_date() -> str:
    """Returns today's date."""
    return date.today().isoformat()


def lookup_faq(question: str) -> str:
    """Looks up an answer to a common question from a predefined FAQ list."""
    question = question.lower().strip()
    for key in FAQ:
        if key in question:
            return FAQ[key]
    return "I don't have a saved answer for that."


def get_weather(city: str) -> str:
    """Gets the current weather for a given city name."""
    url = f"https://wttr.in/{city}?format=%C+%t"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.read().decode("utf-8")
    except Exception:
        return "Sorry, I couldn't fetch the weather right now."


def search_products_tool(keyword: str = None, category: str = None,
                          max_price: float = None, min_price: float = None,
                          color: str = None) -> str:
    """Searches the store's product catalog. Use this whenever a customer
    asks about items, prices, colors, categories, or availability.
    Returns a JSON list of matching products (name, price, currency, color,
    size, stock, description) — at most 5 results."""
    if not DB_AVAILABLE:
        return json.dumps([])
    try:
        results = db.search_products(
            keyword=keyword, category=category,
            max_price=max_price, min_price=min_price, color=color
        )
        trimmed = [
            {
                "product_id": r["product_id"],
                "name": r["product_name"],
                "category": r["category"],
                "price": r["price"],
                "currency": r["currency"],
                "color": r["color"],
                "size": r["size"],
                "stock": r["stock"],
                "description": r["description"],
            }
            for r in results
        ]
        return json.dumps(trimmed)
    except Exception as e:
        logger.error(f"Product search failed: {e}")
        return json.dumps([])


def find_product_video(message: str):
    """Checks a message for product keywords and returns the matching video filename, if any."""
    message = message.lower().strip()
    if not DB_AVAILABLE:
        return None
    try:
        for product in db.list_products():
            keywords = (product.get("keywords") or "").split(",")
            for keyword in keywords:
                keyword = keyword.strip().lower()
                if keyword and keyword in message:
                    return product.get("video_url")
    except Exception as e:
        logger.error(f"Video lookup failed: {e}")
    return None


DOCUMENT_CHUNKS = [
    "My name is Jawad Ahmad, a student at Malakand University.",
    "I study bioinformatics with a focus on research.",
    "I love playing cricket in my free time.",
    "I enjoy programming and building projects like this assistant.",
]

vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(DOCUMENT_CHUNKS)


def search_document(question: str) -> str:
    """Searches Jawad's personal document for the chunk most relevant to a question."""
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]
    best_index = similarities.argmax()
    return DOCUMENT_CHUNKS[best_index]


def build_memory_context(user_id: str) -> str:
    """Pulls whatever we know about this customer — preferences, notes,
    and a rolling summary — into a short block of text for the prompt.
    Keeps this cheap: no raw message history here, just distilled memory."""
    if not DB_AVAILABLE or not user_id:
        return ""
    try:
        memory = db.get_memory(user_id)
        if not memory:
            return ""
        parts = []
        if memory.get("summary"):
            parts.append(f"Earlier conversation summary: {memory['summary']}")
        if memory.get("preferences"):
            parts.append(f"Customer preferences: {memory['preferences']}")
        if memory.get("important_info"):
            parts.append(f"Other notes about this customer: {memory['important_info']}")
        return "\n".join(parts)
    except Exception as e:
        logger.error(f"Memory lookup failed: {e}")
        return ""


def build_recent_history(conversation_id: str):
    """Returns the last N messages as Gemini Content objects, so the
    model has short-term context without us re-sending the entire
    conversation every time."""
    if not DB_AVAILABLE or not conversation_id:
        return []
    try:
        recent = db.get_recent_messages(conversation_id, limit=12)
        history = []
        for m in recent:
            role = "user" if m["sender"] == "user" else "model"
            history.append(types.Content(role=role, parts=[types.Part(text=m["message"])]))
        return history
    except Exception as e:
        logger.error(f"History lookup failed: {e}")
        return []


def maybe_update_memory(user_id: str, conversation_id: str):
    """Every few messages, distill the conversation into a short rolling
    summary + preferences and save it to customer_memory. This is how the
    assistant 'remembers' a customer across separate conversations without
    us having to resend the full message history every single time.

    Runs only every 6 messages to keep this cheap; safe to fail silently
    since it's a background enhancement, not core to answering.
    """
    if not DB_AVAILABLE or not user_id or user_id == "anonymous" or not conversation_id:
        return
    try:
        total = db.count_messages(conversation_id)
        if total == 0 or total % 6 != 0:
            return

        recent = db.get_recent_messages(conversation_id, limit=20)
        transcript = "\n".join(f"{m['sender']}: {m['message']}" for m in recent)
        existing = db.get_memory(user_id) or {}

        prompt = (
            "You are updating a short internal memory file about a customer for a store's "
            "AI assistant. Given the conversation below and any existing notes, output a "
            "JSON object with exactly these keys: \"summary\" (2-3 sentences on what this "
            "customer has been discussing overall), \"preferences\" (short comma-separated "
            "notes like colors, styles, budget range), \"important_info\" (anything else "
            "worth remembering, e.g. name, sizes, occasions). If a field hasn't changed, "
            "keep it the same as before. Reply with ONLY the JSON object, nothing else.\n\n"
            f"Existing notes:\nsummary: {existing.get('summary', '')}\n"
            f"preferences: {existing.get('preferences', '')}\n"
            f"important_info: {existing.get('important_info', '')}\n\n"
            f"Recent conversation:\n{transcript}"
        )

        response = client.models.generate_content(model="gemini-3.5-flash", contents=prompt)
        raw = response.text.strip().strip("`").lstrip("json").strip()
        parsed = json.loads(raw)

        db.upsert_memory(
            user_id,
            preferences=parsed.get("preferences"),
            important_info=parsed.get("important_info"),
            summary=parsed.get("summary"),
        )
    except Exception as e:
        # Memory updates are a nice-to-have; never let this break the chat.
        logger.error(f"Memory update skipped due to error: {e}")


class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class VoiceRequest(BaseModel):
    audio_base64: str
    mime_type: str = "audio/webm"
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


class ProductIn(BaseModel):
    product_name: str
    category: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = "PKR"
    color: Optional[str] = None
    size: Optional[str] = None
    stock: Optional[int] = 0
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    keywords: Optional[str] = ""


class ProductUpdate(BaseModel):
    product_name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    stock: Optional[int] = None
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    keywords: Optional[str] = None


def check_api_key(x_api_key: str):
    if x_api_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def ensure_conversation(user_id: str, conversation_id: Optional[str]) -> str:
    """Makes sure we have a valid conversation to write into. Creates a new
    one if none was given, or if the given one doesn't belong to this user."""
    db.get_or_create_customer(user_id)
    if conversation_id:
        existing = db.get_conversation(conversation_id)
        if existing and existing["user_id"] == user_id:
            return conversation_id
    return db.create_conversation(user_id)


@app.get("/api/status")
def read_root():
    return {"message": "Hello, your server is running!", "database": DB_AVAILABLE}


@app.get("/settings")
def get_settings():
    return {"assistant_name": ASSISTANT_NAME, "greeting": GREETING}


@app.get("/analytics")
def get_analytics(x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    with db.get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM conversations WHERE is_deleted = 0").fetchone()["c"]
        total_messages = conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"]
        total_customers = conn.execute("SELECT COUNT(*) AS c FROM customers").fetchone()["c"]
    return {
        "total_conversations": total,
        "total_messages": total_messages,
        "total_customers": total_customers,
    }


# Maps each tool name Gemini can call to the actual Python function that
# runs it. Used by the manual function-calling loop below.
TOOL_FUNCTIONS = {
    "get_today_date": get_today_date,
    "lookup_faq": lookup_faq,
    "get_weather": get_weather,
    "search_products_tool": search_products_tool,
}


def generate_reply(message: str, user_id: str = None, conversation_id: str = None) -> str:
    relevant_chunk = search_document(message)
    memory_context = build_memory_context(user_id) if user_id else ""

    full_system_prompt = (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"Here is relevant background information you can use if it helps answer the question:\n"
        f"\"{relevant_chunk}\""
    )
    if memory_context:
        full_system_prompt += f"\n\nWhat you remember about this customer:\n{memory_context}"

    history = build_recent_history(conversation_id) if conversation_id else []
    contents = list(history) if history else [types.Content(role="user", parts=[types.Part(text=message)])]

    config = types.GenerateContentConfig(
        system_instruction=full_system_prompt,
        tools=[get_today_date, lookup_faq, get_weather, search_products_tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    response = client.models.generate_content(model="gemini-3.5-flash", contents=contents, config=config)

    max_turns = 5
    turns = 0
    while response.function_calls and turns < max_turns:
        turns += 1
        contents.append(response.candidates[0].content)

        function_response_parts = []
        for fc in response.function_calls:
            func = TOOL_FUNCTIONS.get(fc.name)
            if func:
                try:
                    result = func(**(fc.args or {}))
                except Exception as e:
                    logger.error(f"Tool '{fc.name}' failed: {e}")
                    result = f"Error running {fc.name}: {e}"
            else:
                result = f"Unknown tool: {fc.name}"
            function_response_parts.append(
                types.Part.from_function_response(name=fc.name, response={"result": result})
            )
        contents.append(types.Content(role="user", parts=function_response_parts))

        response = client.models.generate_content(model="gemini-3.5-flash", contents=contents, config=config)

    return response.text

def text_to_speech(text: str) -> str:
    response = client.models.generate_content(
        model="gemini-3.1-flash-tts-preview",
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
                )
            ),
        )
    )
    audio_data = response.candidates[0].content.parts[0].inline_data.data

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(audio_data)

    wav_bytes = buffer.getvalue()
    return base64.b64encode(wav_bytes).decode("utf-8")


@app.post("/chat")
def chat(request: ChatRequest, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    logger.info(f"Incoming message: {request.message}")

    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Fall back to a per-session anonymous id if the frontend didn't send one,
    # so the app still works even without persistence wired up on that end.
    user_id = request.user_id or "anonymous"
    conversation_id = None

    if DB_AVAILABLE:
        try:
            conversation_id = ensure_conversation(user_id, request.conversation_id)
            db.save_message(conversation_id, "user", request.message, "text")
        except Exception as e:
            logger.error(f"Could not persist incoming message: {e}")
            conversation_id = None  # keep answering even if storage fails

    try:
        reply = generate_reply(request.message, user_id=user_id, conversation_id=conversation_id)
    except Exception as e:
        logger.error(f"Error during chat: {e}")
        return {"reply": "Sorry, I'm having trouble responding right now. Please try again in a moment."}

    if DB_AVAILABLE and conversation_id:
        try:
            db.save_message(conversation_id, "assistant", reply, "text")
            maybe_update_memory(user_id, conversation_id)
        except Exception as e:
            logger.error(f"Could not persist reply: {e}")

        video_file = find_product_video(request.message)
    result = {"reply": reply, "conversation_id": conversation_id}
    if video_file:
        video_url = f"/videos/{video_file}"
        result["video_url"] = video_url
        if DB_AVAILABLE and conversation_id:
            try:
                db.save_message(conversation_id, "assistant", video_url, "video")
            except Exception as e:
                logger.error(f"Could not persist video message: {e}")
    return result


@app.post("/voice-chat")
def voice_chat(request: VoiceRequest, x_api_key: str = Header(None)):
    check_api_key(x_api_key)

    user_id = request.user_id or "anonymous"
    conversation_id = None

    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception as e:
        logger.error(f"Invalid audio payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio data")

    try:
        transcription_response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=request.mime_type),
                "Transcribe this audio to text. Reply with ONLY the transcribed text, nothing else."
            ]
        )
        transcript = transcription_response.text.strip()
        logger.info(f"Voice transcript: {transcript}")
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return {"transcript": "", "reply": "Sorry, I couldn't process that voice message."}

    if DB_AVAILABLE:
        try:
            conversation_id = ensure_conversation(user_id, request.conversation_id)
            db.save_message(conversation_id, "user", transcript, "voice")
        except Exception as e:
            logger.error(f"Could not persist voice message: {e}")
            conversation_id = None

    try:
        reply = generate_reply(transcript, user_id=user_id, conversation_id=conversation_id)
        reply_audio = text_to_speech(reply)
    except Exception as e:
        logger.error(f"Error generating voice reply: {e}")
        return {"transcript": transcript, "reply": "Sorry, I couldn't process that voice message."}

    if DB_AVAILABLE and conversation_id:
        try:
            db.save_message(conversation_id, "assistant", reply, "voice")
            maybe_update_memory(user_id, conversation_id)
        except Exception as e:
            logger.error(f"Could not persist voice reply: {e}")

    return {"transcript": transcript, "reply": reply, "reply_audio": reply_audio, "conversation_id": conversation_id}


# --- Conversation history endpoints ------------------------------------

@app.get("/conversations")
def list_user_conversations(user_id: str, limit: int = Query(20, le=100), offset: int = 0):
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return db.list_conversations(user_id, limit=limit, offset=offset)


@app.post("/conversations")
def start_conversation(user_id: str, title: Optional[str] = None):
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    db.get_or_create_customer(user_id)
    conversation_id = db.create_conversation(user_id, title=title)
    return {"conversation_id": conversation_id}


@app.get("/conversations/{conversation_id}/messages")
def get_conversation_messages(conversation_id: str, user_id: str,
                               limit: int = Query(50, le=200), offset: int = 0):
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    convo = db.get_conversation(conversation_id)
    if not convo or convo["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return db.get_messages_page(conversation_id, limit=limit, offset=offset)


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str, user_id: str):
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    ok = db.delete_conversation(conversation_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}


# --- Admin: product management ------------------------------------------
# Reuses the same shared secret as /analytics for now (single-business
# setup). When this becomes multi-business, swap this for per-business
# admin keys/logins.

@app.get("/admin/products")
def admin_list_products(x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return db.list_products()


@app.post("/admin/products")
def admin_add_product(product: ProductIn, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    product_id = db.add_product(**product.model_dump())
    return {"product_id": product_id}


@app.put("/admin/products/{product_id}")
def admin_update_product(product_id: str, product: ProductUpdate, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    updates = {k: v for k, v in product.model_dump().items() if v is not None}
    ok = db.update_product(product_id, **updates)
    if not ok:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"updated": True}


@app.delete("/admin/products/{product_id}")
def admin_delete_product(product_id: str, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    ok = db.delete_product(product_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Product not found")
    return {"deleted": True}


@app.get("/admin/conversations")
def admin_list_all_conversations(x_api_key: str = Header(None), limit: int = Query(50, le=200)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    with db.get_conn() as conn:
        rows = conn.execute(
            """SELECT c.*, cu.name AS customer_name, cu.email, cu.phone
               FROM conversations c
               LEFT JOIN customers cu ON cu.user_id = c.user_id
               WHERE c.is_deleted = 0
               ORDER BY c.updated_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


@app.get("/admin/conversations/{conversation_id}/messages")
def admin_get_conversation_messages(conversation_id: str, x_api_key: str = Header(None),
                                     limit: int = Query(200, le=500)):
    """Same as /conversations/{id}/messages but for admins: no ownership
    check, since an admin should be able to view any of their business's
    customer conversations."""
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    convo = db.get_conversation(conversation_id)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return db.get_messages_page(conversation_id, limit=limit)


@app.get("/admin/customers")
def admin_list_customers(x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM customers ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


# --- Serve the chat widget (index.html) ---
# This must be registered AFTER all your API routes above,
# otherwise it can override them.
@app.get("/index.html")
def serve_widget():
    return FileResponse("index.html", headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/")
def serve_widget_root():
    return FileResponse("index.html", headers={"Cache-Control": "no-store, must-revalidate"})


# Explicit route for app.js with caching fully disabled. Without this,
# Vercel's global edge network can serve a stale/inconsistent cached copy
# from a different edge location right after a deploy — which is exactly
# what caused the chat to intermittently fail to load until the edge
# cache fully synced. "no-store" forces every request straight to origin.
@app.get("/app.js")
def serve_app_js():
    return FileResponse("app.js", media_type="text/javascript",
                         headers={"Cache-Control": "no-store, must-revalidate"})


# Serves any other files sitting in the same folder (CSS, JS, images)
# e.g. a request for /style.css or /widget.js will be found here automatically.
app.mount("/", StaticFiles(directory=".", html=True), name="static")