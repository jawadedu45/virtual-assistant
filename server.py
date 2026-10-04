from fastapi import FastAPI, Header, HTTPException, Query, BackgroundTasks
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
import time
import random
import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FAST_MODEL = "gemini-1.5-flash-8b"
TTS_MODEL = "gemini-2.5-flash-preview-tts"
LAST_SEARCH_RESULTS = []

def generate_content_with_retry(model: str, contents, config=None, max_retries: int = 3):
    last_error = None
    for attempt in range(max_retries):
        try:
            if config is not None:
                return client.models.generate_content(model=model, contents=contents, config=config)
            return client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            last_error = e
            error_str = str(e)
            is_transient = "503" in error_str or "UNAVAILABLE" in error_str or "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            if not is_transient or attempt == max_retries - 1:
                raise
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Transient error on attempt {attempt + 1}/{max_retries}, retrying in {wait_time:.1f}s: {e}")
            time.sleep(wait_time)
    raise last_error

load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://virtual-assistant-pi-seven.vercel.app",
        "http://localhost:8000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
CUSTOMER_API_KEY = os.getenv("CUSTOMER_API_KEY")

with open("settings.json", "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

ASSISTANT_NAME = SETTINGS["assistant_name"]
GREETING = SETTINGS["greeting"]
FAQ = SETTINGS["faq"]

try:
    db.init_db()
    db.migrate_v2_sales_agent()
    db.migrate_products_from_json("products.json")
    DB_AVAILABLE = True
except Exception as e:
    logger.error(f"Database unavailable at startup: {e}")
    DB_AVAILABLE = False

PERSONALITY_STYLES = {
    "friendly": "warm, approachable, and easygoing — like chatting with a helpful friend who happens to work at the store",
    "professional": "polished, courteous, and efficient — clear and to the point, like a knowledgeable retail associate",
    "luxury": "refined, attentive, and a little indulgent — the tone of a high-end boutique consultant who makes the customer feel valued",
    "energetic": "upbeat, enthusiastic, and quick to get excited about products — lots of positive energy without being over the top",
    "simple": "plain, direct, and easy to follow — short sentences, no jargon, no fluff",
    "persuasive": "confident and compelling — skilled at highlighting value and gently guiding the customer toward a decision",
}

def build_system_prompt(relevant_chunk: str, memory_context: str) -> str:
    biz = db.get_business_settings() if DB_AVAILABLE else None
    ai_name = (biz and biz.get("ai_name")) or ASSISTANT_NAME
    business_name = (biz and biz.get("name")) or ASSISTANT_NAME
    personality_key = (biz and biz.get("personality")) or "friendly"
    personality_desc = PERSONALITY_STYLES.get(personality_key, PERSONALITY_STYLES["friendly"])
    custom_instructions = (biz and biz.get("custom_instructions")) or ""
    supported_languages = (biz and biz.get("supported_languages")) or "en,ur,ps"
    lang_names = {"en": "English", "ur": "Urdu", "ps": "Pashto"}
    languages_list = ", ".join(lang_names.get(code.strip(), code.strip()) for code in supported_languages.split(",") if code.strip())
    prompt = (
        f"You are {ai_name}, an AI sales assistant for {business_name}, built by Jawad — "
        f"not by Google or any other company. If asked who made you, who you work for, or "
        f"what you are, always say you were built by Jawad.\n\n"
        f"PERSONALITY: Your tone is {personality_desc}.\n\n"
        f"LANGUAGE: You support {languages_list}. Detect the language the customer writes in "
        f"and reply naturally in that same language.\n\n"
        f"YOUR JOB: You are a skilled salesperson. Understand what customer wants, ask smart follow-up questions when needed. When you recommend products, explain why they're a good fit, mention color, size, price, discount, stock.\n\n"
        f"GROUNDING — CRITICAL: You can look up real products using the search_products tool. Only ever mention products that came back from that tool. Never invent.\n\n"
        f"IMAGES: If has_image is true, say picture will appear right here in chat. If false, say no photo available.\n\n"
        f"HUMAN HANDOFF: If customer asks for real person, call request_human_tool.\n\n"
        f"TAKING ORDERS: Collect details step by step, call create_order_tool, show summary, ask confirm, then confirm_order_tool.\n\n"
        f"Keep replies natural and conversational."
    )
    if custom_instructions:
        prompt += f"\n\nADDITIONAL INSTRUCTIONS FROM THE BUSINESS OWNER:\n{custom_instructions}"
    prompt += f"\n\nHere is relevant background information:\n\"{relevant_chunk}\""
    if memory_context:
        prompt += f"\n\nWhat you remember about this customer:\n{memory_context}"
    return prompt

def get_today_date() -> str:
    return date.today().isoformat()

def lookup_faq(question: str) -> str:
    question = question.lower().strip()
    for key in FAQ:
        if key in question:
            return FAQ[key]
    return "I don't have a saved answer for that."

def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=%C+%t"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.read().decode("utf-8")
    except Exception:
        return "Sorry, I couldn't fetch the weather right now."

def search_products_tool(keyword: str = None, category: str = None, max_price: float = None, min_price: float = None, color: str = None) -> str:
    global LAST_SEARCH_RESULTS
    if not DB_AVAILABLE:
        return json.dumps([])
    try:
        results = db.search_products(keyword=keyword, category=category, max_price=max_price, min_price=min_price, color=color)
        LAST_SEARCH_RESULTS = results
        trimmed = [
            {
                "product_id": r["product_id"],
                "name": r["product_name"],
                "category": r["category"],
                "price": r["price"],
                "discount_price": r.get("discount_price"),
                "currency": r["currency"],
                "color": r["color"],
                "size": r["size"],
                "stock": r["stock"],
                "brand": r.get("brand"),
                "description": r["description"],
                "has_image": bool(r.get("image_url") or r.get("images")),
                "has_video": bool(r.get("video_url") or r.get("videos")),
                "image_url": r.get("image_url"),
                "video_url": r.get("video_url"),
            }
            for r in results
        ]
        return json.dumps(trimmed)
    except Exception as e:
        logger.error(f"Product search failed: {e}")
        return json.dumps([])

def find_product_video(message: str):
    if LAST_SEARCH_RESULTS:
        for p in LAST_SEARCH_RESULTS:
            if p.get("video_url"):
                return p.get("video_url")
    return None

def find_product_image(message: str):
    if LAST_SEARCH_RESULTS:
        for p in LAST_SEARCH_RESULTS:
            if p.get("image_url"):
                return p.get("image_url")
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
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]
    best_index = similarities.argmax()
    return DOCUMENT_CHUNKS[best_index]

def build_memory_context(user_id: str) -> str:
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
    if not DB_AVAILABLE or not conversation_id:
        return []
    try:
        recent = db.get_recent_messages(conversation_id, limit=8)
        history = []
        for m in recent:
            role = "user" if m["sender"] == "user" else "model"
            history.append(types.Content(role=role, parts=[types.Part(text=m["message"])]))
        return history
    except Exception as e:
        logger.error(f"History lookup failed: {e}")
        return []

def maybe_update_memory(user_id: str, conversation_id: str):
    if not DB_AVAILABLE or not user_id or user_id == "anonymous" or not conversation_id:
        return
    try:
        total = db.count_messages(conversation_id)
        if total == 0 or total % 6!= 0:
            return
        recent = db.get_recent_messages(conversation_id, limit=20)
        transcript = "\n".join(f"{m['sender']}: {m['message']}" for m in recent)
        existing = db.get_memory(user_id) or {}
        prompt = (
            "You are updating a short internal memory file about a customer. Output a JSON object with keys: \"summary\", \"preferences\", \"important_info\". Reply with ONLY JSON.\n\n"
            f"Existing notes:\nsummary: {existing.get('summary', '')}\npreferences: {existing.get('preferences', '')}\nimportant_info: {existing.get('important_info', '')}\n\nRecent conversation:\n{transcript}"
        )
        response = generate_content_with_retry(model=FAST_MODEL, contents=prompt)
        raw = response.text.strip().strip("`").lstrip("json").strip()
        parsed = json.loads(raw)
        db.upsert_memory(user_id, preferences=parsed.get("preferences"), important_info=parsed.get("important_info"), summary=parsed.get("summary"))
    except Exception as e:
        logger.error(f"Memory update skipped: {e}")

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
    discount_price: Optional[float] = None
    images: Optional[str] = None
    videos: Optional[str] = None
    brand: Optional[str] = None

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
    discount_price: Optional[float] = None
    images: Optional[str] = None
    videos: Optional[str] = None
    brand: Optional[str] = None

class BusinessSettingsUpdate(BaseModel):
    ai_name: Optional[str] = None
    name: Optional[str] = None
    logo_url: Optional[str] = None
    personality: Optional[str] = None
    custom_instructions: Optional[str] = None
    supported_languages: Optional[str] = None
    notifications_enabled: Optional[bool] = None
    follow_up_enabled: Optional[bool] = None

def check_api_key(x_api_key: str):
    if x_api_key!= APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

def check_customer_api_key(x_api_key: str):
    if x_api_key!= CUSTOMER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

def ensure_conversation(user_id: str, conversation_id: Optional[str]) -> str:
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
    biz = db.get_business_settings() if DB_AVAILABLE else None
    ai_name = (biz and biz.get("ai_name")) or ASSISTANT_NAME
    return {"assistant_name": ai_name, "greeting": GREETING}

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

TOOL_FUNCTIONS = {
    "get_today_date": get_today_date,
    "lookup_faq": lookup_faq,
    "get_weather": get_weather,
    "search_products_tool": search_products_tool,
}

LEAD_SIGNAL_POINTS = {
    "asked_price": 10,
    "asked_stock": 10,
    "asked_delivery": 8,
    "asked_payment": 8,
    "provided_contact_info": 15,
}

def make_lead_scoring_tool(user_id: str, conversation_id: str = None):
    def log_customer_signal_tool(signal: str) -> str:
        if not DB_AVAILABLE or signal not in LEAD_SIGNAL_POINTS:
            return json.dumps({"ok": False})
        try:
            before = db.get_customer(user_id)
            old_status = before.get("lead_status") if before else "new"
            new_score = db.adjust_lead_score(user_id, LEAD_SIGNAL_POINTS[signal])
            _notify_if_became_hot(user_id, conversation_id, old_status, new_score)
            return json.dumps({"ok": True, "signal": signal, "new_score": new_score})
        except Exception as e:
            logger.error(f"Lead scoring failed: {e}")
            return json.dumps({"ok": False})
    return log_customer_signal_tool

def _notify_if_became_hot(user_id: str, conversation_id: str, old_status: str, new_score: int):
    new_status = db._lead_status_for_score(new_score)
    if old_status!= "hot" and new_status == "hot":
        try:
            customer = db.get_customer(user_id) or {}
            db.create_notification(
                user_id=user_id, conversation_id=conversation_id, priority="hot",
                title="NEW HOT CUSTOMER LEAD 🔥",
                message=f"{customer.get('name') or 'A customer'} (score: {new_score}) is showing strong buying signals.",
            )
        except Exception as e:
            logger.error(f"Could not create hot-lead notification: {e}")

def make_order_tools(user_id: str, conversation_id: str):
    def create_order_tool(product_id: str, color: str = None, size: str = None, quantity: int = 1, customer_name: str = None, customer_phone: str = None, delivery_address: str = None, payment_method: str = None) -> str:
        if not DB_AVAILABLE:
            return json.dumps({"error": "Ordering is temporarily unavailable."})
        product = db.get_product(product_id)
        if not product:
            return json.dumps({"error": "Product not found"})
        price = product.get("discount_price") or product.get("price")
        quantity = quantity or 1
        order_id = db.create_order(
            user_id=user_id, conversation_id=conversation_id,
            product_id=product_id, product_name=product["product_name"],
            color=color or product.get("color"), size=size or product.get("size"),
            quantity=quantity, price=price,
            customer_name=customer_name, customer_phone=customer_phone,
            delivery_address=delivery_address, payment_method=payment_method,
            status="order_pending",
        )
        total_price = (price or 0) * quantity
        return json.dumps({
            "order_id": order_id,
            "product_name": product["product_name"],
            "quantity": quantity,
            "unit_price": price,
            "total_price": total_price,
            "currency": product.get("currency", "PKR"),
            "status": "order_pending",
        })
    def confirm_order_tool(order_id: str) -> str:
        if not DB_AVAILABLE:
            return json.dumps({"error": "Ordering is temporarily unavailable."})
        ok = db.update_order_status(order_id, "order_confirmed")
        if not ok:
            return json.dumps({"error": "Could not find that order."})
        return json.dumps({"order_id": order_id, "status": "order_confirmed"})
    return create_order_tool, confirm_order_tool

def make_human_handoff_tool(user_id: str, conversation_id: str):
    def request_human_tool(reason: str) -> str:
        if DB_AVAILABLE and conversation_id:
            try:
                db.flag_requires_human(conversation_id)
                customer = db.get_customer(user_id) or {}
                db.create_notification(
                    user_id=user_id, conversation_id=conversation_id, priority="human_help",
                    title="⚠️ HUMAN HELP REQUIRED",
                    message=f"{customer.get('name') or 'A customer'} needs human assistance: {reason}",
                )
            except Exception as e:
                logger.error(f"Could not flag human handoff: {e}")
        return json.dumps({"ok": True})
    return request_human_tool

def generate_reply(message: str, user_id: str = None, conversation_id: str = None) -> str:
    global LAST_SEARCH_RESULTS
    LAST_SEARCH_RESULTS = []
    if DB_AVAILABLE and conversation_id:
        convo = db.get_conversation(conversation_id)
        if convo and convo.get("mode") == "human":
            return None
    relevant_chunk = search_document(message)
    memory_context = build_memory_context(user_id) if user_id else ""
    full_system_prompt = build_system_prompt(relevant_chunk, memory_context)
    create_order_tool, confirm_order_tool = make_order_tools(user_id or "anonymous", conversation_id)
    log_customer_signal_tool = make_lead_scoring_tool(user_id or "anonymous", conversation_id)
    request_human_tool = make_human_handoff_tool(user_id or "anonymous", conversation_id)
    tool_functions = dict(TOOL_FUNCTIONS)
    tool_functions["create_order_tool"] = create_order_tool
    tool_functions["confirm_order_tool"] = confirm_order_tool
    tool_functions["log_customer_signal_tool"] = log_customer_signal_tool
    tool_functions["request_human_tool"] = request_human_tool
    history = build_recent_history(conversation_id) if conversation_id else []
    if history:
        contents = history + [types.Content(role="user", parts=[types.Part(text=message)])]
    else:
        contents = [types.Content(role="user", parts=[types.Part(text=message)])]
    config = types.GenerateContentConfig(
        system_instruction=full_system_prompt,
        tools=[get_today_date, lookup_faq, get_weather, search_products_tool, create_order_tool, confirm_order_tool, log_customer_signal_tool, request_human_tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    response = generate_content_with_retry(model=FAST_MODEL, contents=contents, config=config)
    max_turns = 4
    turns = 0
    while response.function_calls and turns < max_turns:
        turns += 1
        contents.append(response.candidates[0].content)
        function_response_parts = []
        for fc in response.function_calls:
            func = tool_functions.get(fc.name)
            if func:
                try:
                    result = func(**(fc.args or {}))
                except Exception as e:
                    result = f"Error: {e}"
            else:
                result = f"Unknown tool: {fc.name}"
            function_response_parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=function_response_parts))
        response = generate_content_with_retry(model=FAST_MODEL, contents=contents, config=config)
    return response.text

def text_to_speech(text: str) -> str:
    response = generate_content_with_retry(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore"))
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
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@app.post("/chat")
def chat(request: ChatRequest, background_tasks: BackgroundTasks, x_api_key: str = Header(None)):
    check_customer_api_key(x_api_key)
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    user_id = request.user_id or "anonymous"
    conversation_id = None
    if DB_AVAILABLE:
        try:
            conversation_id = ensure_conversation(user_id, request.conversation_id)
            db.save_message(conversation_id, "user", request.message, "text")
        except Exception as e:
            logger.error(f"Could not persist incoming: {e}")
            conversation_id = None
    try:
        reply = generate_reply(request.message, user_id=user_id, conversation_id=conversation_id)
    except Exception as e:
        logger.error(f"Error during chat: {e}")
        return {"reply": "Sorry, I'm having trouble responding right now."}
    if reply is None:
        return {"reply": None, "conversation_id": conversation_id, "handed_off": True}
    if DB_AVAILABLE and conversation_id:
        try:
            db.save_message(conversation_id, "assistant", reply, "text")
            background_tasks.add_task(maybe_update_memory, user_id, conversation_id)
        except Exception as e:
            logger.error(f"Could not persist reply: {e}")
    result = {"reply": reply, "conversation_id": conversation_id}
    if LAST_SEARCH_RESULTS:
        first = LAST_SEARCH_RESULTS[0]
        if first.get("image_url"):
            result["image_url"] = first.get("image_url")
        if first.get("video_url"):
            v = first.get("video_url")
            result["video_url"] = f"/videos/{v}" if not str(v).startswith("/videos/") and not str(v).startswith("http") else v
    return result

@app.post("/voice-chat")
def voice_chat(request: VoiceRequest, background_tasks: BackgroundTasks, x_api_key: str = Header(None)):
    check_customer_api_key(x_api_key)
    user_id = request.user_id or "anonymous"
    conversation_id = None
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid audio data")
    try:
        transcription_response = generate_content_with_retry(model=FAST_MODEL, contents=[types.Part.from_bytes(data=audio_bytes, mime_type=request.mime_type), "Transcribe this audio to text. Reply with ONLY the transcribed text."])
        transcript = transcription_response.text.strip()
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {"transcript": "", "reply": "Sorry, I couldn't process that voice message."}
    if DB_AVAILABLE:
        try:
            conversation_id = ensure_conversation(user_id, request.conversation_id)
            db.save_message(conversation_id, "user", transcript, "voice")
        except Exception as e:
            conversation_id = None
    try:
        reply = generate_reply(transcript, user_id=user_id, conversation_id=conversation_id)
        reply_audio = text_to_speech(reply)
    except Exception as e:
        logger.error(f"Voice reply error: {e}")
        return {"transcript": transcript, "reply": "Sorry, I couldn't process that voice message."}
    if DB_AVAILABLE and conversation_id:
        try:
            db.save_message(conversation_id, "assistant", reply, "voice")
            background_tasks.add_task(maybe_update_memory, user_id, conversation_id)
        except Exception as e:
            pass
    result = {"transcript": transcript, "reply": reply, "reply_audio": reply_audio, "conversation_id": conversation_id}
    if LAST_SEARCH_RESULTS:
        first = LAST_SEARCH_RESULTS[0]
        if first.get("image_url"):
            result["image_url"] = first.get("image_url")
        if first.get("video_url"):
            v = first.get("video_url")
            result["video_url"] = f"/videos/{v}" if not str(v).startswith("/videos/") and not str(v).startswith("http") else v
    return result

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
def get_conversation_messages(conversation_id: str, user_id: str, limit: int = Query(50, le=200), offset: int = 0):
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    convo = db.get_conversation(conversation_id)
    if not convo or convo["user_id"]!= user_id:
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
        rows = conn.execute("SELECT c.*, cu.name AS customer_name, cu.email, cu.phone FROM conversations c LEFT JOIN customers cu ON cu.user_id = c.user_id WHERE c.is_deleted = 0 ORDER BY c.updated_at DESC LIMIT?", (limit,)).fetchall()
        return [dict(r) for r in rows]

@app.get("/admin/conversations/{conversation_id}/messages")
def admin_get_conversation_messages(conversation_id: str, x_api_key: str = Header(None), limit: int = Query(200, le=500)):
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
        rows = conn.execute("SELECT * FROM customers ORDER BY updated_at DESC").fetchall()
        return [dict(r) for r in rows]

@app.get("/admin/business-settings")
def admin_get_business_settings(x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    settings = db.get_business_settings()
    if not settings:
        raise HTTPException(status_code=404, detail="Business not found")
    return settings

@app.put("/admin/business-settings")
def admin_update_business_settings(settings: BusinessSettingsUpdate, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    updates = {k: v for k, v in settings.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields provided to update")
    if "notifications_enabled" in updates:
        updates["notifications_enabled"] = int(updates["notifications_enabled"])
    if "follow_up_enabled" in updates:
        updates["follow_up_enabled"] = int(updates["follow_up_enabled"])
    ok = db.update_business_settings(**updates)
    if not ok:
        raise HTTPException(status_code=400, detail="Update failed")
    return db.get_business_settings()

class OrderStatusUpdate(BaseModel):
    status: str

@app.get("/admin/orders")
def admin_list_orders(x_api_key: str = Header(None), status: Optional[str] = None, limit: int = Query(50, le=200)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return db.list_orders(status=status, limit=limit)

@app.put("/admin/orders/{order_id}")
def admin_update_order_status(order_id: str, update: OrderStatusUpdate, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    ok = db.update_order_status(order_id, update.status)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Invalid order_id or status. Valid statuses: {db.ORDER_STATUSES}")
    return {"updated": True}

@app.get("/admin/leads")
def admin_list_leads(x_api_key: str = Header(None), status: Optional[str] = None, limit: int = Query(50, le=200)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return db.list_leads(status=status, limit=limit)

@app.get("/admin/notifications")
def admin_list_notifications(x_api_key: str = Header(None), unread_only: bool = False, limit: int = Query(50, le=200)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return db.list_notifications(unread_only=unread_only, limit=limit)

@app.put("/admin/notifications/{notification_id}/read")
def admin_mark_notification_read(notification_id: str, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    ok = db.mark_notification_read(notification_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"updated": True}

class ConversationModeUpdate(BaseModel):
    mode: str

@app.put("/admin/conversations/{conversation_id}/mode")
def admin_set_conversation_mode(conversation_id: str, update: ConversationModeUpdate, x_api_key: str = Header(None)):
    check_api_key(x_api_key)
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    ok = db.set_conversation_mode(conversation_id, update.mode)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid conversation_id or mode")
    return {"updated": True, "mode": update.mode}

@app.get("/index.html")
def serve_widget():
    return FileResponse("index.html", headers={"Cache-Control": "no-store, must-revalidate"})

@app.get("/")
def serve_widget_root():
    return FileResponse("index.html", headers={"Cache-Control": "no-store, must-revalidate"})

@app.get("/app.js")
def serve_app_js():
    return FileResponse("app.js", media_type="text/javascript", headers={"Cache-Control": "no-store, must-revalidate"})

app.mount("/", StaticFiles(directory=".", html=True), name="static")