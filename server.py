from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

BASE_SYSTEM_PROMPT = (
    f"You are {ASSISTANT_NAME}, built by Jawad — not by Google or any other company. "
    f"If asked who made you, who you work for, or what you are, always say you were built by Jawad. "
    f"Keep replies short, casual, and natural — like a normal conversation, not a formal AI response. "
    f"Detect the language the user writes in — English, Urdu, or Pashto — and always reply in that same language."
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

class ChatRequest(BaseModel):
    message: str

@app.get("/api/status")
def read_root():
    return {"message": "Hello, your server is running!"}

@app.get("/settings")
def get_settings():
    return {"assistant_name": ASSISTANT_NAME, "greeting": GREETING}

@app.post("/chat")
def chat(request: ChatRequest, x_api_key: str = Header(None)):
    if x_api_key != APP_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    logger.info(f"Incoming message: {request.message}")

    relevant_chunk = search_document(request.message)

    full_system_prompt = (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"Here is relevant background information you can use if it helps answer the question:\n"
        f"\"{relevant_chunk}\""
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=request.message,
            config=types.GenerateContentConfig(
                system_instruction=full_system_prompt,
                tools=[get_today_date, lookup_faq, get_weather]
            )
        )
        return {"reply": response.text}
    except Exception as e:
        logger.error(f"Error during chat: {e}")
        return {"reply": "Sorry, I'm having trouble responding right now. Please try again in a moment."}

# --- Serve the chat widget (index.html) ---
# This must be registered AFTER all your API routes above,
# otherwise it can override them.
@app.get("/index.html")
def serve_widget():
    return FileResponse("index.html")

@app.get("/")
def serve_widget_root():
    return FileResponse("index.html")

# Serves any other files sitting in the same folder (CSS, JS, images)
# e.g. a request for /style.css or /widget.js will be found here automatically.
app.mount("/", StaticFiles(directory=".", html=True), name="static")
