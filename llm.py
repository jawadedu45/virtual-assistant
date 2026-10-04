"""
llm.py - the ONLY file that talks to the AI providers.

Order tried for every chat message:
    1) Groq  (openai/gpt-oss-120b)  - fast
    2) Gemini (gemini-3.5-flash)    - backup
    3) Groq  (openai/gpt-oss-20b)   - last-resort backup

If a provider fails (rate limit, outage, timeout...), the next one is used
automatically. A provider that just failed is skipped for a short time
("cooldown") so the next customers don't wait for it to fail again.
"""
import os
import json
import time
import inspect
import logging

from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import types

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Old llama-3.3-70b-versatile / llama-3.1-8b-instant were retired -> use GPT-OSS models.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GROQ_STT_MODEL = os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo")

# GPT-OSS and Gemini 3.x "think" before answering. Thinking uses up tokens,
# so the limits below are higher than before, otherwise the visible answer
# can come back empty.
GROQ_MAX_TOKENS = 2000
GEMINI_MAX_TOKENS = 2048

# max_retries=0 -> do NOT wait and retry inside the SDK; we switch provider instead.
groq_client = Groq(api_key=GROQ_API_KEY, timeout=25.0, max_retries=0) if GROQ_API_KEY else None
gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY, http_options=types.HttpOptions(timeout=40_000))
    if GEMINI_API_KEY else None
)

if not groq_client and not gemini_client:
    logger.error("No AI keys found. Set GROQ_API_KEY and/or GEMINI_API_KEY in .env")


# --------------------------------------------------------------------------
# Cooldown: skip a provider for a bit after it fails
# --------------------------------------------------------------------------
_cooldown_until = {}


def _mark_failed(candidate, error):
    s = str(error).lower()
    rate_limited = any(x in s for x in ("429", "rate", "quota", "resource_exhausted"))
    # A 404 / model_not_found will not fix itself quickly, so rest longer.
    not_found = "404" in s or "model_not_found" in s
    if rate_limited:
        wait = 60
    elif not_found:
        wait = 300
    else:
        wait = 15
    _cooldown_until[candidate] = time.time() + wait


def _order(candidates):
    """Providers that are ready first; ones in cooldown go last (still tried as a last resort)."""
    now = time.time()
    ready = [c for c in candidates if _cooldown_until.get(c, 0) <= now]
    waiting = [c for c in candidates if c not in ready]
    return ready + waiting


def _chat_candidates():
    c = []
    if groq_client:
        c.append(("groq", GROQ_MODEL))
    if gemini_client:
        c.append(("gemini", GEMINI_MODEL))
    if groq_client:
        c.append(("groq", GROQ_FAST_MODEL))
    return c


def _text_candidates():
    c = []
    if groq_client:
        c.append(("groq", GROQ_FAST_MODEL))
    if gemini_client:
        c.append(("gemini", GEMINI_MODEL))
    if groq_client:
        c.append(("groq", GROQ_MODEL))
    return c


# --------------------------------------------------------------------------
# Tool runner: runs the real Python tool functions.
# Results are cached for ONE customer message, so if we have to switch
# provider half-way, an order is never created twice.
# --------------------------------------------------------------------------
def make_tool_runner(tool_functions):
    cache = {}

    def run_tool(name, args):
        func = tool_functions.get(name)
        if not func:
            return f"Unknown tool: {name}"

        args = {k: v for k, v in (args or {}).items() if v is not None and v != ""}
        try:
            sig = inspect.signature(func)
            for k, v in list(args.items()):
                param = sig.parameters.get(k)
                if param is None:
                    args.pop(k)
                    continue
                try:
                    if param.annotation is int:
                        args[k] = int(float(v))
                    elif param.annotation is float:
                        args[k] = float(v)
                except (TypeError, ValueError):
                    args.pop(k)
        except (TypeError, ValueError):
            pass

        key = name + json.dumps(args, sort_keys=True, default=str)
        if key in cache:
            return cache[key]

        try:
            logger.info(f"TOOL CALL: {name} {args}")
            result = str(func(**args))
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            result = f"Error running {name}: {e}"
        cache[key] = result
        return result

    return run_tool


# --------------------------------------------------------------------------
# Groq chat (OpenAI-style messages + tools)
# --------------------------------------------------------------------------
def _run_groq(model, system_prompt, history, tool_schemas, run_tool, max_turns):
    messages = [{"role": "system", "content": system_prompt}] + history

    def ask():
        return groq_client.chat.completions.create(
            model=model, messages=messages, tools=tool_schemas,
            tool_choice="auto", temperature=0.4, max_tokens=GROQ_MAX_TOKENS,
            # Less "thinking" = faster replies (GPT-OSS models support this).
            extra_body={"reasoning_effort": "low"},
        ).choices[0].message

    msg = ask()
    turns = 0
    while msg.tool_calls and turns < max_turns:
        turns += 1
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = run_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        msg = ask()
    return msg.content


# --------------------------------------------------------------------------
# Gemini chat (Content/Part objects + python functions as tools)
# --------------------------------------------------------------------------
def _run_gemini(model, system_prompt, history, tool_callables, run_tool, max_turns):
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part(text=m["content"])],
        )
        for m in history
    ]
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tool_callables,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        max_output_tokens=GEMINI_MAX_TOKENS,
    )

    def log_if_empty(resp):
        # Helps find out WHY Gemini returned nothing (safety block, token limit...)
        try:
            if not resp.function_calls and not (resp.text or "").strip():
                cand = resp.candidates[0] if resp.candidates else None
                logger.warning(
                    f"Gemini empty reply. finish_reason="
                    f"{getattr(cand, 'finish_reason', None)} "
                    f"prompt_feedback={getattr(resp, 'prompt_feedback', None)}"
                )
        except Exception:
            pass

    response = gemini_client.models.generate_content(model=model, contents=contents, config=config)
    turns = 0
    while response.function_calls and turns < max_turns:
        turns += 1
        contents.append(response.candidates[0].content)
        parts = []
        for fc in response.function_calls:
            result = run_tool(fc.name, dict(fc.args or {}))
            parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
        contents.append(types.Content(role="user", parts=parts))
        response = gemini_client.models.generate_content(model=model, contents=contents, config=config)
    log_if_empty(response)
    return response.text


# --------------------------------------------------------------------------
# Public functions used by server.py
# --------------------------------------------------------------------------
def chat_with_tools(system_prompt, history, tool_schemas, tool_callables, tool_functions, max_turns=5):
    """Get the chatbot's reply. history = [{"role": "user"|"assistant", "content": "..."}]"""
    run_tool = make_tool_runner(tool_functions)
    errors = []

    for candidate in _order(_chat_candidates()):
        provider, model = candidate
        try:
            if provider == "groq":
                text = _run_groq(model, system_prompt, history, tool_schemas, run_tool, max_turns)
            else:
                text = _run_gemini(model, system_prompt, history, tool_callables, run_tool, max_turns)
            if text and text.strip():
                logger.info(f"Reply produced by {provider}/{model}")
                return text
            raise ValueError("empty reply")
        except Exception as e:
            logger.warning(f"{provider}/{model} failed, trying next provider: {e}")
            errors.append(f"{provider}/{model}: {e}")
            _mark_failed(candidate, e)

    raise RuntimeError("All AI providers failed -> " + " | ".join(errors))


def generate_text(prompt, json_mode=False):
    """Plain one-shot text generation (used for the customer-memory update)."""
    errors = []
    for candidate in _order(_text_candidates()):
        provider, model = candidate
        try:
            if provider == "groq":
                kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
                r = groq_client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    temperature=0.2, max_tokens=GROQ_MAX_TOKENS,
                    extra_body={"reasoning_effort": "low"}, **kwargs,
                )
                text = r.choices[0].message.content
            else:
                cfg = types.GenerateContentConfig(
                    max_output_tokens=GEMINI_MAX_TOKENS,
                    **({"response_mime_type": "application/json"} if json_mode else {}),
                )
                r = gemini_client.models.generate_content(model=model, contents=prompt, config=cfg)
                text = r.text
            if text and text.strip():
                return text
            raise ValueError("empty reply")
        except Exception as e:
            logger.warning(f"{provider}/{model} failed for text generation: {e}")
            errors.append(f"{provider}/{model}: {e}")
            _mark_failed(candidate, e)
    raise RuntimeError("All AI providers failed -> " + " | ".join(errors))


def transcribe_audio(audio_bytes, mime_type):
    """Voice message -> text. Tries Groq Whisper first, then Gemini."""
    base_mime = (mime_type or "audio/webm").split(";")[0]
    ext = base_mime.split("/")[-1] or "webm"
    errors = []

    if groq_client:
        try:
            result = groq_client.audio.transcriptions.create(
                file=(f"audio.{ext}", audio_bytes), model=GROQ_STT_MODEL,
            )
            return result.text.strip()
        except Exception as e:
            logger.warning(f"Groq transcription failed, trying Gemini: {e}")
            errors.append(f"groq: {e}")

    if gemini_client:
        try:
            r = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=base_mime),
                    "Transcribe this audio to text. Reply with ONLY the transcribed text, nothing else.",
                ],
            )
            return (r.text or "").strip()
        except Exception as e:
            errors.append(f"gemini: {e}")

    raise RuntimeError("Transcription failed -> " + " | ".join(errors))