import asyncio
import json
import logging
import os
import re
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

from openai import OpenAI
from dotenv import load_dotenv

from better_profanity import profanity
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("classroom-monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dashboard_snapshot: List[Dict[str, Any]] = []
flagged_events: List[Dict[str, Any]] = []
session_meta: Dict[str, str] = {"topic": "", "context": "", "allowed_language": ""}
history_lock = asyncio.Lock()
analysis_lock = asyncio.Lock()
connections: set[WebSocket] = set()
connections_lock = asyncio.Lock()
flagged_tokens_by_speaker: Dict[Any, set[str]] = defaultdict(set)
last_analysis_result: Optional[Dict[str, Any]] = None

# Load environment variables from .env if present
load_dotenv()

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://phi4:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("LLM_MODEL_NAME", "phi4"))

ANALYSIS_PROVIDER = os.getenv("ANALYSIS_PROVIDER", "openrouter").lower()
ENABLE_LOCAL_LLM_FALLBACK = os.getenv("ENABLE_LOCAL_LLM_FALLBACK", "1") not in {"0", "false", "False"}

def _read_openrouter_key() -> Optional[str]:
    if os.getenv("OPENROUTER_API_KEY"):
        return os.getenv("OPENROUTER_API_KEY")
    key_path = Path(os.getenv("OPENROUTER_KEY_FILE", "openrouterkey"))
    if key_path.exists():
        try:
            return key_path.read_text().strip()
        except Exception:
            return None
    return None

OPENROUTER_API_KEY = _read_openrouter_key()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "")

llm_client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
openrouter_client = None
if OPENROUTER_API_KEY:
    default_headers = {}
    if OPENROUTER_REFERER:
        default_headers["HTTP-Referer"] = OPENROUTER_REFERER
    if OPENROUTER_APP_TITLE:
        default_headers["X-Title"] = OPENROUTER_APP_TITLE
    openrouter_client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        default_headers=default_headers or None,
    )

profanity.load_censor_words()


def utc_now_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def extract_flagged_words(text: str) -> List[str]:
    lowered = text.lower()

    def _check(candidate: str) -> bool:
        if not candidate:
            return False
        return profanity.contains_profanity(candidate)

    flagged: set[str] = set()

    # Basic word-level check.
    words = re.findall(r"[\w'*]+", lowered)
    for w in words:
        if _check(w):
            flagged.add(w)
        # Handle simple obfuscations like f*ck, f@ck by normalizing common separators.
        deobfuscated = re.sub(r"[*@#$%^&()_+=\\-]", "", w)
        if deobfuscated and _check(deobfuscated):
            flagged.add(deobfuscated)
        star_u = w.replace("*", "u")
        if star_u != w and _check(star_u):
            flagged.add(star_u)

    # Whole-string compact check to catch cases like f*ck -> fck
    compact = re.sub(r"[^a-z]", "", lowered)
    if _check(compact):
        flagged.add(compact)

    return sorted(flagged)


async def register_websocket(ws: WebSocket) -> None:
    async with connections_lock:
        connections.add(ws)


async def unregister_websocket(ws: WebSocket) -> None:
    async with connections_lock:
        connections.discard(ws)


async def broadcast_json(payload: Dict[str, Any]) -> None:
    """Send a JSON payload to all active websocket connections."""
    async with connections_lock:
        targets = list(connections)
    stale: list[WebSocket] = []
    for ws in targets:
        try:
            await ws.send_json(payload)
        except Exception:
            stale.append(ws)
    if stale:
        async with connections_lock:
            for ws in stale:
                connections.discard(ws)


def format_transcript_for_llm(snapshot: List[Dict[str, Any]], max_lines: int = 80, max_chars: int = 4000) -> str:
    """Reduce transcript size to keep LLM context light and avoid OOMs."""
    lines: List[str] = []
    # Use the most recent messages to keep context tight
    recent = snapshot[-max_lines:] if len(snapshot) > max_lines else snapshot
    for idx, item in enumerate(recent):
        speaker = item.get("speaker", "?")
        text = (item.get("text") or "").strip()
        translation = (item.get("translation") or "").strip()
        detected_language = (item.get("detected_language") or "").strip()
        start = item.get("start")
        end = item.get("end")
        timestamp = item.get("timestamp") or ""
        line_parts = [f"S{speaker}: {text}"]
        if translation and translation.lower() != text.lower():
            line_parts.append(f"[translation: {translation}]")
        meta_parts = []
        if detected_language:
            meta_parts.append(f"lang={detected_language}")
        if start is not None and end is not None:
            meta_parts.append(f"{start}-{end}")
        if timestamp:
            meta_parts.append(f"ts={timestamp}")
        suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""
        lines.append(" ".join(line_parts) + suffix)
        joined = "\n".join(lines)
        if len(joined) > max_chars:
            lines.append("...truncated for brevity...")
            break
    return "\n".join(lines) if lines else "No transcript lines provided."


def _call_llm(provider: str, prompt: str, meta: dict) -> dict:
    """Call a specific provider and return a standardized result dict. Runs in a thread."""
    messages = [
        {"role": "system", "content": "Provide concise, factual classroom conversation insights. Do not add extra sections."},
        {"role": "user", "content": prompt},
    ]

    if provider == "openrouter":
        if not openrouter_client:
            raise RuntimeError("OpenRouter client not configured. Set OPENROUTER_API_KEY or OPENROUTER_KEY_FILE.")
        response = openrouter_client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=256,
        )
        content = response.choices[0].message.content if response.choices else ""
        return {
            "analysis": content.strip(),
            "timestamp": utc_now_iso(),
            "meta": meta,
            "model": OPENROUTER_MODEL,
            "provider": "openrouter",
        }

    if provider == "local":
        response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=256,
            extra_body={"options": {"num_ctx": 1024}},
        )
        content = response.choices[0].message.content if response.choices else ""
        return {
            "analysis": content.strip(),
            "timestamp": utc_now_iso(),
            "meta": meta,
            "model": LLM_MODEL,
            "provider": "local",
        }

    raise ValueError(f"Unknown provider {provider}")


async def run_llm_analysis(snapshot, meta):
    """Call the configured LLM (OpenRouter primary, local fallback) for structured analysis."""
    if not snapshot:
        return None

    prompt = f"""You are an educational conversation auditor. Use the transcript and metadata to create a concise report.
Return the output exactly in this format (keep bullets and line breaks):

1. TOPIC ALIGNMENT SCORE (0-10):
   - Give a score for the group's focus.
   - List 2 specific examples where speakers drifted OFF topic (if any).

2. PARTICIPATION BALANCE:
   - Identify the "Dominant Speaker" (who spoke too much).
   - Identify the "Passive Speaker" (who contributed least).
   - Provide a brief critique of the interaction flow (e.g., "Speaker A interrupted frequently").

3. CONCLUSION:
   - One sentence summary of the discussion quality.

Consider the provided topic/context; if off-topic examples are absent, say "No clear off-topic examples." Avoid fabricating content not supported by the transcript.

Session metadata:
- Topic: {meta.get('topic') or 'N/A'}
- Context: {meta.get('context') or 'N/A'}
- Allowed language: {meta.get('allowed_language') or 'auto'}

Transcript:
{format_transcript_for_llm(snapshot)}"""

    primary = ANALYSIS_PROVIDER
    fallbacks = []
    if ENABLE_LOCAL_LLM_FALLBACK and primary != "local":
        fallbacks.append("local")

    tried = []
    last_error = None
    for provider in [primary] + fallbacks:
        tried.append(provider)
        try:
            return await asyncio.to_thread(_call_llm, provider, prompt, meta)
        except Exception as exc:
            logger.warning("LLM analysis failed via %s: %s", provider, exc)
            last_error = str(exc)

    return {
        "analysis": None,
        "error": f"All providers failed ({', '.join(tried)}): {last_error or 'unknown error'}",
        "timestamp": utc_now_iso(),
        "meta": meta,
        "model": OPENROUTER_MODEL if primary == 'openrouter' else LLM_MODEL,
        "provider": primary,
    }

async def handle_snapshot_analysis(snapshot: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    """Compute LLM analysis for the latest snapshot and broadcast it."""
    result = await run_llm_analysis(snapshot, meta)
    if result is None:
        return
    async with analysis_lock:
        global last_analysis_result
        last_analysis_result = result
    await broadcast_json({"type": "analysis", "payload": result})


async def push_history_periodically(websocket: WebSocket, stop_event: asyncio.Event) -> None:
    """Every 30 seconds send the full history snapshot to the connected client."""
    try:
        while not stop_event.is_set():
            await asyncio.sleep(30)
            async with history_lock:
                snapshot = list(dashboard_snapshot)
                meta = dict(session_meta)
            try:
                await websocket.send_json({"type": "history", "history": snapshot, "meta": meta})
                async with analysis_lock:
                    if last_analysis_result:
                        await websocket.send_json({"type": "analysis", "payload": last_analysis_result})
                logger.info("Sent history snapshot with %d entries", len(snapshot))
            except Exception as exc:  # pragma: no cover - transport errors
                logger.warning("Failed to send history snapshot: %s", exc)
                break
    except asyncio.CancelledError:
        # Task cancelled because websocket closed; exit quietly.
        return


@app.websocket("/monitor")
async def monitor(websocket: WebSocket) -> None:
    await websocket.accept()
    await register_websocket(websocket)
    stop_event = asyncio.Event()
    history_task = asyncio.create_task(push_history_periodically(websocket, stop_event))
    logger.info("Frontend monitor connected.")

    try:
        await websocket.send_json({"type": "hello", "message": "Monitor websocket connected"})
        # On connect, share any existing flagged events for display purposes.
        async with history_lock:
            if flagged_events:
                await websocket.send_json({"type": "flagged_bulk", "items": list(flagged_events)})
            if dashboard_snapshot:
                await websocket.send_json({"type": "history", "history": list(dashboard_snapshot), "meta": dict(session_meta)})
        async with analysis_lock:
            if last_analysis_result:
                await websocket.send_json({"type": "analysis", "payload": last_analysis_result})
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON payload"})
                continue

            if data.get("type") == "snapshot":
                items = data.get("items") or []
                aggregate_detected_language = data.get("detected_language")
                allowed_language = data.get("allowed_language") or session_meta.get("allowed_language")
                async with history_lock:
                    session_meta["topic"] = data.get("topic") or session_meta.get("topic") or ""
                    session_meta["context"] = data.get("context") or session_meta.get("context") or ""
                    session_meta["allowed_language"] = data.get("allowed_language") or session_meta.get("allowed_language") or ""
                async with history_lock:
                    dashboard_snapshot.clear()
                    dashboard_snapshot.extend(items)
                normalized_allowed = allowed_language.lower().strip() if allowed_language else ""
                if normalized_allowed and normalized_allowed != "auto" and aggregate_detected_language:
                    if normalized_allowed != aggregate_detected_language.lower().strip():
                        mismatch_entry = {
                            "speaker": "aggregate",
                            "text": "Aggregate language mismatch",
                            "timestamp": utc_now_iso(),
                            "detected_language": aggregate_detected_language,
                            "allowed_language": allowed_language,
                            "topic": session_meta.get("topic"),
                            "context": session_meta.get("context"),
                            "flagged_reason": "language_mismatch_snapshot",
                        }
                        flagged_events.append(mismatch_entry)
                        await broadcast_json({"type": "flagged", "payload": mismatch_entry})
                if normalized_allowed and normalized_allowed != "auto" and items:
                    for item in items:
                        detected_lang = item.get("detected_language")
                        if detected_lang and normalized_allowed != str(detected_lang).lower().strip():
                            mismatch_entry = {
                                "speaker": item.get("speaker"),
                                "text": item.get("text"),
                                "start": item.get("start"),
                                "end": item.get("end"),
                                "timestamp": item.get("timestamp") or utc_now_iso(),
                                "detected_language": detected_lang,
                                "allowed_language": allowed_language,
                                "topic": session_meta.get("topic"),
                                "context": session_meta.get("context"),
                                "flagged_reason": "language_mismatch_snapshot",
                            }
                            flagged_events.append(mismatch_entry)
                            await broadcast_json({"type": "flagged", "payload": mismatch_entry})
                logger.info("Received dashboard snapshot with %d entries", len(items))
                meta_copy: Dict[str, Any] = {}
                async with history_lock:
                    meta_copy = dict(session_meta)
                asyncio.create_task(handle_snapshot_analysis(list(items), meta_copy))
                await websocket.send_json({"type": "ack", "received": True, "mode": "snapshot"})
                continue

            print("got ==> ",data)

            if data.get("type") != "delta":
                # Ignore other message types for profanity processing.
                continue

            text = (data.get("text") or "").strip()
            if not text:
                continue

            speaker = data.get("speaker")
            allowed_language = data.get("allowed_language") or session_meta.get("allowed_language")
            detected_language = data.get("detected_language")
            entry_topic = data.get("topic") or session_meta.get("topic")
            entry_context = data.get("context") or session_meta.get("context")
            entry = {
                "speaker": speaker,
                "text": text,
                "start": data.get("start"),
                "end": data.get("end"),
                "timestamp": data.get("timestamp") or utc_now_iso(),
                "detected_language": detected_language,
                "allowed_language": allowed_language,
                "topic": entry_topic,
                "context": entry_context,
            }
            
            print("Got i websocked :",text)

            flagged_words = extract_flagged_words(text)
            if flagged_words:
                speaker_key = str(speaker) if speaker is not None else "_unknown"
                flagged_words = [w.lower() for w in flagged_words]
                async with history_lock:
                    seen_words = flagged_tokens_by_speaker[speaker_key]
                    new_words = [w for w in flagged_words if w not in seen_words]
                    if new_words:
                        # seen_words.update(new_words)
                        flagged_entry = {**entry, "flagged_words": new_words}
                        # flagged_events.append(flagged_entry)
                        await broadcast_json({"type": "flagged", "payload": flagged_entry})

            if allowed_language and detected_language:
                normalized_allowed = allowed_language.lower().strip()
                if normalized_allowed != "auto" and normalized_allowed != detected_language.lower().strip():
                    mismatch_entry = {
                        **entry,
                        "flagged_reason": "language_mismatch",
                    }
                    flagged_events.append(mismatch_entry)
                    await broadcast_json({"type": "flagged", "payload": mismatch_entry})

            await websocket.send_json({"type": "ack", "received": True})
    except WebSocketDisconnect:
        logger.info("Frontend monitor disconnected.")
    except Exception as exc:  # pragma: no cover - transport errors
        logger.exception("Monitor websocket crashed: %s", exc)
    finally:
        await unregister_websocket(websocket)
        stop_event.set()
        history_task.cancel()
        with suppress(Exception, asyncio.CancelledError):
            await history_task


@app.get("/complete-conversation")
async def get_complete_conversation() -> Dict[str, Any]:
    async with history_lock:
        snapshot = list(dashboard_snapshot)
        meta = dict(session_meta)
    async with analysis_lock:
        analysis = last_analysis_result
    return {"history": snapshot, "meta": meta, "analysis": analysis}


@app.get("/flagged")
async def get_flagged() -> Dict[str, Any]:
    async with history_lock:
        meta = dict(session_meta)
    return {"flagged": flagged_events, "meta": meta}


@app.get("/analysis")
async def get_analysis() -> Dict[str, Any]:
    async with analysis_lock:
        analysis = last_analysis_result
    async with history_lock:
        meta = dict(session_meta)
    return {"analysis": analysis, "meta": meta}
