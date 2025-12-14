import asyncio
import json
import logging
import re
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Dict, List

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
                        await websocket.send_json({"type": "flagged", "payload": mismatch_entry})
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
                            await websocket.send_json({"type": "flagged", "payload": mismatch_entry})
                logger.info("Received dashboard snapshot with %d entries", len(items))
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
                flagged_entry = {**entry, "flagged_words": flagged_words}
                flagged_events.append(flagged_entry)
                await websocket.send_json({"type": "flagged", "payload": flagged_entry})

            if allowed_language and detected_language:
                normalized_allowed = allowed_language.lower().strip()
                if normalized_allowed != "auto" and normalized_allowed != detected_language.lower().strip():
                    mismatch_entry = {
                        **entry,
                        "flagged_reason": "language_mismatch",
                    }
                    flagged_events.append(mismatch_entry)
                    await websocket.send_json({"type": "flagged", "payload": mismatch_entry})

            await websocket.send_json({"type": "ack", "received": True})
    except WebSocketDisconnect:
        logger.info("Frontend monitor disconnected.")
    except Exception as exc:  # pragma: no cover - transport errors
        logger.exception("Monitor websocket crashed: %s", exc)
    finally:
        stop_event.set()
        history_task.cancel()
        with suppress(Exception, asyncio.CancelledError):
            await history_task


@app.get("/complete-conversation")
async def get_complete_conversation() -> Dict[str, Any]:
    async with history_lock:
        snapshot = list(dashboard_snapshot)
        meta = dict(session_meta)
    return {"history": snapshot, "meta": meta}


@app.get("/flagged")
async def get_flagged() -> Dict[str, Any]:
    async with history_lock:
        meta = dict(session_meta)
    return {"flagged": flagged_events, "meta": meta}
