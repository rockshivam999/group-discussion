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
    words = re.findall(r"[\\w'*]+", lowered)
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
    while not stop_event.is_set():
        await asyncio.sleep(30)
        async with history_lock:
            snapshot = list(dashboard_snapshot)
        try:
            await websocket.send_json({"type": "history", "history": snapshot})
            logger.info("Sent history snapshot with %d entries", len(snapshot))
        except Exception as exc:  # pragma: no cover - transport errors
            logger.warning("Failed to send history snapshot: %s", exc)
            break


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
                await websocket.send_json({"type": "history", "history": list(dashboard_snapshot)})
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON payload"})
                continue

            if data.get("type") == "snapshot":
                items = data.get("items") or []
                async with history_lock:
                    dashboard_snapshot.clear()
                    dashboard_snapshot.extend(items)
                logger.info("Received dashboard snapshot with %d entries", len(items))
                await websocket.send_json({"type": "ack", "received": True, "mode": "snapshot"})
                continue

            text = (data.get("text") or "").strip()
            if not text:
                continue

            speaker = data.get("speaker")
            entry = {
                "speaker": speaker,
                "text": text,
                "start": data.get("start"),
                "end": data.get("end"),
                "timestamp": data.get("timestamp") or utc_now_iso(),
            }

            flagged_words = extract_flagged_words(text)
            if flagged_words:
                flagged_entry = {**entry, "flagged_words": flagged_words}
                flagged_events.append(flagged_entry)
                await websocket.send_json({"type": "flagged", "payload": flagged_entry})

            await websocket.send_json({"type": "ack", "received": True})
    except WebSocketDisconnect:
        logger.info("Frontend monitor disconnected.")
    except Exception as exc:  # pragma: no cover - transport errors
        logger.exception("Monitor websocket crashed: %s", exc)
    finally:
        stop_event.set()
        history_task.cancel()
        with suppress(Exception):
            await history_task


@app.get("/complete-conversation")
async def get_complete_conversation() -> Dict[str, Any]:
    async with history_lock:
        snapshot = list(dashboard_snapshot)
    return {"history": snapshot}


@app.get("/flagged")
async def get_flagged() -> Dict[str, Any]:
    return {"flagged": flagged_events}
