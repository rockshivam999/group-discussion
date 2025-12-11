import asyncio
import logging
import time
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .analysis import analyze_live_alerts, analyze_text, compute_dominance, encode_topic, collapse_repetitions
from .config import (
    CORS_ALLOW_ORIGINS,
    DEFAULT_ALLOWED_LANGUAGE,
    DEFAULT_TARGET_DESCRIPTION,
    DEFAULT_TARGET_TOPIC,
    LLM_ENABLE_STUB,
    LLM_SUMMARY_INTERVAL_SECONDS,
    MERGE_WINDOW_SECONDS,
)
from .connections import ConnectionManager
from .sessions import GroupRegistry

# --- Config & Init ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()
registry = GroupRegistry()


class SessionRequest(BaseModel):
    topic: str = DEFAULT_TARGET_TOPIC
    description: str = DEFAULT_TARGET_DESCRIPTION
    allowed_language: str = DEFAULT_ALLOWED_LANGUAGE
    # If provided, backend skips container launch and uses this URL directly
    wlk_ws_url: Optional[str] = None


class SessionResponse(BaseModel):
    group_id: str
    wlk_ws_url: Optional[str]
    analysis_ws_url: str
    allowed_language: str
    topic: str
    description: str


class IngestEvent(BaseModel):
    text: str
    lang: Optional[str] = None
    speaker: Optional[object] = None  # accept int/str
    timestamp: Optional[float] = None
    source: str = "wlk"
    raw_payload: Optional[dict] = None


@app.get("/")
async def root():
    return {"status": "ok", "message": "Classroom Sentinel backend running"}


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.post("/groups/{group_id}/session", response_model=SessionResponse)
async def create_session(group_id: str, payload: SessionRequest):
    """
    Start (or refresh) a group session and spin up a per-group WhisperLiveKit container.
    """
    topic_text, target_emb = encode_topic(payload.topic, payload.description)
    session = registry.ensure_session(
        group_id=group_id,
        topic=topic_text,
        description=payload.description,
        allowed_language=payload.allowed_language,
        embedding=target_emb,
    )
    if payload.wlk_ws_url:
        session.wlk_ws_url = payload.wlk_ws_url

    # Spawn summary scheduler on first session creation if not already running
    if not getattr(app.state, "summary_task", None):
        app.state.summary_task = asyncio.create_task(summary_scheduler())

    logger.info("Session created for %s with WLK %s", group_id, session.wlk_ws_url)
    print(f"[Session] {group_id} -> WLK {session.wlk_ws_url}")

    return SessionResponse(
        group_id=group_id,
        wlk_ws_url=session.wlk_ws_url,
        analysis_ws_url=f"/ws/group/{group_id}",
        allowed_language=session.allowed_language,
        topic=session.topic,
        description=session.description,
    )


@app.post("/groups/{group_id}/events")
async def ingest_event(group_id: str, event: IngestEvent):
    """
    Ingest a WhisperLiveKit transcript event from the client and fan it out with alerts.
    """
    session = registry.sessions.get(group_id)
    if not session:
        raise HTTPException(status_code=404, detail="Group session not found. Create a session first.")

    try:
        entry = await process_event(group_id, event, session)
        return {"status": "ok", "entry": entry}
    except Exception as exc:
        logger.exception("Error ingesting event for %s: %s", group_id, exc)
        raise HTTPException(status_code=500, detail="Failed to process event")


@app.get("/groups/{group_id}/history")
async def history(group_id: str):
    return {"group_id": group_id, "history": registry.get_history(group_id)}


@app.websocket("/ws/teacher")
async def websocket_teacher(websocket: WebSocket):
    await manager.connect_teacher(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        manager.disconnect_teacher(websocket)


@app.websocket("/ws/group/{group_id}")
async def websocket_group(websocket: WebSocket, group_id: str):
    await manager.connect_group(group_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_group(group_id, websocket)


# Backwards-compatible alias for students
@app.websocket("/ws/student/{group_id}")
async def websocket_student(websocket: WebSocket, group_id: str):
    return await websocket_group(websocket, group_id)


async def process_event(group_id: str, event: IngestEvent, session) -> Dict:
    ts = event.timestamp or time.time()
    speaker_val = event.speaker
    speaker = str(speaker_val) if speaker_val is not None else "unknown"
    lang = event.lang or "unknown"
    if event.raw_payload is not None:
        logger.info("Raw WLK payload for %s (speaker=%s): %s", group_id, speaker, event.raw_payload)
    base_entry: Dict = {
        "group_id": group_id,
        "source": event.source or "wlk",
        "speaker": speaker,
        "text": collapse_repetitions(event.text or ""),
        "lang": lang,
        "timestamp": ts,
        "target_topic": session.topic,
        "target_description": session.description,
        "allowed_language": session.allowed_language,
    }

    alerts = []
    prev_text = session.last_text_by_speaker.get(speaker, "")
    if base_entry["text"].startswith(prev_text):
        new_segment = base_entry["text"][len(prev_text):].strip()
    else:
        new_segment = base_entry["text"].strip()

    new_words_count = len(new_segment.split()) if new_segment else 0

    # Profanity on only the new segment (fallback to full text if we somehow missed diff)
    if new_segment or base_entry["text"]:
        segment_for_profanity = new_segment or base_entry["text"]
        prof_alerts = analyze_live_alerts(
            text=segment_for_profanity,
            language=lang,
            allowed_language=session.allowed_language,
            check_language=False,
            check_profanity=True,
        )
        for a in prof_alerts:
            if a.get("type") == "OFFENSIVE":
                a["msg"] = f"{a['msg']} (speaker={speaker}, lang={lang})"
        alerts.extend(prof_alerts)

    # Language check every +10 new words (buffered)
    if new_words_count > 0:
        session.lang_word_buffer += new_words_count
        if session.lang_word_buffer >= 10:
            alerts.extend(
                analyze_live_alerts(
                    text=new_segment or base_entry["text"],
                    language=lang,
                    allowed_language=session.allowed_language,
                    check_language=True,
                    check_profanity=False,
                )
            )
            session.lang_word_buffer = 0

    session.last_text_by_speaker[speaker] = base_entry["text"]
    logger.info("Raw transcript event for %s (speaker=%s): %s", group_id, speaker, event.text)

    history = registry.get_history(group_id)
    merged = False
    if history:
        last = history[-1]
        same_speaker = (last.get("speaker") or "unknown") == speaker
        close_in_time = ts - (last.get("timestamp") or 0) < MERGE_WINDOW_SECONDS
        if same_speaker and close_in_time:
            prev_text = (last.get("text") or "").strip()
            curr_text = (event.text or "").strip()
            if curr_text.startswith(prev_text):
                merged_text = curr_text
            elif prev_text.startswith(curr_text):
                merged_text = prev_text
            else:
                merged_text = f"{prev_text} {curr_text}".strip()
            base_entry["text"] = merged_text
            merged = True
            history[-1] = base_entry

    prospective_history = (history if merged else history + [base_entry]) if history is not None else [base_entry]
    dominance_state, dominance_speaker = ("PENDING_LLM", None)

    entry = {
        **base_entry,
        "alerts": alerts,
        "topic_score": "",
        "dominance_state": dominance_state,
        "dominance_speaker": dominance_speaker,
        "speech_ratio": None,
        "silence": False,
        "chunk_seconds": None,
    }

    if merged:
        registry.update_last(group_id, entry)
    else:
        registry.append_history(group_id, entry)
    await manager.broadcast_alert(entry)
    logger.info("Ingested event for %s (merged=%s): %s", group_id, merged, entry)
    return entry


async def summary_scheduler():
    """
    Periodically collect recent history and hand it to a larger model (stubbed).
    """
    if not LLM_ENABLE_STUB:
        logger.info("LLM summary scheduler disabled (LLM_ENABLE_STUB=0)")
        return
    logger.info("Starting summary scheduler")
    while True:
        await asyncio.sleep(LLM_SUMMARY_INTERVAL_SECONDS)
        for gid, session in list(registry.sessions.items()):
            pending = [e for e in session.history if e.get("timestamp", 0) > session.last_summary_at]
            if not pending:
                continue
            text_blob = "\n".join(
                f"[{e.get('speaker') or 'unknown'}] {e.get('text')}" for e in pending if e.get("text")
            )
            summary = await send_to_big_model_stub(gid, text_blob, session.topic, session.description)
            payload = {
                "group_id": gid,
                "source": "llm",
                "text": summary.get("summary", "LLM summary"),
                "alerts": summary.get("alerts", []),
                "timestamp": time.time(),
                "dominance_state": summary.get("dominance", "PENDING"),
                "dominance_speaker": summary.get("dominant_speaker"),
                "topic_score": "",
                "lang": "unknown",
            }
            await manager.broadcast_alert(payload)
            registry.update_summary_timestamp(gid)


async def send_to_big_model_stub(group_id: str, text_blob: str, topic: str, description: str):
    """
    Placeholder hook to send text to an LLM / external API.
    """
    import random

    summary = f"LLM summary (mock): {text_blob[:200]}..."
    dominant_speaker = f"spk{random.randint(1,3)}"
    alerts = []
    if random.random() > 0.5:
        alerts.append({"type": "TOPIC_LLM", "msg": "Possible off-topic drift (mocked)"})
    dominance = random.choice(["DOMINATING", "BALANCED", "QUIET"])
    result = {
        "summary": summary,
        "alerts": alerts,
        "dominance": dominance,
        "dominant_speaker": dominant_speaker,
    }
    logger.info(
        "LLM stub for group %s | chars=%d | topic=%s | desc=%s | result=%s",
        group_id,
        len(text_blob),
        topic,
        description,
        result,
    )
    return result
