import asyncio
import logging
import time
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .analysis import analyze_text, compute_dominance, encode_topic
from .config import (
    CORS_ALLOW_ORIGINS,
    DEFAULT_ALLOWED_LANGUAGE,
    DEFAULT_TARGET_DESCRIPTION,
    DEFAULT_TARGET_TOPIC,
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
    speaker: Optional[str] = None
    timestamp: Optional[float] = None
    source: str = "wlk"


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

    entry = await process_event(group_id, event, session)
    return {"status": "ok", "entry": entry}


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
    base_entry: Dict = {
        "group_id": group_id,
        "source": event.source or "wlk",
        "speaker": event.speaker,
        "text": event.text,
        "lang": event.lang or "unknown",
        "timestamp": ts,
        "target_topic": session.topic,
        "target_description": session.description,
        "allowed_language": session.allowed_language,
    }

    alerts, topic_score = analyze_text(
        text=event.text,
        language=event.lang or "unknown",
        target_embedding=session.target_embedding,
        target_topic=session.topic,
        allowed_language=session.allowed_language,
    )

    prospective_history = registry.get_history(group_id) + [base_entry]
    dominance_state, dominance_speaker = compute_dominance(prospective_history)

    entry = {
        **base_entry,
        "alerts": alerts,
        "topic_score": f"{topic_score:.2f}",
        "dominance_state": dominance_state,
        "dominance_speaker": dominance_speaker,
        "speech_ratio": None,
        "silence": False,
        "chunk_seconds": None,
    }

    registry.append_history(group_id, entry)
    await manager.broadcast_alert(entry)
    logger.info("Ingested event for %s: %s", group_id, entry)
    return entry


async def summary_scheduler():
    """
    Periodically collect recent history and hand it to a larger model (stubbed).
    """
    logger.info("Starting summary scheduler")
    while True:
        await asyncio.sleep(90)
        for gid, session in list(registry.sessions.items()):
            pending = [e for e in session.history if e.get("timestamp", 0) > session.last_summary_at]
            if not pending:
                continue
            text_blob = "\n".join(
                f"[{e.get('speaker') or 'unknown'}] {e.get('text')}" for e in pending if e.get("text")
            )
            await send_to_big_model_stub(gid, text_blob, session.topic, session.description)
            registry.update_summary_timestamp(gid)


async def send_to_big_model_stub(group_id: str, text_blob: str, topic: str, description: str):
    """
    Placeholder hook to send text to an LLM / external API.
    """
    logger.info(
        "LLM stub for group %s | chars=%d | topic=%s | desc=%s",
        group_id,
        len(text_blob),
        topic,
        description,
    )
