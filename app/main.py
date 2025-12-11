import asyncio
import logging
import time
from typing import Dict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    CORS_ALLOW_ORIGINS,
    DEFAULT_ALLOWED_LANGUAGE,
    DEFAULT_TARGET_DESCRIPTION,
    DEFAULT_TARGET_TOPIC,
    LLM_ENABLE_STUB,
    LLM_SUMMARY_INTERVAL_SECONDS,
)
from .connections import ConnectionManager
from .event_processor import EventProcessor, build_entry_from_summary
from .llm import summarize_history
from .models import IngestEvent, SessionRequest, SessionResponse
from .sessions import GroupRegistry

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
processor = EventProcessor(registry=registry, manager=manager)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Classroom Sentinel backend"}


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.post("/groups/{group_id}/session", response_model=SessionResponse)
async def create_session(group_id: str, payload: SessionRequest):
    session = registry.ensure_session(
        group_id=group_id,
        topic=payload.topic or DEFAULT_TARGET_TOPIC,
        description=payload.description or DEFAULT_TARGET_DESCRIPTION,
        allowed_language=payload.allowed_language or DEFAULT_ALLOWED_LANGUAGE,
        embedding=None,
    )
    if payload.wlk_ws_url:
        session.wlk_ws_url = payload.wlk_ws_url

    if not getattr(app.state, "summary_task", None):
        app.state.summary_task = asyncio.create_task(summary_scheduler())

    logger.info("Session created for %s with WLK %s", group_id, session.wlk_ws_url)
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
    session = registry.sessions.get(group_id)
    if not session:
        raise HTTPException(status_code=404, detail="Group session not found. Create a session first.")

    try:
        entry = await processor.process_event(group_id, event, session)
        return {"status": "ok", "entry": entry}
    except Exception as exc:  # pragma: no cover - runtime safeguards
        logger.exception("Error ingesting event for %s: %s", group_id, exc)
        raise HTTPException(status_code=500, detail="Failed to process event")


@app.get("/groups/{group_id}/history")
async def history(group_id: str):
    return {"group_id": group_id, "history": registry.get_history(group_id)}


@app.websocket("/ws/teacher")
async def websocket_teacher(websocket: WebSocket):
    await manager.connect_teacher(websocket)
    history = registry.get_all_history()
    for entry in history:
        try:
            await websocket.send_json(entry)
        except Exception:
            break
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_teacher(websocket)


@app.websocket("/ws/group/{group_id}")
async def websocket_group(websocket: WebSocket, group_id: str):
    await manager.connect_group(group_id, websocket)
    history = registry.get_history(group_id)
    for entry in history:
        try:
            await websocket.send_json(entry)
        except Exception:
            break
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect_group(group_id, websocket)


async def summary_scheduler():
    if not LLM_ENABLE_STUB:
        logger.info("LLM summary scheduler disabled (LLM_ENABLE_STUB=0)")
        return
    logger.info("Starting summary scheduler")
    while True:
        await asyncio.sleep(LLM_SUMMARY_INTERVAL_SECONDS)
        for gid, session in list(registry.sessions.items()):
            pending = registry.get_pending_history(gid, session.last_summary_at)
            if not pending:
                continue
            summary = await summarize_history(gid, pending, session.topic, session.description)
            payload = build_entry_from_summary(gid, summary)
            await manager.broadcast_alert(payload)
            registry.update_summary_timestamp(gid)
