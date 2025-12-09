import json
import logging
import time
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .analysis import (
    ASR_MODEL,
    TOPIC_MODEL,
    analyze_text,
    compute_dominance,
    encode_topic,
)
from .audio_utils import convert_webm_to_wav, pcm_to_wav
from .config import CORS_ALLOW_ORIGINS, DEFAULT_TARGET_DESCRIPTION, DEFAULT_TARGET_TOPIC, SILENCE_THRESHOLD_SECONDS
from .connections import ConnectionManager

# --- Config & Init ---
logging.basicConfig(level=logging.INFO)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()


@app.get("/")
async def root():
    return {"status": "ok", "message": "Classroom Sentinel backend running"}


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/teacher")
async def websocket_teacher(websocket: WebSocket):
    await manager.connect_teacher(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        manager.disconnect_teacher(websocket)


@app.websocket("/ws/student/{group_id}")
async def websocket_student(websocket: WebSocket, group_id: str):
    await websocket.accept()
    logging.info("Group %s connected", group_id)

    target_topic = DEFAULT_TARGET_TOPIC
    target_description = DEFAULT_TARGET_DESCRIPTION
    topic_text, target_emb = encode_topic(target_topic, target_description)

    header_prefix: bytes | None = None
    pcm_buffer = bytearray()
    PCM_BYTES_PER_SECOND = 32000  # 16kHz * 2 bytes mono
    PCM_TARGET = PCM_BYTES_PER_SECOND * 5  # ~5 seconds
    silence_accum = 0.0

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                raw_bytes = bytes(message["bytes"] or b"")
            elif "text" in message:
                # Accept JSON metadata to set topic/description
                try:
                    data = json.loads(message["text"])
                    new_topic = data.get("topic")
                    new_desc = data.get("description", "")
                    if new_topic:
                        target_topic = new_topic
                        target_description = new_desc or target_description
                        topic_text, target_emb = encode_topic(target_topic, target_description)
                        logging.info("Group %s updated topic: %s", group_id, topic_text)
                except Exception:
                    pass
                continue

            # Skip if no audio payload
            if not raw_bytes or len(raw_bytes) < 512:
                continue

            # Ensure each chunk has EBML header; prepend from first chunk if missing
            if header_prefix is None:
                header_prefix = raw_bytes
                data_bytes = raw_bytes
            else:
                if len(raw_bytes) >= 2 and raw_bytes[0] == 0x1A and raw_bytes[1] == 0x45:
                    header_prefix = raw_bytes
                    data_bytes = raw_bytes
                else:
                    data_bytes = header_prefix + raw_bytes

            try:
                wav_file = convert_webm_to_wav(data_bytes)
                wav_bytes = wav_file.getvalue()
                pcm_data = wav_bytes[44:] if len(wav_bytes) > 44 else b""
                if pcm_data:
                    pcm_buffer.extend(pcm_data)
            except Exception as exc:
                logging.error("Error decoding audio for group %s: %s", group_id, exc)
                continue

            # Process when we have ~5 seconds of PCM
            if len(pcm_buffer) >= PCM_TARGET:
                try:
                    chunk_duration = len(pcm_buffer) / PCM_BYTES_PER_SECOND
                    wav_chunk = pcm_to_wav(bytes(pcm_buffer))

                    segments, info = ASR_MODEL.transcribe(wav_chunk, vad_filter=True)
                    full_text = " ".join([s.text for s in segments])
                    detected_lang = info.language or "unknown"

                    speech_duration = sum(max(0.0, s.end - s.start) for s in segments)
                    speech_ratio = min(1.0, speech_duration / chunk_duration) if chunk_duration else 0.0
                    dominance_state = compute_dominance(speech_duration, chunk_duration)

                    silence_flag = False
                    if speech_duration < 0.2:
                        silence_accum += chunk_duration
                        if silence_accum >= SILENCE_THRESHOLD_SECONDS:
                            silence_flag = True
                    else:
                        silence_accum = 0.0

                    alerts, topic_score = analyze_text(full_text, detected_lang, target_emb, topic_text)
                    if silence_flag:
                        alerts.append(
                            {
                                "type": "SILENCE",
                                "msg": f"No meaningful speech for ~{SILENCE_THRESHOLD_SECONDS:.0f}s",
                            }
                        )

                    timestamp = time.time()
                    payload: Dict = {
                        "group_id": group_id,
                        "text": full_text,
                        "lang": detected_lang,
                        "alerts": alerts,
                        "topic_score": f"{topic_score:.2f}",
                        "timestamp": timestamp,
                        "speech_ratio": round(speech_ratio, 2),
                        "dominance_state": dominance_state,
                        "silence": silence_flag,
                        "chunk_seconds": round(chunk_duration, 2),
                        "target_topic": target_topic,
                        "target_description": target_description,
                    }

                    await manager.broadcast_alert(payload)
                    await websocket.send_json({"type": "analysis", **payload})
                    logging.info("Processed Group %s: %s", group_id, json.dumps(payload))
                except Exception as exc:
                    logging.error("Error processing audio for group %s: %s", group_id, exc)
                finally:
                    pcm_buffer = bytearray()

    except WebSocketDisconnect:
        logging.info("Group %s disconnected", group_id)
