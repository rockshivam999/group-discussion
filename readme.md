# Classroom Monitor

Classroom monitoring stack with live ASR (WhisperLiveKit), backend moderation/LLM analysis, and a React teacher dashboard.

## Services (docker-compose.yml)
- `whisperlivekit`: Streams audio → diarized/translated transcripts (diart backend).
- `backend`: FastAPI websocket/REST for moderation, snapshot storage, profanity/language checks, and LLM-based 30s analysis.
- `frontend`: React/Vite UI served by Nginx.
- `phi4` (optional): Ollama server for local Phi-4 (fallback LLM).
- `phi4-init` (optional): One-shot pull of the Phi-4 model.

## Requirements
- Docker + Docker Compose
- Internet access to pull models/images
- **Hugging Face token** with access to the gated pyannote models (used by diart):
  - Accept the license for `pyannote/segmentation-3.0` on Hugging Face.
  - Create a token and provide it (see Setup).
- **LLM keys**:
  - Primary: OpenRouter API key (`OPENROUTER_API_KEY` or `openrouterkey` file).
  - Optional: Local Phi-4 via Ollama (no key enforced; `ollama` placeholder is fine).

## Setup
1) Copy env template and fill values:
   ```bash
   cp .env.example .env
   ```
   - Set `OPENROUTER_API_KEY` (or keep it in `openrouterkey`).
   - Set `OPENROUTER_REFERER`/`OPENROUTER_APP_TITLE` if you want attribution on OpenRouter.
   - For local fallback, leave `ENABLE_LOCAL_LLM_FALLBACK=1` and ensure `phi4` runs; otherwise set `ANALYSIS_PROVIDER=openrouter`.

2) Provide Hugging Face token for diarization:
   - Accept pyannote license on HF, then create a token.
   - Put the token in `WhisperLiveKit/hf_token_temp` (single line). The WhisperLiveKit Dockerfile copies it into the image.

3) Build and run:
   ```bash
   docker compose up --build
   ```
   - Frontend: http://localhost:5173
   - Backend API/WS: http://localhost:8000 (WS at `/monitor`)
   - WhisperLiveKit WS (ASR): ws://localhost:8100/asr
   - Ollama/Phi-4: http://localhost:11434

## Model terms & gates
- Diarization uses pyannote models (`pyannote/segmentation-3.0`), which require accepting the HF license and providing a token.
- Whisper base models download on first run; ensure you agree to their licenses per Hugging Face/OpenAI terms.
- Phi-4 via Ollama pulls from Ollama registry; accept any model terms on first pull if prompted.

## LLM analysis behavior
- Primary provider: `ANALYSIS_PROVIDER` (`openrouter` default).
- Fallback: `ENABLE_LOCAL_LLM_FALLBACK=1` will try local Phi-4 if the primary fails.
- Env variables are loaded from `.env` (python-dotenv).
- To rely only on OpenRouter, set `ANALYSIS_PROVIDER=openrouter` and either leave `ENABLE_LOCAL_LLM_FALLBACK=0` or omit/comment the `phi4` services in `docker-compose.yml`.

## Frontend & API endpoints
- Student view: `http://localhost:5173/` (connects to WhisperLiveKit WS at `ws://localhost:8100/asr`).
- Teacher dashboard: `http://localhost:5173/teacher` (consumes backend data below).
- Backend APIs (FastAPI):
  - WebSocket (teacher feed): `ws://localhost:8000/monitor`
    - Receives snapshots (`type: "snapshot"`) from frontend; broadcasts flagged items and LLM analysis (`type: "analysis"`).
  - REST:
    - `GET /complete-conversation` → `{ history, meta, analysis }`
    - `GET /flagged` → `{ flagged, meta }`
    - `GET /analysis` → `{ analysis, meta }`

## Health/verification commands
- OpenRouter ping (host):
  ```bash
  curl https://openrouter.ai/api/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <YOUR_OPENROUTER_API_KEY>" \
    -H "HTTP-Referer: http://localhost" \
    -H "X-Title: ClassroomMonitor" \
    -d '{"model":"openai/gpt-4o-mini","messages":[{"role":"user","content":"Ping from Classroom Monitor. Reply OK."}],"max_tokens":32}'
  ```
- Local Phi-4 ping (host):
  ```bash
  curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ollama" \
    -d '{"model":"phi4","messages":[{"role":"user","content":"Ping from Classroom Monitor. Reply OK."}],"max_tokens":32}'
  ```

## Typical flow
1) Frontend connects to WhisperLiveKit WS and streams mic audio.
2) Backend receives transcript deltas/snapshots over WS, flags profanity/language mismatches, and stores snapshots.
3) Every snapshot triggers an LLM call (OpenRouter primary, local fallback) to produce the 30s analysis shown on the teacher dashboard at `/teacher`.

## Notes
- If outbound network is blocked, set `ANALYSIS_PROVIDER=local` and keep `phi4` running.
- Ensure Hugging Face token is baked into the `whisperlivekit` image (via `hf_token_temp`) before building.
