FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG EXTRAS
ARG HF_PRECACHE_DIR
ARG HF_TKN_FILE="/Users/prem/.cache/huggingface/token"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        build-essential \
        python3-dev \
        libsndfile1 \
        libportaudio2 && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    torch==2.1.1 \
    torchaudio==2.1.1 \
    torchvision==0.16.1 \
    diart==0.9.2 \
    whisperlivekit${EXTRAS:+[$EXTRAS]} \
    --extra-index-url https://download.pytorch.org/whl/cpu
COPY . .
COPY hf_token_temp /tmp/hf_token_temp
# Install WhisperLiveKit directly, allowing for optional dependencies
# RUN if [ -n "$EXTRAS" ]; then \
#       echo "Installing with extras: [$EXTRAS]"; \
#       pip install --no-cache-dir whisperlivekit[$EXTRAS]; \
#     else \
#       echo "Installing base package only"; \
#       pip install --no-cache-dir whisperlivekit; \
#     fi

# Enable in-container caching for Hugging Face models
VOLUME ["/root/.cache/huggingface/hub"]

# Conditionally copy a local pre-cache from the build context
RUN if [ -n "$HF_PRECACHE_DIR" ]; then \
      echo "Copying Hugging Face cache from $HF_PRECACHE_DIR"; \
      mkdir -p /root/.cache/huggingface/hub && \
      cp -r $HF_PRECACHE_DIR/* /root/.cache/huggingface/hub; \
    else \
      echo "No local Hugging Face cache specified, skipping copy"; \
    fi

# Conditionally copy a Hugging Face token if provided
RUN if [ -f "/tmp/hf_token_temp" ]; then \
      echo "Copying token..."; \
      mkdir -p /root/.cache/huggingface && \
      cp /tmp/hf_token_temp /root/.cache/huggingface/token; \
    fi
# Expose port for the transcription server
EXPOSE 8000

ENTRYPOINT ["whisperlivekit-server","--host", "0.0.0.0","--diarization","--diarization-backend", "diart"]

CMD ["--model", "base","--language", "en","--segmentation-model", "pyannote/segmentation-3.0","--embedding-model", "speechbrain/spkrec-ecapa-voxceleb"]


# docker pull ayushdh96/whisper-diarization