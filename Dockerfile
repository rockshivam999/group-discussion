FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (ffmpeg is crucial for audio)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Quieten tokenizers warning in container
ENV TOKENIZERS_PARALLELISM=false

# Pre-download models to avoid timeout on first run
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
