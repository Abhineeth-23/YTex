FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Preload models (faster first run)
RUN python -c "from faster_whisper import WhisperModel; from sentence_transformers import SentenceTransformer; WhisperModel('base', compute_type='int8'); SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]