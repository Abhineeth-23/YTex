FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload models (faster first run)
RUN python - <<EOF
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
WhisperModel("base", compute_type="int8")
SentenceTransformer("all-MiniLM-L6-v2")
EOF

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
