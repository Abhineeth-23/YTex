from fastapi import FastAPI
from faster_whisper import WhisperModel
import yt_dlp, uuid, os, re
import numpy as np
from sentence_transformers import SentenceTransformer
from summa import summarizer
from dotenv import load_dotenv
import logging
from groq import Groq
from sklearn.cluster import KMeans

# =========================
# ENV
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
if client is None:
    logging.warning("GROQ_API_KEY not set; LLM title generation will be disabled and fallback summarization will be used.")

app = FastAPI()

# =========================
# MODELS
# =========================
whisper = WhisperModel("base", compute_type="int8")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# UTIL
# =========================
def sec_to_time(sec: float) -> str:
    return f"{int(sec//60)}:{int(sec%60):02d}"

# =========================
# TRANSCRIBE
# =========================
def transcribe(video_id: str):
    url = f"https://www.youtube.com/watch?v={video_id}"
    uid = str(uuid.uuid4())
    audio = f"{uid}.wav"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": uid + ".%(ext)s",
        "quiet": True,
        "postprocessors": [{"key": "FFmpegExtractAudio","preferredcodec": "wav"}]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    segments, _ = whisper.transcribe(audio)
    os.remove(audio)
    return [{"start": s.start, "text": s.text.strip()} for s in segments]

# =========================
# FALLBACK CLEANER
# =========================
def clean_title(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()[:5]
    return " ".join(words).title() if words else "Overview"

# =========================
# LLM TITLES (GROQ)
# =========================
def llm_generate_title(block: str):
    try:
        if client is None:
            return None

        prompt = f"""
You are generating structured chapter headings for a tutorial video.

Your job:
From the transcript below, identify the PRIMARY ACTION, STEP, or CONCEPT.
Return ONLY a short professional chapter title.

Rules:
- 2–6 words
- Verb + Noun or Noun Phrase
- Abstract the meaning (do NOT paraphrase speech)
- Sound like course topics:
  Cooking Rice
  Chopping Vegetables
  Preparing Ingredients
  Adding Eggs
  Seasoning Food
  Final Plating

Transcript:
{block}
"""

        r = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role":"user","content":prompt}],
            temperature=0.10,      # very low = forces structure
            max_tokens=20
        )

        return r.choices[0].message.content.strip()

    except Exception as e:
        print("LLM ERROR:", e)
        return None

# =========================
# CHAPTER ENGINE
# =========================
def generate_chapters(transcript):
    # Clean transcript
    cleaned = [t for t in transcript if len(t["text"]) > 15]
    texts = [t["text"] for t in cleaned]
    times = [t["start"] for t in cleaned]

    # Embed all segments
    embeddings = embedder.encode(texts)

    # Decide number of chapters (YouTube style)
    k = min(12, max(5, len(texts)//20))

    # Global clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    # Group clusters
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append({
            "text": texts[i],
            "time": times[i]
        })

    chapters = []
    for items in clusters.values():
        cluster_text = " ".join([x["text"] for x in items])[:1200]
        start_time = min(x["time"] for x in items)

        # LLM semantic abstraction
        title = llm_generate_title(cluster_text)
        if not title or len(title) < 8:
            title = clean_title(cluster_text)

        chapters.append({
            "time": start_time,
            "title": title
        })

    # Sort chronologically
    chapters.sort(key=lambda x: x["time"])

    return [f"{sec_to_time(c['time'])} {c['title']}" for c in chapters]
# =========================
# API
# =========================
@app.post("/chapters")
def chapters(video_id: str):
    return {"video_id": video_id, "chapters": generate_chapters(transcribe(video_id))}
