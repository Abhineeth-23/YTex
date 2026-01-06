from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
import yt_dlp
import uuid
import os
import json
import logging
import numpy as np

# =========================
# ENV + LOGGING
# =========================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = Groq(api_key=GROQ_API_KEY)

# =========================
# FASTAPI
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # chrome-extension:// allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELS (loaded once)
# =========================
whisper = WhisperModel("base", compute_type="int8")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# CONSTANTS
# =========================
FALLBACK_TOPICS = [
    "Introduction",
    "Background Context",
    "Early Challenges",
    "Key Technology",
    "Industry Impact",
    "Business Strategy",
    "Market Dominance",
    "Future Outlook",
]

# =========================
# UTILS
# =========================
def sec_to_time(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"

# =========================
# TRANSCRIPTION
# =========================
def transcribe(video_id: str):
    url = f"https://www.youtube.com/watch?v={video_id}"
    uid = str(uuid.uuid4())
    audio = f"{uid}.wav"

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": uid + ".%(ext)s",
        "quiet": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    segments, _ = whisper.transcribe(audio)
    os.remove(audio)

    cleaned = [
        {"start": s.start, "text": s.text.strip()}
        for s in segments
        if len(s.text.strip()) > 10
    ]

    return cleaned

# =========================
# OUTLINE TRANSCRIPT (TOKEN SAFE)
# =========================
def build_outline_transcript(segments, max_lines=40):
    step = max(1, len(segments) // max_lines)
    return "\n".join(
        segments[i]["text"]
        for i in range(0, len(segments), step)
    )

# =========================
# LLM: TOPIC PLANNING ONLY
# =========================
def llm_generate_chapters(outline_text: str):
    prompt = f"""
    You are a professional YouTube video editor.

    Given the following condensed transcript outline, generate
    clear, topic-based chapter titles in chronological order.

    Rules:
    - 5 to 8 chapters total
    - Each title must be 3–6 words
    - Titles must describe TOPICS, not speech
    - Return ONLY valid JSON in this format:

    [
    {{
        "title": "Chapter title",
        "start_hint": "short phrase indicating where this topic begins"
    }}
    ]

    Transcript outline:
    {outline_text}
    """

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
        max_tokens=600,
    )

    content = response.choices[0].message.content.strip()
    return json.loads(content)

# =========================
# ALIGN CHAPTERS TO TIMESTAMPS
# =========================
def align_chapters(chapters, segments):
    texts = [s["text"] for s in segments]
    times = [s["start"] for s in segments]

    embeddings = embedder.encode(texts, normalize_embeddings=True)

    final = []
    used_times = set()

    for ch in chapters:
        hint_emb = embedder.encode(
            [ch["start_hint"]],
            normalize_embeddings=True,
        )[0]

        sims = np.dot(embeddings, hint_emb)
        idx = int(np.argmax(sims))
        ts = times[idx]

        if ts in used_times:
            continue

        used_times.add(ts)
        final.append(f"{sec_to_time(ts)} {ch['title']}")

    return final

# =========================
# FALLBACK (NEVER GARBAGE)
# =========================
def fallback_chapters(segments):
    step = max(1, len(segments) // len(FALLBACK_TOPICS))
    chapters = []

    for i, title in enumerate(FALLBACK_TOPICS):
        idx = min(i * step, len(segments) - 1)
        chapters.append(
            f"{sec_to_time(segments[idx]['start'])} {title}"
        )

    return chapters

# =========================
# API
# =========================
@app.post("/chapters")
def chapters(video_id: str):
    try:
        segments = transcribe(video_id)

        if not segments:
            return {
                "video_id": video_id,
                "chapters": ["0:00 Overview"],
            }

        outline = build_outline_transcript(segments)

        try:
            plan = llm_generate_chapters(outline)
            chapter_list = align_chapters(plan, segments)

            if len(chapter_list) < 3:
                raise RuntimeError("Weak LLM output")

        except Exception as e:
            logger.warning("LLM failed, using fallback: %s", e)
            chapter_list = fallback_chapters(segments)

        return {
            "video_id": video_id,
            "chapters": chapter_list,
        }

    except Exception as e:
        logger.exception("Hard failure")
        return {
            "video_id": video_id,
            "chapters": ["0:00 Failed to analyze video"],
        }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_model": GROQ_MODEL,
    }
