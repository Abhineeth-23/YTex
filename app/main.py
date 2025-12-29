from fastapi import FastAPI, HTTPException
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq
import yt_dlp, uuid, os, json, re, logging
import numpy as np

# =========================
# ENV
# =========================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

client = Groq(api_key=GROQ_API_KEY)

logging.basicConfig(level=logging.INFO)

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
    return f"{int(sec // 60)}:{int(sec % 60):02d}"

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
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
        ]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    segments, _ = whisper.transcribe(audio)
    os.remove(audio)

    return [
        {"start": s.start, "text": s.text.strip()}
        for s in segments
        if len(s.text.strip()) > 10
    ]

# =========================
# BUILD FULL TRANSCRIPT
# =========================
def build_full_transcript(segments):
    return "\n".join(
        f"[{sec_to_time(s['start'])}] {s['text']}"
        for s in segments
    )

# =========================
# LLM CHAPTER PLANNER
# =========================
def llm_generate_chapters(transcript_text: str):
    prompt = f"""
You are a professional YouTube video editor.

Analyze the following full transcript and divide it into
clear, topic-wise chapters.

Rules:
- 5 to 10 chapters total
- Each chapter must represent a distinct topic
- Chapters must follow the video flow
- Output ONLY valid JSON in this exact format:

[
  {{
    "title": "Chapter title (3–6 words)",
    "start_hint": "Short phrase spoken at the start of this chapter"
  }}
]

Transcript:
{transcript_text}
"""

    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise RuntimeError("LLM did not return valid JSON")

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
            normalize_embeddings=True
        )[0]

        sims = np.dot(embeddings, hint_emb)
        best_idx = int(np.argmax(sims))

        ts = times[best_idx]
        if ts in used_times:
            continue  # prevent duplicate timestamps

        used_times.add(ts)
        final.append(f"{sec_to_time(ts)} {ch['title']}")

    return final

# =========================
# API
# =========================
@app.post("/chapters")
def chapters(video_id: str):
    try:
        segments = transcribe(video_id)
        transcript_text = build_full_transcript(segments)

        chapter_plan = llm_generate_chapters(transcript_text)
        chapter_list = align_chapters(chapter_plan, segments)

        return {
            "video_id": video_id,
            "chapters": chapter_list
        }

    except Exception as e:
        logging.exception("Chapter generation failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    }
