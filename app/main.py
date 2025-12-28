from fastapi import FastAPI
from faster_whisper import WhisperModel
import yt_dlp, uuid, os
import numpy as np
from sentence_transformers import SentenceTransformer
from summa import summarizer
import re
from openai import OpenAI

# =========================
# APP INIT
# =========================
app = FastAPI()

# =========================
# MODELS
# =========================
whisper = WhisperModel("base", compute_type="int8")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# LLM CLIENT (OPTIONAL)
# =========================
USE_LLM = bool(os.getenv("OPENAI_API_KEY"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if USE_LLM else None

# =========================
# UTILS
# =========================
def sec_to_time(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"

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
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav"
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    segments, _ = whisper.transcribe(audio)
    os.remove(audio)

    return [{"start": s.start, "text": s.text.strip()} for s in segments]

# =========================
# RULE-BASED CLEANER (FALLBACK)
# =========================
def clean_title(text: str) -> str:
    text = text.lower()

    # remove fillers
    text = re.sub(
        r"\b(so|well|okay|now|oh|alright|basically|just|actually|like|kind of|sort of)\b",
        "",
        text
    )

    # remove self references
    text = re.sub(r"\b(i am|i'm|we are|we're|you are|you're|i|we|you)\b", "", text)

    # remove weak verb phrases
    text = re.sub(
        r"(going to|want to|need to|trying to|talk about|show you|explain|walk through)",
        "",
        text
    )

    # normalize
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # definition shortcut
    if "what is" in text:
        after = text.split("what is", 1)[1].strip()
        topic = after.split()[:4]
        if topic:
            return ("What Is " + " ".join(topic)).title()

    words = text.split()[:6]
    title = " ".join(words).title()

    return title if len(title) >= 8 else "Overview"

# =========================
# LLM TITLE GENERATOR
# =========================
def llm_generate_title(block: str) -> str | None:
    if not USE_LLM:
        return None

    try:
        prompt = f"""
Create a concise YouTube chapter title (3–6 words).
No filler. No first-person language. No punctuation at the end.

Text:
{block}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate short YouTube chapter titles."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=20
        )

        title = response.choices[0].message.content.strip()
        return title if 3 <= len(title.split()) <= 7 else None

    except Exception as e:
        print("LLM error:", e)
        return None

# =========================
# CHAPTER ENGINE
# =========================
def generate_chapters(transcript):
    # ---- CLEAN TRANSCRIPT ----
    clean = []
    for t in transcript:
        txt = t["text"].lower()
        if len(txt) < 15:
            continue
        if any(x in txt for x in ["okay", "yeah", "huh", "correct"]):
            continue
        clean.append(t)

    texts = [t["text"] for t in clean]
    times = [t["start"] for t in clean]

    if len(texts) < 10:
        return [f"{sec_to_time(times[i])} {texts[i][:60]}" for i in range(len(texts))]

    # ---- SEMANTIC SIMILARITY ----
    emb = embedder.encode(texts)
    sims = [np.dot(emb[i], emb[i + 1]) for i in range(len(emb) - 1)]
    threshold = np.mean(sims) - 0.35 * np.std(sims)

    # ---- INITIAL CUTS ----
    cuts = [0]
    for i, s in enumerate(sims):
        if s < threshold:
            cuts.append(i + 1)
    cuts.append(len(texts))

    # ---- MIN GAP FILTER (30s) ----
    filtered = [cuts[0]]
    for c in cuts[1:]:
        if c >= len(times) or times[c] - times[filtered[-1]] >= 30:
            filtered.append(c)

    cuts = filtered[:12]

    # ---- GENERATE TITLES ----
    chapters = []
    for i in range(len(cuts) - 1):
        block = " ".join(texts[cuts[i]:cuts[i + 1]])[:800]

        # LLM first
        title = llm_generate_title(block)

        # fallback 1: summarizer
        if not title:
            title = summarizer.summarize(block, ratio=0.2)

        # fallback 2: rule-based cleaner
        if not title or len(title.strip()) < 10:
            title = clean_title(block)

        chapters.append(f"{sec_to_time(times[cuts[i]])} {title}")

    return chapters

# =========================
# API
# =========================
@app.post("/chapters")
def chapters(video_id: str):
    transcript = transcribe(video_id)
    return {
        "video_id": video_id,
        "chapters": generate_chapters(transcript)
    }
