from fastapi import FastAPI
from faster_whisper import WhisperModel
import yt_dlp, uuid, os
import numpy as np
from sentence_transformers import SentenceTransformer
from summa import summarizer

app = FastAPI()

# ===== MODELS =====
whisper = WhisperModel("base", compute_type="int8")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ===== UTIL =====
def sec_to_time(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"

# ===== TRANSCRIBE =====
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

# ===== CHAPTER ENGINE =====
def generate_chapters(transcript):
    clean = []
    for t in transcript:
        txt = t["text"].lower()
        if len(txt) < 15: continue
        if any(x in txt for x in ["okay","yeah","huh","correct"]): continue
        clean.append(t)

    texts = [t["text"] for t in clean]
    times = [t["start"] for t in clean]

    emb = embedder.encode(texts)
    sims = [np.dot(emb[i], emb[i+1]) for i in range(len(emb)-1)]
    threshold = np.mean(sims) - 0.35*np.std(sims)

    cuts = [0]
    for i, s in enumerate(sims):
        if s < threshold:
            cuts.append(i+1)
    cuts.append(len(texts))

    cuts = cuts[:12]

    chapters = []
    for i in range(len(cuts)-1):
        block = " ".join(texts[cuts[i]:cuts[i+1]])[:800]
        title = summarizer.summarize(block, ratio=0.2).split(".")[0]
        chapters.append(f"{sec_to_time(times[cuts[i]])} {title.title()}")

    return chapters

# ===== API =====
@app.post("/chapters")
def chapters(video_id: str):
    transcript = transcribe(video_id)
    return {
        "video_id": video_id,
        "chapters": generate_chapters(transcript)
    }
