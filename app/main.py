from fastapi import FastAPI, Body, HTTPException
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

# Basic logging configuration (override with environment/runner settings as needed)
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
def llm_generate_title(block: str, use_llm: bool = True):
    try:
        if not use_llm or client is None:
            return None

        prompt = f"""
You are a concise title generator. Extract a short YouTube chapter title (3–6 words) from the transcript below.
Output exactly one short title only, no punctuation or surrounding quotes, and no explanation.
Use imperative verb+noun or short noun phrase style. If unsure, return an empty string.

Transcript:
{block}
"""
        messages = [
            {"role": "system", "content": "You are a concise title generator that outputs a single short phrase as a YouTube chapter title."},
            {"role": "user", "content": prompt},
        ]
        logging.debug("LLM prompt (truncated): %s", prompt[:800])

        r = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=messages,
            temperature=0.08,
            max_tokens=18
        )

        title = r.choices[0].message.content.strip()
        title = title.replace("\n", " ").strip()
        title = re.sub(r'^[^\w]*|[^\w]*$', '', title)
        words = title.split()
        if len(words) > 6:
            title = " ".join(words[:6])
        if len(title) == 0:
            return None

        logging.debug("LLM title result: %s", title)
        return title

    except Exception as e:
        logging.warning("LLM ERROR: %s", e)
        return None

# =========================
# CHAPTER ENGINE
# =========================
def generate_chapters(transcript, use_llm: bool = True):
    # Clean transcript
    segments = [t for t in transcript if len(t["text"]) > 15]
    if not segments:
        return ["0:00 Overview"]

    texts = [t["text"] for t in segments]
    times = [t["start"] for t in segments]

    # Embed & normalize
    embeddings = np.array(embedder.encode(texts))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_norm = embeddings / norms

    # Number of topics
    k = min(10, max(3, len(texts) // 20))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(emb_norm)

    # === MERGE CONSECUTIVE SAME-TOPIC SEGMENTS ===
    merged = []
    cur_label = labels[0]
    cur_start = times[0]
    cur_texts = [texts[0]]

    for i in range(1, len(labels)):
        if labels[i] == cur_label:
            cur_texts.append(texts[i])
        else:
            merged.append({
                "start": cur_start,
                "text": " ".join(cur_texts)
            })
            cur_label = labels[i]
            cur_start = times[i]
            cur_texts = [texts[i]]

    merged.append({
        "start": cur_start,
        "text": " ".join(cur_texts)
    })

    # === TITLE GENERATION ===
    chapters = []
    for block in merged:
        text_block = block["text"][:1200]

        title = llm_generate_title(text_block, use_llm=use_llm)
        if not title:
            title = summarizer.summarize(text_block, ratio=0.25)
        if not title or len(title) < 6:
            title = clean_title(text_block)

        chapters.append(f"{sec_to_time(block['start'])} {title}")

    return chapters


# =========================
# API
# =========================
@app.post("/chapters")
def chapters(video_id: str):
    return {"video_id": video_id, "chapters": generate_chapters(transcribe(video_id))}

@app.post("/debug")
def debug(payload: dict = Body(...)):
    """Return clustering details for a provided transcript (list of {start, text}).
    Use this endpoint to inspect which segments were chosen and what titles were generated.
    Accepts either raw list body or JSON object {"transcript": [...]}
    """
    # support both raw list body and object with 'transcript' key
    if isinstance(payload, dict) and "transcript" in payload and isinstance(payload["transcript"], list):
        transcript = payload["transcript"]
    elif isinstance(payload, list):
        transcript = payload
    else:
        raise HTTPException(status_code=422, detail="Invalid payload: expected list or {'transcript': [...]}")

    cleaned = [t for t in transcript if len(t.get("text", "")) > 15]
    texts = [t["text"] for t in cleaned]
    times = [t["start"] for t in cleaned]

    if not texts:
        return {"error": "no valid segments"}

    embeddings = np.array(embedder.encode(texts))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_norm = embeddings / norms

    k = min(12, max(2, len(texts)//12))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(emb_norm)

    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    out_clusters = []
    for label, indices in clusters.items():
        centroid = kmeans.cluster_centers_[label]
        c_norm = centroid / (np.linalg.norm(centroid) + 1e-9)
        sims = np.dot(emb_norm[indices], c_norm)
        ranked = sorted(zip(indices, sims), key=lambda x: x[1], reverse=True)
        top_n = [i for i, s in ranked[:3]]
        top_texts = [texts[i] for i in sorted(top_n, key=lambda i: times[i])]
        cluster_text = " ".join(top_texts)
        title = llm_generate_title(cluster_text, use_llm=True) or summarizer.summarize(cluster_text, ratio=0.2) or clean_title(cluster_text)

        out_clusters.append({
            "label": int(label),
            "size": len(indices),
            "top_indices": top_n,
            "top_times": [times[i] for i in top_n],
            "top_texts": top_texts,
            "cluster_text": cluster_text,
            "title": title
        })

    out_clusters.sort(key=lambda x: min(x["top_times"]) if x["top_times"] else 0)
    return {"k": k, "clusters": out_clusters}

@app.post("/compare")
def compare(payload: dict = Body(...)):
    """Return titles generated with and without LLM (A/B).
    Accepts either raw list body or JSON object {"transcript": [...]}
    """
    if isinstance(payload, dict) and "transcript" in payload and isinstance(payload["transcript"], list):
        transcript = payload["transcript"]
    elif isinstance(payload, list):
        transcript = payload
    else:
        raise HTTPException(status_code=422, detail="Invalid payload: expected list or {'transcript': [...]}")

    cleaned = [t for t in transcript if len(t.get("text", "")) > 15]
    with_llm = generate_chapters(cleaned, use_llm=True)
    without_llm = generate_chapters(cleaned, use_llm=False)
    return {"with_llm": with_llm, "without_llm": without_llm}

@app.get("/health")
def health():
    return {"llm_enabled": bool(client), "model": os.getenv("GROQ_MODEL", None)}

#done