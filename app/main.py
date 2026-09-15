import json
import logging
import os
import re
import tempfile
import uuid

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from groq import Groq
from dotenv import load_dotenv
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

# =========================
# ENV + LOGGING
# =========================
load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured on backend. Please set it in server environment variables."
        )
    return Groq(api_key=api_key)

# =========================
# CACHE DIRECTORY
# =========================
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

VIDEO_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{11}$")

# =========================
# FASTAPI
# =========================
app = FastAPI(title="YT Smart Chapterizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://.*|https?://.*)$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)



# =========================
# MODELS (Lazy-loaded on demand to preserve RAM)
# =========================
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info("Initializing WhisperModel (base, int8)...")
        _whisper_model = WhisperModel("base", compute_type="int8")
    return _whisper_model

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
    # 1. Attempt to fetch pre-existing captions directly (Takes ~1 second)
    try:
        logger.info("Attempting to fetch pre-existing subtitles for video: %s", video_id)
        data = None
        # Try direct get_transcript (fastest)
        try:
            data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
        except Exception as e_direct:
            logger.info("Direct transcript fetch failed: %s, inspecting available transcripts...", e_direct)
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                t = None
                try:
                    t = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
                except Exception:
                    try:
                        t = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
                    except Exception:
                        t = next(iter(transcript_list), None)
                        if t and t.is_translatable:
                            t = t.translate('en')
                if t:
                    data = t.fetch()
            except Exception as e_list:
                logger.info("List transcripts failed: %s", e_list)

        if data:
            cleaned = [
                {"start": item["start"], "text": item["text"].strip()}
                for item in data
                if len(item.get("text", "").strip()) > 3
            ]
            if cleaned:
                logger.info("Successfully loaded %d captions from YouTube transcript API", len(cleaned))
                return cleaned
    except Exception as e:
        logger.warning("Subtitles fetch failed: %s. Falling back to Whisper transcription.", e)

    # 2. Fallback: Audio download + optimized Whisper transcription
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_template = os.path.join(tmpdir, "audio.%(ext)s")

            ydl_opts = {
                "format": "worstaudio/worst",
                "outtmpl": audio_template,
                "quiet": True,
                "extractor_args": {
                    "youtube": {
                        "player_client": ["default", "-android_sdkless"]
                    }
                },
                "http_headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            audio_file = None
            for file in os.listdir(tmpdir):
                if file.startswith("audio."):
                    audio_file = os.path.join(tmpdir, file)
                    break

            if not audio_file or not os.path.exists(audio_file):
                logger.warning("Audio file could not be downloaded via yt-dlp")
                return []

            logger.info("Transcribing audio using Whisper (beam_size=1, VAD enabled)")
            whisper = get_whisper()
            segments, _ = whisper.transcribe(audio_file, beam_size=1, vad_filter=True)
            
            cleaned = [
                {"start": s.start, "text": s.text.strip()}
                for s in segments
                if len(s.text.strip()) > 3
            ]
            return cleaned
    except Exception as e:
        logger.warning("Fallback audio download and transcription failed: %s", e)
        return []

# =========================
# OUTLINE TRANSCRIPT (TOKEN SAFE)
# =========================
def build_outline_transcript(segments, max_chars=20000):
    if not segments:
        return ""

    full_text = "\n".join(f"[{sec_to_time(s['start'])}] {s['text']}" for s in segments)
    if len(full_text) <= max_chars:
        return full_text

    # If transcript is very long, sample contiguous sentence pairs to maintain local context
    logger.info("Transcript exceeds %d characters, condensing...", max_chars)
    step = max(1, len(segments) // 50)
    chunks = []
    for i in range(0, len(segments), step):
        chunk = " ".join(s["text"] for s in segments[i:i+2])
        chunks.append(f"[{sec_to_time(segments[i]['start'])}] {chunk}")
    return "\n".join(chunks)

# =========================
# LLM: TOPIC PLANNING ONLY
# =========================
def safe_json_loads(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def llm_generate_chapters(outline_text: str):
    prompt = f"""You are a professional YouTube video editor.

Given the following timestamped transcript outline, generate 5 to 8 clear, topic-based chapter titles in strictly chronological order.
Rules:
- 5 to 8 chapters total
- Each title must be 3 to 6 words describing the overarching TOPIC, not speech verbatim
- Extract the exact start timestamp (format "M:SS" or "MM:SS") from the outline for each chapter
- Return ONLY valid JSON in this format:

[
  {{
    "timestamp": "0:00",
    "title": "Introduction & Overview"
  }}
]

Transcript outline:
{outline_text}
"""

    client = get_groq_client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.15,
        max_tokens=600,
    )

    content = response.choices[0].message.content.strip()
    return safe_json_loads(content)

# =========================
# FALLBACK (NEVER GARBAGE)
# =========================
def fallback_chapters(segments):
    n = len(segments)
    if n < 3:
        return ["0:00 Video Start"]

    # Space 4-6 dynamic sequential parts across segments
    num_chapters = min(6, max(3, n // 10))
    step = max(1, n // num_chapters)
    chapters = []
    for i in range(num_chapters):
        idx = min(i * step, n - 1)
        ts = segments[idx]["start"]
        part_name = f"Part {i + 1}" if i > 0 else "Introduction"
        chapters.append(f"{sec_to_time(ts)} {part_name}")

    return chapters

# =========================
# API
# =========================
@app.post("/chapters")
def chapters(video_id: str = Query(...)):
    # 1. Regex validation for YouTube Video ID format
    if not VIDEO_ID_REGEX.match(video_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube Video ID format"
        )

    # 2. Check local JSON cache
    cache_path = os.path.join(CACHE_DIR, f"{video_id}.json")
    if os.path.exists(cache_path):
        logger.info("Serving chapters for video %s from cache", video_id)
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read cache for video %s: %s", video_id, e)

    # 3. Process video
    try:
        segments = transcribe(video_id)

        if not segments:
            res = {
                "video_id": video_id,
                "chapters": ["0:00 Overview", "1:00 Discussion", "2:00 Key Insights", "3:00 Conclusion"],
            }
            return res

        outline = build_outline_transcript(segments)

        try:
            plan = llm_generate_chapters(outline)
            chapter_list = []
            if isinstance(plan, list):
                for item in plan:
                    if isinstance(item, dict):
                        ts = str(item.get("timestamp", "")).strip()
                        title = str(item.get("title", "")).strip()
                        if ts and title:
                            chapter_list.append(f"{ts} {title}")

            if len(chapter_list) < 2:
                raise RuntimeError("Weak LLM output")

        except Exception as e:
            logger.warning("LLM failed, using fallback: %s", e)
            chapter_list = fallback_chapters(segments)

        res = {
            "video_id": video_id,
            "chapters": chapter_list,
        }

        # 4. Save to cache
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to write cache for video %s: %s", video_id, e)

        return res

    except Exception as e:
        logger.exception("Hard failure while generating chapters for video %s", video_id)
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze video",
        )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_model": GROQ_MODEL,
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
    }

# =========================
# MOUNT LANDING PAGE & DOWNLOADS
# =========================
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
if os.path.exists(DOCS_DIR):
    app.mount("/", StaticFiles(directory=DOCS_DIR, html=True), name="site")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)