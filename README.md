# YT Smart Chapterizer ⚡

[![Deploy & Zero-Leak Security Pipeline](https://github.com/Abhineeth-23/YTex/actions/workflows/deploy.yml/badge.svg)](https://github.com/Abhineeth-23/YTex/actions/workflows/deploy.yml)

**YT Smart Chapterizer** generates topic-based, clickable chapters and timestamps for YouTube videos using AI. It pairs a **FastAPI backend** (running Whisper transcription, Groq LLM inference, and SentenceTransformer vector alignment) with a **Chrome Extension (Manifest V3)** that integrates directly with YouTube watch pages.

---

## 🔒 Zero-Leak Security Architecture

* **Zero Client Keys**: End users do **NOT** need an API key or an account. Anyone can install and use the extension immediately.
* **Server-Side Key Isolation**: Your `GROQ_API_KEY` is stored strictly in the private memory of your cloud backend (Render) environment variables. It is **never** sent to clients, logged in error responses, or committed to GitHub.
* **Strict Parameter Sanitization**: The API strictly enforces that the incoming `video_id` is an 11-character alphanumeric YouTube ID (`^[a-zA-Z0-9_-]{11}$`). Users cannot pass custom prompts to your LLM, preventing prompt injection or API abuse.
* **Smart Caching**: Once a video is analyzed, chapters are cached locally on disk (`app/cache/`) to eliminate duplicate Groq API requests.
* **Automated CI/CD Secret Scanning**: GitHub Actions automatically audits all code on every push to guarantee no `.env` or sensitive key patterns are ever tracked.

---

## 🚀 Deployment Guide

### 1. Deploy the Backend to Render (Free Docker Tier)
1. Fork or push this repository to GitHub.
2. Log in to [Render.com](https://render.com) and click **New +** -> **Blueprint**.
3. Connect your repository. Render will automatically detect [`render.yaml`](./render.yaml).
4. Render will prompt you for `GROQ_API_KEY`:
   * Enter your Groq API key (starts with `gsk_...`).
   * Leave `GROQ_MODEL` as `llama-3.1-8b-instant`.
5. Click **Apply**. Render will build the Docker container and provide a live URL (e.g., `https://yt-smart-chapterizer.onrender.com`).

### 2. Connect Automated Deployments in GitHub
1. In your Render Dashboard, go to your deployed service -> **Settings** -> scroll to **Deploy Hook**.
2. Copy the Deploy Hook URL.
3. In your GitHub repository, go to **Settings** -> **Secrets and variables** -> **Actions** -> **New repository secret**.
4. Name the secret `RENDER_DEPLOY_HOOK_URL` and paste the URL.
5. Whenever you push to `main`, GitHub Actions will automatically test, package, and trigger a fresh deployment!

### 3. Enable GitHub Pages for User Downloads & Docs
1. In your GitHub repository, go to **Settings** -> **Pages**.
2. Under **Build and deployment** -> **Source**, select **GitHub Actions**.
3. Push to `main` (or run the workflow manually under the **Actions** tab).
4. Your landing site and download portal will be live at `https://<username>.github.io/<repo>/`!

---

## 💻 Chrome Extension Installation (For Any User)

1. Download **`yt-smart-chapterizer.zip`** from your GitHub Pages portal or the repository.
2. Extract the ZIP file into a folder on your computer.
3. Open Google Chrome (or Brave, Edge) and go to `chrome://extensions`.
4. Turn ON **Developer mode** (toggle in the top-right corner).
5. Click **Load unpacked** and select the extracted `extension` folder.
6. Open any YouTube video and click the **Smart Chapters** extension icon in your toolbar!

---

## 🛠️ Local Development

### 1. Clone & Setup Python Environment
```bash
git clone https://github.com/Abhineeth-23/YTex.git
cd YTex

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` and insert your `GROQ_API_KEY`.

### 3. Run FastAPI Locally
```bash
uvicorn app.main:app --reload --port 8000
```
Visit `http://localhost:8000/health` to verify that the service is running.

---

## 📁 Repository Structure

```
.
├── .github/workflows/
│   └── deploy.yml            # CI/CD pipeline (Secret check, Pages deploy, Render hook)
├── app/
│   ├── main.py               # FastAPI backend (Whisper, Groq LLM, Vector DP alignment)
│   └── cache/                # Cached chapter outputs (git-ignored)
├── docs/
│   ├── index.html            # Neo-Brutalist landing & download page for GitHub Pages
│   └── style.css             # Sage & Lime styling
├── extension/
│   ├── manifest.json         # Chrome MV3 manifest
│   ├── background.js         # Service worker with Render production endpoint
│   ├── popup.html            # Extension popup widget
│   ├── popup.js              # State management & player timeline jumping
│   └── style.css             # Extension UI stylesheet
├── Dockerfile                # CPU-optimized PyTorch + Whisper container
├── render.yaml               # Render Infrastructure-as-Code Blueprint
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
└── README.md                 # Documentation
```
