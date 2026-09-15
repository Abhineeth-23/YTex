# Specification: Production Deployment & Zero-Leak Security Architecture

**Date**: 2026-09-15  
**Project**: YT Smart Chapterizer (YTex)  
**Status**: Approved  

---

## 1. Executive Summary

This specification defines the deployment pipeline and security architecture for **YT Smart Chapterizer**. 
The system allows any end-user to install and use the Chrome Extension for generating YouTube video chapters powered by the backend AI and Groq LLM, without requiring users to have their own API keys, while strictly guaranteeing that the host's `GROQ_API_KEY` and infrastructure credentials are never exposed, leaked, or accessible.

---

## 2. Security Architecture & Threat Model

### 2.1 Zero-Leak Key Protection
* **Server-Side Key Storage**: The `GROQ_API_KEY` exists exclusively inside the private memory of the Render Docker container environment variable. It is NEVER sent to the client, logged in output, or returned in error payloads.
* **Client Isolation**: The Chrome Extension communicates strictly via `POST /chapters?video_id={id}`. It has zero knowledge of or interaction with Groq.
* **Repository Safety**:
  * `.env` is permanently excluded via `.gitignore` and `.dockerignore`.
  * A template `.env.example` provides documentation without secrets.
  * CI/CD security scanning automatically scans files to prevent accidental commits containing API keys.

### 2.2 Endpoint Abuse Prevention
* **Input Validation**: The backend strictly validates `video_id` via regex (`^[a-zA-Z0-9_-]{11}$`). Any non-compliant request is rejected immediately with HTTP 400.
* **Fixed Prompt Boundaries**: Users cannot pass arbitrary text or prompts to the LLM. The server solely passes the verified video transcript to a hardcoded prompt template, eliminating prompt injection and free-LLM proxy abuse.
* **Caching**: Processed videos are cached in `app/cache/` to eliminate duplicate Groq API consumption for identical videos.
* **Rate Limiting**: An in-memory rate limiter protects the `/chapters` endpoint against spam and denial-of-service attempts.

---

## 3. Backend Deployment (Render)

### 3.1 Render Blueprint (`render.yaml`)
* Defined at the repository root as Infrastructure-as-Code.
* Deploys the FastAPI application inside Docker on Render's free tier with automatic HTTPS/TLS.
* Environment variable `GROQ_API_KEY` is configured with `sync: false` to require manual entry via Render's secure dashboard.
* Health check endpoint: `GET /health`.

### 3.2 CORS & Port Handling
* Updates `app/main.py` CORS configuration to allow Chrome extensions (`chrome-extension://*`) and production domains without conflicting with credential headers.
* Binds port flexibly using `$PORT` (defaulting to 8000) to adhere to cloud container standards.

---

## 4. Chrome Extension Production Configuration

### 4.1 Production API Routing
* `extension/background.js` points by default to the production Render URL (`https://yt-smart-chapterizer.onrender.com/chapters`), with fallback/configurable support in `chrome.storage.local` for local development.
* `extension/manifest.json` updates `host_permissions` to include `https://*.onrender.com/*` so network requests succeed under Manifest V3 security rules.

---

## 5. GitHub Pages Landing & Download Portal

### 5.1 Location & Aesthetics
* Hosted in `docs/` or deployed via GitHub Actions Pages artifact.
* Follows the extension's Sage & Lime Neo-Brutalist visual identity (Google Font *Outfit*, dark card widgets, crisp pill tags).

### 5.2 Content & User Flow
* Clear hero headline explaining the tool.
* Direct **"Download Extension (.zip)"** button.
* 3-step illustrated install guide:
  1. Download and extract the `.zip`.
  2. Open `chrome://extensions` and enable **Developer mode**.
  3. Click **Load unpacked** and select the folder.
* Privacy pledge: no user browsing data or login credentials collected.

---

## 6. GitHub Actions CI/CD Pipeline (`.github/workflows/deploy.yml`)

### 6.1 Triggering
* Push to `main` branch or manual `workflow_dispatch`.

### 6.2 Workflow Jobs
1. **Security & Secret Audit**:
   * Asserts `.env` is uncommitted.
   * Scans tracked files for accidental key exposure.
2. **Build & Package Extension**:
   * Creates `yt-smart-chapterizer.zip` containing `extension/`.
   * Bundles the ZIP into the GitHub Pages publication directory (`docs/`).
3. **Deploy to GitHub Pages**:
   * Uses `actions/upload-pages-artifact` and `actions/deploy-pages` to publish the documentation and download portal.
4. **Trigger Render Deployment**:
   * Securely pings `${{ secrets.RENDER_DEPLOY_HOOK_URL }}` via `curl` to trigger zero-downtime container rebuilds on Render.
