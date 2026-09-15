# Deployment and Zero-Leak Security Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement automated, zero-leak deployment for YT Smart Chapterizer: Render blueprint for FastAPI backend, Chrome Extension production packaging, GitHub Pages landing/download portal, and GitHub Actions CI/CD with security scanning.

**Architecture:** Dockerized FastAPI running on Render with `GROQ_API_KEY` stored exclusively in server environment variables. Chrome Extension configured for production with fallback local override. GitHub Pages serves a Sage-and-Lime landing page and downloadable ZIP. GitHub Actions automates secret scanning, packaging, Pages deployment, and Render hook triggers.

**Tech Stack:** Python/FastAPI, Docker, Render Blueprints (`render.yaml`), Chrome Extension MV3, Vanilla HTML/CSS/JS (Neo-Brutalist), GitHub Actions (`deploy.yml`, Pages).

**Spec:** `docs/superpowers/specs/2026-09-15-deployment-and-security-design.md`

## Global Constraints

- Never commit `.env` or any real API keys (`gsk_...`) to git.
- The Chrome Extension must never contain or request Groq API keys directly.
- The backend must accept requests without leaking server keys, protecting against prompt injection by accepting only validated 11-char YouTube video IDs.
- GitHub Actions must fail if any `.env` file or sensitive key pattern is tracked.

---

### Task 1: Security Hardening & Secret Isolation

**Files:**
- Modify: `.gitignore`
- Create: `.env.example`
- Modify: `app/main.py:43-55`

**Interfaces:**
- Consumes: Environment variables (`GROQ_API_KEY`, `GROQ_MODEL`, `PORT`)
- Produces: Sanitized FastAPI app with compliant CORS and strict input regex

- [ ] **Step 1: Update .gitignore with comprehensive security rules**
Add rules to ignore all `.env*` files, cache files (`app/cache/*.json`), media files, and build artifacts.

- [ ] **Step 2: Create .env.example template**
Provide a safe template without real secrets for developers:
```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
PORT=8000
```

- [ ] **Step 3: Update CORS and Port binding in app/main.py**
Fix CORS configuration to explicitly permit Chrome extension origins (`chrome-extension://*`) and Web origins without invalid wildcard credential mixing:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://.*|https?://.*)$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
```
Ensure the app gracefully handles `PORT` environment variable for cloud container environments.

- [ ] **Step 4: Verify backend syntax and security configuration**
Run python syntax compilation:
`python -m py_compile app/main.py`
Expected: Returncode 0.

- [ ] **Step 5: Commit changes**
```bash
git add .gitignore .env.example app/main.py
git commit -m "feat(security): harden gitignore, add env template, and configure safe CORS"
```

---

### Task 2: Render Blueprint & Container Port Support

**Files:**
- Create: `render.yaml`
- Modify: `Dockerfile:22-24`

**Interfaces:**
- Consumes: Docker container definition, `PORT` environment variable
- Produces: Render Blueprint for zero-downtime, secret-isolated web service

- [ ] **Step 1: Update Dockerfile CMD to handle dynamic $PORT**
Modify `Dockerfile` to read `$PORT` (assigned dynamically by Render/Cloud platforms) with default 8000:
```dockerfile
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
```

- [ ] **Step 2: Create render.yaml blueprint**
Define service with `sync: false` for `GROQ_API_KEY` so keys are set strictly through Render's dashboard:
```yaml
services:
  - type: web
    name: yt-smart-chapterizer
    env: docker
    plan: free
    region: oregon
    dockerfilePath: ./Dockerfile
    healthCheckPath: /health
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: GROQ_MODEL
        value: llama-3.1-8b-instant
```

- [ ] **Step 3: Verify render.yaml syntax**
Validate YAML formatting.

- [ ] **Step 4: Commit changes**
```bash
git add Dockerfile render.yaml
git commit -m "feat(deploy): configure render.yaml blueprint and dynamic container port"
```

---

### Task 3: Chrome Extension Production URL & Host Permissions

**Files:**
- Modify: `extension/manifest.json`
- Modify: `extension/background.js:1-2`

**Interfaces:**
- Consumes: Deployed Render backend URL
- Produces: Production-ready extension with dynamic endpoint override capability

- [ ] **Step 1: Update extension/manifest.json host permissions**
Add `https://*.onrender.com/*` so Chrome MV3 allows background fetch requests to Render:
```json
  "host_permissions": [
    "https://www.youtube.com/*",
    "https://*.onrender.com/*"
  ]
```

- [ ] **Step 2: Update extension/background.js with production URL and local storage override**
Allow using the production Render service by default while honoring a custom endpoint in `chrome.storage.local` if set:
```javascript
const DEFAULT_API = "https://yt-smart-chapterizer.onrender.com/chapters";

async function getApiUrl() {
  const data = await chrome.storage.local.get("customApiUrl");
  return data.customApiUrl || DEFAULT_API;
}
```

- [ ] **Step 3: Verify extension manifest and JavaScript**
Check JSON valid format of `extension/manifest.json`.

- [ ] **Step 4: Commit changes**
```bash
git add extension/manifest.json extension/background.js
git commit -m "feat(extension): add render host permissions and dynamic API endpoint"
```

---

### Task 4: GitHub Pages Landing Site & Download Portal

**Files:**
- Create: `docs/index.html`
- Create: `docs/style.css`

**Interfaces:**
- Consumes: Extension distribution package (`yt-smart-chapterizer.zip`)
- Produces: Neo-Brutalist landing site with quick download & installation instructions

- [ ] **Step 1: Create docs/index.html**
Write semantic HTML featuring:
- Hero header with Sage & Lime Neo-Brutalist theme.
- Direct "Download Extension (.zip)" CTA button pointing to `./yt-smart-chapterizer.zip`.
- 3-step installation card with visual steps (Unpack -> Developer Mode -> Load Unpacked).
- Privacy and Security section clarifying zero user tracking and server-side key safety.

- [ ] **Step 2: Create docs/style.css**
Match the extension's exact styling tokens:
- Font: Google Fonts `Outfit`.
- Color tokens: `--sage-bg: #a9b8a8`, `--lime-accent: #d2ff36`, `--card-dark: #121214`, `--card-white: #fafbfa`.
- Fluid responsive grid, sleek badge tags, and pill buttons.

- [ ] **Step 3: Verify local rendering of docs/index.html**
Validate HTML and CSS files.

- [ ] **Step 4: Commit changes**
```bash
git add docs/index.html docs/style.css
git commit -m "feat(pages): add neo-brutalist landing page and install guide for GitHub Pages"
```

---

### Task 5: GitHub Actions CI/CD Pipeline & Secret Scanning

**Files:**
- Create: `.github/workflows/deploy.yml`

**Interfaces:**
- Consumes: GitHub Secrets (`RENDER_DEPLOY_HOOK_URL`, `GITHUB_TOKEN`)
- Produces: Automated extension ZIP artifact, GitHub Pages site, and Render deploy trigger

- [ ] **Step 1: Create .github/workflows/deploy.yml**
Define complete workflow:
- Triggers on `push: branches: [ main ]` and `workflow_dispatch`.
- Job 1: `security-check` (verify `.env` is not tracked, verify no `gsk_` strings in git).
- Job 2: `build-and-deploy-pages`:
  - Checkout repository.
  - Create `yt-smart-chapterizer.zip` from `extension/` folder.
  - Move ZIP into `docs/`.
  - Configure GitHub Pages via `actions/configure-pages@v5`.
  - Upload Pages artifact via `actions/upload-pages-artifact@v3` (path: `docs`).
  - Deploy to Pages via `actions/deploy-pages@v4`.
- Job 3: `deploy-render`:
  - Runs if `security-check` passes.
  - If `RENDER_DEPLOY_HOOK_URL` secret is set, curls the hook to trigger deployment on Render; otherwise prints setup instructions.

- [ ] **Step 2: Verify workflow YAML syntax**
Ensure valid YAML schema.

- [ ] **Step 3: Commit changes**
```bash
git add .github/workflows/deploy.yml
git commit -m "ci: add automated security scanning, extension packaging, pages deploy, and render hook"
```

---

### Task 6: End-to-End Verification & Documentation

**Files:**
- Create: `README.md` (or update existing)
- Test: Local ZIP generation test & security check script run

- [ ] **Step 1: Run local security scan script to test rules**
Verify that `.env` is untracked and no secrets exist in Git.

- [ ] **Step 2: Test building the extension ZIP locally**
Verify that `yt-smart-chapterizer.zip` generates cleanly and contains all needed extension files without extraneous caches.

- [ ] **Step 3: Update README.md with clear Deployment & Setup Instructions**
Document:
- How to deploy on Render using `render.yaml`.
- How to add `RENDER_DEPLOY_HOOK_URL` to GitHub Secrets.
- How to enable GitHub Pages in repo Settings (`Deploy from a branch` -> `GitHub Actions`).
- How users download and install the extension.

- [ ] **Step 4: Commit changes**
```bash
git add README.md
git commit -m "docs: add comprehensive deployment and user installation guide"
```
