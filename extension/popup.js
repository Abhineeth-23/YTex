let currentTab = null;
let currentVideoId = null;
let statusInterval = null;

const LOADING_STEPS = [
  "Downloading video audio from YouTube…",
  "Transcribing speech using offline AI…",
  "Structuring transcripts into outline…",
  "Aligning topics to timestamps via vector embeddings…",
  "Finalizing premium video chapters…"
];

document.addEventListener("DOMContentLoaded", init);

async function init() {
  // Bind close buttons
  document.getElementById("closeBtn").onclick = () => window.close();
  
  const regenerateBtn = document.getElementById("regenerateBtn");
  if (regenerateBtn) {
    regenerateBtn.onclick = handleRegenerate;
  }
  
  const retryBtn = document.getElementById("retryBtn");
  if (retryBtn) {
    retryBtn.onclick = startGeneration;
  }

  // Find active tab
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  currentTab = tab;

  if (!tab || !tab.url || !tab.url.includes("youtube.com/watch")) {
    showError("Please open a YouTube watch page");
    return;
  }

  currentVideoId = new URL(tab.url).searchParams.get("v");
  if (!currentVideoId) {
    showError("No video ID found in URL");
    return;
  }

  // Listen to background updates
  chrome.runtime.onMessage.addListener((message) => {
    if (message.action === "STATUS_UPDATE" && message.videoId === currentVideoId) {
      handleStatusResponse({ status: message.status, chapters: message.chapters, error: message.error });
    }
  });

  // Query state on startup
  const status = await chrome.runtime.sendMessage({ action: "GET_STATUS", videoId: currentVideoId });
  handleStatusResponse(status);
}

function handleStatusResponse(response) {
  stopLoadingCycle();

  if (response.status === "success") {
    showUIState("success");
    renderChapters(response.chapters);
  } else if (response.status === "loading") {
    showUIState("loading");
    startLoadingCycle();
  } else if (response.status === "error") {
    showError(response.error || "Analysis failed");
  } else {
    // Start job immediately if it hasn't run yet
    startGeneration();
  }
}

async function startGeneration() {
  showUIState("loading");
  startLoadingCycle();
  const response = await chrome.runtime.sendMessage({ action: "START_ANALYSIS", videoId: currentVideoId });
  if (response.status !== "loading") {
    handleStatusResponse(response);
  }
}

async function handleRegenerate() {
  showUIState("loading");
  startLoadingCycle();
  const response = await chrome.runtime.sendMessage({ action: "REGENERATE_ANALYSIS", videoId: currentVideoId });
  if (response.status !== "loading") {
    handleStatusResponse(response);
  }
}

function showUIState(state) {
  const loadingEl = document.getElementById("loadingSection");
  const errorEl = document.getElementById("errorSection");
  const chaptersEl = document.getElementById("chapters");
  const actionsEl = document.getElementById("actionsSection");

  loadingEl.style.display = state === "loading" ? "flex" : "none";
  errorEl.style.display = state === "error" ? "flex" : "none";
  chaptersEl.style.display = state === "success" ? "flex" : "none";
  if (actionsEl) {
    actionsEl.style.display = state === "success" ? "flex" : "none";
  }
}

function showError(msg) {
  showUIState("error");
  document.getElementById("errorMsg").innerText = msg;
}

function startLoadingCycle() {
  const statusTextEl = document.getElementById("statusText");
  let step = 0;
  statusTextEl.innerText = LOADING_STEPS[0];

  statusInterval = setInterval(() => {
    step = (step + 1) % LOADING_STEPS.length;
    statusTextEl.innerText = LOADING_STEPS[step];
  }, 5000);
}

function stopLoadingCycle() {
  if (statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
}

function renderChapters(chapters) {
  const list = document.getElementById("chapters");
  list.innerHTML = "";

  chapters.forEach(line => {
    const [time, ...titleParts] = line.split(" ");
    const title = titleParts.join(" ");
    const seconds = parseTime(time);

    const el = document.createElement("div");
    el.className = "chapter";

    el.innerHTML = `
      <div class="thumb">
        <img src="https://img.youtube.com/vi/${currentVideoId}/mqdefault.jpg" alt="Thumbnail">
        <div class="time">${time}</div>
      </div>
      <div class="meta">
        <div class="title">${title}</div>
      </div>
    `;

    el.onclick = () => jumpTo(seconds);
    list.appendChild(el);
  });
}

function jumpTo(sec) {
  if (!currentTab || !currentTab.id) return;
  chrome.scripting.executeScript({
    target: { tabId: currentTab.id },
    func: (seconds) => {
      const video = document.querySelector('video');
      if (video) {
        video.currentTime = seconds;
        video.play();
      }
    },
    args: [sec]
  });
}

function parseTime(t) {
  const parts = t.split(":").map(Number);
  if (parts.length === 3) {
    // HH:MM:SS
    const [h, m, s] = parts;
    return h * 3600 + m * 60 + s;
  } else if (parts.length === 2) {
    // MM:SS
    const [m, s] = parts;
    return m * 60 + s;
  }
  return 0;
}