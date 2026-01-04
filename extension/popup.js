const API = "http://localhost:8000/chapters";
let busy = false;

document.addEventListener("DOMContentLoaded", init);

async function init() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab || !tab.url.includes("youtube.com/watch")) {
    setStatus("Open a YouTube video");
    return;
  }

  const videoId = new URL(tab.url).searchParams.get("v");
  if (!videoId) {
    setStatus("Invalid YouTube video");
    return;
  }

  generateChapters(videoId, tab);
}

async function generateChapters(videoId, tab) {
  if (busy) return;
  busy = true;

  setStatus("Analyzing video…");

  try {
    const res = await fetch(`${API}?video_id=${videoId}`, { method: "POST" });

    if (!res.ok) {
      const errText = await res.text();
      console.error("Backend error:", errText);
      setStatus("Backend error. Check server logs.");
      busy = false;
      return;
    }

    const data = await res.json();

    if (!data.chapters || !Array.isArray(data.chapters)) {
      console.error("Invalid response:", data);
      setStatus("Invalid response from backend");
      busy = false;
      return;
    }

    renderChapters(data.chapters, tab);
    setStatus("");

  } catch (e) {
    console.error("Fetch failed:", e);
    setStatus("Failed to connect to backend");
  }

  busy = false;
}

function renderChapters(chapters, tab) {
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
        <img src="https://img.youtube.com/vi/${getVideoId(tab.url)}/mqdefault.jpg">
        <div class="time">${time}</div>
      </div>
      <div class="meta">
        <div class="title">${title}</div>
      </div>
    `;

    el.onclick = () => jumpTo(tab, seconds);
    list.appendChild(el);
  });
}

function jumpTo(sec) {
  chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
    const base = tab.url.split("&t=")[0];
    chrome.tabs.update(tab.id, {
      url: `${base}&t=${sec}`
    });
  });
}

function parseTime(t) {
  const [m, s] = t.split(":").map(Number);
  return m * 60 + s;
}

function getVideoId(url) {
  return new URL(url).searchParams.get("v");
}

function setStatus(text) {
  document.getElementById("status").innerText = text;
}

document.getElementById("closeBtn").onclick = () => {
  window.close();
};