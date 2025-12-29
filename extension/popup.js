let busy = false;

const API = "http://localhost:8000/chapters";

document.getElementById("generate").onclick = async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab.url.includes("youtube.com/watch")) {
    return alert("Open a YouTube video first.");
  }

  const videoId = new URL(tab.url).searchParams.get("v");
  if (!videoId) return alert("Invalid YouTube URL");

  generateChapters(videoId);
};

async function generateChapters(videoId) {
    if (busy) return;
    busy = true;
  const status = document.getElementById("status");
  status.innerText = "Analyzing video…";

  const res = await fetch(`${API}?video_id=${videoId}`, { method: "POST" });
  const data = await res.json();

  const list = document.getElementById("chapters");
  list.innerHTML = "";
  status.innerText = "";

  data.chapters.forEach(line => {
    const [time, ...title] = line.split(" ");
    const sec = parseTime(time);

    const li = document.createElement("li");
    li.innerText = line;
    li.onclick = () => jump(sec);
    list.appendChild(li);
  });
  busy = false;
}

function parseTime(t) {
  const [m, s] = t.split(":").map(Number);
  return m * 60 + s;
}

function jump(sec) {
  chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
    chrome.tabs.update(tabs[0].id, {
      url: tabs[0].url.split("&t=")[0] + "&t=" + sec
    });
  });
}
