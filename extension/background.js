// Production endpoint deployed on Render with fallback to user-configured custom endpoint
const DEFAULT_API_ENDPOINT = "https://yt-smart-chapterizer.onrender.com/chapters";

async function getApiEndpoint() {
  try {
    const data = await chrome.storage.local.get("customApiUrl");
    if (data.customApiUrl && data.customApiUrl.trim()) {
      return data.customApiUrl.trim();
    }
  } catch (e) {
    console.warn("Could not retrieve customApiUrl from storage:", e);
  }
  return DEFAULT_API_ENDPOINT;
}

// In-memory record of running background fetches
const activeJobs = {};

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const { action, videoId } = message;

  if (action === "GET_STATUS") {
    getStatus(videoId).then(sendResponse);
    return true; // Keep message channel open for asynchronous reply
  }

  if (action === "START_ANALYSIS") {
    startAnalysis(videoId, false).then(sendResponse);
    return true;
  }

  if (action === "REGENERATE_ANALYSIS") {
    startAnalysis(videoId, true).then(sendResponse);
    return true;
  }
});

async function getStatus(videoId) {
  if (activeJobs[videoId]) {
    return { status: "loading" };
  }

  const storageData = await chrome.storage.local.get(videoId);
  if (storageData[videoId]) {
    return { status: "success", chapters: storageData[videoId] };
  }

  return { status: "idle" };
}

async function startAnalysis(videoId, forceBypassCache = false) {
  if (activeJobs[videoId]) {
    return { status: "loading" };
  }

  if (!forceBypassCache) {
    const storageData = await chrome.storage.local.get(videoId);
    if (storageData[videoId]) {
      return { status: "success", chapters: storageData[videoId] };
    }
  } else {
    // Clear storage to force recalculation
    await chrome.storage.local.remove(videoId);
  }

  // Register background job
  activeJobs[videoId] = true;

  // Run the HTTP request detached from popup lifecycle
  fetchChapters(videoId)
    .then(async (chapters) => {
      await chrome.storage.local.set({ [videoId]: chapters });
      delete activeJobs[videoId];
      // Notify popup if it is open
      chrome.runtime.sendMessage({ action: "STATUS_UPDATE", videoId, status: "success", chapters }).catch(() => {
        // Silent catch: popup is closed
      });
    })
    .catch((err) => {
      console.error("Background error fetching chapters:", err);
      delete activeJobs[videoId];
      // Notify popup of error
      chrome.runtime.sendMessage({ action: "STATUS_UPDATE", videoId, status: "error", error: err.message }).catch(() => {
        // Silent catch: popup is closed
      });
    });

  return { status: "loading" };
}

async function fetchChapters(videoId) {
  const endpoint = await getApiEndpoint();
  const url = endpoint.includes("?")
    ? `${endpoint}&video_id=${encodeURIComponent(videoId)}`
    : `${endpoint}?video_id=${encodeURIComponent(videoId)}`;

  const res = await fetch(url, { method: "POST" });
  if (!res.ok) {
    let errText = "Server Error";
    try {
      const errJson = await res.json();
      errText = errJson.detail || JSON.stringify(errJson);
    } catch {
      try {
        errText = await res.text();
      } catch {}
    }
    throw new Error(errText);
  }
  const data = await res.json();
  if (!data.chapters || !Array.isArray(data.chapters)) {
    throw new Error("Invalid backend JSON response structure");
  }
  return data.chapters;
}
