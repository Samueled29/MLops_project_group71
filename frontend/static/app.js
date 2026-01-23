const ENDPOINT = "/predict";

const fileInput = document.getElementById("fileInput");
const runBtn = document.getElementById("runBtn");

const previewImg = document.getElementById("previewImg");
const placeholder = document.getElementById("placeholder");

const resultBox = document.getElementById("result");
const predEl = document.getElementById("pred");
const confEl = document.getElementById("conf");

const errorBox = document.getElementById("error");

let file = null;
let objectUrl = null;

function resetUI() {
  resultBox.classList.add("hidden");
  errorBox.classList.add("hidden");
  errorBox.textContent = "";

  predEl.textContent = "—";
  predEl.classList.remove("ok", "bad");
  confEl.textContent = "—";
}

function setPreview(f) {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;

  if (!f) {
    previewImg.classList.add("hidden");
    previewImg.removeAttribute("src");
    placeholder.classList.remove("hidden");
    return;
  }

  objectUrl = URL.createObjectURL(f);
  previewImg.src = objectUrl;
  previewImg.classList.remove("hidden");
  placeholder.classList.add("hidden");
}

fileInput.addEventListener("change", () => {
  resetUI();
  file = fileInput.files?.[0] ?? null;

  if (!file) {
    runBtn.disabled = true;
    setPreview(null);
    return;
  }

  // basic type check
  if (file.type && !file.type.startsWith("image/")) {
    file = null;
    runBtn.disabled = true;
    setPreview(null);
    errorBox.textContent = "Please upload an image file.";
    errorBox.classList.remove("hidden");
    return;
  }

  runBtn.disabled = false;
  setPreview(file);
});

runBtn.addEventListener("click", async () => {
  if (!file) return;

  resetUI();
  runBtn.disabled = true;
  runBtn.textContent = "Checking...";

  try {
    const form = new FormData();
    form.append("file", file); // must match FastAPI param name

    const res = await fetch(ENDPOINT, { method: "POST", body: form });
    const data = await res.json().catch(() => null);

    if (!res.ok || !data) {
      errorBox.textContent = `Request failed (HTTP ${res.status}).`;
      errorBox.classList.remove("hidden");
      return;
    }

    const prediction = String(data.prediction ?? "—");
    const confidence = Number(data.confidence);

    predEl.textContent = prediction.toUpperCase();
    if (prediction === "healthy") predEl.classList.add("ok");
    if (prediction === "rotten") predEl.classList.add("bad");

    if (Number.isFinite(confidence)) {
      confEl.textContent = `${Math.round(confidence * 100)}%`;
    } else {
      confEl.textContent = "—";
    }

    resultBox.classList.remove("hidden");
  } catch (e) {
    errorBox.textContent = "Network error. Please try again.";
    errorBox.classList.remove("hidden");
  } finally {
    runBtn.disabled = !file;
    runBtn.textContent = "Check image";
  }
});