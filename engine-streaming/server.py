#!/usr/bin/env python3
"""
server.py — Streaming PDF-to-audiobook web server.

Upload a PDF → audio chunks are generated one by one →
browser starts playing chunk 1 while chunks 2, 3, 4... are still generating.

Usage:
    python3 server.py

Then open: http://localhost:8080

Setup (same as audiobook.py):
    pip install TTS pymupdf flask flask-sock
    brew install ffmpeg
"""

import os
import re
import json
import uuid
import time
import shutil
import threading
import subprocess
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, send_file, abort
from flask_sock import Sock

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# MODEL     = "tts_models/en/ljspeech/tacotron2-DDC"          # fast, ~100 MB
MODEL       = "tts_models/multilingual/multi-dataset/xtts_v2"  # best quality
SPEAKER     = "Daisy Studious"
LANGUAGE    = "en"
CHUNK_WORDS = 80
HOST        = "0.0.0.0"
PORT        = 8080
JOBS_DIR    = Path("jobs")          # temp storage for job files
MAX_UPLOAD_MB = 50
# ──────────────────────────────────────────────────────────────────────────────

app  = Flask(__name__)
sock = Sock(app)

JOBS_DIR.mkdir(exist_ok=True)

# Global TTS instance — loaded once at startup
tts_model = None
tts_lock  = threading.Lock()   # only one TTS job at a time


# ─── TTS loader ───────────────────────────────────────────────────────────────

def load_tts():
    global tts_model
    print(f"  Loading TTS model: {MODEL}")
    from TTS.api import TTS
    tts_model = TTS(model_name=MODEL, progress_bar=False, gpu=False)
    print("  Model ready.\n")


# ─── Text helpers ──────────────────────────────────────────────────────────────

def extract_text(pdf_path: Path) -> str:
    import fitz
    doc   = fitz.open(str(pdf_path))
    pages = [p.get_text() for p in doc if p.get_text().strip()]
    raw   = "\n".join(pages)
    # Clean
    raw = re.sub(r"^\s*\d+\s*$", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"-\n", "", raw)
    raw = re.sub(r"\n+", " ", raw)
    raw = re.sub(r" {2,}", " ", raw)
    return raw.strip()


def chunk_text(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?…])\s+", text)
    chunks, current, count = [], [], 0
    for sent in sentences:
        w = len(sent.split())
        if count + w > CHUNK_WORDS and current:
            chunks.append(" ".join(current))
            current, count = [sent], w
        else:
            current.append(sent)
            count += w
    if current:
        chunks.append(" ".join(current))
    return chunks


# ─── Job runner (runs in background thread) ────────────────────────────────────

def run_job(job_id: str, pdf_path: Path):
    """Convert PDF chunk by chunk, saving each as an MP3 as soon as it's ready."""
    job_dir   = JOBS_DIR / job_id
    state_file = job_dir / "state.json"

    def set_state(data: dict):
        state_file.write_text(json.dumps(data))

    set_state({"status": "extracting", "chunks_ready": [], "total": 0, "error": None})

    try:
        # Extract + chunk
        text   = extract_text(pdf_path)
        chunks = chunk_text(text)
        total  = len(chunks)
        set_state({"status": "converting", "chunks_ready": [], "total": total, "error": None})

        ready = []
        with tts_lock:
            for i, chunk in enumerate(chunks):
                wav_path = job_dir / f"chunk_{i:05d}.wav"
                mp3_path = job_dir / f"chunk_{i:05d}.mp3"

                # Generate WAV
                try:
                    kwargs = {"text": chunk, "file_path": str(wav_path)}
                    if SPEAKER:
                        kwargs["speaker"] = SPEAKER
                    if LANGUAGE and "multilingual" in MODEL:
                        kwargs["language"] = LANGUAGE
                    tts_model.tts_to_file(**kwargs)
                except Exception as e:
                    print(f"  [job {job_id}] chunk {i} failed: {e}")
                    continue

                # WAV → MP3
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(wav_path),
                     "-codec:a", "libmp3lame", "-qscale:a", "4", str(mp3_path)],
                    capture_output=True
                )
                wav_path.unlink(missing_ok=True)

                ready.append(i)
                set_state({"status": "converting", "chunks_ready": ready, "total": total, "error": None})

        set_state({"status": "done", "chunks_ready": ready, "total": total, "error": None})

    except Exception as e:
        set_state({"status": "error", "chunks_ready": [], "total": 0, "error": str(e)})

    finally:
        pdf_path.unlink(missing_ok=True)


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/upload", methods=["POST"])
def upload():
    if "pdf" not in request.files:
        return jsonify({"error": "No file"}), 400

    f = request.files["pdf"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Must be a PDF"}), 400

    job_id  = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir()

    pdf_path = job_dir / "input.pdf"
    f.save(str(pdf_path))

    # Start conversion in background thread
    t = threading.Thread(target=run_job, args=(job_id, pdf_path), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    state_file = JOBS_DIR / job_id / "state.json"
    if not state_file.exists():
        abort(404)
    return jsonify(json.loads(state_file.read_text()))


@app.route("/chunk/<job_id>/<int:chunk_index>")
def get_chunk(job_id, chunk_index):
    mp3 = JOBS_DIR / job_id / f"chunk_{chunk_index:05d}.mp3"
    if not mp3.exists():
        abort(404)
    return send_file(str(mp3), mimetype="audio/mpeg")


@app.route("/download/<job_id>")
def download_full(job_id):
    """Stitch all chunks into one MP3 and return it for download."""
    job_dir    = JOBS_DIR / job_id
    state_file = job_dir / "state.json"
    if not state_file.exists():
        abort(404)

    state = json.loads(state_file.read_text())
    if state["status"] != "done":
        return jsonify({"error": "Not ready yet"}), 400

    chunks  = sorted(job_dir.glob("chunk_*.mp3"))
    concat  = job_dir / "concat_list.txt"
    output  = job_dir / "full.mp3"

    concat.write_text("\n".join(f"file '{c.resolve()}'" for c in chunks))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(concat), "-codec:a", "copy", str(output)],
        capture_output=True
    )
    return send_file(str(output), as_attachment=True,
                     download_name="audiobook.mp3", mimetype="audio/mpeg")


# ─── Inline HTML/JS player ─────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PDF to Audiobook</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0f0f0f; color: #eee;
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; padding: 2rem;
  }
  h1 { font-size: 1.8rem; margin-bottom: .4rem; }
  p.sub { color: #888; margin-bottom: 2rem; font-size: .95rem; }

  .card {
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 16px; padding: 2rem; width: 100%; max-width: 520px;
  }

  /* Upload area */
  #drop-zone {
    border: 2px dashed #333; border-radius: 12px;
    padding: 2.5rem; text-align: center; cursor: pointer;
    transition: border-color .2s;
  }
  #drop-zone:hover, #drop-zone.drag-over { border-color: #7c6af7; }
  #drop-zone .icon { font-size: 2.5rem; margin-bottom: .5rem; }
  #drop-zone p { color: #888; font-size: .9rem; }
  #drop-zone strong { color: #ccc; }
  #file-input { display: none; }

  /* Progress */
  #progress-section { display: none; margin-top: 1.5rem; }
  .progress-bar-bg {
    background: #2a2a2a; border-radius: 99px;
    height: 6px; margin: .8rem 0;
  }
  .progress-bar-fill {
    background: linear-gradient(90deg, #7c6af7, #a78bfa);
    height: 6px; border-radius: 99px;
    transition: width .4s ease;
  }
  #status-text { font-size: .85rem; color: #888; }

  /* Player */
  #player-section { display: none; margin-top: 1.5rem; }
  audio { width: 100%; border-radius: 8px; }

  .chunk-list {
    margin-top: 1rem; max-height: 180px; overflow-y: auto;
    font-size: .8rem; color: #666;
  }
  .chunk-item { padding: .25rem .5rem; border-radius: 4px; cursor: pointer; }
  .chunk-item:hover { background: #222; color: #ccc; }
  .chunk-item.playing { color: #a78bfa; }
  .chunk-item.ready { color: #888; }
  .chunk-item.pending { color: #444; cursor: default; }

  /* Download button */
  #download-btn {
    display: none; margin-top: 1rem; width: 100%;
    padding: .75rem; background: #7c6af7; color: #fff;
    border: none; border-radius: 10px; font-size: 1rem;
    cursor: pointer; transition: background .2s;
  }
  #download-btn:hover { background: #6a58e0; }

  .reset-btn {
    margin-top: 1rem; width: 100%;
    padding: .6rem; background: transparent; color: #555;
    border: 1px solid #333; border-radius: 10px; font-size: .9rem;
    cursor: pointer; transition: color .2s, border-color .2s;
  }
  .reset-btn:hover { color: #ccc; border-color: #555; }
</style>
</head>
<body>
<h1>PDF → Audiobook</h1>
<p class="sub">Upload a PDF and start listening within minutes</p>

<div class="card">
  <!-- Upload -->
  <div id="drop-zone" onclick="document.getElementById('file-input').click()">
    <div class="icon">📄</div>
    <strong>Click to upload or drag & drop</strong>
    <p>PDF files only · max 50 MB</p>
  </div>
  <input type="file" id="file-input" accept=".pdf">

  <!-- Progress -->
  <div id="progress-section">
    <div id="status-text">Uploading...</div>
    <div class="progress-bar-bg">
      <div class="progress-bar-fill" id="progress-fill" style="width:0%"></div>
    </div>
  </div>

  <!-- Player -->
  <div id="player-section">
    <audio id="audio-player" controls></audio>
    <div class="chunk-list" id="chunk-list"></div>
  </div>

  <button id="download-btn" onclick="downloadFull()">⬇ Download full MP3</button>
  <button class="reset-btn" id="reset-btn" style="display:none" onclick="reset()">Convert another file</button>
</div>

<script>
let jobId       = null;
let totalChunks = 0;
let readyChunks = [];
let currentChunk = 0;
let pollTimer   = null;
let chunkQueue  = [];
let isPlaying   = false;

const dropZone  = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const audio     = document.getElementById("audio-player");

// ── Drag & drop ──
dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});
fileInput.addEventListener("change", e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

// ── Upload ──
async function handleFile(file) {
  if (!file.name.endsWith(".pdf")) return alert("Please upload a PDF file.");
  const fd = new FormData();
  fd.append("pdf", file);

  showProgress("Uploading...");

  const res  = await fetch("/upload", { method: "POST", body: fd });
  const data = await res.json();
  if (data.error) return alert(data.error);

  jobId = data.job_id;
  pollTimer = setInterval(poll, 2000);
}

// ── Polling ──
async function poll() {
  const res   = await fetch(`/status/${jobId}`);
  const state = await res.json();

  totalChunks = state.total || 0;
  readyChunks = state.chunks_ready || [];

  const pct = totalChunks > 0 ? Math.round(readyChunks.length / totalChunks * 100) : 0;
  setProgress(pct);

  if (state.status === "extracting") {
    setStatusText("Extracting text from PDF...");
  } else if (state.status === "converting") {
    setStatusText(`Generating audio… ${readyChunks.length} / ${totalChunks} chunks ready`);
    buildChunkList();
    // Auto-start playing when first chunk is ready
    if (readyChunks.length > 0 && !isPlaying) startPlaying();
  } else if (state.status === "done") {
    clearInterval(pollTimer);
    setStatusText(`Done — ${totalChunks} chunks generated`);
    setProgress(100);
    buildChunkList();
    document.getElementById("download-btn").style.display = "block";
    document.getElementById("reset-btn").style.display    = "block";
    if (!isPlaying) startPlaying();
  } else if (state.status === "error") {
    clearInterval(pollTimer);
    setStatusText(`Error: ${state.error}`);
  }
}

// ── Player ──
function startPlaying() {
  if (readyChunks.length === 0) return;
  isPlaying    = true;
  currentChunk = readyChunks[0];
  playChunk(currentChunk);
  document.getElementById("player-section").style.display = "block";
}

function playChunk(index) {
  highlightChunk(index);
  audio.src = `/chunk/${jobId}/${index}`;
  audio.play().catch(() => {});
}

audio.addEventListener("ended", () => {
  // Find next ready chunk after current
  const sorted = [...readyChunks].sort((a, b) => a - b);
  const pos    = sorted.indexOf(currentChunk);
  if (pos >= 0 && pos + 1 < sorted.length) {
    currentChunk = sorted[pos + 1];
    playChunk(currentChunk);
  } else {
    // Not ready yet — wait and retry
    setTimeout(() => {
      const next = currentChunk + 1;
      if (readyChunks.includes(next)) {
        currentChunk = next;
        playChunk(currentChunk);
      }
    }, 2000);
  }
});

function buildChunkList() {
  const list = document.getElementById("chunk-list");
  list.innerHTML = "";
  for (let i = 0; i < totalChunks; i++) {
    const div = document.createElement("div");
    div.id    = `chunk-${i}`;
    const isReady = readyChunks.includes(i);
    div.className = `chunk-item ${isReady ? "ready" : "pending"}`;
    div.textContent = `${isReady ? "▶" : "○"} Segment ${i + 1}`;
    if (isReady) div.onclick = () => { currentChunk = i; playChunk(i); };
    list.appendChild(div);
  }
  highlightChunk(currentChunk);
}

function highlightChunk(index) {
  document.querySelectorAll(".chunk-item").forEach(el => el.classList.remove("playing"));
  const el = document.getElementById(`chunk-${index}`);
  if (el) { el.classList.add("playing"); el.scrollIntoView({ block: "nearest" }); }
}

// ── Download ──
function downloadFull() {
  window.location.href = `/download/${jobId}`;
}

// ── UI helpers ──
function showProgress(msg) {
  document.getElementById("progress-section").style.display = "block";
  setStatusText(msg);
}
function setProgress(pct) {
  document.getElementById("progress-fill").style.width = pct + "%";
}
function setStatusText(msg) {
  document.getElementById("status-text").textContent = msg;
}
function reset() {
  clearInterval(pollTimer);
  jobId = null; totalChunks = 0; readyChunks = []; currentChunk = 0; isPlaying = false;
  audio.pause(); audio.src = "";
  document.getElementById("progress-section").style.display = "none";
  document.getElementById("player-section").style.display   = "none";
  document.getElementById("download-btn").style.display     = "none";
  document.getElementById("reset-btn").style.display        = "none";
  document.getElementById("chunk-list").innerHTML            = "";
  document.getElementById("progress-fill").style.width      = "0%";
  fileInput.value = "";
}
</script>
</body>
</html>
"""

# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 56)
    print("  STREAMING AUDIOBOOK SERVER")
    print("═" * 56)

    if not shutil.which("ffmpeg"):
        print("  ✗ FFmpeg not found — install it first")
        exit(1)

    load_tts()
    print(f"  Open in browser: http://localhost:{PORT}\n")
    app.run(host=HOST, port=PORT, debug=False)
