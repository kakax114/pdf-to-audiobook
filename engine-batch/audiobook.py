#!/usr/bin/env python3
"""
audiobook.py — Convert a PDF to an MP3 audiobook using Coqui TTS.

Usage:
    python audiobook.py mybook.pdf

Resume if interrupted:
    Just run the same command again — completed chunks are skipped automatically.

First-run setup:
    pip install TTS pymupdf
    brew install ffmpeg          # macOS
    # or: https://ffmpeg.org/download.html  (Windows/Linux)
"""

import sys
import re
import json
import time
import shutil
import subprocess
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Uncomment the second MODEL line for near-ElevenLabs quality (~1.8 GB download)
# MODEL     = "tts_models/en/ljspeech/tacotron2-DDC"          # fast, ~100 MB
MODEL       = "tts_models/multilingual/multi-dataset/xtts_v2"  # best quality

SPEAKER     = "Daisy Studious"  # xtts_v2 voice — see full list below
LANGUAGE    = "en"    # only used by multilingual models
CHUNK_CHARS = 220     # xtts_v2 has a ~250 char hard limit; stay safely under it
DISK_WARN_GB = 5      # warn if free disk space is below this (WAV files are large)
MAX_PAGES   = None    # set to an integer (e.g. 10) to only convert the first N pages
# ──────────────────────────────────────────────────────────────────────────────


# ─── Helpers ──────────────────────────────────────────────────────────────────

def banner(text):
    width = 62
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)

def section(label):
    print(f"\n▶ {label}")

def ok(msg):   print(f"  ✓ {msg}")
def warn(msg): print(f"  ⚠ {msg}")
def fail(msg): print(f"  ✗ {msg}")

def fmt_eta(seconds):
    if seconds < 60:    return f"{int(seconds)}s"
    if seconds < 3600:  return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m}m"

def progress_bar(current, total, eta):
    pct    = int(current / total * 100)
    filled = pct // 5
    bar    = "█" * filled + "░" * (20 - filled)
    print(f"\r  Chunk {current}/{total}  [{bar}]  {pct}%  ETA: {eta}   ",
          end="", flush=True)


# ─── Stage 1: Dependency check ────────────────────────────────────────────────

def check_dependencies():
    section("Stage 1/6 · Dependency check")
    errors = []

    try:
        import fitz
        ok("PyMuPDF (fitz)")
    except ImportError:
        fail("PyMuPDF missing  →  pip install pymupdf")
        errors.append("pymupdf")

    try:
        import TTS  # noqa: F401
        ok("Coqui TTS")
    except ImportError:
        fail("Coqui TTS missing  →  pip install TTS")
        errors.append("TTS")

    if shutil.which("ffmpeg"):
        ok(f"FFmpeg  ({shutil.which('ffmpeg')})")
    else:
        fail("FFmpeg missing  →  brew install ffmpeg  or  https://ffmpeg.org")
        errors.append("ffmpeg")

    if errors:
        print(f"\n  Fix the {len(errors)} issue(s) above and re-run.")
        sys.exit(1)


# ─── Stage 2: PDF extraction ──────────────────────────────────────────────────

def extract_and_clean(pdf_path: Path) -> str:
    section("Stage 2/6 · PDF extraction")
    import fitz

    doc     = fitz.open(str(pdf_path))
    pages   = []
    skipped = 0

    page_limit = MAX_PAGES if MAX_PAGES else len(doc)
    for page in list(doc)[:page_limit]:
        text = page.get_text()
        if text.strip():
            pages.append(text)
        else:
            skipped += 1

    if skipped:
        warn(f"{skipped} pages skipped (empty / scanned images — no text layer)")

    raw  = "\n".join(pages)
    text = _clean(raw)

    words = len(text.split())
    mins  = words / 150          # ~150 wpm speaking rate
    h, m  = int(mins // 60), int(mins % 60)

    ok(f"Pages extracted : {len(pages)}")
    ok(f"Word count      : {words:,}")
    ok(f"Est. duration   : {h}h {m}m")
    return text


def _clean(text: str) -> str:
    # Remove standalone page numbers (arabic or roman numerals)
    text = re.sub(r"^\s*[IVXLCDM]*\d*[IVXLCDM]*\s*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    # Remove TOC/index lines — text followed by leader dots and a page number
    # e.g. "Chapter One ........... 12"  or  "PART TWO . . . . 88"
    text = re.sub(r"^.{0,120}\.{2,}[\s\d]+$", "", text, flags=re.MULTILINE)
    # Remove lines that are nothing but dots, dashes, underscores (decorative rules)
    text = re.sub(r"^\s*[.\-_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Repair hyphenated line-breaks  (e.g. "some-\nthing" → "something")
    text = re.sub(r"-\n", "", text)
    # Collapse all remaining newlines to a single space
    text = re.sub(r"\n+", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ─── Stage 3: Chunking ────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    section("Stage 3/6 · Chunking text")

    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?…])\s+", text)

    chunks = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        sep = 1 if current else 0   # space that joins sentences

        if current and current_len + sep + sent_len > CHUNK_CHARS:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
            sep = 0

        if sent_len > CHUNK_CHARS:
            # Single sentence too long — hard-split it
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            for i in range(0, sent_len, CHUNK_CHARS):
                chunks.append(sent[i:i + CHUNK_CHARS])
        else:
            current.append(sent)
            current_len += sep + sent_len

    if current:
        chunks.append(" ".join(current))

    avg = sum(len(c) for c in chunks) // max(len(chunks), 1)
    ok(f"Chunks created  : {len(chunks)}")
    ok(f"Avg chunk size  : ~{avg} chars")
    return chunks


# ─── Stage 4: Load TTS model ──────────────────────────────────────────────────

def load_model():
    section("Stage 4/6 · Loading TTS model")
    print(f"  Model    : {MODEL}")
    print("  (First run will download the model — cached for all future runs)")

    from TTS.api import TTS
    tts = TTS(model_name=MODEL, progress_bar=True, gpu=False)
    ok("Model ready")
    return tts


# ─── Stage 5: Generate WAV chunks ─────────────────────────────────────────────

def generate_chunks(tts, chunks: list[str], tmp_dir: Path, resume_path: Path) -> list[Path]:
    section("Stage 5/6 · Generating audio")

    # Load resume state
    done: set[int] = set()
    if resume_path.exists():
        done = set(json.loads(resume_path.read_text()))
        if done:
            ok(f"Resuming — {len(done)}/{len(chunks)} chunks already done")

    total     = len(chunks)
    wav_files = [tmp_dir / f"chunk_{i:06d}.wav" for i in range(total)]
    t0        = time.time()
    n_done    = 0    # chunks completed this session
    any_work  = False

    print()
    for i, chunk in enumerate(chunks):
        if i in done:
            continue

        any_work = True
        remaining = total - len(done) - n_done
        eta = fmt_eta((time.time() - t0) / max(n_done, 1) * remaining) \
              if n_done >= 2 else "calculating..."
        progress_bar(i + 1, total, eta)

        try:
            kwargs: dict = {"text": chunk, "file_path": str(wav_files[i])}
            if SPEAKER:
                kwargs["speaker"] = SPEAKER
            if LANGUAGE and "multilingual" in MODEL:
                kwargs["language"] = LANGUAGE
            tts.tts_to_file(**kwargs)
        except Exception as exc:
            print()   # break progress line
            warn(f"Chunk {i} failed ({exc}) — skipping")
            continue

        done.add(i)
        n_done += 1
        resume_path.write_text(json.dumps(sorted(done)))

    if any_work:
        print()   # newline after final \r progress bar
    ok("Audio generation complete")
    return wav_files


# ─── Stage 6: Stitch + encode MP3 ─────────────────────────────────────────────

def stitch_to_mp3(wav_files: list[Path], output_mp3: Path, tmp_dir: Path):
    section("Stage 6/6 · Encoding MP3")

    existing = [f for f in wav_files if f.exists()]
    ok(f"Stitching {len(existing)} chunks")

    # Write ffmpeg concat list
    concat = tmp_dir / "concat_list.txt"
    concat.write_text("\n".join(f"file '{f.resolve()}'" for f in existing))

    print(f"  Encoding  → {output_mp3.name}")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat),
            "-codec:a", "libmp3lame", "-qscale:a", "2",   # VBR ~190 kbps
            str(output_mp3),
        ],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        fail("FFmpeg failed:")
        print(result.stderr)
        sys.exit(1)

    mb = output_mp3.stat().st_size / 1024**2
    ok(f"MP3 saved  {mb:.1f} MB  →  {output_mp3.resolve()}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python audiobook.py <book.pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        fail(f"File not found: {pdf_path}")
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        fail(f"Expected a .pdf file, got: {pdf_path.suffix}")
        sys.exit(1)

    output_mp3 = pdf_path.with_suffix(".mp3")
    tmp_dir    = pdf_path.parent / f".{pdf_path.stem}_tmp"
    tmp_dir.mkdir(exist_ok=True)
    resume     = tmp_dir / "progress.json"

    banner(f"AUDIOBOOK CONVERTER  ·  {pdf_path.name}  →  {output_mp3.name}")

    # Pre-flight disk check
    free_gb = shutil.disk_usage(pdf_path.parent).free / 1024**3
    if free_gb < DISK_WARN_GB:
        warn(f"Only {free_gb:.1f} GB free — recommend {DISK_WARN_GB} GB for WAV temp files")
        ans = input("  Continue anyway? [y/N]: ").strip().lower()
        if ans != "y":
            sys.exit(0)

    check_dependencies()
    text   = extract_and_clean(pdf_path)
    chunks = chunk_text(text)
    tts    = load_model()
    wavs   = generate_chunks(tts, chunks, tmp_dir, resume)
    stitch_to_mp3(wavs, output_mp3, tmp_dir)

    print("\n  Cleaning up temp WAV files...")
    shutil.rmtree(tmp_dir)

    banner(f"DONE  ·  {output_mp3.name}")


if __name__ == "__main__":
    main()
