"""
ABDA Phase 1 — Live Demo Backend (FastAPI).

Architecture:
  POST /api/analyze   ← upload media (audio or video)
      ↓
  save temp file
  extract audio (if video)
  normalise to 16 kHz mono
      ↓  (independent branches)
  ├── speech pipeline (existing)
  ├── facial pipeline (existing, if video)
  └── Whisper STT → text demo adapter
      ↓
  integrated display dict
  (no fusion, no BDI prediction)
      ↓
  JSON response → UI

GET /                 ← serve the demo UI (static HTML)

IMPORTANT:
  - Live Whisper transcription is marked "live_inference" — NOT CMU-MOSEI ground truth.
  - The validated CMU-MOSEI text pipeline is NOT called from here.
  - A separate lightweight demo text adapter is used instead.
  - OpenFace unavailability is always represented explicitly.
  - No fake fusion scores are returned.
  - No BDI/depression prediction is returned.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---- Project root must be on sys.path so pipelines resolve. ----------------
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.services.media import extract_audio_from_video, normalise_audio
from app.services.speech_service import run_speech_on_file
from app.services.face_service import run_face_on_file
from app.services.text_service import run_demo_text_adapter
from app.services.integration_service import assemble_demo_record

# ---------------------------------------------------------------------------

app = FastAPI(
    title="ABDA Multimodal Demo",
    description="ABDA Phase 1 live-demo backend — no fusion, no BDI prediction.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static UI from app/static/
_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Thread pool for CPU-bound modality processing.
_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# Temp directory for uploaded files.
_UPLOAD_DIR = Path(tempfile.gettempdir()) / "abda_demo_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the live-demo HTML interface."""
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        return JSONResponse(
            {"error": "UI not found. Run the app from the project root."},
            status_code=404,
        )
    return FileResponse(str(index_path))


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "ABDA Demo Backend"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Analyze an uploaded audio or video file.

    Steps:
      1. Save temporary upload.
      2. Extract / normalise audio to 16 kHz mono.
      3. In parallel: run speech pipeline + facial pipeline (if video).
      4. Run Whisper STT (marked as live_inference).
      5. Run lightweight demo text adapter on transcript.
      6. Assemble and return integrated display record.

    Returns JSON with all modality results.
    Does NOT perform fusion.
    Does NOT predict BDI/depression.
    """
    session_id = str(uuid.uuid4())[:8]
    suffix = Path(file.filename or "upload.mp4").suffix.lower() or ".mp4"
    upload_path = _UPLOAD_DIR / f"abda_{session_id}{suffix}"

    # Save upload.
    try:
        with upload_path.open("wb") as fout:
            content = await file.read()
            fout.write(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload save failed: {exc}")

    try:
        result = await _process_upload(upload_path, session_id, suffix)
    finally:
        try:
            upload_path.unlink(missing_ok=True)
        except Exception:
            pass

    return JSONResponse(result)


async def _process_upload(
    upload_path: Path,
    session_id: str,
    suffix: str,
) -> Dict[str, Any]:
    """Coordinate the parallel processing pipeline."""
    loop = asyncio.get_event_loop()
    is_video = suffix in {".mp4", ".mov", ".avi", ".webm", ".mkv"}

    # -- Step 1: audio path --
    if is_video:
        audio_path = _UPLOAD_DIR / f"abda_{session_id}.wav"
        try:
            await loop.run_in_executor(
                _EXECUTOR,
                extract_audio_from_video,
                upload_path,
                audio_path,
            )
        except Exception as exc:
            audio_path = None
    else:
        audio_path = upload_path

    # -- Step 2: normalise audio --
    if audio_path and audio_path.exists():
        norm_path = _UPLOAD_DIR / f"abda_{session_id}_norm.wav"
        try:
            await loop.run_in_executor(
                _EXECUTOR,
                normalise_audio,
                audio_path,
                norm_path,
            )
            if is_video:
                try:
                    audio_path.unlink(missing_ok=True)
                except Exception:
                    pass
            audio_path = norm_path
        except Exception:
            pass

    # -- Step 3: STT (Whisper) — needed before speech and text --
    transcript_result: Dict[str, Any] = {"text": "", "status": "unavailable", "source": "live_inference"}
    if audio_path and audio_path.exists():
        try:
            transcript_result = await loop.run_in_executor(
                _EXECUTOR,
                _run_whisper,
                audio_path,
            )
        except Exception as exc:
            transcript_result = {"text": "", "status": "error", "source": "live_inference", "error": str(exc)}

    # -- Step 4: parallel speech + facial --
    transcript_text = transcript_result.get("text", "") or ""

    speech_future = loop.run_in_executor(
        _EXECUTOR,
        run_speech_on_file,
        audio_path,
        transcript_text,
        session_id,
    ) if (audio_path and audio_path.exists()) else _done_future(loop, {"status": "unavailable"})

    face_future = loop.run_in_executor(
        _EXECUTOR,
        run_face_on_file,
        upload_path if is_video else None,
        session_id,
    ) if is_video else _done_future(loop, {"status": "not_applicable"})

    speech_result, face_result = await asyncio.gather(speech_future, face_future)

    # -- Step 5: demo text adapter --
    text_result: Dict[str, Any] = {"status": "unavailable"}
    if transcript_text.strip():
        try:
            text_result = await loop.run_in_executor(
                _EXECUTOR,
                run_demo_text_adapter,
                transcript_text,
            )
        except Exception as exc:
            text_result = {"status": "error", "error": str(exc)}

    # -- Step 6: assemble --
    record = assemble_demo_record(
        session_id=session_id,
        transcript=transcript_result,
        text=text_result,
        speech=speech_result,
        facial=face_result,
    )

    # Cleanup normalised audio.
    if audio_path and audio_path != upload_path:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass

    return record


def _run_whisper(audio_path: Path) -> Dict[str, Any]:
    """
    Run Whisper 'tiny' on the audio file for live demo transcription.

    IMPORTANT: this is live_inference, NOT CMU-MOSEI ground truth.
    """
    try:
        import whisper
        model = whisper.load_model("tiny")
        result = model.transcribe(str(audio_path), fp16=False)
        return {
            "text": result.get("text", "").strip(),
            "status": "available",
            "source": "live_inference",
            "model": "openai/whisper-tiny",
            "note": "Live demo transcription. NOT CMU-MOSEI ground-truth text.",
        }
    except Exception as exc:
        return {
            "text": "",
            "status": "error",
            "source": "live_inference",
            "error": str(exc),
        }


def _done_future(loop: asyncio.AbstractEventLoop, value: Any):
    """Return a future that is already resolved with value."""
    future = loop.create_future()
    future.set_result(value)
    return future


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start():
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    start()
