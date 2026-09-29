"""
ABDA Phase 1 — Media service.

Audio extraction from video and normalisation to 16 kHz mono.
This is only for the live-demo; it does NOT modify the research pipelines.
"""

from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

import librosa
import soundfile as sf
import numpy as np


def extract_audio_from_video(video_path: Path, output_wav: Path) -> Path:
    """
    Extract audio from video to a WAV file using ffmpeg.

    Falls back to librosa if ffmpeg is unavailable.
    """
    video_path = Path(video_path)
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    # Try ffmpeg first (best quality).
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(output_wav),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and output_wav.exists():
            return output_wav
        # Fall through to librosa on failure.

    # librosa fallback.
    audio, sr = librosa.load(str(video_path), sr=16000, mono=True)
    sf.write(str(output_wav), audio.astype(np.float32), 16000, subtype="PCM_16")
    return output_wav


def normalise_audio(input_path: Path, output_path: Path, target_sr: int = 16000) -> Path:
    """
    Normalise audio to 16 kHz mono float32 WAV.

    This matches the format expected by the existing speech pipeline.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio, sr = librosa.load(str(input_path), sr=target_sr, mono=True)
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError(f"Audio is empty after loading: {input_path}")

    sf.write(str(output_path), audio, target_sr, subtype="PCM_16")
    return output_path
