"""
ABDA Phase 1 — Speech/audio extraction utilities.

Refactored from the Phase 1 Audio_Processing notebook:
- 16 kHz mono loading
- Librosa energy-based VAD
- speech-region selection for acoustic features

This module contains no participant-specific hard-coded values.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def load_audio(
    audio_path: str | Path,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at the target sampling rate."""
    audio_path = str(audio_path)

    audio, sr = librosa.load(
        audio_path,
        sr=target_sr,
        mono=True,
    )

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    return audio, int(sr)


def detect_voice_activity(
    audio: np.ndarray,
    sr: int = 16000,
    top_db: float = 28.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> list[tuple[int, int]]:
    """
    Detect active-speech regions using Librosa's energy-based split.

    The parameters match the Phase 1 notebook.
    """
    intervals = librosa.effects.split(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    return [(int(start), int(end)) for start, end in intervals]


def concatenate_intervals(
    audio: np.ndarray,
    intervals: list[tuple[int, int]],
) -> np.ndarray:
    """Concatenate non-empty sample intervals."""
    chunks = [audio[start:end] for start, end in intervals if end > start]

    if not chunks:
        return np.asarray([], dtype=np.float32)

    return np.concatenate(chunks).astype(np.float32)


def get_speech_audio(
    audio: np.ndarray,
    speech_intervals: list[tuple[int, int]],
    min_samples: int = 512,
) -> np.ndarray:
    """
    Return concatenated speech audio, falling back to full audio when the
    detected speech signal is too short.
    """
    if speech_intervals:
        speech_audio = concatenate_intervals(audio, speech_intervals)
    else:
        speech_audio = np.asarray([], dtype=np.float32)

    if len(speech_audio) < min_samples:
        return audio

    return speech_audio


def audio_summary(
    audio: np.ndarray,
    sr: int,
) -> dict:
    """Basic input-audio summary."""
    return {
        "sampling_rate": int(sr),
        "duration_sec": float(len(audio) / sr),
        "peak_amplitude": float(np.max(np.abs(audio))),
        "rms_amplitude": float(np.sqrt(np.mean(audio ** 2))),
    }
