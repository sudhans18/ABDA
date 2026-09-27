"""
ABDA Phase 1 — Speech/audio feature extraction.

Feature groups retained from the Phase 1 notebook:
- MFCC 1–13: mean/std/min/max
- F0/pitch statistics
- jitter, shimmer, HNR
- pause dynamics
- speaking rate
- RMS and spectral centroid
- estimated audio quality

Total feature count for a session with a transcript:
    52 MFCC + 6 F0 + 3 voice-quality + 4 pause + 1 speaking-rate
    + 4 spectral = 70 features
"""

from __future__ import annotations

from typing import Optional

import librosa
import numpy as np


def extract_mfcc_features(
    audio: np.ndarray,
    sr: int,
    speech_intervals,
    n_mfcc: int = 13,
) -> dict:
    """Extract mean/std/min/max for MFCC coefficients 1–13."""
    from .extraction import get_speech_audio

    speech_audio = get_speech_audio(audio, speech_intervals)

    mfccs = librosa.feature.mfcc(
        y=speech_audio,
        sr=sr,
        n_mfcc=n_mfcc,
    )

    features = {}
    for i in range(n_mfcc):
        idx = i + 1
        trajectory = mfccs[i, :]

        features[f"mfcc_{idx}_mean"] = float(np.mean(trajectory))
        features[f"mfcc_{idx}_std"] = float(np.std(trajectory))
        features[f"mfcc_{idx}_min"] = float(np.min(trajectory))
        features[f"mfcc_{idx}_max"] = float(np.max(trajectory))

    return features


def extract_pitch_features(
    audio: np.ndarray,
    sr: int,
) -> dict:
    """Extract F0 statistics using Praat-Parselmouth."""
    import parselmouth

    metrics = {
        "f0_mean": 0.0,
        "f0_std": 0.0,
        "f0_min": 0.0,
        "f0_max": 0.0,
        "f0_range": 0.0,
        "f0_variability": 0.0,
    }

    sound = parselmouth.Sound(
        audio,
        sampling_frequency=sr,
    )

    pitch = sound.to_pitch(
        pitch_floor=75.0,
        pitch_ceiling=500.0,
    )

    pitch_values = pitch.selected_array["frequency"]
    voiced_f0 = pitch_values[pitch_values > 0]

    if len(voiced_f0) == 0:
        return metrics

    mean_v = float(np.mean(voiced_f0))
    std_v = float(np.std(voiced_f0))
    min_v = float(np.min(voiced_f0))
    max_v = float(np.max(voiced_f0))

    metrics.update(
        {
            "f0_mean": mean_v,
            "f0_std": std_v,
            "f0_min": min_v,
            "f0_max": max_v,
            "f0_range": max_v - min_v,
            "f0_variability": std_v / mean_v if mean_v > 0 else 0.0,
        }
    )

    return metrics


def extract_voice_quality_features(
    audio: np.ndarray,
    sr: int,
) -> dict:
    """Extract Praat local jitter, local shimmer and mean HNR."""
    import parselmouth
    from parselmouth.praat import call

    metrics = {
        "jitter_local": 0.0,
        "shimmer_local": 0.0,
        "hnr_mean": 0.0,
    }

    sound = parselmouth.Sound(
        audio,
        sampling_frequency=sr,
    )

    point_process = call(
        sound,
        "To PointProcess (periodic, cc)",
        75.0,
        500.0,
    )

    jitter = call(
        point_process,
        "Get jitter (local)",
        0.0,
        0.0,
        0.0001,
        0.02,
        1.3,
    )

    shimmer = call(
        [sound, point_process],
        "Get shimmer (local)",
        0.0,
        0.0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )

    harmonicity = call(
        sound,
        "To Harmonicity (cc)",
        0.01,
        75.0,
        0.1,
        4.5,
    )

    hnr = call(
        harmonicity,
        "Get mean",
        0.0,
        0.0,
    )

    metrics["jitter_local"] = (
        float(jitter) if np.isfinite(jitter) else 0.0
    )
    metrics["shimmer_local"] = (
        float(shimmer) if np.isfinite(shimmer) else 0.0
    )
    metrics["hnr_mean"] = (
        float(hnr) if np.isfinite(hnr) else 0.0
    )

    return metrics


def extract_pause_features(
    speech_intervals,
    total_samples: int,
    sr: int,
    min_pause_sec: float = 0.15,
) -> dict:
    """Extract count, total duration, mean duration and rate of pauses."""
    metrics = {
        "pause_count": 0.0,
        "pause_total_duration": 0.0,
        "pause_mean_duration": 0.0,
        "pause_rate": 0.0,
    }

    total_sec = total_samples / float(sr)

    if total_sec <= 0 or not speech_intervals:
        return metrics

    pauses = []

    # Pre-speech silence.
    if speech_intervals[0][0] > 0:
        duration = speech_intervals[0][0] / sr
        if duration >= min_pause_sec:
            pauses.append(duration)

    # Internal silences.
    for i in range(len(speech_intervals) - 1):
        duration = (
            speech_intervals[i + 1][0] - speech_intervals[i][1]
        ) / sr

        if duration >= min_pause_sec:
            pauses.append(duration)

    # Post-speech silence.
    if speech_intervals[-1][1] < total_samples:
        duration = (
            total_samples - speech_intervals[-1][1]
        ) / sr

        if duration >= min_pause_sec:
            pauses.append(duration)

    if pauses:
        metrics["pause_count"] = float(len(pauses))
        metrics["pause_total_duration"] = float(np.sum(pauses))
        metrics["pause_mean_duration"] = float(np.mean(pauses))
        metrics["pause_rate"] = float(len(pauses) / total_sec)

    return metrics


def extract_speaking_rate(
    total_duration_sec: float,
    transcript: Optional[str] = None,
) -> dict:
    """
    Speaking rate in words/second.

    A transcript is required. If it is unavailable, the feature is 0.0 rather
    than being guessed from audio alone.
    """
    if transcript is None or total_duration_sec <= 0:
        return {"speaking_rate_wps": 0.0}

    words = len(str(transcript).strip().split())

    return {
        "speaking_rate_wps": float(words / total_duration_sec)
        if words > 0
        else 0.0
    }


def extract_spectral_features(
    audio: np.ndarray,
    sr: int,
    speech_intervals,
) -> dict:
    """Extract RMS energy and spectral centroid statistics."""
    from .extraction import get_speech_audio

    speech_audio = get_speech_audio(audio, speech_intervals)

    rms = librosa.feature.rms(y=speech_audio)[0]
    centroid = librosa.feature.spectral_centroid(
        y=speech_audio,
        sr=sr,
    )[0]

    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
    }


def compute_audio_quality_score(
    audio: np.ndarray,
    speech_intervals,
) -> dict:
    """
    Estimate audio quality using the Phase 1 heuristic.

    The resulting SNR is explicitly an estimate, not a calibrated recording
    quality measurement.
    """
    total_len = len(audio)

    if total_len == 0:
        return {
            "speech_ratio": 0.0,
            "snr_estimate": 0.0,
            "quality_score": 0.0,
        }

    speech_samples = sum(
        end - start for start, end in speech_intervals
    )
    speech_ratio = min(
        1.0,
        float(speech_samples / total_len),
    )

    speech_mask = np.zeros(total_len, dtype=bool)
    for start, end in speech_intervals:
        speech_mask[start:end] = True

    sig_energy = (
        np.mean(audio[speech_mask] ** 2)
        if np.any(speech_mask)
        else 1e-9
    )

    noise_energy = (
        np.mean(audio[~speech_mask] ** 2)
        if np.any(~speech_mask)
        else 1e-9
    )

    snr = 10.0 * np.log10(
        max(sig_energy / (noise_energy + 1e-9), 1e-3)
    )

    snr_norm = 1.0 / (
        1.0 + np.exp(-0.2 * (snr - 10.0))
    )

    quality_score = float(
        np.clip(
            0.4 * speech_ratio + 0.6 * snr_norm,
            0.0,
            1.0,
        )
    )

    return {
        "speech_ratio": float(speech_ratio),
        "snr_estimate": float(snr),
        "quality_score": quality_score,
    }


def extract_all_speech_features(
    audio: np.ndarray,
    sr: int,
    speech_intervals,
    transcript: Optional[str] = None,
) -> tuple[dict, dict]:
    """Run the complete Phase 1 speech feature extraction."""
    feature_blocks = [
        extract_mfcc_features(
            audio,
            sr,
            speech_intervals,
        ),
        extract_pitch_features(
            audio,
            sr,
        ),
        extract_voice_quality_features(
            audio,
            sr,
        ),
        extract_pause_features(
            speech_intervals,
            total_samples=len(audio),
            sr=sr,
        ),
        extract_speaking_rate(
            len(audio) / sr,
            transcript=transcript,
        ),
        extract_spectral_features(
            audio,
            sr,
            speech_intervals,
        ),
    ]

    features = {}
    for block in feature_blocks:
        features.update(block)

    quality = compute_audio_quality_score(
        audio,
        speech_intervals,
    )

    return features, quality


def build_speech_session_record(
    audio: np.ndarray,
    sr: int,
    speech_intervals,
    participant_id: str,
    session_id: str,
    timestamp: str,
    source_dataset: str,
    language: str,
    transcript: str | None = None,
    feature_version: str = "phase1_v1",
) -> dict:
    """Create the speech-side record expected by the future common schema."""
    features, quality = extract_all_speech_features(
        audio=audio,
        sr=sr,
        speech_intervals=speech_intervals,
        transcript=transcript,
    )

    record = {
        "participant_id": participant_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "source_dataset": source_dataset,
        "language": language,
        "modality_available": True,
        "quality_score": quality["quality_score"],
        "feature_version": feature_version,
        "features": features,
        # Retain quality components so the integration layer can inspect them.
        "speech_ratio": quality["speech_ratio"],
        "snr_estimate": quality["snr_estimate"],
    }

    return record
