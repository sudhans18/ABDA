"""
ABDA Phase 1 — Speech service for the live demo.

Calls the EXISTING speech pipeline (speech.run_file.run_speech_file).
Does NOT modify speech feature definitions or thresholds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def run_speech_on_file(
    audio_path: Optional[Path],
    transcript: str,
    session_id: str,
) -> Dict[str, Any]:
    """
    Run the existing speech pipeline on the given audio file.

    Returns a display-ready dict with features and quality indicators.
    If audio_path is None or missing, returns status='unavailable'.
    """
    if audio_path is None or not Path(audio_path).exists():
        return {"status": "unavailable"}

    try:
        from speech.run_file import run_speech_file
    except ImportError:
        return {"status": "error", "error": "speech module not importable"}

    try:
        record = run_speech_file(
            audio_path=audio_path,
            participant_id=f"demo_{session_id}",
            session_id=f"demo_{session_id}",
            segment_id=None,
            label_idx=None,
            start=None,
            end=None,
            transcript=transcript or None,
            source_dataset="live_demo",
            language="English",
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    features = record.get("features", {})
    return {
        "status": "available",
        "feature_dim": len(features),
        "quality_score": round(float(record.get("quality_score", 0)), 4),
        "speech_ratio": round(float(record.get("speech_ratio", 0)), 4),
        "snr_estimate": round(float(record.get("snr_estimate", 0)), 4),
        "feature_version": record.get("feature_version", "phase1_v1"),
        # Representative feature groups for display (not full 70-D vector).
        "feature_summary": _summarise_features(features),
        "feature_names": list(features.keys()),
    }


def _summarise_features(features: dict) -> dict:
    """Extract named feature summaries for the UI display."""
    return {
        "f0_mean": _safe(features.get("f0_mean")),
        "f0_std": _safe(features.get("f0_std")),
        "speaking_rate_wps": _safe(features.get("speaking_rate_wps")),
        "jitter_local": _safe(features.get("jitter_local")),
        "shimmer_local": _safe(features.get("shimmer_local")),
        "hnr_mean": _safe(features.get("hnr_mean")),
        "pause_count": _safe(features.get("pause_count")),
        "pause_total_duration": _safe(features.get("pause_total_duration")),
        "rms_mean": _safe(features.get("rms_mean")),
        "spectral_centroid_mean": _safe(features.get("spectral_centroid_mean")),
        "mfcc_1_mean": _safe(features.get("mfcc_1_mean")),
        "mfcc_2_mean": _safe(features.get("mfcc_2_mean")),
    }


def _safe(v) -> Optional[float]:
    try:
        f = float(v)
        return None if f != f else round(f, 5)
    except (TypeError, ValueError):
        return None
