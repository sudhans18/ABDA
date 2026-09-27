"""
ABDA Phase 1 — Speech pipeline validation.
"""

from __future__ import annotations

import numpy as np


def validate_audio(audio: np.ndarray, sr: int) -> None:
    """Validate the loaded audio signal."""
    if not isinstance(sr, int) or sr <= 0:
        raise ValueError(f"Invalid sampling rate: {sr}")

    if audio.ndim != 1:
        raise ValueError(f"Expected mono 1-D audio; got {audio.shape}")

    if len(audio) == 0:
        raise ValueError("Audio signal is empty.")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinite values.")


def validate_speech_intervals(
    intervals,
    audio_length: int,
) -> None:
    """Validate VAD intervals."""
    previous_end = 0

    for start, end in intervals:
        if start < 0 or end <= start:
            raise ValueError(f"Invalid VAD interval: {(start, end)}")

        if end > audio_length:
            raise ValueError(
                f"VAD interval exceeds audio length: {(start, end)}"
            )

        if start < previous_end:
            raise ValueError("VAD intervals are overlapping/unsorted.")

        previous_end = end


def validate_speech_features(
    features: dict,
    quality: dict,
    expected_feature_count: int = 70,
) -> dict:
    """
    Validate the complete Phase 1 speech feature dictionary.

    Speaking rate is still represented as one feature even if no transcript
    was supplied; in that case its value is 0.0 and should be treated as
    unavailable rather than a measured rate.
    """
    if len(features) != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} speech features; "
            f"got {len(features)}."
        )

    values = np.asarray(list(features.values()), dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Speech features contain NaN or Inf.")

    required_quality = {
        "speech_ratio",
        "snr_estimate",
        "quality_score",
    }

    if not required_quality.issubset(quality):
        raise ValueError(
            f"Missing quality fields: "
            f"{sorted(required_quality - set(quality))}"
        )

    if not 0.0 <= float(quality["speech_ratio"]) <= 1.0:
        raise ValueError("speech_ratio must be in [0, 1].")

    if not 0.0 <= float(quality["quality_score"]) <= 1.0:
        raise ValueError("quality_score must be in [0, 1].")

    return {
        "feature_count": len(features),
        "quality_score": float(quality["quality_score"]),
        "all_features_finite": True,
    }


def validate_session_record(record: dict) -> None:
    """Validate the speech-side session record before common integration."""
    required = {
        "participant_id",
        "session_id",
        "timestamp",
        "source_dataset",
        "language",
        "modality_available",
        "quality_score",
        "feature_version",
        "features",
    }

    missing = required - set(record)
    if missing:
        raise ValueError(
            f"Missing session-record fields: {sorted(missing)}"
        )

    if record["modality_available"] is not True:
        raise ValueError(
            "A speech session record must mark modality_available=True."
        )

    if not 0.0 <= float(record["quality_score"]) <= 1.0:
        raise ValueError("Session quality_score must be in [0, 1].")

    validate_speech_features(
        record["features"],
        {
            "speech_ratio": record.get("speech_ratio", 0.0),
            "snr_estimate": record.get("snr_estimate", 0.0),
            "quality_score": record["quality_score"],
        },
    )
