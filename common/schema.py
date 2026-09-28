"""
ABDA Phase 1 — Common multimodal record schema.

This module defines the metadata contract shared by text, speech,
and facial modalities.

Feature extraction remains modality-specific.

The schema does NOT define the fusion method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModalityRecord:
    """
    Common record emitted by a modality-specific pipeline.

    `features` remains a named dictionary during extraction.
    Conversion to a fixed numerical vector happens later at the
    integration/training boundary.
    """

    participant_id: str
    session_id: str

    # Optional segment-level identity.
    segment_id: str | None

    # Segment index inside the source video/session.
    label_idx: int | None

    # Temporal boundaries in seconds.
    start: float | None
    end: float | None

    timestamp: str

    source_dataset: str
    language: str

    modality: str
    modality_available: bool

    quality_score: float

    feature_version: str

    features: dict[str, float]

    # Optional diagnostic information.
    quality_details: dict[str, Any] | None = None
    error: str | None = None

    def validate(self) -> None:
        """Validate the common record contract."""

        if not self.participant_id:
            raise ValueError("participant_id is required.")

        if not self.session_id:
            raise ValueError("session_id is required.")

        if not self.source_dataset:
            raise ValueError("source_dataset is required.")

        if not self.language:
            raise ValueError("language is required.")

        if not self.modality:
            raise ValueError("modality is required.")

        if not 0.0 <= float(self.quality_score) <= 1.0:
            raise ValueError(
                "quality_score must be within [0, 1]."
            )

        if self.start is not None and self.end is not None:
            if self.start < 0:
                raise ValueError("start must be >= 0.")

            if self.end <= self.start:
                raise ValueError(
                    "end must be greater than start."
                )

        if not self.modality_available:
            return

        if not self.features:
            raise ValueError(
                "Available modality must contain features."
            )

        for name, value in self.features.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Feature names must be non-empty strings."
                )

            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Feature '{name}' is not numeric."
                ) from exc

            if numeric_value != numeric_value:
                raise ValueError(
                    f"Feature '{name}' is NaN."
                )

            if numeric_value in (
                float("inf"),
                float("-inf"),
            ):
                raise ValueError(
                    f"Feature '{name}' is infinite."
                )


def make_alignment_key(
    record: ModalityRecord,
) -> tuple[str, int | None]:
    """
    Return the established CMU-MOSEI segment identity.

    For the current text pipeline this corresponds to:
        (segment_id, label_idx)
    """

    return (
        record.segment_id,
        record.label_idx,
    )