"""
ABDA Phase 1 — Multimodal Integration Schema.

Defines the structured multimodal record schema.
No quality-aware fusion. No flattening into one vector.
No invented thresholds or quality scores.

The purpose of this schema is to establish a clean, explicit contract
for what one integrated multimodal record contains, and to make
unavailability of any component explicit (never silent/zero).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Text component
# ---------------------------------------------------------------------------

@dataclass
class TextRecord:
    """
    Holds the text representation retrieved from the validated NLP artifact.

    feature_dim is fixed at 774 (768 LaBSE + 6 scalar NLP features).
    MSTTR-10 may be NaN for short segments; this is intentional and validated.
    """
    # The 774-D feature set, stored as a dict of named scalars + the embedding.
    features: Dict[str, Any]
    feature_dim: int = 774
    status: str = "available"  # "available" | "unavailable" | "error"
    # Source text used to produce features.
    text: Optional[str] = None
    # Validity indicators from the existing text pipeline.
    # Not a numerical quality score -- the text pipeline does not define one.
    validity: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Speech component
# ---------------------------------------------------------------------------

@dataclass
class SpeechQuality:
    """Quality indicators produced by the existing speech pipeline."""
    quality_score: float
    speech_ratio: float
    snr_estimate: float


@dataclass
class SpeechRecord:
    """
    Holds the speech representation produced by the existing speech pipeline.

    feature_dim is fixed at 70 (Phase 1 spec).
    """
    features: Dict[str, float]   # 70 named features
    feature_dim: int = 70
    status: str = "available"    # "available" | "unavailable" | "error"
    quality: Optional[SpeechQuality] = None
    # Passthrough fields from the speech session record.
    source_dataset: Optional[str] = None
    feature_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Facial component
# ---------------------------------------------------------------------------

@dataclass
class OpenFaceStatus:
    """
    Explicit OpenFace availability record.

    IMPORTANT: 'unavailable' must never silently collapse to zero-valued AU
    features. The status field is the authoritative record.
    """
    status: str   # "success" | "unavailable" | "failed" | "not_run"
    error: Optional[str] = None
    csv_path: Optional[str] = None


@dataclass
class FacialRecord:
    """
    Holds the facial representation produced by the existing facial pipeline.

    AU statistics are only populated when OpenFace status == "success".
    face_coverage_percent is in [0, 100] as produced by the existing pipeline.
    """
    # Core MediaPipe-derived feature summaries.
    ear_statistics: Dict[str, float] = field(default_factory=dict)
    gaze_statistics: Dict[str, Any] = field(default_factory=dict)
    head_pose_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    blink_count: int = 0
    blink_rate_per_minute: float = 0.0
    # Quality indicators from the facial pipeline.
    face_coverage_percent: float = 0.0
    quality_indicator: Dict[str, Any] = field(default_factory=dict)
    # OpenFace / AU -- explicit status always required.
    openface: Optional[OpenFaceStatus] = None
    au_statistics: Dict[str, Any] = field(default_factory=dict)
    # Video metadata.
    video_metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "available"     # "available" | "unavailable" | "error"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

@dataclass
class SegmentIdentity:
    """
    Multimodal identity key.

    Alignment is based on (segment_id, label_idx).
    Timestamp metadata is preserved for traceability but does NOT imply
    that equal timestamps constitute alignment evidence.
    """
    source_dataset: str
    segment_id: str
    label_idx: int
    start: float
    end: float


# ---------------------------------------------------------------------------
# Integration summary
# ---------------------------------------------------------------------------

@dataclass
class IntegrationSummary:
    """
    Records the outcome of alignment verification and schema validation.

    Does NOT contain fusion weights or quality-aware fusion outputs.
    """
    alignment_status: str          # "PASS" | "FAIL"
    validation_status: str         # "PASS" | "FAIL"
    alignment_errors: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    modality_availability: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level record
# ---------------------------------------------------------------------------

@dataclass
class MultimodalRecord:
    """
    One integrated multimodal record for a single (segment_id, label_idx).

    No fusion is performed here. The record is a clean container for
    modality-specific representations + quality indicators + integration status.
    """
    identity: SegmentIdentity
    text: Optional[TextRecord] = None
    speech: Optional[SpeechRecord] = None
    facial: Optional[FacialRecord] = None
    integration: Optional[IntegrationSummary] = None

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON export / display."""
        import dataclasses
        import numpy as np

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, float) and (obj != obj):  # NaN
                return None
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        import dataclasses
        return _convert(dataclasses.asdict(self))
