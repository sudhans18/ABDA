"""
ABDA Phase 1 — Integration visualisation helper.

Provides plain-text and dict-based summaries of an integrated record
for display in the CLI and the live-demo backend.

Does NOT perform fusion.
Does NOT invent quality metrics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .schema import MultimodalRecord


def record_to_display_dict(record: MultimodalRecord) -> Dict[str, Any]:
    """
    Convert an integrated MultimodalRecord to a flat display-friendly dict.

    Designed for JSON serialisation to the live-demo frontend.
    Does NOT flatten features into a single vector.
    """
    out: Dict[str, Any] = {}

    # -- Identity --
    ident = record.identity
    out["identity"] = {
        "source_dataset": ident.source_dataset,
        "segment_id": ident.segment_id,
        "label_idx": ident.label_idx,
        "start": ident.start,
        "end": ident.end,
    }

    # -- Text --
    t = record.text
    if t is None:
        out["text"] = {"status": "not_present"}
    else:
        td: Dict[str, Any] = {"status": t.status}
        if t.status == "available":
            td["feature_dim"] = t.feature_dim
            td["text_preview"] = (t.text[:120] + "...") if t.text and len(t.text) > 120 else t.text
            f = t.features
            td["sentiment_normalized"] = _safe_float(f.get("sentiment_normalized"))
            td["word_count"] = _safe_int(f.get("word_count"))
            td["avg_word_length"] = _safe_float(f.get("avg_word_length"))
            td["dependency_tree_depth"] = _safe_int(f.get("dependency_tree_depth"))
            td["avg_dependency_distance"] = _safe_float(f.get("avg_dependency_distance"))
            msttr = f.get("msttr_10")
            td["msttr_10"] = _safe_float(msttr) if msttr is not None else None
            td["labse_dim"] = int(len(np.asarray(f["labse_embedding"]))) if f.get("labse_embedding") is not None else None
            td["validity"] = t.validity
        out["text"] = td

    # -- Speech --
    s = record.speech
    if s is None:
        out["speech"] = {"status": "not_present"}
    else:
        sd: Dict[str, Any] = {"status": s.status}
        if s.status == "available":
            sd["feature_dim"] = s.feature_dim
            if s.quality:
                sd["quality_score"] = _safe_float(s.quality.quality_score)
                sd["speech_ratio"] = _safe_float(s.quality.speech_ratio)
                sd["snr_estimate"] = _safe_float(s.quality.snr_estimate)
            sd["feature_version"] = s.feature_version
            # Expose feature name list (not values, to keep payload small).
            sd["feature_names"] = list(s.features.keys())
        out["speech"] = sd

    # -- Facial --
    fa = record.facial
    if fa is None:
        out["facial"] = {"status": "not_present"}
    else:
        fd: Dict[str, Any] = {"status": fa.status}
        if fa.status == "available":
            fd["face_coverage_percent"] = _safe_float(fa.face_coverage_percent)
            fd["blink_count"] = fa.blink_count
            fd["blink_rate_per_minute"] = _safe_float(fa.blink_rate_per_minute)
            fd["ear_statistics"] = _clean_stats(fa.ear_statistics)
            fd["gaze_statistics"] = {
                "ratio": _clean_stats(fa.gaze_statistics.get("ratio", {})),
                "percentages": fa.gaze_statistics.get("percentages", {}),
            }
            fd["head_pose_statistics"] = {
                axis: _clean_stats(fa.head_pose_statistics.get(axis, {}))
                for axis in ("pitch", "yaw", "roll")
            }
            of = fa.openface
            fd["openface"] = {
                "status": of.status if of else "not_run",
                "error": of.error if of else None,
            }
            fd["au_count"] = len(fa.au_statistics)
            fd["quality_indicator"] = fa.quality_indicator
        out["facial"] = fd

    # -- Integration --
    integ = record.integration
    if integ is None:
        out["integration"] = {"status": "NOT_RUN"}
    else:
        out["integration"] = {
            "alignment_status": integ.alignment_status,
            "validation_status": integ.validation_status,
            "modality_availability": integ.modality_availability,
            "alignment_errors": integ.alignment_errors,
            "validation_errors": integ.validation_errors,
            "fusion": "NOT_IMPLEMENTED",
        }

    return out


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if (f != f) else round(f, 6)   # NaN → None
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _clean_stats(stats: dict) -> dict:
    if not isinstance(stats, dict):
        return {}
    return {k: _safe_float(v) for k, v in stats.items()}
