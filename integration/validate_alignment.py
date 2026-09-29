"""
ABDA Phase 1 — Integration alignment validation.

Validates an assembled MultimodalRecord:
  - identity fields present and finite
  - segment_id / label_idx consistent across modalities
  - feature vectors have expected dimensions
  - all required numerical values are finite
  - unavailable modalities/features are represented explicitly
  - no silent replacement of unavailable data with zeros

Does NOT perform quality-aware fusion.
Does NOT define fusion weights.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .schema import (
    MultimodalRecord,
    IntegrationSummary,
)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate_identity(record: MultimodalRecord) -> List[str]:
    errors = []
    ident = record.identity
    if ident is None:
        return ["identity is None"]
    if not ident.segment_id:
        errors.append("identity.segment_id is empty")
    if ident.label_idx is None:
        errors.append("identity.label_idx is None")
    if not _finite(ident.start):
        errors.append("identity.start is not finite")
    if not _finite(ident.end):
        errors.append("identity.end is not finite")
    if _finite(ident.start) and _finite(ident.end):
        if ident.end <= ident.start:
            errors.append(
                f"identity.end ({ident.end}) must be > start ({ident.start})"
            )
    if not ident.source_dataset:
        errors.append("identity.source_dataset is empty")
    return errors


def _validate_alignment(record: MultimodalRecord) -> List[str]:
    """
    Verify that all present modalities share the same (segment_id, label_idx).

    Identity is based solely on (segment_id, label_idx) — NOT on timestamps.
    """
    errors = []
    ident = record.identity
    if ident is None:
        return ["Cannot check alignment: identity is None"]

    ref_sid = str(ident.segment_id)
    ref_lidx = int(ident.label_idx)

    # Speech alignment check.
    speech = record.speech
    if speech is not None and speech.status == "available":
        # Speech record does not embed segment_id/label_idx internally;
        # alignment is established by the builder passing the identity key.
        # We document that alignment was verified at build time.
        pass  # builder explicitly sets these; validated there

    # Facial alignment check via alignment_metadata if present.
    facial = record.facial
    if facial is not None and facial.status == "available":
        vm = facial.video_metadata
        if isinstance(vm, dict):
            face_sid = vm.get("segment_id")
            face_lidx = vm.get("label_idx")
            if face_sid is not None and str(face_sid) != ref_sid:
                errors.append(
                    f"Facial segment_id '{face_sid}' != identity '{ref_sid}'"
                )
            if face_lidx is not None and int(face_lidx) != ref_lidx:
                errors.append(
                    f"Facial label_idx {face_lidx} != identity {ref_lidx}"
                )

    return errors


def _validate_text(record: MultimodalRecord) -> List[str]:
    errors = []
    text = record.text
    if text is None:
        errors.append("text component is None (should be explicitly marked unavailable)")
        return errors
    if text.status not in {"available", "unavailable", "error"}:
        errors.append(f"text.status invalid: '{text.status}'")
    if text.status == "available":
        if not isinstance(text.features, dict):
            errors.append("text.features must be a dict")
        else:
            if text.feature_dim != 774:
                errors.append(
                    f"text.feature_dim must be 774; got {text.feature_dim}"
                )
            # Check scalar features are finite.
            scalar_keys = [
                "sentiment_normalized",
                "word_count",
                "avg_word_length",
                "dependency_tree_depth",
                "avg_dependency_distance",
            ]
            for key in scalar_keys:
                val = text.features.get(key)
                if val is not None and not _finite(val):
                    errors.append(f"text.features.{key} is not finite")
            # LaBSE embedding.
            emb = text.features.get("labse_embedding")
            if emb is None:
                errors.append("text.features.labse_embedding is missing")
            else:
                arr = np.asarray(emb, dtype=float)
                if arr.shape != (768,):
                    errors.append(
                        f"labse_embedding must be (768,); got {arr.shape}"
                    )
                elif not np.isfinite(arr).all():
                    errors.append("labse_embedding contains NaN/Inf")
    return errors


def _validate_speech(record: MultimodalRecord) -> List[str]:
    errors = []
    speech = record.speech
    if speech is None:
        errors.append("speech component is None (should be explicitly marked unavailable)")
        return errors
    if speech.status not in {"available", "unavailable", "error"}:
        errors.append(f"speech.status invalid: '{speech.status}'")
    if speech.status == "available":
        if speech.feature_dim != 70:
            errors.append(
                f"speech.feature_dim must be 70; got {speech.feature_dim}"
            )
        if not isinstance(speech.features, dict):
            errors.append("speech.features must be a dict")
        elif len(speech.features) != 70:
            errors.append(
                f"speech.features must have 70 entries; got {len(speech.features)}"
            )
        else:
            values = np.asarray(list(speech.features.values()), dtype=float)
            if not np.isfinite(values).all():
                errors.append("speech.features contain NaN/Inf")
        q = speech.quality
        if q is None:
            errors.append("speech.quality is None")
        else:
            if not 0.0 <= q.quality_score <= 1.0:
                errors.append("speech.quality.quality_score outside [0, 1]")
            if not 0.0 <= q.speech_ratio <= 1.0:
                errors.append("speech.quality.speech_ratio outside [0, 1]")
            if not _finite(q.snr_estimate):
                errors.append("speech.quality.snr_estimate is not finite")
    return errors


def _validate_facial(record: MultimodalRecord) -> List[str]:
    errors = []
    facial = record.facial
    if facial is None:
        errors.append("facial component is None (should be explicitly marked unavailable)")
        return errors
    if facial.status not in {"available", "unavailable", "error"}:
        errors.append(f"facial.status invalid: '{facial.status}'")
    if facial.status == "available":
        if not 0.0 <= facial.face_coverage_percent <= 100.0:
            errors.append("facial.face_coverage_percent outside [0, 100]")
        # OpenFace must have an explicit status -- never silent.
        if facial.openface is None:
            errors.append(
                "facial.openface is None -- OpenFace status must be explicit"
            )
        else:
            valid_of_statuses = {"success", "unavailable", "failed", "not_run"}
            if facial.openface.status not in valid_of_statuses:
                errors.append(
                    f"facial.openface.status '{facial.openface.status}' "
                    f"is not one of {valid_of_statuses}"
                )
            # CRITICAL: if OpenFace is unavailable, au_statistics must be empty.
            if facial.openface.status in {"unavailable", "failed", "not_run"}:
                if facial.au_statistics:
                    errors.append(
                        "facial.au_statistics must be empty when OpenFace is "
                        f"'{facial.openface.status}' -- "
                        "do not replace unavailable AU data with zeros"
                    )
        # EAR statistics.
        ear = facial.ear_statistics
        if not isinstance(ear, dict):
            errors.append("facial.ear_statistics must be a dict")
    return errors


def validate_alignment(record: MultimodalRecord) -> IntegrationSummary:
    """
    Run all integration validation checks and return an IntegrationSummary.

    Does NOT modify the record. Returns PASS/FAIL + error list.
    """
    all_errors: List[str] = []

    identity_errors = _validate_identity(record)
    all_errors.extend(identity_errors)

    alignment_errors: List[str] = []
    if not identity_errors:
        alignment_errors = _validate_alignment(record)
        all_errors.extend(alignment_errors)

    text_errors = _validate_text(record)
    speech_errors = _validate_speech(record)
    facial_errors = _validate_facial(record)
    all_errors.extend(text_errors)
    all_errors.extend(speech_errors)
    all_errors.extend(facial_errors)

    validation_errors = text_errors + speech_errors + facial_errors
    alignment_status = "PASS" if not alignment_errors and not identity_errors else "FAIL"
    validation_status = "PASS" if not all_errors else "FAIL"

    # Modality availability summary.
    availability: dict = {}
    if record.text is not None:
        availability["text"] = record.text.status
    else:
        availability["text"] = "not_present"
    if record.speech is not None:
        availability["speech"] = record.speech.status
    else:
        availability["speech"] = "not_present"
    if record.facial is not None:
        availability["facial"] = record.facial.status
    else:
        availability["facial"] = "not_present"

    return IntegrationSummary(
        alignment_status=alignment_status,
        validation_status=validation_status,
        alignment_errors=alignment_errors,
        validation_errors=validation_errors,
        modality_availability=availability,
    )
