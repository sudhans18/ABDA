"""
ABDA Phase 1 — Integration session builder.

Assembles one MultimodalRecord for a given (segment_id, label_idx) by:

  1. Retrieving the text representation from the validated NLP artifact.
  2. Running or retrieving the speech representation.
  3. Running or retrieving the facial representation.
  4. Verifying identity/alignment via (segment_id, label_idx).
  5. Constructing a MultimodalRecord.
  6. Running validate_alignment.

This module does NOT perform quality-aware fusion.
This module does NOT flatten features into a single vector.
This module does NOT invent quality metrics.
This module does NOT modify any existing pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .schema import (
    FacialRecord,
    IntegrationSummary,
    MultimodalRecord,
    OpenFaceStatus,
    SegmentIdentity,
    SpeechQuality,
    SpeechRecord,
    TextRecord,
)
from .validate_alignment import validate_alignment


# ---------------------------------------------------------------------------
# Text retrieval — reads from the validated NLP artifact, no pipeline changes.
# ---------------------------------------------------------------------------

DEFAULT_TEXT_ARTIFACT = Path("data/processed/text/CMU_MOSEI_final_NLP_features.pkl")


def _load_text_artifact(artifact_path: Path) -> pd.DataFrame:
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Text NLP artifact not found: {artifact_path}\n"
            "Run the text pipeline first to produce this file."
        )
    return pd.read_pickle(artifact_path)


def retrieve_text_record(
    segment_id: str,
    label_idx: int,
    artifact_path: Path = DEFAULT_TEXT_ARTIFACT,
) -> TextRecord:
    """
    Retrieve one row from the validated CMU-MOSEI NLP artifact.

    Alignment key: (segment_id, label_idx).
    Does NOT re-run any text processing.
    Does NOT invent a text quality score.
    """
    df = _load_text_artifact(artifact_path)

    # Normalise segment_id type for lookup.
    sid = str(segment_id)
    df["_sid_str"] = df["segment_id"].astype(str)
    row = df[(df["_sid_str"] == sid) & (df["label_idx"] == label_idx)]

    if row.empty:
        return TextRecord(
            features={},
            feature_dim=774,
            status="unavailable",
            text=None,
            validity={"error": f"segment ({sid}, {label_idx}) not found in NLP artifact"},
        )

    if len(row) > 1:
        # Should never happen since text pipeline validates unique keys.
        return TextRecord(
            features={},
            feature_dim=774,
            status="error",
            text=None,
            validity={"error": f"Duplicate key ({sid}, {label_idx}) in NLP artifact"},
        )

    row = row.iloc[0]
    text_str = str(row["text"]) if pd.notna(row["text"]) else None

    # Build feature dict preserving all columns.
    features: dict = {
        "sentiment_normalized": float(row["sentiment_normalized"]),
        "word_count": float(row["word_count"]),
        "avg_word_length": float(row["avg_word_length"]),
        "dependency_tree_depth": float(row["dependency_tree_depth"]),
        "avg_dependency_distance": float(row["avg_dependency_distance"]),
        # MSTTR-10 may be NaN for short segments — keep as-is.
        "msttr_10": row["msttr_10"] if pd.notna(row.get("msttr_10")) else None,
        # LaBSE 768-D.
        "labse_embedding": np.asarray(row["labse_embedding"]),
    }

    # Validity indicators from the existing pipeline (no invented quality score).
    validity = {
        "pipeline": "CMU-MOSEI Phase 1 validated",
        "labse_dim": int(len(features["labse_embedding"])),
        "msttr_available": features["msttr_10"] is not None,
        "text_non_empty": bool(text_str and text_str.strip()),
    }

    return TextRecord(
        features=features,
        feature_dim=774,
        status="available",
        text=text_str,
        validity=validity,
    )


# ---------------------------------------------------------------------------
# Speech retrieval — calls the existing pipeline, no methodology changes.
# ---------------------------------------------------------------------------

def retrieve_speech_record(
    audio_path: Path,
    segment_id: str,
    label_idx: int,
    start: float,
    end: float,
    transcript: Optional[str] = None,
) -> SpeechRecord:
    """
    Run the existing speech pipeline on an audio file and wrap the result.

    Does NOT modify speech feature definitions.
    Does NOT change thresholds.
    """
    try:
        from speech.run_file import run_speech_file
    except ImportError as exc:
        return SpeechRecord(
            features={},
            feature_dim=70,
            status="error",
            quality=None,
        )

    if not Path(audio_path).exists():
        return SpeechRecord(
            features={},
            feature_dim=70,
            status="unavailable",
            quality=None,
        )

    try:
        record = run_speech_file(
            audio_path=audio_path,
            participant_id=str(segment_id),
            session_id=str(segment_id),
            segment_id=str(segment_id),
            label_idx=int(label_idx),
            start=start,
            end=end,
            transcript=transcript,
            source_dataset="CMU-MOSEI",
            language="English",
        )
    except Exception as exc:
        return SpeechRecord(
            features={},
            feature_dim=70,
            status="error",
            quality=None,
        )

    quality = SpeechQuality(
        quality_score=float(record["quality_score"]),
        speech_ratio=float(record["speech_ratio"]),
        snr_estimate=float(record["snr_estimate"]),
    )

    return SpeechRecord(
        features=dict(record["features"]),
        feature_dim=len(record["features"]),
        status="available",
        quality=quality,
        source_dataset=record.get("source_dataset", "CMU-MOSEI"),
        feature_version=record.get("feature_version", "phase1_v1"),
    )


# ---------------------------------------------------------------------------
# Facial retrieval — calls the existing pipeline, no methodology changes.
# ---------------------------------------------------------------------------

def retrieve_facial_record(
    video_path: Path,
    segment_id: str,
    label_idx: int,
    start: float,
    end: float,
) -> FacialRecord:
    """
    Run the existing facial pipeline on a video file and wrap the result.

    Does NOT modify landmark indices, EAR threshold, gaze thresholds,
    head-pose calculation, smoothing window, or AU methodology.
    OpenFace unavailability is ALWAYS represented explicitly -- never zeroed.
    """
    try:
        from face.run_video import process_video
    except ImportError:
        return FacialRecord(status="error")

    if not Path(video_path).exists():
        return FacialRecord(status="unavailable")

    try:
        result = process_video(
            video_path=video_path,
            segment_id=str(segment_id),
            label_idx=int(label_idx),
            start=start,
            end=end,
        )
    except Exception:
        return FacialRecord(status="error")

    # OpenFace status -- always explicit, never silent.
    of_raw = result.get("openface", {})
    of_status = OpenFaceStatus(
        status=of_raw.get("status", "not_run"),
        error=of_raw.get("error"),
        csv_path=of_raw.get("csv_path"),
    )

    # AU statistics -- only populated on success.
    au_stats = {}
    if of_status.status == "success":
        au_stats = result.get("au_statistics", {})
    # If not success, au_stats stays empty.  Critical: do NOT zero-fill.

    # Facial alignment metadata embedded in video_metadata for validation.
    alignment_meta = result.get("alignment_metadata", {})
    video_info = dict(result.get("video", {}))
    video_info["segment_id"] = alignment_meta.get("segment_id", segment_id)
    video_info["label_idx"] = alignment_meta.get("label_idx", label_idx)

    quality_ind = result.get("quality_indicator", {})

    return FacialRecord(
        ear_statistics=result.get("ear_statistics", {}),
        gaze_statistics=result.get("gaze_statistics", {}),
        head_pose_statistics=result.get("head_pose_statistics", {}),
        blink_count=int(result.get("blink_count", 0)),
        blink_rate_per_minute=float(result.get("blink_rate_per_minute", 0.0)),
        face_coverage_percent=float(result.get("face_coverage_percent", 0.0)),
        quality_indicator=quality_ind,
        openface=of_status,
        au_statistics=au_stats,
        video_metadata=video_info,
        status="available",
    )


# ---------------------------------------------------------------------------
# Session builder
# ---------------------------------------------------------------------------

def build_session(
    segment_id: str,
    label_idx: int,
    start: float,
    end: float,
    video_path: Path,
    audio_path: Path,
    transcript: Optional[str] = None,
    text_artifact_path: Path = DEFAULT_TEXT_ARTIFACT,
    source_dataset: str = "CMU-MOSEI",
) -> MultimodalRecord:
    """
    Build one integrated MultimodalRecord for a single segment.

    1. Establishes identity.
    2. Retrieves text from validated artifact.
    3. Processes speech via existing pipeline.
    4. Processes facial via existing pipeline.
    5. Validates alignment.
    6. Returns structured record.

    Does NOT perform fusion.
    Does NOT modify any existing pipeline.
    """
    identity = SegmentIdentity(
        source_dataset=source_dataset,
        segment_id=str(segment_id),
        label_idx=int(label_idx),
        start=float(start),
        end=float(end),
    )

    print(f"\n[integration] Building session: segment_id={segment_id}, label_idx={label_idx}")
    print(f"[integration] start={start}, end={end}, source={source_dataset}")

    # -- Text --
    print("[integration] Retrieving text representation...")
    text_record = retrieve_text_record(
        segment_id=segment_id,
        label_idx=label_idx,
        artifact_path=text_artifact_path,
    )
    print(f"[integration] Text status: {text_record.status}")

    # -- Speech --
    print("[integration] Processing speech...")
    speech_record = retrieve_speech_record(
        audio_path=audio_path,
        segment_id=segment_id,
        label_idx=label_idx,
        start=start,
        end=end,
        transcript=transcript,
    )
    print(f"[integration] Speech status: {speech_record.status}")

    # -- Facial --
    print("[integration] Processing facial features...")
    facial_record = retrieve_facial_record(
        video_path=video_path,
        segment_id=segment_id,
        label_idx=label_idx,
        start=start,
        end=end,
    )
    print(f"[integration] Facial status: {facial_record.status}")
    if facial_record.openface is not None:
        print(f"[integration] OpenFace status: {facial_record.openface.status}")

    # -- Assemble record --
    record = MultimodalRecord(
        identity=identity,
        text=text_record,
        speech=speech_record,
        facial=facial_record,
    )

    # -- Validate --
    print("[integration] Running alignment validation...")
    summary = validate_alignment(record)
    record.integration = summary
    print(f"[integration] Alignment: {summary.alignment_status}")
    print(f"[integration] Validation: {summary.validation_status}")
    if summary.alignment_errors:
        for err in summary.alignment_errors:
            print(f"  [ALIGN ERROR] {err}")
    if summary.validation_errors:
        for err in summary.validation_errors:
            print(f"  [VALID ERROR] {err}")

    return record


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------

def print_record_summary(record: MultimodalRecord) -> None:
    """Print a human-readable summary of the integrated record."""
    sep = "=" * 70
    print()
    print(sep)
    print("ABDA PHASE 1 — INTEGRATED MULTIMODAL RECORD")
    print(sep)

    ident = record.identity
    print(f"\nIDENTITY")
    print(f"  source_dataset : {ident.source_dataset}")
    print(f"  segment_id     : {ident.segment_id}")
    print(f"  label_idx      : {ident.label_idx}")
    print(f"  start          : {ident.start}")
    print(f"  end            : {ident.end}")

    # -- Text --
    print(f"\nTEXT")
    t = record.text
    if t is None:
        print("  status         : not_present")
    else:
        print(f"  status         : {t.status}")
        if t.status == "available":
            print(f"  feature_dim    : {t.feature_dim}")
            print(f"  text           : {t.text[:80] if t.text else '(empty)'}...")
            f = t.features
            print(f"  sentiment      : {f.get('sentiment_normalized', 'n/a'):.4f}")
            print(f"  word_count     : {f.get('word_count', 'n/a')}")
            print(f"  avg_word_len   : {f.get('avg_word_length', 'n/a'):.3f}")
            print(f"  dep_depth      : {f.get('dependency_tree_depth', 'n/a')}")
            msttr = f.get("msttr_10")
            print(f"  msttr_10       : {msttr if msttr is not None else 'NaN (segment < 10 tokens)'}")
            labse = f.get("labse_embedding")
            if labse is not None:
                arr = np.asarray(labse)
                print(f"  labse_dim      : {arr.shape[0]}")
            print(f"  pipeline       : {t.validity.get('pipeline', 'n/a')}")

    # -- Speech --
    print(f"\nSPEECH")
    s = record.speech
    if s is None:
        print("  status         : not_present")
    else:
        print(f"  status         : {s.status}")
        if s.status == "available":
            print(f"  feature_dim    : {s.feature_dim}")
            if s.quality:
                print(f"  quality_score  : {s.quality.quality_score:.4f}")
                print(f"  speech_ratio   : {s.quality.speech_ratio:.4f}")
                print(f"  snr_estimate   : {s.quality.snr_estimate:.4f}")
            print(f"  feature_ver    : {s.feature_version}")

    # -- Facial --
    print(f"\nFACIAL")
    fa = record.facial
    if fa is None:
        print("  status         : not_present")
    else:
        print(f"  status         : {fa.status}")
        if fa.status == "available":
            print(f"  face_coverage  : {fa.face_coverage_percent:.2f}%")
            print(f"  blink_count    : {fa.blink_count}")
            print(f"  blink_rate/min : {fa.blink_rate_per_minute:.3f}")
            ear = fa.ear_statistics
            if ear:
                print(f"  EAR mean/std   : {ear.get('mean', 'n/a'):.3f} / {ear.get('std', 'n/a'):.3f}")
            gaze = fa.gaze_statistics
            if gaze:
                gp = gaze.get("percentages", {})
                print(
                    f"  gaze           : "
                    f"L={gp.get('LEFT', 0):.1f}% "
                    f"C={gp.get('CENTER', 0):.1f}% "
                    f"R={gp.get('RIGHT', 0):.1f}%"
                )
            pose = fa.head_pose_statistics
            if pose:
                p = pose.get("pitch", {})
                y = pose.get("yaw", {})
                r = pose.get("roll", {})
                print(
                    f"  head_pose pitch: mean={p.get('mean', 'n/a'):.3f}  "
                    f"yaw mean={y.get('mean', 'n/a'):.3f}  "
                    f"roll mean={r.get('mean', 'n/a'):.3f}"
                )
            of = fa.openface
            if of:
                print(f"  openface       : status={of.status}")
                if of.error:
                    print(f"  openface_error : {of.error}")
            print(f"  au_statistics  : {len(fa.au_statistics)} AU(s)")

    # -- Integration --
    print(f"\nINTEGRATION")
    integ = record.integration
    if integ is None:
        print("  validation     : NOT RUN")
    else:
        print(f"  alignment      : {integ.alignment_status}")
        print(f"  validation     : {integ.validation_status}")
        avail = integ.modality_availability
        print(f"  text           : {avail.get('text', 'n/a')}")
        print(f"  speech         : {avail.get('speech', 'n/a')}")
        print(f"  facial         : {avail.get('facial', 'n/a')}")
        if integ.alignment_errors:
            for e in integ.alignment_errors:
                print(f"  [ALIGN ERROR]  : {e}")
        if integ.validation_errors:
            for e in integ.validation_errors:
                print(f"  [VALID ERROR]  : {e}")
        print(f"\n  fusion         : NOT IMPLEMENTED (Phase 1 — methodology pending)")

    print()
    print(sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ABDA Phase 1 — Build one integrated multimodal session record."
    )
    parser.add_argument("--segment-id", required=True, type=str)
    parser.add_argument("--label-idx", required=True, type=int)
    parser.add_argument("--start", default=0.311, type=float)
    parser.add_argument("--end", default=3.775, type=float)
    parser.add_argument(
        "--video",
        default=r"C:\Projects\Raw\266396\0.mp4",
        type=Path,
    )
    parser.add_argument(
        "--audio",
        default="data/processed/pilot/266396_0.wav",
        type=Path,
    )
    parser.add_argument(
        "--transcript",
        default="there are two types of people in this world people who like m night films",
        type=str,
    )
    parser.add_argument(
        "--text-artifact",
        default="data/processed/text/CMU_MOSEI_final_NLP_features.pkl",
        type=Path,
    )
    parser.add_argument(
        "--save-json",
        default=None,
        type=Path,
        help="Optional path to save the serialised record as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    record = build_session(
        segment_id=args.segment_id,
        label_idx=args.label_idx,
        start=args.start,
        end=args.end,
        video_path=args.video,
        audio_path=args.audio,
        transcript=args.transcript,
        text_artifact_path=args.text_artifact,
        source_dataset="CMU-MOSEI",
    )

    print_record_summary(record)

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(record.to_dict(), fh, indent=2, ensure_ascii=False, default=str)
        print(f"[integration] Record saved: {out}")

    integ = record.integration
    if integ and integ.validation_status != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
