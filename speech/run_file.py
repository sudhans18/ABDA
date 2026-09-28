"""
ABDA Phase 1 — Speech file runner.

Runs the existing speech feature extractor on one audio file.

This is intentionally separate from CMU-MOSEI media discovery.
The extractor itself remains dataset-agnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .extraction import (
    detect_voice_activity,
    load_audio,
)
from .features import (
    build_speech_session_record,
)
from .validate import (
    validate_audio,
    validate_session_record,
    validate_speech_intervals,
)


def run_speech_file(
    audio_path: str | Path,
    participant_id: str,
    session_id: str,
    segment_id: str | None = None,
    label_idx: int | None = None,
    start: float | None = None,
    end: float | None = None,
    transcript: str | None = None,
    source_dataset: str = "CMU-MOSEI",
    language: str = "English",
) -> dict:
    """Run the existing speech pipeline on one audio file."""

    audio, sr = load_audio(audio_path)

    validate_audio(
        audio,
        sr,
    )

    speech_intervals = detect_voice_activity(
        audio,
        sr,
    )

    validate_speech_intervals(
        speech_intervals,
        len(audio),
    )

    record = build_speech_session_record(
        audio=audio,
        sr=sr,
        speech_intervals=speech_intervals,
        participant_id=participant_id,
        session_id=session_id,
        timestamp="0",
        source_dataset=source_dataset,
        language=language,
        transcript=transcript,
    )

    # Add alignment metadata without modifying the extractor.
    record["segment_id"] = segment_id
    record["label_idx"] = label_idx
    record["start"] = start
    record["end"] = end

    validate_session_record(
        record
    )

    return record


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ABDA Phase 1 speech extraction."
    )

    parser.add_argument(
        "--audio",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--participant-id",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--session-id",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--segment-id",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--label-idx",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--transcript",
        default=None,
        type=str,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    record = run_speech_file(
        audio_path=args.audio,
        participant_id=args.participant_id,
        session_id=args.session_id,
        segment_id=args.segment_id,
        label_idx=args.label_idx,
        transcript=args.transcript,
    )

    print()
    print("=" * 70)
    print("ABDA SPEECH EXTRACTION")
    print("=" * 70)

    print(
        f"Feature count : "
        f"{len(record['features'])}"
    )

    print(
        f"Quality score : "
        f"{record['quality_score']:.4f}"
    )

    print(
        f"Speech ratio : "
        f"{record['speech_ratio']:.4f}"
    )

    print(
        f"SNR estimate : "
        f"{record['snr_estimate']:.4f}"
    )

    print("VALIDATION: PASS")


if __name__ == "__main__":
    main()