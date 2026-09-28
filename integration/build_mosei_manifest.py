"""
ABDA Phase 1 — CMU-MOSEI segment manifest builder.

The validated text representation is used only to obtain the
established CMU-MOSEI segment identity and timestamps.

This script does NOT extract audio or video features.

It creates a reproducible manifest that later modality adapters
can consume.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "segment_id",
    "label_idx",
    "start",
    "end",
]


def load_text_artifact(path: str | Path) -> pd.DataFrame:
    """Load the validated final NLP artifact."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Text artifact not found:\n{path.resolve()}"
        )

    df = pd.read_pickle(path)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Expected the NLP artifact to contain a pandas DataFrame."
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


def validate_text_identity(df: pd.DataFrame) -> None:
    """Validate the established CMU-MOSEI identity."""

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            "Manifest identity/timestamps contain missing values."
        )

    duplicates = df.duplicated(
        subset=["segment_id", "label_idx"]
    ).sum()

    if duplicates:
        raise ValueError(
            "Duplicate CMU-MOSEI alignment keys: "
            f"{duplicates}"
        )

    invalid_intervals = (
        (df["start"] < 0)
        | (df["end"] <= df["start"])
    )

    if invalid_intervals.any():
        raise ValueError(
            "Invalid CMU-MOSEI segment timestamps found."
        )


def build_manifest(
    text_artifact: str | Path,
) -> pd.DataFrame:
    """
    Build the multimodal manifest from the validated text artifact.
    """

    text_df = load_text_artifact(text_artifact)

    validate_text_identity(text_df)

    manifest_columns = [
        "segment_id",
        "label_idx",
        "start",
        "end",
    ]

    manifest = (
        text_df[manifest_columns]
        .copy()
        .sort_values(
            ["segment_id", "label_idx"]
        )
        .reset_index(drop=True)
    )

    # These fields deliberately remain unresolved until the raw
    # audio/video locations are established.
    manifest["audio_path"] = pd.NA
    manifest["video_path"] = pd.NA

    return manifest


def save_manifest(
    manifest: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save manifest as CSV."""

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        output_path,
        index=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the ABDA CMU-MOSEI multimodal segment manifest."
        )
    )

    parser.add_argument(
        "--text-artifact",
        type=str,
        default=(
            "data/processed/"
            "CMU_MOSEI_final_NLP_features.pkl"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/metadata/"
            "CMU_MOSEI_multimodal_manifest.csv"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    manifest = build_manifest(
        args.text_artifact
    )

    save_manifest(
        manifest,
        args.output,
    )

    print()
    print("=" * 70)
    print("CMU-MOSEI MANIFEST")
    print("=" * 70)
    print(f"Rows        : {len(manifest):,}")
    print(
        "Unique videos: "
        f"{manifest['segment_id'].nunique():,}"
    )
    print(
        "Unique keys : "
        f"{manifest[['segment_id', 'label_idx']].drop_duplicates().shape[0]:,}"
    )
    print(
        f"Output      : {args.output}"
    )


if __name__ == "__main__":
    main()