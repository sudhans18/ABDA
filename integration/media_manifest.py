"""
ABDA Phase 1 — CMU-MOSEI raw-media mapping.

This module attaches raw audio/video paths to the established
CMU-MOSEI segment manifest.

It intentionally does NOT assume a filename convention.

A mapping file must explicitly provide:
    segment_id
    audio_path
    video_path

This prevents accidental alignment based on guessed filenames.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


KEY_COLUMNS = [
    "segment_id",
    "label_idx",
]


def load_manifest(
    path: str | Path,
) -> pd.DataFrame:
    """Load the base CMU-MOSEI manifest."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{path.resolve()}"
        )

    df = pd.read_csv(path)

    required = [
        "segment_id",
        "label_idx",
        "start",
        "end",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Manifest missing columns: {missing}"
        )

    return df


def load_media_mapping(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load an explicit raw-media mapping.

    Required:
        segment_id
        label_idx
        audio_path
        video_path
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Media mapping not found:\n{path.resolve()}"
        )

    df = pd.read_csv(path)

    required = [
        "segment_id",
        "label_idx",
        "audio_path",
        "video_path",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Media mapping missing columns: {missing}"
        )

    duplicates = df.duplicated(
        subset=KEY_COLUMNS
    ).sum()

    if duplicates:
        raise ValueError(
            f"Media mapping contains {duplicates} "
            "duplicate alignment keys."
        )

    return df


def attach_media(
    manifest: pd.DataFrame,
    media_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Attach explicitly mapped media paths."""

    merged = manifest.merge(
        media_mapping,
        on=KEY_COLUMNS,
        how="left",
        suffixes=("", "_mapped"),
        validate="one_to_one",
    )

    return merged