"""
ABDA Phase 1 — Text preprocessing.

Reconstructs CMU-MOSEI labelled text segments from the timestamped-word
computational sequence and label computational sequence.

This module intentionally preserves the alignment logic used in the Phase 1
Colab notebook:
    word_end > label_start AND word_start < label_end

The six segments longer than LaBSE's 256-token input limit are excluded from
the embedding-ready dataset, matching the Phase 1 experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


LABEL_NAMES = [
    "sentiment",
    "happiness",
    "sadness",
    "anger",
    "surprise",
    "disgust",
    "fear",
]


def load_csd_sequences(words_path: str | Path, labels_path: str | Path):
    """Load CMU-MOSEI timestamped words and labels using MMSDK."""
    try:
        from mmsdk import mmdatasdk
    except ImportError as exc:
        raise ImportError(
            "CMU-MOSEI preprocessing requires MMSDK. "
            "Install the CMU-MultimodalSDK package."
        ) from exc

    words_dataset = mmdatasdk.mmdataset({"words": str(words_path)})
    labels_dataset = mmdatasdk.mmdataset({"labels": str(labels_path)})
    return words_dataset["words"], labels_dataset["labels"]


def get_words_for_interval(segment_id, start: float, end: float, words_sequence):
    """
    Return timestamped words overlapping [start, end).

    Exact Phase 1 overlap rule:
        word_end > start AND word_start < end

    'sp' is a CMU-MOSEI pause annotation and is excluded from semantic text.
    """
    group = words_sequence.data[segment_id]
    intervals = group["intervals"][:]
    features = group["features"][:]

    selected = []
    for interval, feature in zip(intervals, features):
        word_start = float(interval[0])
        word_end = float(interval[1])

        if word_end > start and word_start < end:
            word = feature[0].decode("utf-8").strip()
            if word.lower() == "sp":
                continue
            selected.append((word_start, word_end, word))

    selected.sort(key=lambda x: x[0])
    return selected


def build_aligned_segments(words_sequence, labels_sequence) -> pd.DataFrame:
    """Build one row per CMU-MOSEI labelled interval."""
    common_ids = sorted(
        set(words_sequence.data.keys()) & set(labels_sequence.data.keys())
    )

    rows = []

    for segment_id in common_ids:
        label_group = labels_sequence.data[segment_id]
        label_intervals = label_group["intervals"][:]
        label_features = label_group["features"][:]

        for label_idx, (interval, label_vector) in enumerate(
            zip(label_intervals, label_features)
        ):
            start = float(interval[0])
            end = float(interval[1])

            selected_words = get_words_for_interval(
                segment_id, start, end, words_sequence
            )
            text = " ".join(word for _, _, word in selected_words)

            vector = np.asarray(label_vector, dtype=np.float32).reshape(-1)
            if len(vector) != len(LABEL_NAMES):
                raise ValueError(
                    f"Unexpected CMU-MOSEI label dimension for {segment_id}: "
                    f"{len(vector)}; expected {len(LABEL_NAMES)}."
                )

            rows.append(
                {
                    "segment_id": segment_id,
                    "label_idx": int(label_idx),
                    "start": start,
                    "end": end,
                    "text": text,
                    "label": vector.tolist(),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No aligned CMU-MOSEI labelled segments were produced.")

    label_matrix = np.vstack(df["label"].to_numpy())
    for i, name in enumerate(LABEL_NAMES):
        df[name] = label_matrix[:, i]

    df = df.drop(columns=["label"])

    # token_count is deliberately added in add_labse_token_count().
    # It must be computed with the exact LaBSE tokenizer because the 256-token
    # filtering decision is part of the Phase 1 artifact definition.

    df = df[
        [
            "segment_id",
            "label_idx",
            "start",
            "end",
            "text",
            *LABEL_NAMES,
        ]
    ]

    return df


def add_labse_token_count(
    df: pd.DataFrame,
    model_name: str = "sentence-transformers/LaBSE",
) -> pd.DataFrame:
    """Compute the exact LaBSE tokenizer length used by the embedding stage."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer

    result = df.copy()
    result["token_count"] = [
        len(
            tokenizer(
                str(text),
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]
        )
        for text in result["text"]
    ]
    return result


def make_embedding_ready(
    df: pd.DataFrame,
    max_length: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split aligned data into LaBSE-compatible and excluded samples.

    Returns:
        embedding_ready, excluded
    """
    if "token_count" not in df.columns:
        raise ValueError("token_count column is required.")

    ready = df.loc[df["token_count"] <= max_length].copy()
    excluded = df.loc[df["token_count"] > max_length].copy()

    return ready.reset_index(drop=True), excluded.reset_index(drop=True)


def clean_transcript(text: str) -> str:
    """Remove CMU-MOSEI 'sp' annotations and normalize whitespace."""
    import re

    text = re.sub(r"\bsp\b", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()
