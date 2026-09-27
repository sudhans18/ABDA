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

import pandas as pd

from .csd import read_csd


LABEL_NAMES = [
    "sentiment",
    "happiness",
    "sadness",
    "anger",
    "surprise",
    "disgust",
    "fear",
]


def load_csd_sequences(
    words_path: str | Path,
    labels_path: str | Path,
):
    """
    Load CMU-MOSEI timestamped words and labels.

    Uses the internal HDF5 CSD reader rather than the
    legacy CMU Multimodal SDK.
    """

    words_path = Path(words_path)
    labels_path = Path(labels_path)

    print("Loading timestamped words...")

    words_dataset = read_csd(words_path)

    print(
        f"  Word sources: "
        f"{len(words_dataset):,}"
    )

    print("Loading labels...")

    labels_dataset = read_csd(labels_path)

    print(
        f"  Label sources: "
        f"{len(labels_dataset):,}"
    )

    return words_dataset, labels_dataset


def build_aligned_segments(
    words_sequence: dict,
    labels_sequence: dict,
) -> pd.DataFrame:
    """
    Reconstruct timestamp-aligned CMU-MOSEI text samples.

    For every labelled temporal interval, select timestamped
    words satisfying:

        word_end > label_start
        AND
        word_start < label_end

    Selected words are sorted chronologically.

    CMU-MOSEI 'sp' pause annotations are excluded.
    """

    common_ids = sorted(
        set(words_sequence.keys())
        &
        set(labels_sequence.keys())
    )

    print(
        f"Common segment IDs: "
        f"{len(common_ids):,}"
    )

    aligned_rows = []

    for segment_id in common_ids:

        word_group = words_sequence[segment_id]
        label_group = labels_sequence[segment_id]

        word_intervals = word_group["intervals"]
        word_features = word_group["features"]

        label_intervals = label_group["intervals"]
        label_features = label_group["features"]

        for label_idx, (
            label_interval,
            label_vector,
        ) in enumerate(
            zip(
                label_intervals,
                label_features,
            )
        ):

            label_start = float(label_interval[0])
            label_end = float(label_interval[1])

            selected_words = []

            for word_interval, word_feature in zip(
                word_intervals,
                word_features,
            ):

                word_start = float(word_interval[0])
                word_end = float(word_interval[1])

                if (
                    word_end > label_start
                    and
                    word_start < label_end
                ):

                    word = word_feature[0]

                    if isinstance(word, bytes):
                        word = word.decode("utf-8")

                    word = str(word).strip()

                    if word.lower() == "sp":
                        continue

                    selected_words.append(
                        (
                            word_start,
                            word_end,
                            word,
                        )
                    )

            selected_words.sort(
                key=lambda x: x[0]
            )

            text = " ".join(
                word
                for _, _, word in selected_words
            )

            aligned_rows.append(
                {
                    "segment_id": segment_id,
                    "label_idx": label_idx,
                    "start": label_start,
                    "end": label_end,
                    "text": text,
                    "label": (
                        label_vector.tolist()
                        if hasattr(label_vector, "tolist")
                        else list(label_vector)
                    ),
                }
            )

    return pd.DataFrame(aligned_rows)


def add_labse_token_count(
    df: pd.DataFrame,
    encoder,
) -> pd.DataFrame:
    """
    Calculate full LaBSE tokenizer length without truncation.

    The resulting column is named `token_count` to preserve
    the validated Notebook 1 schema.
    """

    result = df.copy()

    result["token_count"] = [
        encoder.count_tokens(text)
        for text in result["text"]
    ]

    return result


def make_embedding_ready(
    df: pd.DataFrame,
    max_length: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    if "token_count" not in df.columns:
        raise ValueError(
            "token_count column is required."
        )

    ready = df.loc[
        df["token_count"] <= max_length
    ].copy()

    excluded = df.loc[
        df["token_count"] > max_length
    ].copy()

    return (
        ready.reset_index(drop=True),
        excluded.reset_index(drop=True),
    )
