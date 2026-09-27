"""
ABDA Phase 1 — Text pipeline validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ALIGNMENT_KEYS = ["segment_id", "label_idx"]


def validate_alignment_keys(df: pd.DataFrame) -> None:
    """Validate uniqueness and completeness of the segment alignment key."""
    missing = [c for c in ALIGNMENT_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing alignment columns: {missing}")

    if df[ALIGNMENT_KEYS].isna().any().any():
        raise ValueError("Alignment keys contain missing values.")

    duplicates = df.duplicated(subset=ALIGNMENT_KEYS).sum()
    if duplicates:
        raise ValueError(
            f"Found {duplicates} duplicate (segment_id, label_idx) keys."
        )


def validate_final_nlp_features(df: pd.DataFrame) -> dict:
    """
    Validate the final NLP representation.

    MSTTR-10 NaNs are allowed because they are structurally undefined for
    segments containing fewer than 10 lexical tokens.
    """
    expected_columns = [
        "segment_id",
        "label_idx",
        "start",
        "end",
        "text",
        "sentiment_normalized",
        "word_count",
        "avg_word_length",
        "dependency_tree_depth",
        "avg_dependency_distance",
        "msttr_10",
        "labse_embedding",
    ]

    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing final NLP columns: {missing}")

    validate_alignment_keys(df)

    if df["text"].isna().any() or df["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Final NLP table contains empty text.")

    if not df["sentiment_normalized"].between(-1, 1).all():
        raise ValueError("Sentiment is outside [-1, 1].")

    numeric_columns = [
        "word_count",
        "avg_word_length",
        "dependency_tree_depth",
        "avg_dependency_distance",
    ]

    numeric = df[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Linguistic complexity contains NaN/Inf.")

    if (numeric < 0).any():
        raise ValueError("Linguistic complexity contains negative values.")

    msttr = df["msttr_10"].dropna()
    if len(msttr) and not msttr.between(0, 1).all():
        raise ValueError("MSTTR-10 is outside [0, 1].")

    X = np.vstack(df["labse_embedding"].to_numpy())
    if X.shape != (len(df), 768):
        raise ValueError(
            f"Expected LaBSE matrix {(len(df), 768)}; got {X.shape}."
        )

    if not np.isfinite(X).all():
        raise ValueError("LaBSE contains NaN/Inf.")

    norms = np.linalg.norm(X, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("LaBSE embeddings are not L2-normalized.")

    return {
        "samples": int(len(df)),
        "embedding_dimension": int(X.shape[1]),
        "final_feature_dimension": 774,
        "msttr_valid": int(df["msttr_10"].notna().sum()),
        "msttr_missing": int(df["msttr_10"].isna().sum()),
        "duplicate_keys": int(
            df.duplicated(subset=ALIGNMENT_KEYS).sum()
        ),
    }


def compare_artifacts(
    *dataframes: tuple[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compare alignment keys across multiple feature artifacts.

    Usage:
        compare_artifacts(
            ("labse", labse_df),
            ("sentiment", sentiment_df),
            ("complexity", complexity_df),
        )
    """
    if not dataframes:
        raise ValueError("At least one dataframe is required.")

    key_sets = {}
    rows = []

    for name, df in dataframes:
        validate_alignment_keys(df)
        keys = pd.MultiIndex.from_frame(df[ALIGNMENT_KEYS])
        key_sets[name] = set(keys.tolist())

        rows.append(
            {
                "artifact": name,
                "rows": len(df),
                "unique_keys": len(key_sets[name]),
                "duplicate_keys": int(
                    df.duplicated(subset=ALIGNMENT_KEYS).sum()
                ),
            }
        )

    reference_name = dataframes[0][0]
    reference = key_sets[reference_name]

    for name, keys in key_sets.items():
        if keys != reference:
            raise ValueError(
                f"Alignment mismatch between {reference_name} and {name}: "
                f"missing_from_{name}={len(reference - keys)}, "
                f"extra_in_{name}={len(keys - reference)}"
            )

    return pd.DataFrame(rows)

def validate_alignment_keys(
    df,
    key_columns,
) -> None:
    """Validate uniqueness and completeness of alignment keys."""

    missing = [
        column
        for column in key_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing alignment columns: {missing}"
        )

    if df[list(key_columns)].isna().any().any():
        raise ValueError(
            "Alignment keys contain missing values."
        )

    duplicate_count = (
        df.duplicated(
            subset=list(key_columns)
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            f"alignment keys."
        )

    print("Alignment keys: PASS")
    print(
        f"Unique keys: "
        f"{len(df):,}"
    )


def validate_final_nlp_table(
    df,
    expected_embedding_dim: int = 768,
) -> None:
    """Validate the final Phase 1 NLP feature table."""

    required_columns = [
        "segment_id",
        "label_idx",
        "start",
        "end",
        "text",
        "sentiment_normalized",
        "word_count",
        "avg_word_length",
        "dependency_tree_depth",
        "avg_dependency_distance",
        "msttr_10",
        "labse_embedding",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing final NLP columns: {missing}"
        )

    if len(df) == 0:
        raise ValueError(
            "Final NLP table is empty."
        )

    # ----------------------------------------------------------
    # LaBSE
    # ----------------------------------------------------------

    embeddings = df[
        "labse_embedding"
    ].to_numpy()

    if len(embeddings) != len(df):
        raise ValueError(
            "Embedding count does not match "
            "row count."
        )

    dimensions = {
        len(np.asarray(embedding))
        for embedding in embeddings
    }

    if dimensions != {expected_embedding_dim}:
        raise ValueError(
            f"Unexpected LaBSE dimensions: "
            f"{dimensions}"
        )

    matrix = np.vstack(embeddings)

    if not np.isfinite(matrix).all():
        raise ValueError(
            "LaBSE embeddings contain NaN/Inf."
        )

    # ----------------------------------------------------------
    # Sentiment
    # ----------------------------------------------------------

    sentiment = df[
        "sentiment_normalized"
    ].to_numpy(dtype=float)

    if not np.isfinite(sentiment).all():
        raise ValueError(
            "Sentiment contains NaN/Inf."
        )

    if not ((sentiment >= -1).all() and
            (sentiment <= 1).all()):
        raise ValueError(
            "Sentiment is outside [-1, 1]."
        )

    # ----------------------------------------------------------
    # Complexity
    # ----------------------------------------------------------

    complexity_columns = [
        "word_count",
        "avg_word_length",
        "dependency_tree_depth",
        "avg_dependency_distance",
    ]

    for column in complexity_columns:
        values = df[column].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{column} contains NaN/Inf."
            )

        if (values < 0).any():
            raise ValueError(
                f"{column} contains negative values."
            )

    # ----------------------------------------------------------
    # MSTTR
    # ----------------------------------------------------------

    msttr = df[
        "msttr_10"
    ].dropna().to_numpy(dtype=float)

    if len(msttr) > 0:
        if not np.isfinite(msttr).all():
            raise ValueError(
                "MSTTR contains Inf."
            )

        if not ((msttr >= 0).all() and
                (msttr <= 1).all()):
            raise ValueError(
                "MSTTR values outside [0, 1]."
            )

    # ----------------------------------------------------------
    # Expected dimensionality
    # ----------------------------------------------------------

    expected_feature_dimension = (
        768  # LaBSE
        + 1  # sentiment
        + 4  # complexity
        + 1  # MSTTR
    )

    if expected_feature_dimension != 774:
        raise AssertionError(
            "Internal feature-dimension definition "
            "is inconsistent."
        )

    print("Final NLP table: PASS")
    print(
        f"Rows: {len(df):,}"
    )
    print(
        f"Feature dimension: "
        f"{expected_feature_dimension}"
    )