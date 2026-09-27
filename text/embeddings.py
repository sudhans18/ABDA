"""
ABDA Phase 1 — LaBSE semantic embedding pipeline.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_MODEL = "sentence-transformers/LaBSE"


def load_labse(
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
):
    """Load LaBSE lazily so importing this module does not download weights."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device)
    return model


def encode_texts(
    texts: Iterable[str],
    model,
    batch_size: int = 32,
    normalize_embeddings: bool = True,
    show_progress_bar: bool = True,
) -> np.ndarray:
    """Encode texts using LaBSE."""
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2-D embedding matrix, got {embeddings.shape}.")

    return embeddings


def add_embeddings(
    df: pd.DataFrame,
    model,
    text_column: str = "text",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Return a copy of df with a 768-D `labse_embedding` column."""
    if text_column not in df.columns:
        raise KeyError(f"Missing text column: {text_column}")

    embeddings = encode_texts(
        df[text_column].astype(str).tolist(),
        model=model,
        batch_size=batch_size,
        normalize_embeddings=True,
    )

    expected_dim = model.get_sentence_embedding_dimension()
    if embeddings.shape != (len(df), expected_dim):
        raise ValueError(
            f"Unexpected embedding shape {embeddings.shape}; "
            f"expected {(len(df), expected_dim)}."
        )

    result = df.reset_index(drop=True).copy()
    result["labse_embedding"] = list(embeddings)
    return result


def validate_embeddings(
    embeddings: np.ndarray,
    expected_dim: int = 768,
    require_unit_norm: bool = True,
    atol: float = 1e-4,
) -> None:
    """Validate numerical integrity of an embedding matrix."""
    X = np.asarray(embeddings)

    if X.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D; got {X.shape}.")
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Expected embedding dimension {expected_dim}; got {X.shape[1]}."
        )
    if not np.isfinite(X).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    if require_unit_norm:
        norms = np.linalg.norm(X, axis=1)
        if not np.allclose(norms, 1.0, atol=atol):
            raise ValueError(
                "Embeddings are expected to be L2-normalized, "
                f"but norms range from {norms.min()} to {norms.max()}."
            )
