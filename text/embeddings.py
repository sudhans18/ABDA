"""
ABDA Phase 1 — LaBSE semantic embedding pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

from typing import Iterable

import numpy as np
import pandas as pd

@dataclass
class LaBSEEncoder:
    """
    Reusable LaBSE tokenizer + encoder.

    The same model instance is used for both:
    1. token-length validation
    2. embedding generation
    """

    model_name: str = "sentence-transformers/LaBSE"
    device: str | None = None
    max_length: int = 256

    def __post_init__(self):
        kwargs = {}

        if self.device is not None:
            kwargs["device"] = self.device

        self.model = SentenceTransformer(
            self.model_name,
            **kwargs,
        )

        self.tokenizer = self.model.tokenizer

    def count_tokens(
        self,
        text: str,
    ) -> int:
        """
        Count the full number of LaBSE tokens without truncation.

        This intentionally allows sequences longer than the model's
        256-token input limit because Stage 3 uses this count to
        identify and exclude over-length samples.
        """

        tokenizer = self.tokenizer

        original_max_length = tokenizer.model_max_length

        try:
            # Temporarily disable the tokenizer's warning threshold.
            tokenizer.model_max_length = int(1e9)

            encoded = tokenizer(
                str(text),
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
            )

        finally:
            tokenizer.model_max_length = original_max_length

        return len(encoded["input_ids"])

    def encode(
        self,
        texts,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

def load_labse(
    model_name="sentence-transformers/LaBSE",
    device=None,
    max_length=256,
):
    return LaBSEEncoder(
        model_name=model_name,
        device=device,
        max_length=max_length,
    )


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
    df,
    encoder,
    text_column="text",
    batch_size=32,
):
    result = df.copy()

    embeddings = encoder.encode(
        result[text_column].tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
    )

    result["labse_embedding"] = list(
        embeddings
    )

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
