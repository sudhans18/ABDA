"""
ABDA Phase 1 — NLP feature extraction.

Final Phase 1 text representation:
    768-D LaBSE
    + 1-D continuous sentiment
    + 4-D linguistic complexity
    + 1-D lexical diversity (MSTTR-10)
    = 774 dimensions

Important:
CMU-MOSEI's annotated sentiment is a ground-truth/reference signal used for
evaluation. The production sentiment feature defaults to the selected Cardiff
XLM-R model. Use sentiment_source="cmu_mosei" only when reproducing the
historical Phase 1 artifact exactly.
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd


CARDIFF_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def extract_cardiff_sentiment(
    texts,
    model_name: str = CARDIFF_MODEL,
    device: str | None = None,
    batch_size: int = 32,
    max_length: int = 256,
) -> np.ndarray:
    """
    Produce continuous sentiment = P(positive) - P(negative).

    Output is naturally bounded to [-1, 1].
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}

    positive_idx = next(
        (idx for idx, label in id2label.items() if "positive" in label.lower()),
        None,
    )
    negative_idx = next(
        (idx for idx, label in id2label.items() if "negative" in label.lower()),
        None,
    )

    if positive_idx is None or negative_idx is None:
        raise ValueError(f"Could not identify positive/negative labels: {id2label}")

    scores = []

    texts = list(texts)
    for start in range(0, len(texts), batch_size):
        batch = [str(x) for x in texts[start : start + batch_size]]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded)
            probabilities = torch.softmax(logits.logits, dim=-1).cpu().numpy()

        scores.append(
            probabilities[:, positive_idx] - probabilities[:, negative_idx]
        )

    result = np.concatenate(scores).astype(np.float32)

    if not np.all(np.isfinite(result)):
        raise ValueError("Cardiff sentiment contains non-finite values.")

    return result


def add_sentiment_feature(
    df: pd.DataFrame,
    sentiment_source: Literal["cardiff", "cmu_mosei"] = "cardiff",
    model_name: str = CARDIFF_MODEL,
    batch_size: int = 32,
    max_length: int = 256,
) -> pd.DataFrame:
    """
    Add `sentiment_normalized`.

    - cardiff: model-derived continuous sentiment.
    - cmu_mosei: CMU-MOSEI annotation / 3.0, for historical reproduction.
    """
    result = df.copy()

    if sentiment_source == "cardiff":
        result["sentiment_normalized"] = extract_cardiff_sentiment(
            result["text"].tolist(),
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
        )
    elif sentiment_source == "cmu_mosei":
        if "sentiment" not in result.columns:
            raise KeyError("CMU-MOSEI sentiment annotation is missing.")
        result["sentiment_normalized"] = (
            result["sentiment"].astype(float) / 3.0
        )
    else:
        raise ValueError(
            "sentiment_source must be 'cardiff' or 'cmu_mosei'."
        )

    if not result["sentiment_normalized"].between(-1, 1).all():
        raise ValueError("Sentiment feature is outside [-1, 1].")

    return result


def _surface_features(text: str) -> dict:
    text = str(text).strip()
    words = text.split()

    if not words:
        return {
            "word_count": 0,
            "avg_word_length": 0.0,
        }

    return {
        "word_count": len(words),
        "avg_word_length": float(
            np.mean([len(word) for word in words])
        ),
    }


def _dependency_features(doc) -> dict:
    tokens = [
        token for token in doc
        if not token.is_space and not token.is_punct
    ]

    if len(tokens) <= 1:
        return {
            "dependency_tree_depth": 0,
            "avg_dependency_distance": 0.0,
        }

    def token_depth(token):
        depth = 0
        current = token
        visited = set()

        while current.head != current:
            if current.i in visited:
                break
            visited.add(current.i)
            depth += 1
            current = current.head

        return depth

    depths = [token_depth(token) for token in tokens]
    distances = [
        abs(token.i - token.head.i)
        for token in tokens
        if token.head != token
    ]

    return {
        "dependency_tree_depth": int(max(depths) if depths else 0),
        "avg_dependency_distance": (
            float(np.mean(distances)) if distances else 0.0
        ),
    }


def add_linguistic_complexity(
    df: pd.DataFrame,
    spacy_model: str = "en_core_web_sm",
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Extract the four selected Phase 1 complexity features.

    The implementation matches Notebook 4:
      word_count
      avg_word_length
      dependency_tree_depth
      avg_dependency_distance
    """
    import spacy

    nlp = spacy.load(spacy_model)

    texts = df["text"].astype(str).tolist()
    rows = []

    for text, doc in zip(texts, nlp.pipe(texts, batch_size=batch_size)):
        row = {}
        row.update(_surface_features(text))
        row.update(_dependency_features(doc))
        rows.append(row)

    complexity = pd.DataFrame(rows)

    expected = [
        "word_count",
        "avg_word_length",
        "dependency_tree_depth",
        "avg_dependency_distance",
    ]
    complexity = complexity[expected]

    result = pd.concat(
        [df.reset_index(drop=True), complexity],
        axis=1,
    )

    return result


def lexical_tokens(text: str) -> list[str]:
    """Match the Notebook 5 lexical-token definition."""
    if not isinstance(text, str):
        return []
    return re.findall(r"\b[\w']+\b", text.lower())


def msttr_10(tokens: list[str], segment_size: int = 10) -> float:
    """
    Compute the selected MSTTR-10 implementation.

    Only complete non-overlapping blocks of 10 tokens are used.
    If fewer than 10 tokens are available, return NaN.
    """
    if len(tokens) < segment_size:
        return np.nan

    n_segments = len(tokens) // segment_size
    ttr_values = []

    for i in range(n_segments):
        segment = tokens[
            i * segment_size : (i + 1) * segment_size
        ]
        if len(segment) == 0:
            continue

        ttr_values.append(len(set(segment)) / len(segment))

    if not ttr_values:
        return np.nan

    return float(np.mean(ttr_values))


def add_lexical_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """Add the final MSTTR-10 lexical diversity feature."""
    result = df.copy()

    tokens = result["text"].astype(str).apply(lexical_tokens)
    result["msttr_10"] = tokens.apply(msttr_10)

    valid = result["msttr_10"].dropna()
    if len(valid) and not valid.between(0, 1).all():
        raise ValueError("MSTTR-10 contains values outside [0, 1].")

    return result


def build_final_nlp_features(
    embedding_df: pd.DataFrame,
    sentiment_source: Literal["cardiff", "cmu_mosei"] = "cardiff",
    spacy_model: str = "en_core_web_sm",
    sentiment_model: str = CARDIFF_MODEL,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Build the final 774-D NLP feature representation.

    The returned table keeps session/segment metadata needed for later
    multimodal alignment.
    """
    required = {
        "segment_id",
        "label_idx",
        "start",
        "end",
        "text",
        "labse_embedding",
    }
    missing = required - set(embedding_df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = embedding_df.copy()

    df = add_sentiment_feature(
        df,
        sentiment_source=sentiment_source,
        model_name=sentiment_model,
        batch_size=batch_size,
    )

    df = add_linguistic_complexity(
        df,
        spacy_model=spacy_model,
        batch_size=64,
    )

    df = add_lexical_diversity(df)

    final_columns = [
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

    final = df[final_columns].copy()

    # Explicitly validate the 774-D representation.
    embedding_dim = len(final.iloc[0]["labse_embedding"])
    if embedding_dim != 768:
        raise ValueError(f"Expected 768-D LaBSE; got {embedding_dim}.")

    if len(final) != len(embedding_df):
        raise ValueError("Row count changed during feature extraction.")

    return final


def text_quality_score(
    text: str,
    token_count: int | None = None,
    max_length: int = 256,
) -> float:
    """
    Lightweight text-quality heuristic for the common interface.

    This is an input-quality indicator, not a psychological score.
    It rewards non-empty text and penalizes truncation/very short inputs.
    """
    text = str(text).strip()
    if not text:
        return 0.0

    words = text.split()
    if not words:
        return 0.0

    score = 1.0

    if len(words) < 3:
        score *= 0.75

    if token_count is not None and token_count > max_length:
        score *= 0.5

    return float(np.clip(score, 0.0, 1.0))
