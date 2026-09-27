"""
ABDA Phase 1 — CMU-MOSEI Text Pipeline Runner.

Pipeline:

CMU-MOSEI CSD
    ↓
timestamp alignment
    ↓
LaBSE token filtering
    ↓
LaBSE 768-D embeddings
    ↓
Cardiff XLM-R sentiment
    ↓
linguistic complexity
    ↓
MSTTR-10
    ↓
final 774-D NLP representation
    ↓
validation
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .preprocessing import (
    add_labse_token_count,
    build_aligned_segments,
    load_csd_sequences,
    make_embedding_ready,
)

from .embeddings import (
    add_embeddings,
    load_labse,
    validate_embeddings,
)

from .features import (
    build_final_nlp_features,
)

from .validate import (
    validate_alignment_keys,
    validate_final_nlp_table,
)


def load_config(config_path: str | Path) -> dict:
    """Load YAML pipeline configuration."""

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    return config


def resolve_device(config: dict) -> str | None:
    """Resolve configured runtime device."""

    device = config["runtime"].get("device", "auto")

    if device == "auto":
        return None

    if device not in {"cpu", "cuda"}:
        raise ValueError(
            "runtime.device must be one of: auto, cpu, cuda"
        )

    return device


def save_pickle(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Save DataFrame as pickle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


def save_json(
    data: dict,
    path: Path,
) -> None:
    """Save JSON metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def print_stage(
    stage_number: int,
    title: str,
) -> None:
    """Print a consistent pipeline-stage header."""

    print()
    print("=" * 70)
    print(f"STAGE {stage_number}: {title}")
    print("=" * 70)


def run_pipeline(config_path: str | Path) -> pd.DataFrame:
    """Execute the complete Phase 1 text pipeline."""

    start_time = time.time()

    config = load_config(config_path)

    output_dir = Path(
        config["output"]["directory"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    words_path = Path(
        config["input"]["words_file"]
    )

    labels_path = Path(
        config["input"]["labels_file"]
    )

    save_intermediate = config["runtime"].get(
        "save_intermediate",
        True,
    )

    # ============================================================
    # STAGE 1 — LOAD AND ALIGN CMU-MOSEI
    # ============================================================

    print_stage(
        1,
        "CMU-MOSEI Loading and Timestamp Alignment",
    )

    print(f"Words file : {words_path}")
    print(f"Labels file: {labels_path}")

    words_sequence, labels_sequence = load_csd_sequences(
        words_path,
        labels_path,
    )

    aligned = build_aligned_segments(
        words_sequence,
        labels_sequence,
    )

    print(f"Aligned segments: {len(aligned)}")

    if save_intermediate:
        save_pickle(
            aligned,
            output_dir / "aligned_text.pkl",
        )

    # ============================================================
    # STAGE 2 — LaBSE TOKEN FILTERING
    # ============================================================

    print_stage(
        2,
        "LaBSE Token-Length Filtering",
    )

    labse_config = config["labse"]

    aligned = add_labse_token_count(
        aligned,
        model_name=labse_config["model_name"],
    )

    embedding_ready, excluded = make_embedding_ready(
        aligned,
        max_length=labse_config["max_length"],
    )

    print(f"Total aligned : {len(aligned)}")
    print(f"Embedding ready: {len(embedding_ready)}")
    print(f"Excluded       : {len(excluded)}")

    if save_intermediate:
        save_pickle(
            aligned,
            output_dir / "aligned_with_token_counts.pkl",
        )

        save_pickle(
            embedding_ready,
            output_dir / "embedding_ready.pkl",
        )

        save_pickle(
            excluded,
            output_dir / "excluded_long_segments.pkl",
        )

    # ============================================================
    # STAGE 3 — LaBSE EMBEDDINGS
    # ============================================================

    print_stage(
        3,
        "LaBSE Semantic Embeddings",
    )

    device = resolve_device(config)

    model = load_labse(
        model_name=labse_config["model_name"],
        device=device,
    )

    embedded = add_embeddings(
        embedding_ready,
        model=model,
        text_column="text",
        batch_size=labse_config["batch_size"],
    )

    embedding_matrix = np.vstack(
        embedded["labse_embedding"].to_numpy()
    )

    validate_embeddings(
        embedding_matrix,
        expected_dim=config["validation"][
            "expected_embedding_dimension"
        ],
    )

    print(
        f"LaBSE matrix shape: "
        f"{embedding_matrix.shape}"
    )

    if save_intermediate:
        save_pickle(
            embedded,
            output_dir / "labse_embeddings.pkl",
        )

    # ============================================================
    # STAGE 4 — FINAL NLP FEATURES
    # ============================================================

    print_stage(
        4,
        "Sentiment + Linguistic Complexity + MSTTR",
    )

    sentiment_config = config["sentiment"]
    complexity_config = config["complexity"]

    final = build_final_nlp_features(
        embedded,
        sentiment_source=sentiment_config["source"],
        sentiment_model=sentiment_config["model_name"],
        spacy_model=complexity_config["spacy_model"],
        batch_size=sentiment_config["batch_size"],
    )

    print(f"Final table shape: {final.shape}")

    print()
    print("Final columns:")
    for i, column in enumerate(final.columns):
        print(f"{i:2d} | {column}")

    # ============================================================
    # STAGE 5 — VALIDATION
    # ============================================================

    print_stage(
        5,
        "Final NLP Pipeline Validation",
    )

    validate_alignment_keys(
        final,
        key_columns=[
            "segment_id",
            "label_idx",
        ],
    )

    validate_final_nlp_table(
        final,
        expected_embedding_dim=config["validation"][
            "expected_embedding_dimension"
        ],
    )

    expected_dimension = config["validation"][
        "expected_final_dimension"
    ]

    actual_dimension = (
        768
        + 1
        + 4
        + 1
    )

    if actual_dimension != expected_dimension:
        raise ValueError(
            f"Expected final dimension "
            f"{expected_dimension}, "
            f"but pipeline defines "
            f"{actual_dimension}."
        )

    print()
    print("VALIDATION: PASS")

    # ============================================================
    # STAGE 6 — SAVE FINAL ARTIFACT
    # ============================================================

    print_stage(
        6,
        "Saving Final NLP Representation",
    )

    final_path = (
        output_dir /
        "CMU_MOSEI_final_NLP_features.pkl"
    )

    save_pickle(
        final,
        final_path,
    )

    # Optional CSV representation without embedding expansion.
    csv_view = final.drop(
        columns=["labse_embedding"]
    )

    csv_path = (
        output_dir /
        "CMU_MOSEI_final_NLP_features_metadata.csv"
    )

    csv_view.to_csv(
        csv_path,
        index=False,
    )

    metadata = {
        "pipeline": config["pipeline"],
        "input": {
            "words_file": str(words_path),
            "labels_file": str(labels_path),
        },
        "counts": {
            "aligned_segments": int(len(aligned)),
            "embedding_ready": int(len(embedding_ready)),
            "excluded_long_segments": int(len(excluded)),
            "final_rows": int(len(final)),
        },
        "features": {
            "labse_dimension": 768,
            "sentiment": sentiment_config["source"],
            "linguistic_complexity": [
                "word_count",
                "avg_word_length",
                "dependency_tree_depth",
                "avg_dependency_distance",
            ],
            "lexical_diversity": [
                "msttr_10",
            ],
            "final_dimension": expected_dimension,
        },
        "artifacts": {
            "final_pickle": str(final_path),
            "metadata_csv": str(csv_path),
        },
        "runtime_seconds": round(
            time.time() - start_time,
            2,
        ),
    }

    metadata_path = (
        output_dir /
        "text_pipeline_metadata.json"
    )

    save_json(
        metadata,
        metadata_path,
    )

    print(f"Final artifact: {final_path}")
    print(f"Metadata       : {metadata_path}")
    print(
        f"Runtime        : "
        f"{metadata['runtime_seconds']} seconds"
    )

    return final


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the ABDA Phase 1 CMU-MOSEI text pipeline."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/text_pipeline.yaml",
        help="Path to YAML configuration.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_pipeline(
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
    