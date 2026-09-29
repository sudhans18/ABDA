"""
ABDA Phase 1 — Demo text adapter for the live-demo backend.

IMPORTANT CONSTRAINTS:
  - This adapter does NOT call the validated CMU-MOSEI text pipeline.
  - The CMU-MOSEI pipeline (text/run_pipeline.py) operates on the full
    dataset and must NOT be modified to support arbitrary live transcripts.
  - This adapter produces a LIGHTWEIGHT, separate text representation
    for live-demo display purposes only.
  - Live transcription is NOT CMU-MOSEI ground truth.
  - No LaBSE embedding is computed here (would require ~470 MB model load).
    The adapter notes this explicitly rather than returning zeros.
  - No sentiment model is run (requires Cardiff XLM-R).
    The adapter notes this explicitly.
  - Only the LIGHTWEIGHT features (word_count, avg_word_length, msttr_10,
    dependency_tree_depth, avg_dependency_distance) are computed from the
    transcript using the existing utility functions, WITHOUT running the
    full pipeline.

This is a DEMO ADAPTER -- not a research pipeline.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import numpy as np


def run_demo_text_adapter(transcript: str) -> Dict[str, Any]:
    """
    Compute lightweight text features from a live transcript.

    Returns a display-ready dict marked as 'live_inference'.
    Does NOT run LaBSE or Cardiff XLM-R sentiment models.
    Does NOT modify the CMU-MOSEI text pipeline.
    """
    transcript = str(transcript).strip()
    if not transcript:
        return {
            "status": "unavailable",
            "source": "live_inference",
            "note": "Empty transcript",
        }

    result: Dict[str, Any] = {
        "status": "available",
        "source": "live_inference",
        "note": (
            "Lightweight demo adapter. "
            "LaBSE embedding and Cardiff sentiment NOT computed in live demo "
            "(separate validated CMU-MOSEI pipeline). "
            "Linguistic complexity features computed from live transcript."
        ),
        "transcript_length": len(transcript),
    }

    # -- Surface features (from existing text.features utility definitions) --
    words = transcript.split()
    result["word_count"] = len(words)
    result["avg_word_length"] = (
        float(np.mean([len(w) for w in words])) if words else 0.0
    )

    # -- MSTTR-10 (reuse existing function) --
    try:
        from text.features import lexical_tokens, msttr_10 as _msttr
        tokens = lexical_tokens(transcript)
        result["msttr_10"] = _msttr(tokens)
        if result["msttr_10"] is not None and result["msttr_10"] != result["msttr_10"]:
            result["msttr_10"] = None  # NaN → None for JSON
    except Exception as exc:
        result["msttr_10"] = None
        result["msttr_error"] = str(exc)

    # -- Dependency features via spaCy (lightweight, already installed) --
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(transcript)
            tokens_nlp = [t for t in doc if not t.is_space and not t.is_punct]
            if len(tokens_nlp) > 1:
                def _depth(token):
                    d, cur, seen = 0, token, set()
                    while cur.head != cur:
                        if cur.i in seen:
                            break
                        seen.add(cur.i)
                        d += 1
                        cur = cur.head
                    return d
                depths = [_depth(t) for t in tokens_nlp]
                distances = [abs(t.i - t.head.i) for t in tokens_nlp if t.head != t]
                result["dependency_tree_depth"] = int(max(depths)) if depths else 0
                result["avg_dependency_distance"] = float(np.mean(distances)) if distances else 0.0
            else:
                result["dependency_tree_depth"] = 0
                result["avg_dependency_distance"] = 0.0
        except OSError:
            result["dependency_tree_depth"] = None
            result["avg_dependency_distance"] = None
            result["spacy_note"] = "en_core_web_sm not available"
    except ImportError:
        result["dependency_tree_depth"] = None
        result["avg_dependency_distance"] = None
        result["spacy_note"] = "spacy not available"

    # -- Features NOT computed in demo (explicit, never zeros) --
    result["labse_embedding"] = {
        "status": "not_computed",
        "reason": (
            "LaBSE embedding requires the validated CMU-MOSEI text pipeline. "
            "Not computed for live demo transcripts."
        ),
        "dim": 768,
    }
    result["sentiment_normalized"] = {
        "status": "not_computed",
        "reason": (
            "Cardiff XLM-R sentiment not computed for live demo. "
            "Use the validated text pipeline for CMU-MOSEI data."
        ),
    }

    return result
