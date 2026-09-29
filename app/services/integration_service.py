"""
ABDA Phase 1 — Integration service for the live-demo backend.

Assembles the per-session display dict from modality results.
Does NOT perform fusion.
Does NOT define fusion weights.
Does NOT predict BDI/depression.
"""

from __future__ import annotations

from typing import Any, Dict


def assemble_demo_record(
    session_id: str,
    transcript: Dict[str, Any],
    text: Dict[str, Any],
    speech: Dict[str, Any],
    facial: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Assemble all modality results into one response dict for the UI.

    All modalities are independently reported.
    No fusion is performed.
    No BDI prediction.
    """
    # Availability summary.
    availability = {
        "text": text.get("status", "unavailable"),
        "speech": speech.get("status", "unavailable"),
        "facial": facial.get("status", "not_applicable"),
    }

    alignment_ok = all(
        v in {"available", "not_applicable"}
        for v in availability.values()
    )

    return {
        "session_id": session_id,
        "source": "live_demo",
        "note": (
            "Live demo session. "
            "Transcript is live_inference, NOT CMU-MOSEI ground truth. "
            "No fusion performed. No BDI prediction."
        ),
        "transcript": transcript,
        "text": text,
        "speech": speech,
        "facial": facial,
        "integration": {
            "modality_availability": availability,
            "alignment_status": "PASS" if alignment_ok else "PARTIAL",
            "fusion": "NOT_IMPLEMENTED",
            "prediction": "NOT_IMPLEMENTED",
        },
    }
