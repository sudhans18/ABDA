"""
ABDA Phase 1 — Facial service for the live demo.

Calls the EXISTING facial pipeline (face.run_video.process_video).
Does NOT modify landmark indices, EAR threshold, gaze thresholds,
head-pose calculation, smoothing window, or AU methodology.
OpenFace unavailability is ALWAYS explicit -- never zeroed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def run_face_on_file(
    video_path: Optional[Path],
    session_id: str,
) -> Dict[str, Any]:
    """
    Run the existing facial pipeline on the given video file.

    Returns a display-ready dict.
    If video_path is None or missing, returns status='not_applicable'.
    OpenFace status is always explicit.
    """
    if video_path is None:
        return {"status": "not_applicable"}

    video_path = Path(video_path)
    if not video_path.exists():
        return {"status": "unavailable"}

    try:
        from face.run_video import process_video
    except ImportError:
        return {"status": "error", "error": "face module not importable"}

    try:
        result = process_video(
            video_path=video_path,
            segment_id=f"demo_{session_id}",
            label_idx=0,
            start=None,
            end=None,
        )
    except Exception as exc:
        return {"status": "error", "error": str(exc)}

    # OpenFace -- always explicit.
    of_raw = result.get("openface", {})
    openface = {
        "status": of_raw.get("status", "not_run"),
        "error": of_raw.get("error"),
    }

    # AU statistics -- only if OpenFace succeeded.
    au_stats: dict = {}
    if of_raw.get("status") == "success":
        au_stats = result.get("au_statistics", {})
    # Do NOT zero-fill if unavailable.

    video = result.get("video", {})
    ear = result.get("ear_statistics", {})
    gaze = result.get("gaze_statistics", {})
    pose = result.get("head_pose_statistics", {})

    return {
        "status": "available",
        "face_coverage_percent": round(float(result.get("face_coverage_percent", 0)), 2),
        "blink_count": int(result.get("blink_count", 0)),
        "blink_rate_per_minute": round(float(result.get("blink_rate_per_minute", 0)), 3),
        "video_metadata": {
            "fps": video.get("fps"),
            "frame_count": video.get("frame_count"),
            "duration_seconds": video.get("duration_seconds"),
        },
        "ear_statistics": _clean(ear),
        "gaze_statistics": {
            "ratio": _clean(gaze.get("ratio", {})),
            "percentages": gaze.get("percentages", {}),
        },
        "head_pose_statistics": {
            "pitch": _clean(pose.get("pitch", {})),
            "yaw": _clean(pose.get("yaw", {})),
            "roll": _clean(pose.get("roll", {})),
        },
        "quality_indicator": result.get("quality_indicator", {}),
        "openface": openface,
        "au_count": len(au_stats),
    }


def _clean(stats: dict) -> dict:
    out = {}
    for k, v in stats.items():
        try:
            f = float(v)
            out[k] = None if f != f else round(f, 4)
        except (TypeError, ValueError):
            out[k] = None
    return out
