"""Validation for structured ABDA facial video results."""

from __future__ import annotations

import math
from pathlib import Path


def validate_video_result(result: dict) -> dict:
    """Validate input, frame records, calculated values, and explicit AU status."""
    errors = []
    video = result.get("video")
    if not isinstance(video, dict):
        errors.append("Missing video metadata.")
        video = {}
    path = video.get("video_path")
    if not path or not Path(path).is_file(): errors.append("Video does not exist.")
    if not _positive(video.get("fps")): errors.append("FPS must be greater than zero.")
    if not isinstance(video.get("frame_count"), int) or video.get("frame_count", 0) <= 0: errors.append("Frame count must be greater than zero.")
    frames = result.get("frames")
    if not isinstance(frames, list) or not frames:
        errors.append("No frame processing occurred."); frames = []
    indices = [f.get("frame_index") for f in frames if isinstance(f, dict)]
    if len(indices) != len(set(indices)): errors.append("Duplicate frame indices found.")
    if len(frames) != video.get("frame_count"): errors.append("Frame record count does not match video frame count.")
    for frame in frames:
        if not isinstance(frame, dict): errors.append("Invalid frame record."); continue
        if frame.get("face_detected") not in (0, 1): errors.append("Invalid face_detected value.")
        if frame.get("face_detected") == 1:
            for key in ("ear", "pitch", "yaw", "roll", "gaze_ratio"):
                if not _finite(frame.get(key)): errors.append(f"Non-finite {key} on detected-face frame {frame.get('frame_index')}.")
    coverage = result.get("face_coverage_percent")
    if not _finite(coverage) or not 0 <= float(coverage) <= 100: errors.append("Face coverage must be a percentage in [0, 100].")
    quality = result.get("quality_indicator")
    if not isinstance(quality, dict) or quality.get("metric") != "face_detection_coverage_percent" or not _finite(quality.get("value")):
        errors.append("Invalid face-coverage quality indicator.")
    _validate_stats("EAR", result.get("ear_statistics"), errors)
    _validate_stats("gaze ratio", result.get("gaze_statistics", {}).get("ratio"), errors)
    for axis, stats in result.get("head_pose_statistics", {}).items(): _validate_stats(f"head pose {axis}", stats, errors)
    metadata = result.get("alignment_metadata")
    if not isinstance(metadata, dict): errors.append("Missing alignment metadata.")
    elif any(metadata.get(k) == "" for k in ("segment_id", "label_idx")): errors.append("Invalid supplied identity.")
    openface = result.get("openface")
    if not isinstance(openface, dict) or openface.get("status") not in {"success", "failed", "unavailable", "not_run"}: errors.append("OpenFace status is not explicit.")
    if not isinstance(result.get("au_statistics"), dict): errors.append("AU statistics must be a dictionary.")
    return {"passed": not errors, "errors": errors}


def _positive(value) -> bool: return _finite(value) and float(value) > 0
def _finite(value) -> bool:
    try: return math.isfinite(float(value))
    except (TypeError, ValueError): return False


def _validate_stats(name, stats, errors) -> None:
    if not isinstance(stats, dict): errors.append(f"Missing {name} statistics."); return
    for key, value in stats.items():
        if value is not None and not (isinstance(value, float) and math.isnan(value)) and not _finite(value): errors.append(f"Non-finite {name} {key}.")
