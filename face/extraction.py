"""Reusable MediaPipe video extraction for ABDA facial Phase 1."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np

from .features import TemporalFeatureState, calculate_statistics

MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"


def create_face_landmarker():
    """Create the original VIDEO-mode Face Landmarker configuration."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"face_landmarker.task not found: {MODEL_PATH}")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def process_video_frames(video_path: str | Path, frame_callback: Callable | None = None) -> dict:
    """Extract original facial measurements from every frame of a video file."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise ValueError(f"Video FPS must be positive: {fps}")

    state = TemporalFeatureState()
    frames = []
    face_detected_frames = 0
    gaze_counts = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}
    frame_index = 0
    try:
        with create_face_landmarker() as landmarker:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame_index += 1  # Original pipeline uses one-based frame numbers.
                frame_height, frame_width = frame.shape[:2]
                timestamp_ms = int((frame_index / fps) * 1000)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                record = {"frame_index": frame_index, "timestamp_ms": timestamp_ms,
                          "face_detected": 0, "ear": None, "pitch": None, "yaw": None,
                          "roll": None, "gaze_label": "UNKNOWN", "gaze_ratio": None,
                          "cumulative_blinks": state.blink_count}
                points = None
                if result.face_landmarks:
                    face_detected_frames += 1
                    record["face_detected"] = 1
                    points = [(landmark.x * frame_width, landmark.y * frame_height)
                              for landmark in result.face_landmarks[0]]
                    measurement = state.update(points, frame_width, frame_height)
                    for key in ("ear", "pitch", "yaw", "roll", "gaze_ratio"):
                        value = measurement[key]
                        record[key] = value if np.isfinite(value) else None
                    record["gaze_label"] = measurement["gaze_label"]
                    record["cumulative_blinks"] = measurement["blink_count"]
                    if record["gaze_label"] in gaze_counts:
                        gaze_counts[record["gaze_label"]] += 1
                frames.append(record)
                if frame_callback is not None:
                    frame_callback(frame, record, points)
    finally:
        cap.release()

    total_frames = len(frames)
    duration_seconds = total_frames / fps
    face_coverage = face_detected_frames / total_frames * 100 if total_frames else 0.0
    total_gaze = sum(gaze_counts.values())
    gaze_percentages = {label: (count / total_gaze * 100 if total_gaze else 0.0)
                        for label, count in gaze_counts.items()}
    def values(field):
        return [record[field] for record in frames if record[field] is not None]
    return {
        "video": {"video_path": str(path.resolve()), "fps": fps, "width": width, "height": height,
                  "reported_frame_count": reported_frame_count, "frame_count": total_frames,
                  "duration_seconds": duration_seconds},
        "frames": frames,
        "face_detected_frames": face_detected_frames,
        "face_coverage_percent": float(face_coverage),
        # The original pipeline's only face-quality measure is coverage. Do
        # not introduce a new threshold or quality model around it.
        "quality_indicator": {"metric": "face_detection_coverage_percent", "value": float(face_coverage)},
        "blink_count": state.blink_count,
        "blink_rate_per_minute": float(state.blink_count / (duration_seconds / 60)) if duration_seconds else 0.0,
        "ear_statistics": calculate_statistics(values("ear")),
        "head_pose_statistics": {"pitch": calculate_statistics(values("pitch")),
                                 "yaw": calculate_statistics(values("yaw")),
                                 "roll": calculate_statistics(values("roll"))},
        "gaze_statistics": {"ratio": calculate_statistics(values("gaze_ratio")),
                            "counts": gaze_counts, "percentages": gaze_percentages},
    }
