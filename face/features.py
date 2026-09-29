"""Facial calculations retained from the original ABDA live pipeline."""

from __future__ import annotations

import csv
import math
import os
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# These settings and landmark definitions intentionally match the original
# face/main.py implementation.
EAR_THRESHOLD = 0.21
GAZE_LEFT_THRESHOLD = 0.40
GAZE_RIGHT_THRESHOLD = 0.60
SMOOTHING_WINDOW = 5
GAZE_SMOOTHING_WINDOW = 5

OPENFACE_EXE = r"C:\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
OPENFACE_OUTPUT_DIR = Path(__file__).resolve().parent / "openface_output"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]


def distance(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def calculate_ear(landmarks, eye_indices) -> float:
    """Compute EAR using the original six-point definition."""
    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in eye_indices)
    vertical_1 = distance(p2, p6)
    vertical_2 = distance(p3, p5)
    horizontal = distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def calculate_gaze(landmarks) -> tuple[str, float]:
    """Compute the original iris-to-eye-corner gaze ratio and label."""
    left_iris_x = np.mean([landmarks[i][0] for i in LEFT_IRIS])
    left_corners = [landmarks[i][0] for i in LEFT_EYE_CORNERS]
    left_width = max(left_corners) - min(left_corners)
    if left_width <= 0:
        return "UNKNOWN", np.nan
    left_ratio = (left_iris_x - min(left_corners)) / left_width

    right_iris_x = np.mean([landmarks[i][0] for i in RIGHT_IRIS])
    right_corners = [landmarks[i][0] for i in RIGHT_EYE_CORNERS]
    right_width = max(right_corners) - min(right_corners)
    if right_width <= 0:
        return "UNKNOWN", np.nan
    right_ratio = (right_iris_x - min(right_corners)) / right_width

    gaze_ratio = (left_ratio + right_ratio) / 2.0
    if gaze_ratio < GAZE_LEFT_THRESHOLD:
        return "LEFT", float(gaze_ratio)
    if gaze_ratio > GAZE_RIGHT_THRESHOLD:
        return "RIGHT", float(gaze_ratio)
    return "CENTER", float(gaze_ratio)


def calculate_head_pose(landmarks, width: int, height: int):
    """Compute pitch, yaw, and roll with the original solvePnP model."""
    model_points = np.array(
        [(0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
         (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
         (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)],
        dtype=np.float64,
    )
    image_points = np.array(
        [(landmarks[1][0], landmarks[1][1]),
         (landmarks[152][0], landmarks[152][1]),
         (landmarks[33][0], landmarks[33][1]),
         (landmarks[263][0], landmarks[263][1]),
         (landmarks[61][0], landmarks[61][1]),
         (landmarks[291][0], landmarks[291][1])],
        dtype=np.float64,
    )
    camera_matrix = np.array(
        [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    if sy >= 1e-6:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def calculate_statistics(values) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "min": float(np.min(values)), "max": float(np.max(values))}


class TemporalFeatureState:
    """Per-video version of the original module-global temporal state."""

    def __init__(self) -> None:
        self.pitch_history = deque(maxlen=SMOOTHING_WINDOW)
        self.yaw_history = deque(maxlen=SMOOTHING_WINDOW)
        self.roll_history = deque(maxlen=SMOOTHING_WINDOW)
        self.gaze_history = deque(maxlen=GAZE_SMOOTHING_WINDOW)
        self.eyes_closed = False
        self.blink_count = 0

    @staticmethod
    def smooth_value(history: deque, value: float) -> float:
        history.append(value)
        return sum(history) / len(history)

    def update(self, points, width: int, height: int) -> dict[str, Any]:
        left_ear = calculate_ear(points, LEFT_EYE)
        right_ear = calculate_ear(points, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        if ear < EAR_THRESHOLD:
            if not self.eyes_closed:
                self.eyes_closed = True
        elif self.eyes_closed:
            self.blink_count += 1
            self.eyes_closed = False

        pitch = yaw = roll = np.nan
        pose = calculate_head_pose(points, width, height)
        if pose is not None:
            raw_pitch, raw_yaw, raw_roll = pose
            pitch = self.smooth_value(self.pitch_history, raw_pitch)
            yaw = self.smooth_value(self.yaw_history, raw_yaw)
            roll = self.smooth_value(self.roll_history, raw_roll)

        _, raw_gaze_ratio = calculate_gaze(points)
        gaze_ratio = np.nan
        gaze_label = "UNKNOWN"
        if not np.isnan(raw_gaze_ratio):
            gaze_ratio = self.smooth_value(self.gaze_history, raw_gaze_ratio)
            if gaze_ratio < GAZE_LEFT_THRESHOLD:
                gaze_label = "LEFT"
            elif gaze_ratio > GAZE_RIGHT_THRESHOLD:
                gaze_label = "RIGHT"
            else:
                gaze_label = "CENTER"
        return {"ear": float(ear), "pitch": float(pitch), "yaw": float(yaw),
                "roll": float(roll), "gaze_label": gaze_label,
                "gaze_ratio": float(gaze_ratio), "blink_count": self.blink_count}


def analyze_action_units(csv_file: str | Path | None) -> dict:
    """Preserve the original AU intensity and presence aggregation."""
    if csv_file is None:
        return {}
    try:
        with open(csv_file, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            reader.fieldnames = [c.strip() if c is not None else c for c in (reader.fieldnames or [])]
            rows = [{k.strip(): v for k, v in row.items() if k is not None} for row in reader]
            columns = reader.fieldnames or []
    except (OSError, csv.Error):
        return {}
    intensity_columns = [c for c in columns if c and c.startswith("AU") and c.endswith("_r")]
    presence_columns = [c for c in columns if c and c.startswith("AU") and c.endswith("_c")]
    statistics = {}
    for column in intensity_columns:
        values = _finite_column(rows, column)
        if len(values):
            statistics[column] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                  "min": float(np.min(values)), "max": float(np.max(values)),
                                  "activity_rate": float(np.mean(values > 0) * 100)}
    for column in presence_columns:
        values = _finite_column(rows, column)
        if len(values):
            presence_rate = float(np.mean(values >= 1) * 100)
            intensity_name = column[:-2] + "_r"
            if intensity_name in statistics:
                statistics[intensity_name]["presence_rate"] = presence_rate
            else:
                statistics[column] = {"presence_rate": presence_rate}
    return statistics


def _finite_column(rows, column: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            value = float(row[column])
            if np.isfinite(value):
                values.append(value)
        except (ValueError, TypeError, KeyError):
            pass
    return np.asarray(values, dtype=float)


def run_openface(video_file: str | Path, output_dir: str | Path | None = None) -> dict:
    """Run the original OpenFace command and always return an explicit status."""
    video_path = Path(video_file).resolve()
    exe_path = Path(OPENFACE_EXE)
    if not exe_path.exists():
        return {"status": "unavailable", "csv_path": None, "error": f"OpenFace executable not found: {exe_path}"}
    if not video_path.exists():
        return {"status": "failed", "csv_path": None, "error": f"Video not found: {video_path}"}
    out_dir = Path(output_dir) if output_dir else OPENFACE_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [str(exe_path), "-f", str(video_path), "-out_dir", str(out_dir.resolve()), "-aus", "-q"]
    try:
        completed = subprocess.run(command, cwd=str(exe_path.parent), capture_output=True, text=True)
    except OSError as error:
        return {"status": "failed", "csv_path": None, "error": str(error)}
    if completed.returncode != 0:
        return {"status": "failed", "csv_path": None, "returncode": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr}
    expected = out_dir / f"{video_path.stem}.csv"
    csv_files = list(out_dir.glob("*.csv"))
    if expected.exists():
        csv_path = expected
    elif csv_files:
        csv_path = max(csv_files, key=lambda item: item.stat().st_mtime)
    else:
        return {"status": "failed", "csv_path": None, "returncode": completed.returncode,
                "error": "OpenFace completed but produced no CSV."}
    return {"status": "success", "csv_path": str(csv_path.resolve()), "returncode": completed.returncode}


def save_au_summary(au_statistics: dict, output_path: str | Path) -> None:
    """Write the original AU summary CSV layout."""
    if not au_statistics:
        return
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ActionUnit", "Mean", "Std", "Min", "Max", "ActivityRate", "PresenceRate"])
        for name in sorted(au_statistics):
            stats = au_statistics[name]
            writer.writerow([name, stats.get("mean", ""), stats.get("std", ""), stats.get("min", ""),
                             stats.get("max", ""), stats.get("activity_rate", ""), stats.get("presence_rate", "")])
