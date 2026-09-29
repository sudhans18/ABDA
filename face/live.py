"""Optional webcam wrapper retaining the original live-session workflow."""
from __future__ import annotations
import csv
import cv2
from .extraction import process_video_frames
from .features import analyze_action_units, run_openface, save_au_summary


def run_live_session(recorded_video="abda_session.mp4") -> dict:
    """Record webcam 0, then run the reusable extractor on that recording."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError("Could not open webcam.")
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    writer = cv2.VideoWriter(recorded_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened(): cap.release(); raise RuntimeError("Could not create video file.")
    print(f"Recording video to: {recorded_video}"); print("Press Q to stop the session.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            writer.write(frame); cv2.imshow("ABDA - Facial Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        cap.release(); writer.release(); cv2.destroyAllWindows()
    result = process_video_frames(recorded_video)
    _save_live_session_csv(result, "facial_session.csv")
    openface = run_openface(recorded_video)
    au_statistics = analyze_action_units(openface.get("csv_path")) if openface["status"] == "success" else {}
    save_au_summary(au_statistics, "openface_au_summary.csv")
    result.update({"openface": openface, "au_statistics": au_statistics})
    _save_live_summary(result, "facial_summary.txt")
    return result


def _save_live_session_csv(result, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "FaceDetected", "EAR", "Pitch", "Yaw", "Roll", "Gaze", "GazeRatio", "CumulativeBlinks"])
        for f in result["frames"]:
            writer.writerow([f["frame_index"], f["face_detected"], f["ear"], f["pitch"], f["yaw"], f["roll"], f["gaze_label"], f["gaze_ratio"], f["cumulative_blinks"]])


def _save_live_summary(result, output_path):
    """Keep the original live workflow's textual summary artifact."""
    video, gaze, pose = result["video"], result["gaze_statistics"], result["head_pose_statistics"]
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("========================================\n       ABDA FACIAL SESSION SUMMARY\n========================================\n\n")
        file.write(f"Duration          : {video['duration_seconds']:.2f} sec\nTotal Frames      : {video['frame_count']}\nFace Detected     : {result['face_detected_frames']}\nFace Quality      : {result['face_coverage_percent']:.2f} %\n\n")
        for name, stats in (("PITCH", pose["pitch"]), ("YAW", pose["yaw"]), ("ROLL", pose["roll"]), ("EAR", result["ear_statistics"]), ("GAZE RATIO", gaze["ratio"])):
            file.write(f"{name}\n----------------------------------------\n")
            file.write(f"Mean              : {stats['mean']:.3f}\nStd               : {stats['std']:.3f}\nMin               : {stats['min']:.3f}\nMax               : {stats['max']:.3f}\n\n")
        file.write(f"BLINK\n----------------------------------------\nTotal Blinks      : {result['blink_count']}\nBlink Rate        : {result['blink_rate_per_minute']:.2f} / min\n\n")
        file.write(f"GAZE\n----------------------------------------\nLeft              : {gaze['percentages']['LEFT']:.2f} %\nCenter            : {gaze['percentages']['CENTER']:.2f} %\nRight             : {gaze['percentages']['RIGHT']:.2f} %\n\n")
        file.write(f"OpenFace Status   : {result['openface']['status']}\nAU Statistics     : {len(result['au_statistics'])}\n")
