"""CLI and reusable entry point for one offline CMU-MOSEI facial video."""
from __future__ import annotations
import argparse
from pathlib import Path
from .extraction import process_video_frames
from .features import analyze_action_units, run_openface
from .validate import validate_video_result


def process_video(video_path, segment_id=None, label_idx=None, start=None, end=None) -> dict:
    """Process an existing video with ABDA's unchanged Phase 1 facial metrics."""
    result = process_video_frames(video_path)
    result["alignment_metadata"] = {"segment_id": segment_id, "label_idx": label_idx, "start": start, "end": end}
    openface = run_openface(video_path)
    result["openface"] = openface
    result["au_statistics"] = analyze_action_units(openface.get("csv_path")) if openface["status"] == "success" else {}
    result["validation"] = validate_video_result(result)
    return result


def _stats(stats): return "mean={mean:.3f}, std={std:.3f}, min={min:.3f}, max={max:.3f}".format(**stats)
def print_summary(result):
    video, gaze, pose = result["video"], result["gaze_statistics"], result["head_pose_statistics"]
    print(f"segment_id: {result['alignment_metadata']['segment_id']}")
    print(f"label_idx: {result['alignment_metadata']['label_idx']}")
    print(f"video duration: {video['duration_seconds']:.3f} sec\nFPS: {video['fps']:.6g}\ntotal frames: {video['frame_count']}")
    print(f"face-detection coverage: {result['face_coverage_percent']:.2f}%\nquality indicator: {result['quality_indicator']['metric']}={result['quality_indicator']['value']:.2f}%")
    print(f"blink count/rate: {result['blink_count']} / {result['blink_rate_per_minute']:.3f} per min")
    print(f"EAR statistics: {_stats(result['ear_statistics'])}")
    print(f"gaze statistics: {_stats(gaze['ratio'])}; LEFT={gaze['percentages']['LEFT']:.2f}%, CENTER={gaze['percentages']['CENTER']:.2f}%, RIGHT={gaze['percentages']['RIGHT']:.2f}%")
    print(f"head-pose pitch: {_stats(pose['pitch'])}\nhead-pose yaw: {_stats(pose['yaw'])}\nhead-pose roll: {_stats(pose['roll'])}")
    print(f"OpenFace status: {result['openface']['status']}\nAU statistics: {len(result['au_statistics'])}")
    print("VALIDATION: " + ("PASS" if result['validation']['passed'] else "FAIL"))
    for error in result["validation"]["errors"]: print(f"  - {error}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process one ABDA facial video.")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--segment-id", default=None)
    parser.add_argument("--label-idx", default=None, type=int)
    parser.add_argument("--start", default=None, type=float)
    parser.add_argument("--end", default=None, type=float)
    return parser.parse_args()


def main():
    args = parse_args(); result = process_video(args.video, args.segment_id, args.label_idx, args.start, args.end); print_summary(result)
    if not result["validation"]["passed"]: raise SystemExit(1)
if __name__ == "__main__": main()
