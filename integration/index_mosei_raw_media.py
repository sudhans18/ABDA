"""
Index raw CMU-MOSEI MP4 files.

The raw dataset is expected to have the structure:

RAW_ROOT/
    <segment_id>/
        <local_video_id>.mp4
        <local_video_id>.mp4
        ...

The folder name is treated as the CMU-MOSEI source/video ID.
The MP4 filename is preserved but is NOT assumed to equal label_idx.

Output:
    data/metadata/CMU_MOSEI_raw_media.csv
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

def run_ffprobe(video_path: Path) -> dict:
    """Return basic media metadata using ffprobe."""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )

    return json.loads(result.stdout)


def parse_media_info(video_path: Path) -> dict:
    info = run_ffprobe(video_path)

    streams = info.get("streams", [])
    format_info = info.get("format", {})

    video_streams = [
        s for s in streams
        if s.get("codec_type") == "video"
    ]

    audio_streams = [
        s for s in streams
        if s.get("codec_type") == "audio"
    ]

    video = video_streams[0] if video_streams else {}
    audio = audio_streams[0] if audio_streams else {}

    duration = format_info.get("duration")

    if duration is None:
        duration = video.get("duration")

    fps = None

    if video.get("r_frame_rate"):
        try:
            num, den = video["r_frame_rate"].split("/")
            if float(den) != 0:
                fps = float(num) / float(den)
        except Exception:
            fps = None

    return {
        "duration_sec": float(duration) if duration is not None else None,
        "has_video": bool(video_streams),
        "has_audio": bool(audio_streams),
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": fps,
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "sample_rate": audio.get("sample_rate"),
        "channels": audio.get("channels"),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-root",
        required=True,
        help="Root directory containing CMU-MOSEI raw folders.",
    )

    parser.add_argument(
        "--output",
        default="data/metadata/CMU_MOSEI_raw_media.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    output_path = Path(args.output)

    if not raw_root.exists():
        raise FileNotFoundError(
            f"Raw dataset directory does not exist: {raw_root}"
        )

    mp4_files = sorted(raw_root.rglob("*.mp4"))

    print("=" * 70)
    print("CMU-MOSEI RAW MEDIA INDEX")
    print("=" * 70)
    print(f"Raw root : {raw_root}")
    print(f"MP4 files: {len(mp4_files)}")
    print()

    rows = []

    for video_path in tqdm(mp4_files, desc="Indexing MP4 files"):
        relative = video_path.relative_to(raw_root)

        # First directory component is the CMU-MOSEI source/video ID.
        if len(relative.parts) < 2:
            print(
                f"[WARNING] MP4 is not inside a source folder: "
                f"{video_path}"
            )
            continue

        segment_id = relative.parts[0]
        local_video_id = video_path.stem

        try:
            media = parse_media_info(video_path)
            status = "OK"
        except Exception as exc:
            media = {
                "duration_sec": None,
                "has_video": False,
                "has_audio": False,
                "width": None,
                "height": None,
                "fps": None,
                "video_codec": None,
                "audio_codec": None,
                "sample_rate": None,
                "channels": None,
            }
            status = f"ERROR: {exc}"

        rows.append(
            {
                "segment_id": segment_id,
                "local_video_id": local_video_id,
                "video_path": str(video_path),
                **media,
                "index_status": status,
            }
        )

    df = pd.DataFrame(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print()
    print("=" * 70)
    print("INDEX COMPLETE")
    print("=" * 70)
    print(f"Rows          : {len(df):,}")
    print(f"Unique sources: {df['segment_id'].nunique():,}")

    if len(df):
        print(f"With audio    : {df['has_audio'].sum():,}")
        print(f"With video    : {df['has_video'].sum():,}")

    errors = df[
        df["index_status"].astype(str).str.startswith("ERROR")
    ]

    print(f"Errors        : {len(errors):,}")
    print(f"Output        : {output_path}")

    if len(errors):
        print()
        print("ERROR FILES:")
        print(errors[["segment_id", "video_path", "index_status"]])


if __name__ == "__main__":
    main()