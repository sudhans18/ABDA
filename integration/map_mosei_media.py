"""Deterministic CMU-MOSEI media mapper.

Equal-count sources use the established stable positional mapping.  Unequal
sources are mapped only by the explicit exception CSV; all other manifest rows
are retained as unavailable and all unused raw files are diagnosed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FAILURE_COLUMNS = ["segment_id", "reason", "manifest_count", "raw_count",
                   "label_idx", "local_video_id", "duration_difference_sec"]


def numeric_id(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Raw MP4 filename is not numeric: {value}") from exc


def diagnostic(segment_id, reason, manifest_count, raw_count, label_idx=np.nan,
               local_video_id=np.nan, duration_difference_sec=np.nan):
    return dict(zip(FAILURE_COLUMNS, [segment_id, reason, manifest_count, raw_count,
                                      label_idx, local_video_id, duration_difference_sec]))


def unavailable(m: pd.Series) -> dict:
    start, end = float(m.start), float(m.end)
    return {"segment_id": str(m.segment_id), "label_idx": int(m.label_idx),
            "start": start, "end": end, "chronological_rank": np.nan,
            "local_video_id": np.nan, "video_path": np.nan,
            "media_duration_sec": np.nan, "manifest_duration_sec": end - start,
            "duration_difference_sec": np.nan, "has_video": False, "has_audio": False,
            "media_available": False, "mapping_status": "MEDIA_UNAVAILABLE",
            "duration_validation": "NOT_AVAILABLE"}


def assigned(m: pd.Series, r: pd.Series, rank: int, warning: float) -> dict:
    start, end = float(m.start), float(m.end)
    manifest_duration, media_duration = end - start, float(r.duration_sec)
    difference = abs(media_duration - manifest_duration)
    return {"segment_id": str(m.segment_id), "label_idx": int(m.label_idx),
            "start": start, "end": end, "chronological_rank": rank,
            "local_video_id": r.local_video_id, "video_path": r.video_path,
            "media_duration_sec": media_duration, "manifest_duration_sec": manifest_duration,
            "duration_difference_sec": difference, "has_video": bool(r.has_video),
            "has_audio": bool(r.has_audio), "media_available": True,
            "mapping_status": "MAPPED",
            "duration_validation": "CLOSE" if difference <= warning else "WARNING"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/metadata/CMU_MOSEI_multimodal_manifest.csv")
    parser.add_argument("--raw-index", default="data/metadata/CMU_MOSEI_raw_media.csv")
    parser.add_argument("--output", default="data/metadata/CMU_MOSEI_media_manifest.csv")
    parser.add_argument("--exceptions", default="data/metadata/CMU_MOSEI_media_mapping_exceptions.csv")
    parser.add_argument("--duration-warning", type=float, default=1.0)
    args = parser.parse_args()

    manifest, raw, exceptions = (pd.read_csv(path).copy() for path in
                                 (args.manifest, args.raw_index, args.exceptions))
    required = ((manifest, {"segment_id", "label_idx", "start", "end"}, "Manifest"),
                (raw, {"segment_id", "local_video_id", "video_path", "duration_sec", "has_video", "has_audio"}, "Raw index"),
                (exceptions, {"segment_id", "label_idx", "local_video_id", "reason"}, "Exception table"))
    for frame, columns, name in required:
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    for frame in (manifest, raw, exceptions):
        frame["segment_id"] = frame["segment_id"].astype(str)
    manifest["label_idx"] = manifest["label_idx"].astype(int)
    raw["local_video_id"] = raw["local_video_id"].astype(str)
    raw["numeric_video_id"] = raw["local_video_id"].map(numeric_id)
    exceptions["label_idx"] = exceptions["label_idx"].astype(int)
    exceptions["local_video_id"] = exceptions["local_video_id"].astype(str)
    for frame, keys, name in ((manifest, ["segment_id", "label_idx"], "manifest"),
                              (raw, ["segment_id", "numeric_video_id"], "raw media"),
                              (exceptions, ["segment_id", "label_idx"], "exception manifest"),
                              (exceptions, ["segment_id", "local_video_id"], "exception raw media")):
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate {name} keys: {keys}")
    manifest_keys = set(zip(manifest.segment_id, manifest.label_idx))
    raw_keys = set(zip(raw.segment_id, raw.local_video_id))
    for e in exceptions.itertuples(index=False):
        if (e.segment_id, e.label_idx) not in manifest_keys or (e.segment_id, e.local_video_id) not in raw_keys:
            raise ValueError(f"Exception does not name an existing manifest row and raw file: {e}")

    rows, failures = [], []
    for segment_id in sorted(set(manifest.segment_id) | set(raw.segment_id)):
        mg = manifest[manifest.segment_id == segment_id].copy()
        rg = raw[raw.segment_id == segment_id].copy()
        mc, rc = len(mg), len(rg)
        if not mc:
            failures.extend(diagnostic(segment_id, "RAW_ONLY_SOURCE", 0, rc, local_video_id=r.local_video_id)
                            for r in rg.itertuples(index=False))
            continue
        mo = mg.sort_values(["start", "end", "label_idx"], kind="mergesort").reset_index(drop=True)
        ro = rg.sort_values("numeric_video_id", kind="mergesort").reset_index(drop=True)
        source_exceptions = exceptions[exceptions.segment_id == segment_id]
        if mc == rc:
            if len(source_exceptions):
                raise ValueError(f"Equal-count source unexpectedly has exceptions: {segment_id}")
            matches = [(mo.iloc[i], ro.iloc[i], i) for i in range(mc)]
        elif len(source_exceptions):
            matches = []
            for e in source_exceptions.itertuples(index=False):
                m, r = mg[mg.label_idx == e.label_idx].iloc[0], rg[rg.local_video_id == e.local_video_id].iloc[0]
                rank = int(mo.index[mo.label_idx == e.label_idx][0])
                matches.append((m, r, rank))
        else:
            matches = []
        assigned_labels, assigned_raw_ids = set(), set()
        for m, r, rank in matches:
            row = assigned(m, r, rank, args.duration_warning)
            rows.append(row); assigned_labels.add(int(m.label_idx)); assigned_raw_ids.add(str(r.local_video_id))
            if row["duration_validation"] == "WARNING":
                failures.append(diagnostic(segment_id, "DURATION_WARNING", mc, rc, row["label_idx"], row["local_video_id"], row["duration_difference_sec"]))
        unavailable_reason = "MISSING_RAW_SOURCE" if rc == 0 else "MEDIA_UNAVAILABLE"
        for m in mo.itertuples(index=False):
            if m.label_idx not in assigned_labels:
                rows.append(unavailable(m))
                failures.append(diagnostic(segment_id, unavailable_reason, mc, rc, label_idx=m.label_idx))
        for r in ro.itertuples(index=False):
            if str(r.local_video_id) not in assigned_raw_ids:
                failures.append(diagnostic(segment_id, "EXTRA_RAW_MEDIA", mc, rc, local_video_id=r.local_video_id))

    output = pd.DataFrame(rows).sort_values(["segment_id", "label_idx"], kind="mergesort").reset_index(drop=True)
    failures_df = pd.DataFrame(failures, columns=FAILURE_COLUMNS)
    mapped = output[output.mapping_status == "MAPPED"]
    # Canonical identity and mapping invariants.
    assert len(output) == len(manifest)
    assert not output.duplicated(["segment_id", "label_idx"]).any()
    assert len(output[["segment_id", "label_idx"]].drop_duplicates()) == len(manifest)
    assert mapped.media_available.all()
    assert not output.loc[~output.media_available, "mapping_status"].eq("MAPPED").any()
    assert mapped.video_path.map(lambda path: Path(path).is_file()).all()
    assert mapped.duration_validation.isin(["CLOSE", "WARNING"]).all()
    assert (mapped.loc[mapped.duration_validation == "CLOSE", "duration_difference_sec"] <= args.duration_warning).all()
    assert (mapped.loc[mapped.duration_validation == "WARNING", "duration_difference_sec"] > args.duration_warning).all()

    output_path = Path(args.output); output_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path = output_path.parent / "CMU_MOSEI_media_mapping_failures.csv"
    output.to_csv(output_path, index=False); failures_df.to_csv(failure_path, index=False)
    print("CMU-MOSEI MEDIA MAPPING — POSITIONAL WITH EXCEPTIONS")
    print(f"Manifest rows          : {len(output):,}")
    print(f"Media available        : {int(output.media_available.sum()):,}")
    print(f"Media unavailable      : {int((~output.media_available).sum()):,}")
    print(f"Mapping failures       : {len(failures_df):,}")
    print(f"Extra raw media        : {int(failures_df.reason.isin(['EXTRA_RAW_MEDIA', 'RAW_ONLY_SOURCE']).sum()):,}")
    print(f"Unique sources w/media : {mapped.segment_id.nunique():,}")
    print("Duration validation counts:"); print(output.duration_validation.value_counts().to_string())
    print(f"Output         : {output_path}"); print(f"Failure report : {failure_path}")


if __name__ == "__main__":
    main()
