"""
Minimal CMU-MOSEI CSD reader.

CMU-MOSEI computational sequence files are HDF5 files
with a .csd extension.

We only implement the read functionality required by
the ABDA text pipeline. This intentionally avoids a
dependency on the legacy CMU Multimodal SDK.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _decode(value: Any) -> Any:
    """Decode bytes / numpy byte strings recursively."""

    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")

    return value


def _decode_array(array: np.ndarray) -> np.ndarray:
    """Decode a numpy array containing byte strings."""

    if array.dtype.kind == "S":
        return array.astype(str)

    if array.dtype.kind == "O":
        return np.array(
            [
                _decode(item)
                for item in array
            ],
            dtype=object,
        )

    return array


def _find_group(
    h5file: h5py.File,
    group_name: str,
) -> h5py.Group:
    """
    Find a group anywhere below the HDF5 root.
    """

    if group_name in h5file:
        obj = h5file[group_name]

        if isinstance(obj, h5py.Group):
            return obj

    found = None

    def visitor(name, obj):
        nonlocal found

        if found is not None:
            return

        if (
            isinstance(obj, h5py.Group)
            and name.split("/")[-1] == group_name
        ):
            found = obj

    h5file.visititems(visitor)

    if found is None:
        raise KeyError(
            f"Could not find HDF5 group '{group_name}'."
        )

    return found


def _read_sequence_group(
    group: h5py.Group,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Read one computational-sequence group.

    Expected structure for each source:

        source_id/
            intervals
            features

    Returns:

        {
            source_id: {
                "intervals": ndarray,
                "features": ndarray
            }
        }
    """

    result = {}

    for source_id in group.keys():

        source = group[source_id]

        if not isinstance(source, h5py.Group):
            continue

        if "intervals" not in source:
            continue

        if "features" not in source:
            continue

        intervals = np.asarray(
            source["intervals"]
        )

        features = np.asarray(
            source["features"]
        )

        result[_decode(source_id)] = {
            "intervals": intervals,
            "features": _decode_array(features),
        }

    return result


def read_csd(
    path: str | Path,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Read a CMU-MOSEI .csd computational sequence.

    Parameters
    ----------
    path:
        Path to a .csd file.

    Returns
    -------
    dict
        Mapping from source/video IDs to intervals/features.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"CSD file not found:\n{path.resolve()}"
        )

    if not path.is_file():
        raise ValueError(
            f"CSD path is not a file:\n{path.resolve()}"
        )

    with h5py.File(
        path,
        "r",
    ) as h5file:

        data_group = _find_group(
            h5file,
            "data",
        )

        result = _read_sequence_group(
            data_group
        )

    if not result:
        raise ValueError(
            f"No computational-sequence data found "
            f"in:\n{path}"
        )

    return result


def inspect_csd(
    path: str | Path,
    max_sources: int = 5,
) -> None:
    """Print a compact summary of a CSD file."""

    data = read_csd(path)

    print()
    print("=" * 70)
    print("CSD INSPECTION")
    print("=" * 70)

    print(f"File: {path}")
    print(f"Sources: {len(data):,}")

    for i, (
        source_id,
        source_data,
    ) in enumerate(data.items()):

        if i >= max_sources:
            break

        intervals = source_data[
            "intervals"
        ]

        features = source_data[
            "features"
        ]

        print()
        print(f"Source: {source_id}")
        print(
            f"  intervals shape: "
            f"{intervals.shape}"
        )
        print(
            f"  features shape : "
            f"{features.shape}"
        )