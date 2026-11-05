"""Portable raw NPZ frames and atomic episode receipts; no pickle payloads."""

import json
import os
from pathlib import Path

import numpy as np
import torch


def to_numpy(value):
    """Materialize raw arrays only at the explicit disk-output boundary."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def flatten_arrays(tree, prefix=""):
    if isinstance(tree, dict):
        result = {}
        for key, value in tree.items():
            result.update(flatten_arrays(value, f"{prefix}/{key}".strip("/")))
        return result
    array = to_numpy(tree)
    if array.dtype.hasobject:
        raise TypeError(f"Cannot save object payload at {prefix}")
    return {prefix: array}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(json_value(value), stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_frame(path, observation, **extra):
    path = Path(path)
    arrays = flatten_arrays({"observation": observation, **extra})
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    # Verify dtype, values and key identity before exposing the file.
    with np.load(temporary, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError("NPZ key readback mismatch")
        for key, array in arrays.items():
            if saved[key].dtype != array.dtype:
                raise ValueError(f"NPZ dtype readback mismatch: {key}")
            np.testing.assert_array_equal(saved[key], array)
    temporary.replace(path)


def append_attempt(root, record):
    with (Path(root) / "attempts.jsonl").open("a") as stream:
        stream.write(json.dumps(json_value(record), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def json_value(value):
    """Out-of-view projections have NaN coordinates; encode them as JSON null."""
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
