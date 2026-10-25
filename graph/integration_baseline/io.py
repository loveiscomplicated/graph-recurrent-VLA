"""Local artifact I/O; the existing lossless Vision serializer is reused."""
import hashlib
import json
from pathlib import Path

import numpy as np

from visual.runtime.test.run_vision_feature_edge import read_packet, save_packet
from visual.runtime.test.run_variable_rate_shape_first import packet_digest


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(serial(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    path.write_text(text)
    if json.loads(path.read_text()) != json.loads(text):
        raise AssertionError('JSON readback')


def save_npz(path, **arrays):
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as restored:
        if set(restored.files) != set(arrays):
            raise AssertionError('NPZ keys')
        for key, original in arrays.items():
            original = np.asarray(original)
            actual = restored[key]
            if (actual.shape != original.shape or actual.dtype != original.dtype
                    or actual.tobytes() != original.tobytes()):
                raise AssertionError('NPZ byte readback: ' + key)
