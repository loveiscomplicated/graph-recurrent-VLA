import importlib.metadata
import json
from pathlib import Path
import random
import tomllib

import numpy as np
import torch


DEFAULT_CONFIG = Path(__file__).with_name("config.toml")


def config(path=DEFAULT_CONFIG):
    with open(path, "rb") as f:
        return tomllib.load(f)


def seed_all(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def versions():
    import sys
    return {"python": sys.version, **{p: importlib.metadata.version(p) for p in
             ("torch", "mani_skill", "sapien", "gymnasium", "h5py", "matplotlib")},
            "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available()}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
