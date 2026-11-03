"""Task-local protocol, durable files and resolution-aware strict boxes."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

REPO = Path(os.environ.get('TASK10I_REPO', Path(__file__).resolve().parents[3]))
SOURCE = Path(__file__).resolve().parent
NAME = 'resolution_perception_graph_20261103_v1'
ROOT = Path(os.environ.get('TASK10I_ROOT', REPO / 'outputs/task10i/stage4a2' / NAME))
REMOTE = os.environ.get('TASK10I_REMOTE_ROOT', '/workspace/task10i_' + NAME)
REMOTE_REPO = '/workspace/task10i_stage4a2_newdev_WB_HANDOFF_20261101_v1/source'
REMOTE_PY = '/workspace/task10i_stage4a2_qwen35_verifier_20261102_v1/env/bin/python'
CHECKPOINT = '/workspace/graph-recurrent-VLA/visual/checkpoints/sam2.1_hiera_small.pt'
CHECKPOINT_SHA = '6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38'
CONFIG = 'configs/sam2.1/sam2.1_hiera_s.yaml'
SIZES = (128, 512, 768, 1024)
CAMERAS = ('base_camera', 'hand_camera')
FIELDS = ('node_input', 'node_type', 'edge_index', 'edge_attr', 'edge_type')
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
       '-p', '26452', 'root@151.237.25.16']
MODEL = 'gpt-6.1-terra'
BUDGET = 20.0
MAX_OUTPUT = 5120
SEED = 42
os.environ.update(WANDB_MODE='disabled', HF_HUB_OFFLINE='1',
                  TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0, str(REPO))


def now():
    return datetime.now(timezone.utc).isoformat()


def serial(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False, default=serial)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=serial,
                                     allow_nan=False).encode()).hexdigest()


def array_sha(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def csvwrite(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, default=serial) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()})


def validate_box(value, width, height):
    if value is None:
        return dict(status='NULL', normalized=None, pixels=None)
    valid = (isinstance(value, list) and len(value) == 4
             and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1000
                     for v in value))
    if not valid or not (value[0] < value[2] and value[1] < value[3]):
        return dict(status='INVALID_BOX', normalized=None, pixels=None)
    return dict(status='VALID', normalized=value,
                pixels=[value[0]*width/1000, value[1]*height/1000,
                        value[2]*width/1000, value[3]*height/1000])


def strict_parse(text, width, height):
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError('positive image dimensions required')
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError('duplicate key')
            d[k] = v
        return d
    try:
        value = json.loads(text, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
        if not isinstance(value, dict) or set(value) != {'peg', 'box'}:
            raise ValueError('exact keys required')
        return dict(json_valid=True, **{k: validate_box(value[k], width, height)
                                       for k in ('peg', 'box')})
    except (ValueError, TypeError):
        return dict(json_valid=False, **{k: dict(status='JSON_INVALID', normalized=None, pixels=None)
                                        for k in ('peg', 'box')})


def accounted(state):
    return sum(j['accounted_usd'] for j in state['jobs'].values())


def reserve_job(state, job, bound, limit=BUDGET):
    """Caller persists this intent before HTTP; any existing intent is never resent."""
    if job['id'] in state['jobs']:
        return False
    if accounted(state) + bound > limit:
        state['stop_reason'] = 'BUDGET_LIMIT'
        return False
    state['jobs'][job['id']] = dict(job, status='RESERVED', reserved_usd=bound,
                                   accounted_usd=bound, started_utc=now())
    return True


def logger(stage):
    from utils.experiment_logger import ExperimentLogger
    return ExperimentLogger(team='integration', model=NAME, seed=SEED, job_type='evaluate',
                            root=ROOT / 'experiments',
                            config=dict(stage=stage, protocol_sha256=sha(ROOT/'protocol.json')))


def assert_source():
    p = read(ROOT/'protocol.json')
    for name, expected in p['source_sha256'].items():
        assert sha(SOURCE/name) == expected, ('experiment source changed', name)
    return p


def image_id(scene, size, camera):
    return f'{scene}__{size}__{camera}'


def job_id(image, repeat):
    return f'r{repeat}__{image}'


def stats(values):
    import numpy as np
    a = np.asarray([v for v in values if v is not None and math.isfinite(v)], float)
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p95=float(np.quantile(a, .95)) if len(a) else None)


def balanced_order(index):
    # Williams square: balanced positions and first-order carryover in four rounds.
    return ((128,512,1024,768), (512,768,128,1024),
            (768,1024,512,128), (1024,128,768,512))[index % 4]
