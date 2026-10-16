from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import sapien
import torch
import yaml

from envs.maniskill import get_gt_state, make_peg_insertion_env
from envs.maniskill.wrappers import peg_task

CONTROL_MODE = "pd_ee_delta_pose"
HORIZON = 250
ACTION_DIM = 7
CONFIG_DIR = Path(__file__).resolve().parent / "configs"
OUTPUT_ROOT = ROOT / "outputs" / "recurrent"
DATA_DIR = OUTPUT_ROOT / "data"

RAW_KEYS = {
    "tcp_pose": 7,
    "left_tip": 3,
    "right_tip": 3,
    "gripper": 2,
    "peg_pose": 7,
    "hole_pose": 7,
    "peg_half": 3,
    "hole_radius": 1,
}


def load_config(name) -> dict:
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / name
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def make_env(horizon: int = HORIZON):
    from mani_skill.utils.registration import TimeLimitWrapper

    env = make_peg_insertion_env(control_mode=CONTROL_MODE, obs_mode="state")
    return TimeLimitWrapper(env.unwrapped, horizon)


def to_sapien(pose) -> sapien.Pose:
    return sapien.Pose(pose.p[0].cpu().numpy(), pose.q[0].cpu().numpy())


def raw_gt(env) -> dict:
    s = get_gt_state(env)
    return {
        "tcp_pose": s.ee_pose[0].cpu().numpy(),
        "left_tip": s.left_fingertip_pos[0].cpu().numpy(),
        "right_tip": s.right_fingertip_pos[0].cpu().numpy(),
        "gripper": s.gripper_qpos[0].cpu().numpy(),
        "peg_pose": s.peg_pose[0].cpu().numpy(),
        "hole_pose": s.hole_pose[0].cpu().numpy(),
        "peg_half": s.peg_half_sizes[0].cpu().numpy(),
        "hole_radius": s.hole_radius.reshape(-1)[:1].cpu().numpy(),
    }


def shift_hole(env, dx: float, dy: float = 0.0) -> None:
    task = peg_task(env)
    box = to_sapien(task.box.pose)
    box.set_p(box.p + np.array([dx, dy, 0.0]))
    task.box.set_pose(box)


def is_success(info) -> bool:
    return bool(info["success"].reshape(-1)[0].item())


def to_action_tensor(action: np.ndarray, env) -> torch.Tensor:
    return torch.as_tensor(action, dtype=torch.float32, device=env.unwrapped.device).reshape(1, -1)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def versions() -> dict:
    import gymnasium
    import mani_skill

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mani_skill": mani_skill.__version__,
        "sapien": sapien.__version__,
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def environment_record(horizon: int = HORIZON) -> dict:
    return {
        "env_id": "PegInsertionSide-v1",
        "control_mode": CONTROL_MODE,
        "obs_mode": "state",
        "sim_backend": "physx_cpu",
        "horizon": horizon,
    }


def experiment_logger(**kwargs):
    try:
        from utils.experiment_logger import ExperimentLogger
    except ImportError:
        return NullLogger()
    return ExperimentLogger(team="recurrent", **kwargs)


class NullLogger:
    run_id = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def log(self, *args, **kwargs):
        pass

    def record(self, **kwargs):
        pass

    def link_output(self, *args, **kwargs):
        pass
