from __future__ import annotations

import numpy as np

FEATURE_LAYOUT = [
    ("tcp_pos", 3),
    ("tcp_quat", 4),
    ("left_tip_rel_tcp", 3),
    ("right_tip_rel_tcp", 3),
    ("gripper_qpos", 2),
    ("peg_pos_rel_tcp", 3),
    ("peg_quat", 4),
    ("peg_head_rel_hole", 3),
    ("hole_pos_rel_tcp", 3),
    ("hole_quat", 4),
    ("peg_half_sizes", 3),
    ("hole_radius", 1),
    ("obs_fresh", 1),
    ("obs_age", 1),
]
OBS_DIM = sum(n for _, n in FEATURE_LAYOUT)
FRESH_IDX = OBS_DIM - 2
AGE_IDX = OBS_DIM - 1
AGE_SCALE = 20.0


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


def rotvec_to_quat(rv: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rv, axis=-1, keepdims=True)
    axis = rv / np.maximum(angle, 1e-9)
    return np.concatenate([np.cos(angle / 2), axis * np.sin(angle / 2)], -1)


def canon_quat(q: np.ndarray) -> np.ndarray:
    return np.where(q[..., :1] < 0, -q, q)


def build_features(raw: dict, fresh: np.ndarray, age: np.ndarray) -> np.ndarray:
    tcp_p = raw["tcp_pose"][..., :3]
    peg = raw["peg_pose"]
    hole = raw["hole_pose"]
    half_len = raw["peg_half"][..., :1]
    head_local = np.concatenate([half_len, np.zeros_like(half_len), np.zeros_like(half_len)], -1)
    peg_head = peg[..., :3] + quat_rotate(peg[..., 3:], head_local)
    parts = [
        tcp_p,
        canon_quat(raw["tcp_pose"][..., 3:]),
        raw["left_tip"] - tcp_p,
        raw["right_tip"] - tcp_p,
        raw["gripper"],
        peg[..., :3] - tcp_p,
        canon_quat(peg[..., 3:]),
        peg_head - hole[..., :3],
        hole[..., :3] - tcp_p,
        canon_quat(hole[..., 3:]),
        raw["peg_half"],
        raw["hole_radius"],
        fresh[..., None],
        np.minimum(age, AGE_SCALE)[..., None] / AGE_SCALE,
    ]
    return np.concatenate(parts, -1).astype(np.float32)
