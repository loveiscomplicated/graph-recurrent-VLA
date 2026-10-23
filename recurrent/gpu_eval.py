"""Tensor evaluation primitives. NumPy is used only for legacy RNG draws."""

from __future__ import annotations

import time

import numpy as np
import torch

from envs.maniskill import get_gt_state
from recurrent.corruption import Corruptor
from recurrent.features import AGE_SCALE
from recurrent.stages import THRESHOLDS


def rotate(q, v):
    t = 2 * torch.linalg.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.linalg.cross(q[..., 1:], t, dim=-1)


def multiply(a, b):
    return torch.cat(
        (
            a[:1] * b[:1] - (a[1:] * b[1:]).sum().reshape(1),
            a[:1] * b[1:] + b[:1] * a[1:] + torch.linalg.cross(a[1:], b[1:]),
        )
    )


def raw_tensor(env):
    s = get_gt_state(env)
    return {
        "tcp_pose": s.ee_pose,
        "left_tip": s.left_fingertip_pos,
        "right_tip": s.right_fingertip_pos,
        "gripper": s.gripper_qpos,
        "peg_pose": s.peg_pose,
        "hole_pose": s.hole_pose,
        "peg_half": s.peg_half_sizes,
        "hole_radius": s.hole_radius[:, None],
    }


def features(r, fresh, age):
    tcp, peg, hole = r["tcp_pose"], r["peg_pose"], r["hole_pose"]
    canon = lambda q: torch.where(q[..., :1] < 0, -q, q)
    local = torch.zeros_like(r["peg_half"])
    local[..., 0] = r["peg_half"][..., 0]
    head = peg[..., :3] + rotate(peg[..., 3:], local)
    return torch.cat(
        (
            tcp[..., :3],
            canon(tcp[..., 3:]),
            r["left_tip"] - tcp[..., :3],
            r["right_tip"] - tcp[..., :3],
            r["gripper"],
            peg[..., :3] - tcp[..., :3],
            canon(peg[..., 3:]),
            head - hole[..., :3],
            hole[..., :3] - tcp[..., :3],
            canon(hole[..., 3:]),
            r["peg_half"],
            r["hole_radius"],
            fresh[..., None],
            age.clamp(max=AGE_SCALE)[..., None] / AGE_SCALE,
        ),
        -1,
    )


class TensorCorruptor(Corruptor):
    def _zeros(self, value):
        return torch.zeros_like(value)

    def _noisy(self, pose, scale):
        out = pose.clone()
        c = self.cfg
        if c["pos_noise"] > 0:
            out[:3] += out.new_tensor(self.rng.normal(0, c["pos_noise"] * scale, 3))
        if c["rot_noise"] > 0:
            # Draw order and float64 rotvec conversion match the reference exactly.
            from recurrent.features import rotvec_to_quat

            dq = out.new_tensor(
                rotvec_to_quat(self.rng.normal(0, c["rot_noise"] * scale, 3))
            )
            out[3:] = multiply(dq, out[3:])
        return out


def freeze_state(new, old, active, kind, root=True):
    if new is None:
        return None
    if isinstance(new, torch.Tensor):
        dim = 1 if root and kind in ("gru", "lstm") else 0
        mask = active.reshape([len(active) if i == dim else 1 for i in range(new.ndim)])
        return torch.where(mask, new, torch.zeros_like(new) if old is None else old)
    return type(new)(
        freeze_state(
            s,
            None if old is None else old[i],
            active,
            kind,
            root=root and kind == "lstm",
        )
        for i, s in enumerate(new)
    )


class TensorPolicy:
    def __init__(self, policy, device):
        self.model = policy.model.to(device)
        self.kind = policy.name
        self.stats = {
            k: torch.as_tensor(v, device=device) for k, v in policy.stats.items()
        }
        self.state = None

    @torch.inference_mode()
    def act(self, raw, fresh, age, active):
        s = self.stats
        x = (features(raw, fresh, age) - s["obs_mean"]) / s["obs_std"]
        x *= s.get("obs_keep", torch.ones_like(s["obs_mean"]))
        out, state = self.model(x[:, None], self.state)
        self.state = freeze_state(state, self.state, active, self.kind)
        out = out[:, 0]
        return torch.cat(
            (
                (out[:, :6] * s["act_std"] + s["act_mean"]).clamp(-1, 1),
                torch.where(out[:, 6:] > 0, 1.0, -1.0),
            ),
            -1,
        )


def stage_snapshots(env, raw):
    p, h, half = raw["peg_pose"], raw["hole_pose"], raw["peg_half"]
    local = torch.zeros_like(half)
    local[:, 0] = half[:, 0]
    head = p[:, :3] + rotate(p[:, 3:], local)
    inv = h[:, 3:].clone()
    inv[:, 1:] *= -1
    rel = rotate(inv, head - h[:, :3])
    axis = torch.zeros_like(half)
    axis[:, 0] = 1
    angle = (
        (rotate(p[:, 3:], axis) * rotate(h[:, 3:], axis)).sum(-1).clamp(-1, 1).acos()
    )
    transverse = rel[:, 1:].norm(dim=-1)
    radius = raw["hole_radius"][:, 0]
    depth = half[:, 0]
    data = {
        "grasped": env.unwrapped.agent.is_grasping(env.unwrapped.peg),
        "lifted": p[:, 2] - half[:, 2] > THRESHOLDS["lift_height_m"],
        "transverse": transverse,
        "progress": ((rel[:, 0] + depth) / (2 * depth)).clamp(0, 1),
        "alignment": (transverse <= THRESHOLDS["alignment_transverse_m"])
        & (angle <= THRESHOLDS["alignment_axis_rad"])
        & ((rel[:, 0] + depth).abs() <= THRESHOLDS["alignment_entry_distance_m"]),
        "insertion": (rel[:, 0] >= -depth)
        & (rel[:, 1].abs() <= radius)
        & (rel[:, 2].abs() <= radius),
    }
    # Only compact event/metric controls cross to host; geometry remains on device.
    data = {k: v.cpu().tolist() for k, v in data.items()}
    return [{k: v[i] for k, v in data.items()} for i in range(len(p))]


def run_batch(env, policy, seeds, cfg, cond, horizon, fixtures):
    from mani_skill.utils.structs.pose import Pose

    from recurrent.evaluate import LEVEL_INDEX
    from recurrent.initial_states import restore
    from recurrent.stages import StageTracker

    reset_start = time.perf_counter()
    errors = restore(env, fixtures, seeds)
    reset_s = time.perf_counter() - reset_start
    device = env.unwrapped.device
    b = len(seeds)
    policy.state = None
    corr = [
        TensorCorruptor(
            cfg, np.random.default_rng([s, LEVEL_INDEX.get(cfg["name"], 9), 7])
        )
        for s in seeds
    ]
    trackers = [StageTracker() for _ in seeds]
    for tr, s in zip(trackers, stage_snapshots(env, raw_tensor(env))):
        tr.update_snapshot(s)
    active = torch.ones(b, dtype=torch.bool, device=device)
    live = [True] * b
    shifts = [None] * b
    blackouts = [None] * b
    steps = [horizon] * b
    success = [False] * b
    deltas = [[] for _ in seeds]
    previous = None
    inference = []
    env_steps = 0
    torch.cuda.synchronize()
    start = time.perf_counter()
    for t in range(horizon):
        pose = env.unwrapped.box.pose.raw_pose.clone()
        changed = False
        for i in range(b):
            if not live[i]:
                continue
            transport = trackers[i].transport_step
            if (
                cond.get("shift_dx")
                and shifts[i] is None
                and transport is not None
                and t >= transport + cond.get("shift_delay", 10)
            ):
                pose[i, 0] += (1 if seeds[i] % 2 == 0 else -1) * cond["shift_dx"]
                shifts[i] = t
                changed = True
            if cond.get("blackout") and blackouts[i] is None and transport is not None:
                corr[i].force_drop(cond["blackout"])
                blackouts[i] = t
        if changed:
            env.unwrapped.box.set_pose(Pose.create(pose))
            env.unwrapped.scene._gpu_apply_all()
            env.unwrapped.scene._gpu_fetch_all()
        raw = raw_tensor(env)
        observed = {k: v.clone() for k, v in raw.items()}
        fresh = torch.ones(b, device=device)
        age = torch.zeros(b, device=device)
        for i in range(b):
            if not live[i]:
                continue
            r, f, a = corr[i].step({k: v[i] for k, v in raw.items()})
            for k in ("peg_pose", "hole_pose"):
                observed[k][i] = r[k]
            fresh[i] = f
            age[i] = a
        torch.cuda.synchronize()
        tick = time.perf_counter()
        action = policy.act(observed, fresh, age, active)
        torch.cuda.synchronize()
        inference.append((time.perf_counter() - tick) * 1000)
        if previous is not None:
            delta = (action[:, :6] - previous[:, :6]).norm(dim=-1).cpu().tolist()
            for i in range(b):
                if live[i]:
                    deltas[i].append(delta[i])
        previous = action.clone()
        action = action.masked_fill(~active[:, None], 0)
        env_steps += sum(live)
        _, _, terminated, truncated, info = env.step(action)
        snaps = stage_snapshots(env, raw_tensor(env))
        succ = info["success"].cpu().tolist()
        ended = (terminated | truncated).cpu().tolist()
        for i in range(b):
            if not live[i]:
                continue
            trackers[i].update_snapshot(snaps[i], succ[i])
            if succ[i] or ended[i]:
                live[i] = False
                active[i] = False
                steps[i] = t + 1
                success[i] = bool(succ[i])
        if not any(live):
            break
    torch.cuda.synchronize()
    rollout_s = time.perf_counter() - start
    rows = []
    for i, seed in enumerate(seeds):
        rows.append(
            dict(
                episode_seed=seed,
                success=success[i],
                steps=steps[i],
                shift_applied=shifts[i] is not None,
                shift_step=shifts[i],
                steps_after_shift=steps[i] - shifts[i]
                if success[i] and shifts[i] is not None
                else None,
                blackout_applied=blackouts[i] is not None,
                blackout_step=blackouts[i],
                mean_action_delta=float(np.mean(deltas[i])) if deltas[i] else 0.0,
                mean_infer_ms=None,
                **trackers[i].summary(),
            )
        )
    return rows, {
        "reset_s": reset_s,
        "rollout_s": rollout_s,
        "active_env_steps": env_steps,
        "physical_env_steps": (t + 1) * b,
        "batch_infer_ms": float(np.mean(inference)),
        "initial_max_errors": errors,
    }
