from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from recurrent.collect import save
from recurrent.common import (
    CONTROL_MODE,
    DATA_DIR,
    HORIZON,
    OUTPUT_ROOT,
    RAW_KEYS,
    environment_record,
    make_env,
    raw_gt,
    resolve,
    sha256,
    versions,
)

ENV_ID = "PegInsertionSide-v1"
DEMO_DIR = OUTPUT_ROOT / "official"
OBS_TOLERANCE = 2e-5


def raw_path(root: Path) -> Path:
    return root / ENV_ID / "motionplanning" / "trajectory.h5"


def converted_path(raw: Path) -> Path:
    return raw.with_name(f"{raw.stem}.state.{CONTROL_MODE}.physx_cpu.h5")


def download(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "mani_skill.utils.download_demo", ENV_ID, "-o", str(root)], check=True)
    return raw_path(root)


def convert(raw: Path, workers: int, count) -> Path:
    from mani_skill.trajectory.replay_trajectory import Args, main as official_replay

    request = Args(
        traj_path=str(raw.resolve()),
        use_first_env_state=True,
        target_control_mode=CONTROL_MODE,
        obs_mode="state",
        save_traj=True,
        num_envs=workers,
        sim_backend="physx_cpu",
        count=count,
    )
    cwd = Path.cwd()
    try:
        official_replay(request)
    finally:
        os.chdir(cwd)
    return converted_path(raw)


def state_at(group, index):
    return {k: state_at(v, index) if isinstance(v, h5py.Group) else np.array(v[index]) for k, v in group.items()}


def reset_episode(env, episode):
    kwargs = json.loads(json.dumps(episode.get("reset_kwargs", {})))
    seed = kwargs.get("seed", episode["episode_seed"])
    if isinstance(seed, list):
        seed = seed[0]
    kwargs["seed"] = seed
    kwargs["options"] = dict(kwargs.get("options") or {}, reconfigure=True)
    return env.reset(**kwargs)


def live_obs(env) -> np.ndarray:
    return env.unwrapped.get_obs().cpu().numpy().reshape(-1)


def initial_state(env, episode, observations, source, source_meta):
    if source is None:
        return env.unwrapped.get_state_dict(), float(np.max(np.abs(live_obs(env) - observations[0])))
    best = None
    for candidate in source_meta.get(episode["episode_seed"], []):
        state = state_at(source[f"traj_{candidate}/env_states"], 0)
        env.unwrapped.set_state_dict(state)
        error = float(np.max(np.abs(live_obs(env) - observations[0])))
        if best is None or error < best[1]:
            best = (state, error)
    if best is None:
        return env.unwrapped.get_state_dict(), float("inf")
    return best


def extract(converted: Path, source: Path | None, count, horizon: int, log=print):
    meta = json.loads(converted.with_suffix(".json").read_text())
    if meta["env_info"]["env_id"] != ENV_ID:
        raise ValueError("wrong task")
    episodes = [e for e in meta["episodes"] if e.get("success", True)]
    episodes = episodes[:count] if count else episodes
    src = h5py.File(source, "r") if source else None
    src_meta = {}
    if source:
        for e in json.loads(source.with_suffix(".json").read_text())["episodes"]:
            src_meta.setdefault(e["episode_seed"], []).append(e["episode_id"])
    env = make_env(horizon)
    kept, rejected = [], []
    try:
        with h5py.File(converted, "r") as f:
            for e in episodes:
                group = f[f"traj_{e['episode_id']}"]
                actions = np.array(group["actions"], dtype=np.float32)
                observations = np.array(group["obs"])
                if actions.ndim != 2 or actions.shape[1] != 7 or len(observations) != len(actions) + 1:
                    rejected.append({"episode_id": e["episode_id"], "reason": "shape"})
                    continue
                reset_episode(env, e)
                s0, err0 = initial_state(env, e, observations, src, src_meta)
                obs = {k: [] for k in RAW_KEYS}
                worst = err0
                ok = err0 <= OBS_TOLERANCE
                for t in range(len(actions)):
                    if not ok:
                        break
                    env.unwrapped.set_state_dict(s0 if t == 0 else state_at(group["env_states"], t))
                    error = float(np.max(np.abs(live_obs(env) - observations[t])))
                    worst = max(worst, error)
                    if error > OBS_TOLERANCE:
                        ok = False
                        break
                    for k, v in raw_gt(env).items():
                        obs[k].append(v)
                status = "PASS" if ok else "FAIL"
                log(f"episode {e['episode_id']} seed {e['episode_seed']} T {len(actions)} max_obs_error {worst:.2e} {status}")
                if not ok:
                    rejected.append({"episode_id": e["episode_id"], "reason": "obs_mismatch", "max_obs_error": worst})
                    continue
                kept.append({
                    "seed": int(e["episode_seed"]),
                    "episode_id": int(e["episode_id"]),
                    "obs": {k: np.stack(v).astype(np.float32) for k, v in obs.items()},
                    "act": np.clip(actions, -1.0, 1.0),
                    "max_obs_error": worst,
                })
    finally:
        env.close()
        if src is not None:
            src.close()
    return kept, rejected, len(episodes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(DEMO_DIR))
    p.add_argument("--download", action="store_true")
    p.add_argument("--convert", action="store_true")
    p.add_argument("--raw", default=None)
    p.add_argument("--converted", default=None)
    p.add_argument("--source", default=None)
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--out", default=str(DATA_DIR / "official_demos.npz"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out = resolve(args.out)
    if out.exists() and not args.force:
        print(f"skip {out} exists")
        return
    root = resolve(args.root)
    raw = resolve(args.raw) if args.raw else raw_path(root)
    if args.download:
        raw = download(root)
    converted = resolve(args.converted) if args.converted else converted_path(raw)
    if args.convert or not converted.exists():
        if not raw.exists():
            raise SystemExit(f"missing {raw}; run with --download or pass --raw/--converted")
        converted = convert(raw, args.workers, args.count)
    source = resolve(args.source) if args.source else (raw if raw.exists() and raw != converted else None)
    t0 = time.time()
    kept, rejected, total = extract(converted, source, args.count, args.horizon)
    if not kept:
        raise SystemExit("no episode passed state restoration")
    meta = {
        "source": "official",
        "converted": str(converted),
        "converted_sha256": sha256(converted),
        "initial_state_source": str(source) if source else None,
        "args": vars(args),
        "environment": environment_record(args.horizon),
        "versions": versions(),
        "kept": len(kept),
        "total": total,
        "rejected": rejected,
    }
    save(out, kept, meta)
    print(f"official demos {len(kept)}/{total}")
    print(f"saved {out} frames {sum(len(r['act']) for r in kept)} time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
