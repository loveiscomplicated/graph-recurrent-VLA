"""Run the installed official replay utility with absolute paths/macOS setup."""
import argparse
import os
from pathlib import Path

from .common import config, DEFAULT_CONFIG, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traj-path", type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--workers", type=int, default=2, help="CPU replay subprocesses, not batched CPU environments")
    args = p.parse_args()
    cfg = config(args.config)
    trajectory = args.traj_path.resolve()
    if not trajectory.exists():
        p.error(f"missing trajectory: {trajectory}")
    # Importing the factory configures the local MoltenVK ICD before SAPIEN.
    from envs.maniskill import make_peg_insertion_env
    from mani_skill.trajectory.replay_trajectory import Args, main as official_replay
    request = Args(traj_path=str(trajectory), use_first_env_state=True,
                   target_control_mode=cfg["control_mode"], obs_mode="state", save_traj=True,
                   num_envs=args.workers, sim_backend="physx_cpu", count=args.count)
    original_cwd = Path.cwd()
    try:
        official_replay(request)
    finally:
        os.chdir(original_cwd)
    write_json(trajectory.parent/"replay_request.json", {"trajectory": str(trajectory),
               "count": args.count, "cpu_workers": args.workers, "control_mode": cfg["control_mode"],
               "obs_mode": "state", "sim_backend": "physx_cpu", "use_first_env_state": True})


if __name__ == "__main__":
    main()
