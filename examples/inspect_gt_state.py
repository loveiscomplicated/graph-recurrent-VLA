"""Print live simulator GT fields without choosing model features."""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.maniskill import get_gt_state, make_peg_insertion_env


def main() -> None:
    env = make_peg_insertion_env(obs_mode="state")
    try:
        env.reset(seed=0)
        state = get_gt_state(env)
        for field in fields(state):
            value = getattr(state, field.name)
            sample = value[0].detach().cpu().flatten()[:7].tolist()
            print(f"{field.name:22} shape={tuple(value.shape)!s:10} dtype={str(value.dtype):13} device={value.device} sample={[round(float(x), 4) for x in sample]}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
