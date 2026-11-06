from __future__ import annotations

import argparse

from recurrent.common import ACTION_DIM
from recurrent.features import OBS_DIM
from recurrent.models import REGISTRY, build, model_config, num_params
from recurrent.overrides import parse_overrides


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(REGISTRY))
    p.add_argument("--set", nargs="*", default=[])
    args = p.parse_args()
    overrides = parse_overrides(args.set)
    counts = {}
    for kind in args.models:
        cfg = model_config(kind, overrides.get(kind))
        counts[kind] = num_params(build(kind, OBS_DIM, ACTION_DIM, cfg))
    base = min(counts.values())
    for kind, n in counts.items():
        print(f"{kind} {n} x{n / base:.2f}")


if __name__ == "__main__":
    main()
