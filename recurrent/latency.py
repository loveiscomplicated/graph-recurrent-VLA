from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from recurrent.common import ACTION_DIM, OUTPUT_ROOT, resolve, versions
from recurrent.features import OBS_DIM
from recurrent.models import REGISTRY, build, model_config, num_params
from recurrent.overrides import parse_overrides


def measure(kind, cfg, device, threads, repeats, warmup):
    torch.set_num_threads(threads)
    model = build(kind, OBS_DIM, ACTION_DIM, cfg).to(device).eval()
    x = torch.randn(1, 1, OBS_DIM, device=device)
    cuda = device.type == "cuda"
    state = None
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            _, state = model(x, state)
        for _ in range(repeats):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, state = model(x, state)
            if cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    t = np.asarray(times)
    return {"model": kind, "device": device.type, "threads": threads, "params": num_params(model),
            "median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)), "mean_ms": float(t.mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(REGISTRY))
    p.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    p.add_argument("--repeats", type=int, default=300)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--out", default=str(OUTPUT_ROOT / "latency.json"))
    args = p.parse_args()
    overrides = parse_overrides(args.set)
    rows = []
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    for kind in args.models:
        cfg = model_config(kind, overrides.get(kind))
        for device in devices:
            for threads in (args.threads if device.type == "cpu" else [1]):
                r = measure(kind, cfg, device, threads, args.repeats, args.warmup)
                rows.append(r)
                print(f"{kind} {device.type} threads {threads} median {r['median_ms']:.3f}ms p90 {r['p90_ms']:.3f}ms")
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "batch": 1, "obs_dim": OBS_DIM, "repeats": args.repeats, "versions": versions()}, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
