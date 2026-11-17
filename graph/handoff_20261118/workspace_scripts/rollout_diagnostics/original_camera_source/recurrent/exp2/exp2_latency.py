import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from exp2_train import build


def detach(state):
    if state is None:
        return None
    if torch.is_tensor(state):
        return state.detach()
    return type(state)(detach(s) for s in state)


@torch.no_grad()
def measure(ck, device, n, warmup):
    model = build(ck["kind"], ck["obs_dim"], 7 * ck["chunk"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    xs = torch.randn(n + warmup, 1, 1, ck["obs_dim"], device=device)
    state, times = None, []
    for i in range(n + warmup):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, state = model(xs[i], state)
        out = out.cpu()
        t1 = time.perf_counter()
        state = detach(state)
        if i >= warmup:
            times.append((t1 - t0) * 1000)
    t = np.array(times)
    return {"p50_ms": round(float(np.percentile(t, 50)), 3), "p95_ms": round(float(np.percentile(t, 95)), 3),
            "max_ms": round(float(t.max()), 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="/work/exp2/runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    a = p.parse_args()
    torch.set_num_threads(1)
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    results = {}
    print("type  params  " + "  ".join(f"{d.type}_p50 {d.type}_p95" for d in devices))
    for path in sorted(Path(a.runs).glob(f"*_s{a.seed}/last.pt")):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        name = path.parent.name[: -len(f"_s{a.seed}")]
        row = {d.type: measure(ck, d, a.n, a.warmup) for d in devices}
        results[name] = {"params": ck["params"], **row}
        print(f"{name}  {ck['params']}  " + "  ".join(f"{row[d.type]['p50_ms']} {row[d.type]['p95_ms']}" for d in devices), flush=True)
    Path(a.runs).joinpath(f"latency_s{a.seed}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
