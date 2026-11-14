import sys
import time
from pathlib import Path

import numpy as np
import torch

from exp2_train import build, features
from kin_filter import KinFilter

torch.set_num_threads(1)
RUNS = Path(sys.argv[1])


def detach(s):
    if s is None:
        return None
    if torch.is_tensor(s):
        return s.detach()
    return type(s)(detach(x) for x in s)


class Dec:
    def __init__(self, ck):
        self.model = build(ck["kind"], ck["obs_dim"], 7 * ck["chunk"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        st = ck["stats"]
        self.mean, self.std, self.keep = (np.asarray(st[k], np.float32) for k in ("obs_mean", "obs_std", "obs_keep"))
        self.am, self.asd = np.asarray(st["act_mean"], np.float32), np.asarray(st["act_std"], np.float32)
        self.hf, self.r6, self.chunk = ck["hole_frame"], bool(ck.get("rot6")), ck["chunk"]
        self.kf = KinFilter(**ck["kin"]) if isinstance(ck.get("kin"), dict) else None
        self.state = None

    @torch.no_grad()
    def act(self, s, counts):
        if self.kf is not None:
            s = self.kf.step(s, counts)
        f = features(s[None].astype(np.float32), self.hf, self.r6)
        x = ((f - self.mean) / self.std * self.keep).astype(np.float32)
        out, st = self.model(torch.from_numpy(x)[None], self.state)
        self.state = detach(st)
        a = out[0, 0].view(self.chunk, 7).numpy()[0]
        return np.concatenate([a[:6] * self.asd + self.am, [1.0 if a[6] > 0 else -1.0]])


def fake_episode(rng, n=150):
    s = np.zeros((n, 36), np.float32)
    s[:, 0:3] = [0.1, 0.0, 0.2] + rng.normal(0, 0.01, (n, 3))
    s[:, 3:7] = [0.0, 1.0, 0.0, 0.0]
    s[:, 13:15] = np.linspace(0.04, 0.02, n)[:, None]
    s[:, 15:18] = rng.normal(0, 0.05, (n, 3))
    s[:, 18:22] = [1.0, 0.0, 0.0, 0.0]
    s[:, 22:25] = rng.normal(0, 0.05, (n, 3))
    s[:, 25:28] = rng.normal(0, 0.1, (n, 3))
    s[:, 28:32] = [1.0, 0.0, 0.0, 0.0]
    s[:, 32:35] = [0.08, 0.01, 0.01]
    s[:, 35] = 0.012
    return s


rng = np.random.default_rng(0)
for typ in ("mamba_step", "mlp_step"):
    cks = [torch.load(RUNS / f"{typ}_s{i}" / "last.pt", map_location="cpu", weights_only=False) for i in range(5)]
    for label, k in (("single", 1), ("ensemble5", 5)):
        ts = []
        for ep in range(5):
            decs = [Dec(c) for c in cks[:k]]
            for s in fake_episode(rng):
                t0 = time.perf_counter()
                acts = [d.act(s, (99, 99)) for d in decs]
                np.mean(acts, 0)
                ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts[20:])
        print(f"{typ} {label} steps {len(ts)} ms mean {ts.mean():.2f} p99 {np.percentile(ts, 99):.2f} max {ts.max():.2f} budget_50ms {'PASS' if np.percentile(ts, 99) < 50 else 'FAIL'}")
