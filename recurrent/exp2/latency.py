import json
import time
from pathlib import Path

import numpy as np
import torch

from exp2_eval import Decoder, Ensemble, load_ck

torch.set_num_threads(1)
spec = json.load(open("/workspace/ens.json"))
with np.load("/workspace/data_kin/train.npz", allow_pickle=False) as d:
    x, off, counts = d["state36_transformer"], d["episode_offsets"], d["counts"]
for name in ("ens_mamba5", "ens_mlp5"):
    decs = [Decoder(load_ck(Path(dd) / f"{m}_s{sd}" / "last.pt"), "step") for dd, m, sd in spec[name]]
    for label, dec in ((name, Ensemble(decs)), (name + "_single", decs[0])):
        ts = []
        for ep in range(5):
            dec.reset()
            for t in range(off[ep], off[ep + 1]):
                t0 = time.perf_counter()
                dec.act(x[t], counts[t])
                ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts[20:])
        print(f"{label} steps {len(ts)} ms mean {ts.mean():.2f} p99 {np.percentile(ts, 99):.2f} max {ts.max():.2f} budget_50ms {'PASS' if np.percentile(ts, 99) < 50 else 'FAIL'}")
