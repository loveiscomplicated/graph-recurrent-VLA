import glob
import json
import sys

import numpy as np
from scipy import stats

for d in sys.argv[1:]:
    rs = [json.loads(l) for f in glob.glob(f"{d}/results_*.jsonl") for l in open(f)]
    for c in sorted({r["cand"] for r in rs}):
        err = sum("error" in r for r in rs if r["cand"] == c)
        by = {}
        for r in rs:
            if r["cand"] == c and "none" in r:
                by.setdefault(r["train_seed"], []).append(r["none"]["success"])
        p = np.array([np.mean(v) for v in by.values()]) * 100
        h = stats.t.ppf(0.975, len(p) - 1) * p.std(ddof=1) / np.sqrt(len(p)) if len(p) > 1 else float("nan")
        eps = sum(len(v) for v in by.values())
        print(f"{d.rstrip('/').split('/')[-1]} {c} {p.mean():.1f}+-{h:.1f} seeds {len(p)} eps {eps} errors {err} {'OK' if eps == 150 and err == 0 else 'INCOMPLETE'}")
