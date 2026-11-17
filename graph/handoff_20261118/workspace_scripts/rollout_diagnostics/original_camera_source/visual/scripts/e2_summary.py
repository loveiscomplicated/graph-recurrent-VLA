"""E2 run.json 집계 → results/v1/E2/table.md"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1data  # noqa: E402

rows = []
for f in sorted(glob.glob(f"{v1data.ROOT}/results/v1/E2/runs/*/eval/run.json"), key=os.path.getmtime):
    R = json.load(open(f))
    E = R["episodes"]
    tr = np.concatenate([e["track_ms"] for e in E.values()])
    pc = np.concatenate([e["pc_ms"] for e in E.values()])
    tot = tr + pc
    r = dict(variant=R["variant"], track_p50=np.median(tr), pc_p50=np.median(pc), p50=np.median(tot), p95=np.percentile(tot, 95),
             max=tot.max(), fps=1000 / np.median(tot), gpu=max(e["gpu_peak_gb"] for e in E.values()))
    for o in ("peg", "box"):
        r[f"{o}_iou"] = v1data.boot_ci([np.array(e[f"{o}_iou"], float) for e in E.values()])["mean"]
        if f"{o}_eq_iou_mean" in next(iter(E.values())):
            r[f"{o}_eq"] = np.mean([e[f"{o}_eq_iou_mean"] for e in E.values()])
            r[f"{o}_ident"] = np.mean([e[f"{o}_eq_identical_frac"] for e in E.values()])
    rows.append(r)
L = ["| variant | track p50 | pc p50 | **합계 p50 / p95 / max (ms)** | FPS (p50) | GPU peak GB | peg / box IoU (GT) | offline 대비 mask IoU peg / box (비트 동일 프레임 비율) |",
     "|---|---|---|---|---|---|---|---|"]
for r in rows:
    eq = f"{r.get('peg_eq', float('nan')):.4f} / {r.get('box_eq', float('nan')):.4f} ({r.get('peg_ident', float('nan')):.2f} / {r.get('box_ident', float('nan')):.2f})"
    L.append(f"| {r['variant']} | {r['track_p50']:.1f} | {r['pc_p50']:.1f} | **{r['p50']:.1f} / {r['p95']:.1f} / {r['max']:.0f}** | {r['fps']:.1f} | {r['gpu']:.2f} | "
             f"{r['peg_iou']:.3f} / {r['box_iou']:.3f} | {eq} |")
open(f"{v1data.ROOT}/results/v1/E2/table.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
