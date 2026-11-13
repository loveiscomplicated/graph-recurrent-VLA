import glob
import json
import sys

import numpy as np
from scipy import stats


def seg(a, lo, hi):
    a = np.asarray(a[lo:hi], float)
    return float(a.mean()) if len(a) else np.nan


def main():
    rows = [json.loads(l) for d in sys.argv[1:] for f in glob.glob(f"{d}/results_*.jsonl") for l in open(f)]
    for cand in sorted({r["cand"] for r in rows if "none" in r}):
        eps = [r["none"] for r in rows if r["cand"] == cand and "none" in r]
        k, n = sum(e["success"] for e in eps), len(eps)
        lo, hi = stats.binomtest(k, n).proportion_ci(0.95, "wilson") if n else (np.nan, np.nan)
        print(f"{cand} success {k}/{n} = {100 * k / max(n, 1):.1f}% wilson {100 * lo:.0f}-{100 * hi:.0f}")
        by = {}
        for e in eps:
            m = e.get("motion") or {}
            by.setdefault(m.get("status", "none"), []).append(e["success"])
        print("  by_status " + " ".join(f"{s}:{sum(v)}/{len(v)}" for s, v in sorted(by.items())))
        acc = {"before": [], "during": [], "after": []}
        lat = {"before": [], "during": [], "after": []}
        for e in eps:
            m = e.get("motion") or {}
            if m.get("first_step") is None:
                continue
            a, b = m["first_step"], m["last_step"] + 1
            f = np.asarray(e.get("fxyz_mm") or [[0, 0, 0]], float)
            fl = np.linalg.norm(f[:, 1:], axis=1)
            for name, (x, y) in (("before", (0, a)), ("during", (a, b)), ("after", (b, len(e["ehole_mm"])))):
                acc[name].append(seg(e["ehole_mm"], x, y))
                lat[name].append(seg(fl, x, y))
        for name in ("before", "during", "after"):
            v, w = np.array(acc[name]), np.array(lat[name])
            if len(v):
                print(f"  {name} raw_hole_err_mm median {np.nanmedian(v):.2f} mean {np.nanmean(v):.2f}  filtered_tip_hole_lateral_mm median {np.nanmedian(w):.2f}")


if __name__ == "__main__":
    main()
