import glob
import json
import sys

import numpy as np


def main():
    rows = [json.loads(l) for f in glob.glob(f"{sys.argv[1]}/**/*.jsonl", recursive=True) for l in open(f)]
    for cand in sorted({r["cand"] for r in rows}):
        eps = [r["none"] for r in rows if r["cand"] == cand and r["none"].get("fxyz_mm")]
        print(f"\n{cand} eps {len(eps)} success {sum(e['success'] for e in eps)}")
        groups = {"success": [], "fail": []}
        for e in eps:
            fr = np.array(e["fxyz_mm"])
            rel = np.array(e["final_rel_mm"]) if e.get("final_rel_mm") else None
            idx = np.arange(len(fr))
            win = fr[(idx >= 80) & (idx < 120)] if len(fr) > 80 else fr[-20:]
            lat = np.linalg.norm(win[:, 1:].mean(0))
            groups["success" if e["success"] else "fail"].append((lat, np.abs(win[:, 1]).mean(), np.abs(win[:, 2]).mean(), np.abs(win[:, 0]).mean()))
        for tag, g in groups.items():
            if not g:
                continue
            g = np.array(g)
            print(f"  {tag} n {len(g)} filtered_lat_80_120 median {np.median(g[:, 0]):.2f} under_1.5mm {100 * np.mean(g[:, 0] < 1.5):.1f}% "
                  f"abs_y {np.median(g[:, 1]):.2f} abs_z {np.median(g[:, 2]):.2f} abs_axial {np.median(g[:, 3]):.2f}")
        fail = np.array(groups["fail"]) if groups["fail"] else np.zeros((0, 4))
        if len(fail):
            print(f"  fail with filtered_lat under 1.5mm (decoder side) {100 * np.mean(fail[:, 0] < 1.5):.1f}%  over 3mm (perception side) {100 * np.mean(fail[:, 0] > 3):.1f}%")


if __name__ == "__main__":
    main()
