import glob
import json
import sys

import numpy as np

BUCKETS = [(0, 40), (40, 80), (80, 120), (120, 250)]


def by_bucket(eps, key):
    out = []
    for lo, hi in BUCKETS:
        v = [x for e in eps for x in e[key][lo:hi]]
        out.append(round(float(np.mean(v)), 1) if v else None)
    return out


def main():
    d = sys.argv[1]
    data = sys.argv[2] if len(sys.argv) > 2 else "/workspace/data_kin/train.npz"
    rows = [json.loads(l) for f in glob.glob(f"{d}/**/*.jsonl", recursive=True) for l in open(f)]
    with np.load(data, allow_pickle=False) as z:
        g, gt, off = z["state36_transformer"], z["state36_gt"], z["episode_offsets"]
    step = np.concatenate([np.arange(e - s) for s, e in zip(off[:-1], off[1:])])
    for name, sl in (("tip_hole", slice(22, 25)), ("peg", slice(15, 18)), ("hole", slice(25, 28))):
        e = np.linalg.norm(g[:, sl] - gt[:, sl], axis=1) * 1000
        print(f"offline_raw {name} by_steps {[round(float(e[(step >= lo) & (step < hi)].mean()), 1) for lo, hi in BUCKETS]}")
    for cand in sorted({r["cand"] for r in rows}):
        eps = [r["none"] for r in rows if r["cand"] == cand]
        ok = [e for e in eps if e["success"]]
        bad = [e for e in eps if not e["success"]]
        nolift = [e for e in bad if not e["lifted"]]
        print(f"\n{cand} eps {len(eps)} success {len(ok)} fail_lifted {len(bad) - len(nolift)} fail_not_lifted {len(nolift)}")
        for tag, grp in (("success", ok), ("fail", bad)):
            if not grp:
                continue
            print(f"  {tag} steps_median {int(np.median([e['steps'] for e in grp]))}")
            for key in ("err_mm", "epeg_mm", "ehole_mm"):
                print(f"  {tag} {key} by_steps {by_bucket(grp, key)}")
        for tag, grp in (("success", ok), ("fail", bad)):
            hb = [np.mean(e["ehole_mm"][80:]) for e in grp if len(e["ehole_mm"]) > 80]
            pb = [np.mean(e["epeg_mm"][80:]) for e in grp if len(e["epeg_mm"]) > 80]
            if hb:
                print(f"  {tag} late_ehole_ep_median {np.median(hb):.1f} late_epeg_ep_median {np.median(pb):.1f}")


if __name__ == "__main__":
    main()
