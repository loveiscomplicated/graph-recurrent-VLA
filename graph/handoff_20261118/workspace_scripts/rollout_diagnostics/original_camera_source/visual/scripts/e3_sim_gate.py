"""E3: shadow 기록 위에서 수락 규칙(게이트 + 2 프레임 확인, 소실 구간마다 첫 수락)을 시뮬레이션해 임계값을 고른다 (dev 전용)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e3_fit_gate import feat, prob  # noqa: E402
from reanchor_v1 import box_iou  # noqa: E402


def simulate(C, gate, obj, thr, mode="lr", sim_thr=0.6):
    by = {}
    for c in C:
        if c["obj"] == obj:
            by.setdefault((c["set"], c["ep"]), {}).setdefault(c["t"], []).append(c)
    good = bad = segs = 0
    g = gate[obj]
    for key, frames in by.items():
        ts = sorted(frames)
        prev_t, prev_box, done, seg_start = None, None, False, None
        for t in ts:
            if prev_t is None or t - prev_t > 1:
                segs += 1
                done = False
                pend = None
            prev_t = t
            if done:
                continue
            cs = frames[t]
            if mode == "lr":
                ps = prob(np.array(g["w"]), np.array(g["mu"]), np.array(g["sd"]), np.array([feat(c) for c in cs]))
                ok = [(p, c) for p, c in zip(ps, cs) if p >= thr]
            else:
                ok = [((c["sim"] or -1), c) for c in cs if (c["sim"] or -1) >= sim_thr]
            if not ok:
                pend = None
                continue
            best = max(ok, key=lambda x: x[0])[1]
            if pend is not None and pend[0] == t - 1 and box_iou(best["box"] if "box" in best else [0, 0, 1, 1], pend[1]) >= 0.5:
                done = True
                if best["gt_iou"] >= 0.5:
                    good += 1
                else:
                    bad += 1
            pend = (t, best.get("box", [0, 0, 1, 1]))
    return good, bad, segs


if __name__ == "__main__":
    C = json.load(open(sys.argv[1]))["cands"]
    gate = json.load(open(sys.argv[2]))
    for obj in ("peg", "box"):
        print(f"== {obj}")
        for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            print(f"  LR p≥{thr}: 정답 수락 / 오답 수락 / 구간 = {simulate(C, gate, obj, thr)}")
        for st in (0.5, 0.6, 0.7, 0.75, 0.8):
            print(f"  DINO sim≥{st} only: {simulate(C, gate, obj, 0, 'sim', st)}")
