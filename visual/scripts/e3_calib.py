"""E3: shadow 실행(후보를 받아들이지 않고 기록만) 결과로 트리거 정밀도와 후보 게이트 임계값을 dev 에서 정한다."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1data  # noqa: E402
from eval_v1 import iou_seq, load_masks  # noqa: E402


def main(run_root, sets):
    trig_stats, cands = {}, []
    for s in sets:
        rd = f"{run_root}/{s}"
        if not os.path.exists(f"{rd}/run.json"):
            continue
        run = json.load(open(f"{rd}/run.json"))
        base, pert = (s.split("_", 1) + [None])[:2] if s.startswith(("dev_patch", "dev_dropout", "eval_patch", "eval_dropout")) else (s, None)
        for ep, r in run["episodes"].items():
            d = v1data.load(base, ep, pert)
            m, _, _ = load_masks(f"{rd}/{ep}_masks.npz")
            ious = {o: iou_seq(m[o], d["base_camera/seg"] == int(d[f"{o}_id"])) for o in ("peg", "box")}
            seen = set()
            for e in r["events"]:
                if e["kind"] != "detect":
                    continue
                io = ious[e["obj"]][e["t"]]
                bad = (not np.isfinite(io) and e["gt_visible"]) or (np.isfinite(io) and io < 0.5)
                for tg in e["trig"]:
                    if (e["obj"], e["t"], tg) in seen:
                        continue
                    seen.add((e["obj"], e["t"], tg))
                    ts = trig_stats.setdefault(f"{e['obj']}:{tg}", dict(n=0, tracker_bad=0, gt_visible=0))
                    ts["n"] += 1
                    ts["tracker_bad"] += int(bool(bad))
                    ts["gt_visible"] += int(e["gt_visible"])
                for c in e["cands"]:
                    cands.append(dict(set=s, ep=ep, t=e["t"], obj=e["obj"], trig="+".join(e["trig"]), tracker_iou=None if not np.isfinite(io) else float(io),
                                      **{k: c.get(k) for k in ("sim", "area_r", "size_r", "dist", "score", "gt_iou", "box")}))
    print("== 트리거 (detect 이벤트 기준): n / 그 순간 tracker 가 틀림(IoU<0.5 또는 GT 보이는데 mask 없음) / GT 보임")
    for k, v in sorted(trig_stats.items()):
        print(f"  {k:12s} n={v['n']:4d} tracker_bad={v['tracker_bad']:4d} ({v['tracker_bad'] / max(1, v['n']):.2f}) gt_visible={v['gt_visible']}")
    C = cands
    print(f"== 후보 {len(C)}개, 정답(gt_iou≥0.5) {sum(c['gt_iou'] >= 0.5 for c in C)}개")
    for o in ("peg", "box"):
        pos = [c for c in C if c["obj"] == o and c["gt_iou"] >= 0.5]
        neg = [c for c in C if c["obj"] == o and c["gt_iou"] < 0.5]
        print(f"-- {o}: pos {len(pos)} neg {len(neg)}")
        for f in ("sim", "area_r", "size_r", "dist", "score"):
            pv_ = np.array([c[f] for c in pos if c[f] is not None], float)
            nv = np.array([c[f] for c in neg if c[f] is not None], float)
            if len(pv_) and len(nv):
                print(f"   {f:7s} pos p10/50/90 {np.percentile(pv_, [10, 50, 90]).round(3)}  neg p10/50/90 {np.percentile(nv, [10, 50, 90]).round(3)}")
        for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
            tp = sum(1 for c in pos if (c["sim"] or -1) >= thr)
            fp = sum(1 for c in neg if (c["sim"] or -1) >= thr)
            print(f"   sim≥{thr}: 정답 통과 {tp}/{len(pos)}, 오답 통과 {fp}/{len(neg)}")
    json.dump(dict(trig=trig_stats, cands=C), open(f"{run_root}/calib.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
