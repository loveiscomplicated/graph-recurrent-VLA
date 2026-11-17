"""E8: 도메인 강인성 — HSV 색 기준선 vs SAM2 tracking (box), 기본 렌더 / lighting 교란 / default shader.

HSV 기준선: dev(minimal shader, 기본 조명)의 GT box 픽셀 HSV 분포 2–98 percentile 범위 → 그 범위 픽셀의 최대 연결 성분 = box.
  (peg 는 흰 꼬리가 흰 로봇 팔과 색으로 구분되지 않아 HSV 기준선을 만들지 않았다)
SAM2 쪽은 track_v1 결과(--runs 의 <tag>/<set>) 를 eval_v1 과 같은 방식으로 평가한다.
Usage: python scripts/e8_domain.py --sets eval eval_lighting eval_shader --sam-tag sam2s_none
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1data  # noqa: E402
from eval_v1 import iou_seq, load_masks  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fit_hsv(set_name="dev", stride=10):
    vals = []
    for ep in v1data.ep_list(set_name):
        d = v1data.load(set_name, ep)
        for t in range(0, len(d["frame_idx"]), stride):
            m = d["base_camera/seg"][t] == int(d["box_id"])
            if m.sum() > 200:
                vals.append(cv2.cvtColor(d["base_camera/rgb"][t], cv2.COLOR_RGB2HSV)[m])
    v = np.concatenate(vals)
    lo, hi = np.percentile(v, 2, 0), np.percentile(v, 98, 0)
    return lo.astype(np.uint8), hi.astype(np.uint8)


def hsv_mask(rgb, lo, hi):
    m = cv2.inRange(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV), lo, hi) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lab == k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["eval", "eval_lighting", "eval_shader"])
    ap.add_argument("--sam-tag", default="sam2s_none")
    ap.add_argument("--runs", default=f"{ROOT}/results/v1/E1/runs")
    ap.add_argument("--out", default=f"{ROOT}/results/v1/E8")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lo, hi = fit_hsv()
    res = dict(hsv_lo=lo.tolist(), hsv_hi=hi.tolist(), sets={})
    for s in args.sets:
        eps = v1data.ep_list(s)
        hs, sm = [], []
        for ep in eps:
            d = v1data.load(s, ep)
            seg, rgb = d["base_camera/seg"], d["base_camera/rgb"]
            gt = seg == int(d["box_id"])
            hm = np.stack([hsv_mask(x, lo, hi) for x in rgb])
            hs.append(iou_seq(hm, gt))
            f = f"{args.runs}/{args.sam_tag}/{s}/{ep}_masks.npz"
            if os.path.exists(f):
                m, _, _ = load_masks(f)
                sm.append(iou_seq(m["box"], gt))
        r = dict(n_ep=len(eps), hsv_box=v1data.boot_ci(hs))
        if sm:
            r["sam2_box"] = v1data.boot_ci(sm)
            r["n_ep_sam2"] = len(sm)
        res["sets"][s] = r
        print(s, json.dumps(r), flush=True)
    json.dump(res, open(f"{args.out}/domain.json", "w"), indent=1)


if __name__ == "__main__":
    main()
