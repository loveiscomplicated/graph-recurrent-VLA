"""E0/E1: v1 세트의 초기 bbox (모델 경로, GT 대체 없음) + GT bbox 대비 IoU (평가용).

규칙 (dev seed 100-109 에서 정함)
  1. 물체별 GDINO 후보 (화면 1/4 이상 제외, 상위 5개)
  2. peg/box **공동 배정**: 두 후보의 box IoU < 0.3 이어야 하고, 각 물체는 "미검출"(점수 TAU) 도 고를 수 있다.
     점수 합이 최대인 조합을 고른다. (v0 top-1 은 "cardboard box" 가 peg 를 잡거나, 가려진 peg 대신 box 를 잡는 경우가 있었음)
  3. frame 0 에서 미검출인 물체는 5 프레임마다 다시 검출 → 2번 연속 같은 자리(IoU ≥ 0.5)면 그 프레임에서 초기화 (늦은 초기화)
     다른 물체가 이미 배정된 경우 그 물체 bbox 와 겹치는 후보는 제외한다 (frame 0 bbox 기준; box 는 kinematic 이라 거의 고정)
GT 는 init IoU 기록에만 쓴다.
출력: results/v1/init/init_boxes_<set>.json  {ep: {peg: {box, frame, score, iou_gt, gt_box, scans}, box: {...}}}

Usage: python scripts/init_v1.py --sets dev eval eval_box_move eval_lighting
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1data  # noqa: E402
from grounding import GDino, gt_bbox, iou  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPTS = {"peg": "red and white block", "box": "cardboard box"}  # Step 3 선택 프롬프트 (seed 0-2 에서 고름)
MAX_BOX_FRAC = 0.25
TAU = 0.3  # "미검출" 점수
EXCL_IOU = 0.3
SCAN_EVERY = 5


def cands(g, rgb, obj, k=5):
    H, W = rgb.shape[:2]
    boxes, scores = g(rgb, PROMPTS[obj])
    return [(b, s) for b, s in zip(boxes, scores) if (b[2] - b[0]) * (b[3] - b[1]) < MAX_BOX_FRAC * H * W][:k]


def assign(cp, cb):
    """cp/cb: [(box, score)] → (peg (box,score)|None, box (box,score)|None)"""
    best, best_s = (None, None), 2 * TAU
    for p, b in itertools.product(cp + [None], cb + [None]):
        if p is not None and b is not None and iou(p[0], b[0]) >= EXCL_IOU:
            continue
        s = (p[1] if p else TAU) + (b[1] if b else TAU)
        if s > best_s:
            best, best_s = (p, b), s
    return best


def detect_initial(g, rgb_seq):
    """causal: frame 0 공동 배정, 미검출 물체는 SCAN_EVERY 프레임마다 재검출 (2회 연속 확인)."""
    cp, cb = cands(g, rgb_seq[0], "peg"), cands(g, rgb_seq[0], "box")
    p, b = assign(cp, cb)
    res = {"peg": dict(det=p, frame=0 if p else None, scans=1), "box": dict(det=b, frame=0 if b else None, scans=1)}
    for obj in ("peg", "box"):
        if res[obj]["det"] is not None:
            continue
        other = "box" if obj == "peg" else "peg"
        ob = res[other]["det"][0] if res[other]["det"] is not None else None
        pend = None
        for t in range(SCAN_EVERY, len(rgb_seq), SCAN_EVERY):
            res[obj]["scans"] += 1
            c = [x for x in cands(g, rgb_seq[t], obj) if x[1] >= TAU and (ob is None or iou(x[0], ob) < EXCL_IOU)]
            if not c:
                pend = None
                continue
            if pend is not None and iou(c[0][0], pend[0]) >= 0.5:
                res[obj].update(det=c[0], frame=t)
                break
            pend = c[0]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out", default=f"{ROOT}/results/v1/init")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(8)
    g = GDino()
    for s in args.sets:
        res = {}
        for ep in v1data.ep_list(s):
            d = v1data.load(s, ep)
            rgb = d[f"{v1data.CAM}/rgb"]
            seg = d[f"{v1data.CAM}/seg"]
            det = detect_initial(g, rgb)
            r = {}
            for obj in ("peg", "box"):
                x = det[obj]
                f = x["frame"]
                gt = gt_bbox(seg[f if f is not None else 0] == int(d[f"{obj}_id"]))
                b = x["det"][0] if x["det"] else None
                r[obj] = dict(box=None if b is None else [round(float(v), 1) for v in b],
                              score=round(float(x["det"][1]), 3) if x["det"] else None, frame=f, scans=x["scans"],
                              iou_gt=round(iou(b, gt), 3) if (b is not None and gt is not None) else (0.0 if b is not None else None),
                              gt_box=gt, gt_visible_f0=gt_bbox(seg[0] == int(d[f"{obj}_id"])) is not None)
            res[ep] = r
            print(s, ep, {o: (r[o]["iou_gt"], r[o]["score"], r[o]["frame"]) for o in r}, flush=True)
        json.dump(res, open(f"{args.out}/init_boxes_{s}.json", "w"), indent=1)
        for obj in PROMPTS:
            v = [res[e][obj]["iou_gt"] or 0.0 for e in res]
            late = [res[e][obj]["frame"] for e in res if res[e][obj]["frame"] not in (0,)]
            print(f"[{s}] {obj}: mean init IoU {np.mean(v):.3f}, hit@0.5 {np.mean(np.array(v) >= 0.5):.2f} ({len(v)} ep), "
                  f"late/none init: {late}")


if __name__ == "__main__":
    main()
