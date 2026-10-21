"""v1 실패 갤러리: 설정 결과에서 peg(또는 box) IoU 가 낮은 에피소드 / 프레임을 GT 윤곽과 함께 그린다.
Usage: python scripts/gallery_v1.py --run results/v1/E1/runs/sam2s_none/eval --set eval --out results/v1/E1/gallery_sam2s_none.jpg
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1data  # noqa: E402
from eval_v1 import iou_seq, load_masks  # noqa: E402
from track_sam2 import label_txt, overlay  # noqa: E402


def panel(d, masks, t, ious, tag=""):
    seg = d["base_camera/seg"][t]
    gt = {o: seg == int(d[f"{o}_id"]) for o in ("peg", "box")}
    im = overlay(d["base_camera/rgb"][t], {o: masks[o][t] for o in ("peg", "box")}, gt)
    im = cv2.resize(im, (320, 320))
    label_txt(im, [f"{tag} t={t}", f"peg {ious['peg'][t]:.2f} box {ious['box'][t]:.2f}"], y0=16)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", default="eval")
    ap.add_argument("--perturb", default=None)
    ap.add_argument("--obj", default="peg")
    ap.add_argument("--n-ep", type=int, default=5)
    ap.add_argument("--eps", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eps = a.eps or v1data.ep_list(a.set)
    scores = []
    cache = {}
    for ep in eps:
        if not os.path.exists(f"{a.run}/{ep}_masks.npz"):
            continue
        d = v1data.load(a.set, ep, a.perturb)
        m, _, _ = load_masks(f"{a.run}/{ep}_masks.npz")
        ious = {o: iou_seq(m[o], d["base_camera/seg"] == int(d[f"{o}_id"])) for o in ("peg", "box")}
        scores.append((np.nanmean(ious[a.obj]) if np.isfinite(ious[a.obj]).any() else 1.0, ep))
        cache[ep] = (d, m, ious)
    rows = []
    for sc, ep in sorted(scores)[:a.n_ep] if not a.eps else [(0, e) for e in a.eps]:
        d, m, ious = cache[ep]
        v = ious[a.obj]
        T = len(v)
        # 대표 프레임: IoU 최저 3개(간격 10) + 처음/끝
        order = [t for t in np.argsort(np.nan_to_num(v, nan=2.0)) if np.isfinite(v[t])]
        pick = []
        for t in order:
            if all(abs(t - p) >= 10 for p in pick):
                pick.append(int(t))
            if len(pick) == 3:
                break
        ts = sorted(set([0] + pick + [T - 1]))[:5]
        while len(ts) < 5:
            ts.append(ts[-1])
        rows.append(np.concatenate([panel(d, m, t, ious, f"{ep} {str(d.get('policy', ''))}") for t in ts], 1))
    cv2.imwrite(a.out, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("worst", sorted(scores)[:a.n_ep])


if __name__ == "__main__":
    main()
