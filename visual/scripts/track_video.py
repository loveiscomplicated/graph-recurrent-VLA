"""추적 비교 데모 영상: 같은 에피소드에서 두 설정의 mask 를 나란히 (초록 = GT 윤곽, 프레임별 IoU, 교란 구간 표시).
저장된 mask npz (track_v1 / e3_run 형식) 만 쓴다.
Usage: python scripts/track_video.py --set eval --perturb dropout --runs results/v1/E3/runs/none/eval_dropout results/v1/E3/runs/v1/eval_dropout \
           --labels "no re-anchor" "re-anchor v1" --eps ep203 --out cache/videos --name 10_reanchor_dropout
"""
import argparse
import os
import sys

import cv2
import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1data  # noqa: E402
from eval_v1 import iou_seq, load_masks  # noqa: E402
from track_sam2 import overlay  # noqa: E402


def text(img, lines, y0=20, s=0.5):
    for i, l in enumerate(lines):
        cv2.putText(img, l, (6, y0 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, l, (6, y0 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="eval")
    ap.add_argument("--perturb", default=None)
    ap.add_argument("--runs", nargs=2, required=True)
    ap.add_argument("--labels", nargs=2, required=True)
    ap.add_argument("--eps", nargs="+", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for ep in a.eps:
        d = v1data.load(a.set, ep, a.perturb)
        rgb, seg = d["base_camera/rgb"], d["base_camera/seg"]
        gt = {o: seg == int(d[f"{o}_id"]) for o in ("peg", "box")}
        ms = [load_masks(f"{r}/{ep}_masks.npz")[0] for r in a.runs]
        ious = [{o: iou_seq(m[o], gt[o]) for o in ("peg", "box")} for m in ms]
        pf = d["perturb_frames"]
        with imageio.get_writer(f"{a.out}/{a.name}_{a.set}_{ep}.mp4", fps=10, macro_block_size=1, quality=7) as w:
            for t in range(len(rgb)):
                tiles = []
                for m, io, lab in zip(ms, ious, a.labels):
                    im = overlay(rgb[t], {"peg": m["peg"][t], "box": m["box"][t]}, {"peg": gt["peg"][t], "box": gt["box"][t]})
                    f = lambda x: "  -  " if np.isnan(x) else f"{x:.2f}"
                    text(im, [lab, f"peg IoU {f(io['peg'][t])}   box IoU {f(io['box'][t])}"])
                    tiles.append(im)
                img = np.concatenate(tiles, 1)
                bar = np.full((52, img.shape[1], 3), 30, np.uint8)
                pert = f"  [{a.perturb.upper()} active]" if a.perturb and pf[t] else ""
                cv2.putText(bar, f"{a.title} | {a.set} {ep} | t={t} ({t / 20:.1f}s){pert}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 0) if pert else (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(bar, "red = peg mask, blue = box mask, green outline = ground truth (IoU only where object visible)", (8, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                w.append_data(np.concatenate([img, bar], 0))
        print("wrote", ep, flush=True)


if __name__ == "__main__":
    main()
