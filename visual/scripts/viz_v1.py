"""E0: v1 에피소드 몽타주 (base_camera RGB + GT 윤곽, 시간순 N 프레임). 정책/교란 동작 확인용."""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_sam2 import label_txt  # noqa: E402


def montage(path, frames=None, n=8, cam="base_camera", extra_frames=(), perturb=None, set_name=None):
    if perturb:
        import v1data
        d = v1data.load(set_name, os.path.basename(path)[:-4], perturb)
    else:
        d = dict(np.load(path))
    T = len(d["frame_idx"])
    ts = frames or sorted(set(np.linspace(0, T - 1, n).astype(int).tolist() + list(extra_frames)))
    tiles = []
    for t in ts:
        im = d[f"{cam}/rgb"][t].copy()
        for oid, col in ((int(d["peg_id"]), (255, 0, 0)), (int(d["box_id"]), (0, 255, 0))):
            cs, _ = cv2.findContours((d[f"{cam}/seg"][t] == oid).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, col, 1)
        im = cv2.resize(im, (256, 256))
        label_txt(im, [f"t={t} succ={int(d['success'][t])}"])
        tiles.append(im)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)]
    return np.concatenate(rows, 0), str(d.get("policy", "")), str(d.get("meta", ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--around", type=int, nargs="*", default=[])
    ap.add_argument("--cam", default="base_camera")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for p in a.npz:
        img, pol, meta = montage(p, n=a.n, cam=a.cam, extra_frames=a.around)
        name = os.path.basename(os.path.dirname(p)) + "_" + os.path.basename(p)[:-4]
        cv2.imwrite(f"{a.out}/{name}_{a.cam}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(name, pol, meta)
