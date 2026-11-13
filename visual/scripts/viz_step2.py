"""Step 2 검증: 수집된 에피소드의 RGB / depth / GT mask 대표 프레임 + 기하 일관성 체크.

- 대표 프레임 패널: RGB | depth(colormap) | GT seg overlay(peg=빨강, box=파랑) + GT pose 투영점
- 기하 체크: peg GT mask 픽셀을 depth+intrinsic으로 역투영 → cam2world → peg local frame 으로 옮겨
  peg 직육면체(half_sizes) 안에 들어오는 비율, 그리고 peg/box 가시 포인트 centroid 와 GT pose 거리
- peg/box 가시 픽셀 수 곡선 (가림 구간 확인용), 에피소드 1개 영상

Usage:
  source env.sh
  python scripts/viz_step2.py --data data/step2 --out results/step2
"""
import argparse
import glob
import json
import os

import cv2
import imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

PEG_C, BOX_C = np.array([255, 40, 40]), np.array([40, 90, 255])


def pose_mat(p):
    """[x y z qw qx qy qz] -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def backproject(depth, K, mask):
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u]
    x = (u + 0.5 - K[0, 2]) * z / K[0, 0]  # pixel center 기준
    y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1)


def to_world(pts, c2w):
    return pts @ c2w[:3, :3].T + c2w[:3, 3]


def project(pw, ext, K):
    pc = ext[:3, :3] @ pw + ext[:3, 3]
    return K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]


def depth_vis(d, dmax=1.5):
    v = np.clip(d / dmax, 0, 1)
    img = (plt.get_cmap("turbo")(v)[..., :3] * 255).astype(np.uint8)
    img[d <= 0] = 0
    return img


def overlay(rgb, seg, peg_id, box_id, a=0.55):
    out = rgb.astype(np.float32).copy()
    for i, c in ((peg_id, PEG_C), (box_id, BOX_C)):
        m = seg == i
        out[m] = (1 - a) * out[m] + a * c
    return out.astype(np.uint8)


def panel(d, cam, t, peg_id, box_id):
    rgb, dep, seg = d[f"{cam}/rgb"][t], d[f"{cam}/depth"][t], d[f"{cam}/seg"][t]
    K, ext = d[f"{cam}/intrinsic"][t], d[f"{cam}/extrinsic"][t]
    ov = overlay(rgb, seg, peg_id, box_id)
    for key, col in (("peg_pose", (255, 255, 0)), ("box_pose", (0, 255, 255)), ("box_hole_pose", (0, 255, 0))):
        u, v = project(d[key][t][:3], ext, K)
        if 0 <= u < rgb.shape[1] and 0 <= v < rgb.shape[0]:
            cv2.drawMarker(ov, (int(u), int(v)), col, cv2.MARKER_CROSS, 18, 2)
    row = np.concatenate([rgb, depth_vis(dep), ov], 1)
    txt = f"{cam} t={t} succ={int(d['success'][t])} peg_px={(seg == peg_id).sum()} box_px={(seg == box_id).sum()}"
    cv2.putText(row, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return row


def geom_check(d, cam, peg_id, box_id):
    """peg 마스크 역투영 포인트가 GT peg 직육면체 안에 있는 비율 + centroid 오차 (프레임별)."""
    hs = d["peg_half_sizes"]
    rows = []
    for t in range(len(d["frame_idx"])):
        dep, seg, K, c2w = (d[f"{cam}/{k}"][t] for k in ("depth", "seg", "intrinsic", "cam2world"))
        r = dict(t=t)
        for name, oid, key in (("peg", peg_id, "peg_pose"), ("box", box_id, "box_pose")):
            pw = to_world(backproject(dep, K, seg == oid), c2w)
            r[f"{name}_n"] = len(pw)
            if len(pw) == 0:
                continue
            r[f"{name}_centroid_err"] = float(np.linalg.norm(pw.mean(0) - d[key][t][:3]))
            if name == "peg":
                Tinv = np.linalg.inv(pose_mat(d[key][t]))
                pl = pw @ Tinv[:3, :3].T + Tinv[:3, 3]
                r["peg_inside_frac"] = float((np.abs(pl) <= hs + 0.003).all(1).mean())  # 3mm tol
                r["peg_max_outside_mm"] = float(np.clip(np.abs(pl) - hs, 0, None).max() * 1000)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/step2")
    ap.add_argument("--out", default="results/step2")
    ap.add_argument("--video-ep", default="ep000")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    report = {}
    for f in sorted(glob.glob(os.path.join(args.data, "ep*.npz"))):
        name = os.path.basename(f)[:-4]
        d = dict(np.load(f))
        peg_id, box_id = int(d["peg_id"]), int(d["box_id"])
        T = len(d["frame_idx"])
        cams = sorted({k.split("/")[0] for k in d if "/" in k})
        # 대표 프레임: 시작, 접근(25%), grasp 후(50%), 정렬(75%), 첫 성공, 마지막
        succ = int(np.argmax(d["success"])) if d["success"].any() else T - 1
        idx = sorted({0, T // 4, T // 2, 3 * T // 4, succ, T - 1})
        for cam in cams:
            sheet = np.concatenate([panel(d, cam, t, peg_id, box_id) for t in idx], 0)
            imageio.imwrite(os.path.join(args.out, f"{name}_{cam}_frames.jpg"),
                            cv2.resize(sheet, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), quality=90)
        # 기하 체크 (base_camera)
        rows = geom_check(d, "base_camera", peg_id, box_id)
        pr = [r for r in rows if r["peg_n"] > 50]
        report[name] = dict(
            frames=T, cams=cams, first_success=succ if d["success"].any() else None,
            image_hw=list(d["base_camera/rgb"].shape[1:3]),
            K_base=d["base_camera/intrinsic"][0].round(2).tolist(),
            base_cam_static=bool(np.allclose(d["base_camera/extrinsic"], d["base_camera/extrinsic"][0], atol=1e-5)),
            depth_valid_frac=float((d["base_camera/depth"] > 0).mean()),
            depth_range_m=[float(d["base_camera/depth"][d["base_camera/depth"] > 0].min()),
                           float(d["base_camera/depth"].max())],
            peg_inside_frac_mean=float(np.mean([r["peg_inside_frac"] for r in pr])),
            peg_inside_frac_min=float(np.min([r["peg_inside_frac"] for r in pr])),
            peg_max_outside_mm=float(np.max([r["peg_max_outside_mm"] for r in pr])),
            peg_centroid_err_mean_m=float(np.mean([r["peg_centroid_err"] for r in pr])),
            box_centroid_err_mean_m=float(np.mean([r["box_centroid_err"] for r in rows if r["box_n"] > 0])),
            peg_px_min=int(min(r["peg_n"] for r in rows)), peg_px_max=int(max(r["peg_n"] for r in rows)),
            box_px_min=int(min(r["box_n"] for r in rows)), box_px_max=int(max(r["box_n"] for r in rows)),
        )
        print(name, json.dumps(report[name]))
        # 가시 픽셀 수 곡선 (base / hand)
        fig, ax = plt.subplots(figsize=(8, 3))
        for cam, ls in zip(cams, ("-", "--")):
            ax.plot([(s == peg_id).sum() for s in d[f"{cam}/seg"]], "r" + ls, label=f"peg ({cam})")
            ax.plot([(s == box_id).sum() for s in d[f"{cam}/seg"]], "b" + ls, label=f"box ({cam})")
        ax.axvline(succ, color="g", lw=1, label="first success")
        ax.set_yscale("symlog")
        ax.set_xlabel("frame")
        ax.set_ylabel("GT mask pixels")
        ax.legend(fontsize=7, ncol=3)
        ax.set_title(f"{name}: visible GT mask pixels (512x512)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"{name}_visibility.png"), dpi=100)
        plt.close(fig)
        # 영상 1개 (gitignore 대상)
        if name == args.video_ep:
            with imageio.get_writer(os.path.join(args.out, f"{name}_overlay.mp4"), fps=20) as w:
                for t in range(T):
                    rows_ = [panel(d, c, t, peg_id, box_id) for c in cams]
                    w.append_data(cv2.resize(np.concatenate(rows_, 0), None, fx=0.5, fy=0.5))
    with open(os.path.join(args.out, "verify.json"), "w") as fp:
        json.dump(report, fp, indent=1)


if __name__ == "__main__":
    main()
