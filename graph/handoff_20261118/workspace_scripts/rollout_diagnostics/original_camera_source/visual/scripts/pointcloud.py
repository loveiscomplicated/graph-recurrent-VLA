"""Step 5: SAM2 mask + depth + intrinsic/extrinsic → 물체별 point cloud (+ GT 대비 평가, 3D 렌더링).

처리 (프레임 × 물체마다):
  1. mask 경계 erosion (ERODE_PX, depth 불연속 경계의 flying pixel 제거). erosion 후 MIN_PTS 미만이면 원래 mask 사용
  2. depth > 0 픽셀 역투영 (pixel center 규약 (u+0.5-cx), Step 2와 동일) → 카메라 좌표계 (CV: x→, y↓, z 전방)
  3. FPS_PRE 점으로 random 선다운샘플 (이후 단계 시간 상한; 512² 에서 물체당 역투영 점은 최대 ~1.2만)
  4. statistical outlier 제거 (k=SOR_K 이웃 평균거리 > mean + SOR_STD·std 제거, scipy cKDTree)
  5. 떨어진 작은 cluster 제거 (반경 CLUSTER_R 연결 성분 중 최대 cluster 의 CLUSTER_KEEP 배 미만이면서
     최대 cluster 와 CLUSTER_GAP 이상 떨어진 것). **기본 OFF, --cluster 로 켬**:
     SOR 은 '덩어리째 섞여 들어온 점'(mask 가 gripper finger·테이블로 번진 부분)을 못 지움.
     하지만 팔에 가려 쪼개진 box 의 실제 조각도 지워서 전체 평균 오차는 OFF 가 더 낮음 (REPORT.md Step 5 ablation)
  6. 최대 MAX_PTS 점으로 FPS 다운샘플
  7. cam2world 로 points_world 도 저장

출력 (출력 포맷 초안): results/step5/<ep>/f{frame:04d}_{object}.npz
  points (N,3) f32 [cam], points_world (N,3) f32, score (N,) f32 = 1.0, object str, frame_idx, timestamp,
  tracking_confidence = sigmoid(SAM2 object_score_logit), + camera, intrinsic, cam2world, n_mask_px, n_raw_pts
  mask 가 비면 N=0 인 npz 를 그대로 저장 (Graph 쪽에서 confidence 로 판단)

평가 (GT 가시 px ≥ VIS_MIN 이고 예측 point cloud 가 비지 않은 프레임, world 좌표):
  err_gtcloud : 예측 cloud centroid ↔ GT mask 전체 역투영 cloud centroid (주 지표; 보이는 표면끼리 비교)
  err_pose    : 예측 cloud centroid ↔ GT pose 원점 (참고; 보이는 표면만 찍히므로 수 cm bias 가 정상)
  err_pose_gt : GT mask cloud centroid ↔ GT pose 원점 (위 bias 의 기준값)
  peg_inside  : 예측 peg 점 중 GT pose 의 peg 직육면체(half_sizes + 3 mm) 안에 있는 비율

Usage:
  source env.sh
  python scripts/pointcloud.py --masks results/step4/small --out results/step5
"""
import argparse
import json
import os
import time

import cv2
import matplotlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAM = "base_camera"
OBJECTS = ("peg", "box")
COLORS = {"peg": np.array([255, 40, 40]), "box": np.array([40, 90, 255])}
VIS_MIN = 200  # Step 3/4 와 동일
ERODE_PX = 2
MIN_PTS = 50
SOR_K = 16
SOR_STD = 2.0
MAX_PTS = 1024
FPS_PRE = 2048
CLUSTER_R = 0.015  # m
CLUSTER_KEEP = 0.2
CLUSTER_GAP = 0.05  # m
INSIDE_TOL = 0.003  # m


def pose_mat(p):
    """[x y z qw qx qy qz] -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def backproject(depth, K, mask):
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u]
    x = (u + 0.5 - K[0, 2]) * z / K[0, 0]
    y = (v + 0.5 - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1).astype(np.float32)


def to_world(pts, c2w):
    return (pts @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)


def sor_filter(pts, k=SOR_K, std_ratio=SOR_STD):
    if len(pts) <= k:
        return pts
    d, _ = cKDTree(pts).query(pts, k=k + 1)
    md = d[:, 1:].mean(1)
    return pts[md <= md.mean() + std_ratio * md.std()]


def drop_small_clusters(pts, r=CLUSTER_R, keep=CLUSTER_KEEP, gap=CLUSTER_GAP):
    if len(pts) < 2:
        return pts
    pairs = cKDTree(pts).query_pairs(r, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), bool), (pairs[:, 0], pairs[:, 1])), shape=(len(pts),) * 2)
    _, lab = connected_components(g, directed=False)
    cnt = np.bincount(lab)
    main = cnt.argmax()
    drop = np.zeros(len(cnt), bool)
    small = np.nonzero(cnt < keep * cnt[main])[0]
    if len(small):
        dist, _ = cKDTree(pts[lab == main]).query(pts)
        for c in small:
            drop[c] = dist[lab == c].min() > gap
    return pts[~drop[lab]]


def fps(pts, n, rng):
    if len(pts) <= n:
        return pts
    x, y, z = (np.ascontiguousarray(pts[:, i]) for i in range(3))
    idx = np.empty(n, np.int64)
    idx[0] = rng.integers(len(pts))
    dist = np.full(len(pts), np.inf, np.float32)
    for i in range(1, n):
        j = idx[i - 1]
        np.minimum(dist, (x - x[j]) ** 2 + (y - y[j]) ** 2 + (z - z[j]) ** 2, out=dist)
        idx[i] = dist.argmax()
    return pts[idx]


def extract(mask, depth, K, c2w, rng, erode_px=ERODE_PX, max_pts=MAX_PTS, cluster=False):
    """mask (H,W) bool → (points_cam, points_world, info). 비어 있으면 (0,3)."""
    info = dict(n_mask_px=int(mask.sum()))
    m = mask
    if erode_px > 0 and info["n_mask_px"]:
        er = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=erode_px) > 0
        if (er & (depth > 0)).sum() >= MIN_PTS:
            m = er
    pts = backproject(depth, K, m)
    info["n_raw_pts"] = len(pts)
    if len(pts) > FPS_PRE:
        pts = pts[rng.choice(len(pts), FPS_PRE, replace=False)]
    pts = sor_filter(pts)
    info["n_sor_pts"] = len(pts)
    if cluster:
        n = len(pts)
        pts = drop_small_clusters(pts)
        info["cluster_removed_frac"] = round(1 - len(pts) / n, 4) if n else 0.0
    pts = fps(pts, max_pts, rng)
    return pts, to_world(pts, c2w), info


def sigmoid(x):
    return float(1 / (1 + np.exp(-x))) if np.isfinite(x) else 0.0


def save_npz(path, obj, t, d, pc, pw, conf, info):
    np.savez_compressed(
        path, points=pc.astype(np.float32), points_world=pw.astype(np.float32),
        score=np.ones(len(pc), np.float32), object=np.array(obj), frame_idx=np.int64(t),
        timestamp=np.float64(d["timestamp"][t]), tracking_confidence=np.float32(conf), camera=np.array(CAM),
        intrinsic=d[f"{CAM}/intrinsic"][t].astype(np.float32), cam2world=d[f"{CAM}/cam2world"][t].astype(np.float32),
        n_mask_px=np.int64(info["n_mask_px"]), n_raw_pts=np.int64(info.get("n_raw_pts", 0)))


def evaluate_frame(d, t, obj, pw, gt_id):
    """GT 대비 오차 (world). GT 가 안 보이거나 예측이 비면 None 항목."""
    seg, dep = d[f"{CAM}/seg"][t], d[f"{CAM}/depth"][t]
    K, c2w = d[f"{CAM}/intrinsic"][t], d[f"{CAM}/cam2world"][t]
    gt_mask = seg == gt_id
    r = dict(gt_px=int(gt_mask.sum()), n=len(pw))
    pose = d[f"{obj}_pose"][t]
    if r["gt_px"] >= VIS_MIN:
        gw = to_world(backproject(dep, K, gt_mask), c2w)
        r["err_pose_gt"] = float(np.linalg.norm(gw.mean(0) - pose[:3]))
        if len(pw):
            r["err_gtcloud"] = float(np.linalg.norm(pw.mean(0) - gw.mean(0)))
            r["err_pose"] = float(np.linalg.norm(pw.mean(0) - pose[:3]))
    if obj == "peg" and len(pw):
        Tinv = np.linalg.inv(pose_mat(pose))
        pl = pw @ Tinv[:3, :3].T + Tinv[:3, 3]
        r["peg_inside"] = float((np.abs(pl) <= d["peg_half_sizes"] + INSIDE_TOL).all(1).mean())
    return r


def cuboid_edges(pose, hs):
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * hs
    T = pose_mat(pose)
    c = c @ T[:3, :3].T + T[:3, 3]
    return [(c[i], c[j]) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]


def render_3d(ax, clouds, gt_clouds, d, t, view, lim):
    """clouds/gt_clouds: {obj: (N,3) world}. GT 는 회색 점, 예측은 물체 색, GT peg cuboid 는 초록 선."""
    for obj, g in gt_clouds.items():
        if len(g):
            g = g[:: max(1, len(g) // 1500)]
            ax.scatter(*g.T, s=0.3, c="0.6", alpha=0.4)
    for obj, p in clouds.items():
        if len(p):
            ax.scatter(*p.T, s=1.5, c=[COLORS[obj] / 255], depthshade=False)
            ax.scatter(*p.mean(0), s=60, c="k", marker="x")
    for a, b in cuboid_edges(d["peg_pose"][t], d["peg_half_sizes"]):
        ax.plot(*np.stack([a, b]).T, c="g", lw=0.8)
    for key, mk in (("peg_pose", "o"), ("box_pose", "s")):
        ax.scatter(*d[key][t][:3], s=40, facecolors="none", edgecolors="g", marker=mk)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])  # noqa: E702
    ax.set_box_aspect(np.ptp(lim, 1))
    ax.view_init(*view)
    ax.tick_params(labelsize=5)


def scene_limits(d):
    c = np.concatenate([d["peg_pose"][:, :3], d["box_pose"][:, :3]])
    lo, hi = c.min(0) - 0.15, c.max(0) + 0.15
    lo[2] = max(lo[2], -0.02)
    return np.stack([lo, hi], 1)


def mask_overlay(rgb, masks, a=0.5):
    out = rgb.astype(np.float32).copy()
    for name, m in masks.items():
        out[m] = (1 - a) * out[m] + a * COLORS[name]
    return out.astype(np.uint8)


def render_frame(d, t, masks, clouds, conf, lim, title=""):
    """RGB+mask | 3D oblique | 3D top-down → (H,W,3) uint8."""
    seg, dep = d[f"{CAM}/seg"][t], d[f"{CAM}/depth"][t]
    K, c2w = d[f"{CAM}/intrinsic"][t], d[f"{CAM}/cam2world"][t]
    ids = {"peg": int(d["peg_id"]), "box": int(d["box_id"])}
    gt_clouds = {o: to_world(backproject(dep, K, seg == ids[o]), c2w) for o in OBJECTS}
    fig = plt.figure(figsize=(12, 4), dpi=90)
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.imshow(mask_overlay(d[f"{CAM}/rgb"][t], {o: masks[o][t] for o in OBJECTS}))
    ax0.set_title(f"{title} t={t}  conf peg {conf['peg']:.2f} box {conf['box']:.2f}", fontsize=8)
    ax0.axis("off")
    for i, (view, name) in enumerate((((25, -150), "oblique"), ((89, -90), "top-down"))):
        ax = fig.add_subplot(1, 3, 2 + i, projection="3d")
        render_3d(ax, clouds, gt_clouds, d, t, view, lim)
        ax.set_title(f"world, {name}: pred peg(red)/box(blue), GT gray, peg GT cuboid green, x=centroid",
                     fontsize=6)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def stats(x):
    x = np.asarray([v for v in x if v is not None], float)
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), mean=round(float(x.mean()), 4), median=round(float(np.median(x)), 4),
                p95=round(float(np.percentile(x, 95)), 4), max=round(float(x.max()), 4))


def process_episode(d, masks, logits, out_dir, rng, cluster=False):
    """masks {obj: (T,H,W)}, logits {obj: (T,)} → npz 저장, per-frame 평가, timing. returns (rows, clouds, conf, ms)."""
    os.makedirs(out_dir, exist_ok=True)
    T = len(d["frame_idx"])
    ids = {"peg": int(d["peg_id"]), "box": int(d["box_id"])}
    rows, clouds, confs, ms = [], [], [], []
    for t in range(T):
        dep, K, c2w = d[f"{CAM}/depth"][t], d[f"{CAM}/intrinsic"][t], d[f"{CAM}/cam2world"][t]
        row, cl, cf = dict(t=t), {}, {}
        for obj in OBJECTS:
            t0 = time.perf_counter()
            pc, pw, info = extract(masks[obj][t], dep, K, c2w, rng, cluster=cluster)
            ms.append((time.perf_counter() - t0) * 1000)
            cf[obj] = sigmoid(logits[obj][t])
            save_npz(f"{out_dir}/f{t:04d}_{obj}.npz", obj, t, d, pc, pw, cf[obj], info)
            row[obj] = {**info, **evaluate_frame(d, t, obj, pw, ids[obj]), "conf": round(cf[obj], 3)}
            cl[obj] = pw
        rows.append(row)
        clouds.append(cl)
        confs.append(cf)
    return rows, clouds, confs, ms


def summarize(rows):
    s = {}
    for obj in OBJECTS:
        r = [x[obj] for x in rows]
        vis = [x for x in r if x["gt_px"] >= VIS_MIN]
        s[obj] = dict(
            frames=len(r), gt_visible=len(vis),
            empty_when_visible=sum(x["n"] == 0 for x in vis),
            nonempty_when_hidden=sum(x["n"] > 0 and x["gt_px"] < VIS_MIN for x in r),
            n_points=stats([x["n"] for x in r if x["n"]]),
            err_gtcloud_m=stats([x.get("err_gtcloud") for x in vis]),
            err_pose_m=stats([x.get("err_pose") for x in vis]),
            err_pose_gt_m=stats([x.get("err_pose_gt") for x in vis]),
        )
        if obj == "peg":
            s[obj]["peg_inside"] = stats([x.get("peg_inside") for x in vis])
    return s


def error_plot(all_rows, path):
    fig, axes = plt.subplots(len(all_rows), 1, figsize=(12, 3 * len(all_rows)), squeeze=False)
    for ax, (ep, rows) in zip(axes[:, 0], all_rows.items()):
        x = np.arange(len(rows))
        for obj, c in (("peg", "r"), ("box", "b")):
            g = lambda k: np.array([r[obj].get(k, np.nan) for r in rows], float)  # noqa: E731
            ax.plot(x, g("err_gtcloud") * 100, c + ".-", ms=3, label=f"{obj} centroid vs GT-mask cloud")
            ax.plot(x, g("err_pose") * 100, c + ":", lw=1, label=f"{obj} centroid vs GT pose")
            ax.plot(x, g("err_pose_gt") * 100, c + "--", lw=0.7, alpha=0.5, label=f"{obj} GT-mask cloud vs GT pose")
        ax.set_ylabel("error [cm]")
        ax.set_ylim(0, 20)
        ax2 = ax.twinx()
        ax2.plot(x, [r["peg"].get("peg_inside", np.nan) for r in rows], "g-", lw=1, label="peg inside GT cuboid")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("peg inside frac")
        ax.set_title(ep)
        if ax is axes[0, 0]:
            ax.legend(fontsize=6, ncol=3, loc="upper left")
            ax2.legend(fontsize=6, loc="upper right")
    axes[-1, 0].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def pick_frames(d, rows):
    """대표 프레임: 시작 / grasp 후 / peg centroid 오차 최대 / 마지막."""
    T = len(rows)
    errs = np.array([r["peg"].get("err_gtcloud", -1) for r in rows])
    cand = [0, T // 3, int(errs.argmax()) if errs.max() > 0 else T // 2, T - 1]
    return sorted(set(cand))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{ROOT}/data/step2")
    ap.add_argument("--masks", default=f"{ROOT}/results/step4/small")
    ap.add_argument("--eps", nargs="+", default=["ep000", "ep001", "ep002"])
    ap.add_argument("--out", default=f"{ROOT}/results/step5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt-masks", action="store_true",
                    help="SAM2 mask 대신 GT seg mask 입력 (처리 단계 자체의 오차 분리용, confidence=1)")
    ap.add_argument("--cluster", action="store_true", help="떨어진 작은 cluster 제거 켜기 (ablation)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    result = dict(camera=CAM, masks="GT seg" if args.gt_masks else args.masks,
                  params=dict(erode_px=ERODE_PX, sor_k=SOR_K, sor_std=SOR_STD, max_pts=MAX_PTS, fps_pre=FPS_PRE,
                              vis_min_px=VIS_MIN, inside_tol_m=INSIDE_TOL,
                              cluster_r=CLUSTER_R if args.cluster else None, cluster_keep=CLUSTER_KEEP,
                              cluster_gap=CLUSTER_GAP),
                  episodes={})
    all_rows, all_ms = {}, []
    for ep in args.eps:
        d = dict(np.load(f"{args.data}/{ep}.npz"))
        if args.gt_masks:
            masks = {o: d[f"{CAM}/seg"] == int(d[f"{o}_id"]) for o in OBJECTS}
            logits = {o: np.full(len(d["frame_idx"]), 20.0) for o in OBJECTS}
        else:
            m = np.load(f"{args.masks}/{ep}_masks.npz")
            masks = {o: m[o] for o in OBJECTS}
            logits = {o: m[f"{o}_logit"] for o in OBJECTS}
        rows, clouds, confs, ms = process_episode(d, masks, logits, f"{args.out}/{ep}", rng,
                                                 cluster=args.cluster)
        all_rows[ep], all_ms = rows, all_ms + ms
        s = summarize(rows)
        s["extract_ms"] = dict(mean=round(float(np.mean(ms)), 1), p95=round(float(np.percentile(ms, 95)), 1))
        result["episodes"][ep] = s
        print(ep, json.dumps({o: {k: s[o][k] for k in ("err_gtcloud_m", "err_pose_m")} for o in OBJECTS}),
              s["extract_ms"])
        lim = scene_limits(d)
        imgs = [render_frame(d, t, masks, clouds[t], confs[t], lim, ep) for t in pick_frames(d, rows)]
        cv2.imwrite(f"{args.out}/{ep}_pointcloud.jpg", cv2.cvtColor(np.concatenate(imgs, 0), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        json.dump(rows, open(f"{args.out}/{ep}_frames.json", "w"))
    error_plot(all_rows, f"{args.out}/centroid_error.png")
    result["all"] = summarize([r for rows in all_rows.values() for r in rows])
    result["all"]["extract_ms_per_object"] = dict(mean=round(float(np.mean(all_ms)), 1),
                                                  p95=round(float(np.percentile(all_ms, 95)), 1))
    json.dump(result, open(f"{args.out}/pointcloud.json", "w"), indent=1)
    print("ALL", json.dumps(result["all"]))


if __name__ == "__main__":
    main()
