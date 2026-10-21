"""E6: point 별 task relevance — GT 라벨 대비 AUROC / AP (E9 export 를 입력으로).

GT 라벨 (평가 전용, GT pose 사용)
  peg: 삽입부 = peg 로컬 x > L − 3 cm (head 끝), 또는 잡는 부분 = TCP 에서 peg 축으로 2 cm 이내 (grasp 이후)
  box: hole 입구 주변 = box 면(hole 입구 평면, |x_hole + L| < 1 cm) 위에서 hole 중심까지 거리 < hole 반폭 + 1.5 cm
후보 점수 (모델 경로: export 의 추정 pose / 점 / TCP 만 사용)
  const   : 전부 1.0 (현재 v0/v1 출력) → AUROC 0.5
  geom    : 같은 규칙을 추정 peg_pose / hole_pose 로 계산 (부드러운 sigmoid)
  dist_tcp: peg 는 TCP 와의 거리 (가까울수록 높음), box 는 추정 hole 중심과의 거리
Usage: python scripts/e6_relevance.py --export data/v1/export/eval
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_v1 as pv  # noqa: E402
import v1data  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def auroc(score, y):
    y = y.astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort") + 1
    # 동점 평균 순위
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ap(score, y):
    o = np.argsort(-score)
    y = y[o].astype(float)
    if y.sum() == 0:
        return np.nan
    prec = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float((prec * y).sum() / y.sum())


def sig(x, s=0.005):
    return 1 / (1 + np.exp(-x / s))


def labels_scores(P, Tpeg, Thole, Ttcp, L, r, kind, grasped):
    """P world points → label / score 계산에 쓰는 공통 함수 (Tpeg/Thole 은 GT 또는 추정)."""
    if kind == "peg":
        q = (P - Tpeg[:3, 3]) @ Tpeg[:3, :3]
        head = q[:, 0] - (L - 0.03)
        a = Tpeg[:3, 0]
        g = Ttcp[:3, 3] - Tpeg[:3, 3]
        xg = g @ a
        grasp = 0.02 - np.abs(q[:, 0] - xg) if grasped else np.full(len(P), -1.0)
        return np.maximum(head, grasp)
    q = (P - Thole[:3, 3]) @ Thole[:3, :3]
    face = 0.01 - np.abs(q[:, 0] + L)
    rad = (r + 0.003 + 0.015) - np.linalg.norm(q[:, 1:], axis=1)
    return np.minimum(face, rad)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--export", default=f"{ROOT}/data/v1/export/eval")
    ap_.add_argument("--set", default="eval")
    ap_.add_argument("--stride", type=int, default=3)
    ap_.add_argument("--out", default=f"{ROOT}/results/v1/E6")
    a = ap_.parse_args()
    os.makedirs(a.out, exist_ok=True)
    acc = {o: {k: [] for k in ("const", "geom", "dist")} for o in ("peg", "box")}
    per = {o: {k: [] for k in ("geom", "dist")} for o in ("peg", "box")}
    for epd in sorted(glob.glob(f"{a.export}/ep*")):
        ep = os.path.basename(epd)
        z0 = np.load(f"{v1data.DATA}/{a.set}/{ep}.npz")
        L, r = float(z0["peg_half_sizes"][0]), float(z0["peg_half_sizes"][1])
        fs = sorted(glob.glob(f"{epd}/f*.npz"))[::a.stride]
        for f in fs:
            z = np.load(f)
            Ttcp = pv.T_from7(z["tcp_pose"])
            grasped = float(np.linalg.norm(z["gt_tcp_pose"][:3] - z["gt_peg_pose"][:3])) < L + 0.02
            for o in ("peg", "box"):
                if not z[f"{o}_valid"]:
                    continue
                P = z[f"{o}_points_world"][z[f"{o}_valid_pts"]]
                y = labels_scores(P, pv.T_from7(z["gt_peg_pose"]), pv.T_from7(z["gt_box_hole_pose"]), Ttcp, L, r, o, grasped) > 0
                key = "peg_pose" if o == "peg" else "hole_pose"
                if z[f"{key}_age_steps"] > 100 or not z[key].any():
                    continue
                Tp_e = pv.T_from7(z["peg_pose"]) if z["peg_pose"].any() else np.eye(4)  # 아직 추정 없음 → 안 쓰이는 쪽만 단위행렬
                Th_e = pv.T_from7(z["hole_pose"]) if z["hole_pose"].any() else np.eye(4)
                g = sig(labels_scores(P, Tp_e, Th_e, Ttcp, L, r, o, grasped))
                ref = Ttcp[:3, 3] if o == "peg" else Th_e[:3, 3]
                dsc = -np.linalg.norm(P - ref, axis=1)
                if y.any() and (~y).any():
                    per[o]["geom"].append(auroc(g, y))
                    per[o]["dist"].append(auroc(dsc, y))
                acc[o]["const"].append((np.ones(len(P)), y))
                acc[o]["geom"].append((g, y))
                acc[o]["dist"].append((dsc, y))
    res = {}
    for o in ("peg", "box"):
        res[o] = {}
        for k, lst in acc[o].items():
            if not lst:
                continue
            s = np.concatenate([x[0] for x in lst])
            y = np.concatenate([x[1] for x in lst])
            res[o][k] = dict(auroc_pooled=round(auroc(s, y), 4), ap_pooled=round(ap(s, y), 4), pos_rate=round(float(y.mean()), 4), n_pts=int(len(y)))
            if k in per[o] and per[o][k]:
                v = np.array(per[o][k])
                res[o][k]["auroc_frame_median"] = round(float(np.nanmedian(v)), 4)
        print(o, json.dumps(res[o]))
    json.dump(res, open(f"{a.out}/relevance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
