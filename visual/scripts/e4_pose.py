"""E4: peg 6D pose / hole pose 추정 정확도 (pose_v1.py) — mask 입력은 GT 또는 tracker 결과.

입력 mask
  --masks gt          : GT seg (pose fitting 자체의 오차)
  --masks <run dir>   : track_v1 결과 (<run dir>/<ep>_masks.npz)
크기 prior: --size known (task 파라미터 L, r) | est (관측 추정)
smoothing: raw 와 smooth(이상치 게이트 + 위치 지수 평활) 둘 다 기록

지표 (GT pose 와 비교, world)
  peg: 중심 위치 오차 mm, 축 방향 오차 deg (head 방향 포함), head 끝 위치 오차 mm. 추정 실패율.
  hole: 위치 오차 mm, 축 오차 deg. box 가 보이는 프레임 기준. 실패율.
  가림 구간별(peg 의 clean / gripper / insertion) 오차.
  비교 기준선: cloud centroid (v0 출력) 의 GT 중심 대비 오차.
출력: results/v1/E4/<name>.json, 표 md
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_v1 as pv  # noqa: E402
import v1data  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def erode(m, px=2):
    import cv2
    return cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=px) > 0


class Smoother:
    """이상치 게이트(점프 > gate 면 hold, 연속 hold_max 번이면 수용) + 위치 지수 평활."""

    def __init__(self, alpha=0.6, gate=0.03, hold_max=2):
        self.T, self.alpha, self.gate, self.hold, self.hold_max = None, alpha, gate, 0, hold_max

    def __call__(self, T):
        if T is None:
            return self.T
        if self.T is None:
            self.T = T.copy()
            return self.T
        if np.linalg.norm(T[:3, 3] - self.T[:3, 3]) > self.gate and self.hold < self.hold_max:
            self.hold += 1
            return self.T
        self.hold = 0
        out = T.copy()
        out[:3, 3] = self.alpha * T[:3, 3] + (1 - self.alpha) * self.T[:3, 3]
        self.T = out
        return out


def peg_err(T, gt7, L):
    G = pv.T_from7(gt7)
    pos = np.linalg.norm(T[:3, 3] - G[:3, 3]) * 1000
    ang = pv.signed_axis_deg(T[:3, 0], G[:3, 0])
    head = np.linalg.norm((T[:3, 3] + L * T[:3, 0]) - (G[:3, 3] + L * G[:3, 0])) * 1000
    return pos, ang, head


def hole_err(T, gt7):
    G = pv.T_from7(gt7)
    return np.linalg.norm(T[:3, 3] - G[:3, 3]) * 1000, pv.signed_axis_deg(T[:3, 0], G[:3, 0])


def run_ep(d, masks, size, seed=0, fast=False):
    rgb, dep, K, c2w = d["base_camera/rgb"], d["base_camera/depth"], d["base_camera/intrinsic"], d["base_camera/cam2world"]
    T = len(rgb)
    Lgt, rgt = float(d["peg_half_sizes"][0]), float(d["peg_half_sizes"][1])
    pe = pv.PegEstimator(*(Lgt, rgt) if size == "known" else (None, None), seed=seed, fast=fast)
    he = pv.HoleEstimator(*(Lgt, rgt) if size == "known" else (None, None), cam_pos=c2w[0][:3, 3])
    ps, hs = Smoother(), Smoother(alpha=0.3, gate=0.015, hold_max=3)
    rows = []
    for t in range(T):
        r = dict(t=t)
        for obj in ("peg", "box"):
            m = masks[obj][t]
            r[f"{obj}_px"] = int(m.sum())
        mp = erode(masks["peg"][t]) if masks["peg"][t].sum() > 300 else masks["peg"][t]
        P, v, u = pv.backproject_px(dep[t], K[t], mp)
        Pw = pv.to_world(P, c2w[t])
        t0 = time.perf_counter()
        pe.set_view(dep[t], K[t], c2w[t], masks["peg"][t])
        Tp, info = pe(Pw, pv.head_flag(rgb[t][v, u])) if len(Pw) else (None, dict(reason="empty"))
        r["peg_ms"] = (time.perf_counter() - t0) * 1000
        Ts = ps(Tp)
        L = pe.L if pe.L is not None else Lgt
        if Tp is not None:
            r["peg"] = peg_err(Tp, d["peg_pose"][t], L)
            r["peg_rms_mm"] = info["rms_mm"]
        if Ts is not None:
            r["peg_s"] = peg_err(Ts, d["peg_pose"][t], L)
        if len(Pw):
            r["centroid_mm"] = float(np.linalg.norm(Pw.mean(0) - d["peg_pose"][t][:3]) * 1000)
        mb = erode(masks["box"][t]) if masks["box"][t].sum() > 300 else masks["box"][t]
        Pb, _, _ = pv.backproject_px(dep[t], K[t], mb)
        Pbw = pv.to_world(Pb, c2w[t])
        t0 = time.perf_counter()
        Th, hinfo = he(Pbw, dep[t], K[t], c2w[t], masks["box"][t]) if len(Pbw) >= 200 else (None, dict(reason="empty"))
        r["hole_ms"] = (time.perf_counter() - t0) * 1000
        Ths = hs(Th)
        if Th is not None:
            r["hole"] = hole_err(Th, d["box_hole_pose"][t])
            r["hole_half_mm"] = hinfo["hole_half_mm"]
        else:
            r["hole_fail"] = hinfo.get("reason")
        if Ths is not None:
            r["hole_s"] = hole_err(Ths, d["box_hole_pose"][t])
        rows.append(r)
    return rows, dict(L_est=pe.L, r_est=pe.r, L_gt=Lgt, r_gt=rgt)


def agg(vals):
    v = np.asarray(vals, float)
    if not len(v):
        return None
    return dict(n=len(v), mean=round(float(v.mean()), 2), median=round(float(np.median(v)), 2),
                p90=round(float(np.percentile(v, 90)), 2), max=round(float(v.max()), 1))


def summarize(all_rows, all_ph):
    S = {}
    for key in ("peg", "peg_s"):
        for j, nm in enumerate(("pos_mm", "axis_deg", "head_mm")):
            S[f"{key}_{nm}"] = agg([r[key][j] for rows in all_rows.values() for r in rows if key in r and r["peg_vis"]])
            for ph in ("clean", "gripper", "insertion"):
                S[f"{key}_{nm}_{ph}"] = agg([r[key][j] for ep, rows in all_rows.items() for r in rows
                                             if key in r and r["peg_vis"] and all_ph[ep][ph][r["t"]]])
    S["peg_fail_rate"] = round(float(np.mean([("peg" not in r) for rows in all_rows.values() for r in rows if r["peg_vis"]])), 4)
    S["centroid_mm"] = agg([r["centroid_mm"] for rows in all_rows.values() for r in rows if "centroid_mm" in r and r["peg_vis"]])
    for key in ("hole", "hole_s"):
        for j, nm in enumerate(("pos_mm", "axis_deg")):
            S[f"{key}_{nm}"] = agg([r[key][j] for rows in all_rows.values() for r in rows if key in r and r["box_vis"]])
    S["hole_fail_rate"] = round(float(np.mean([("hole" not in r) for rows in all_rows.values() for r in rows if r["box_vis"]])), 4)
    S["peg_ms"] = agg([r["peg_ms"] for rows in all_rows.values() for r in rows if "peg" in r])
    S["hole_ms"] = agg([r["hole_ms"] for rows in all_rows.values() for r in rows if "hole" in r])
    # 에피소드별 hole: 첫 20 프레임 중앙값 (box 정지 → 1회 추정으로 충분한지)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev")
    ap.add_argument("--masks", default="gt")
    ap.add_argument("--size", default="known", choices=["known", "est"])
    ap.add_argument("--eps", nargs="*", default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fast", action="store_true", help="이전 자세 refine 우선 (tracking 경로)")
    ap.add_argument("--out", default=f"{ROOT}/results/v1/E4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from eval_v1 import load_masks  # SAM2 를 import 하므로 여기서 (e7 은 SAM2 없는 venv 에서 erode/peg_err 만 씀)
    from track_sam2 import phase_labels
    eps = args.eps or v1data.ep_list(args.set)
    all_rows, all_ph, sizes = {}, {}, {}
    for ep in eps:
        d = v1data.load(args.set, ep)
        seg = d["base_camera/seg"]
        if args.masks == "gt":
            masks = {o: seg == int(d[f"{o}_id"]) for o in ("peg", "box")}
        else:
            masks, _, _ = load_masks(f"{args.masks}/{ep}_masks.npz")
        rows, sz = run_ep(d, masks, args.size, fast=args.fast)
        Tn = len(seg)
        gpx = {o: (seg == int(d[f"{o}_id"])).reshape(Tn, -1).sum(1) for o in ("peg", "box")}
        for r in rows:
            r["peg_vis"] = bool(gpx["peg"][r["t"]] >= 200)
            r["box_vis"] = bool(gpx["box"][r["t"]] >= 200)
        ph = phase_labels(d, int(d["peg_id"]))
        all_ph[ep] = {"clean": ~(ph["arm_hidden"] | ph["gripper"] | ph["insertion"]), "gripper": ph["gripper"] & ~ph["insertion"],
                      "insertion": ph["insertion"]}
        all_rows[ep] = rows
        sizes[ep] = sz
        pe = [r["peg"][0] for r in rows if "peg" in r and r["peg_vis"]]
        he = [r["hole"][0] for r in rows if "hole" in r and r["box_vis"]]
        print(ep, f"peg pos med {np.median(pe) if pe else -1:.1f} mm, hole pos med {np.median(he) if he else -1:.1f} mm", sz, flush=True)
    S = summarize(all_rows, all_ph)
    S["sizes"] = sizes
    S["args"] = vars(args)
    json.dump(dict(summary=S, rows={ep: rows for ep, rows in all_rows.items()}), open(f"{args.out}/{args.name}.json", "w"),
              default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    print(json.dumps({k: v for k, v in S.items() if k not in ("sizes",)}, indent=0)[:3000])


if __name__ == "__main__":
    main()
