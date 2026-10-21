"""E9: Graph/Controller 팀용 Vision 출력 export (GT 와 같은 시간축).

입력: tracker 결과(track_v1 형식 masks npz) + v1 데이터. 출력: data/v1/export/<set>/<ep>/f{t:04d}.npz (git 제외), 에피소드마다 meta.json.
프레임당 npz 1 개, 물체 o ∈ {peg, box}:
  {o}_points_world (1024,3) f32 : world 좌표 (모자라면 0 padding)
  {o}_points_ee    (1024,3) f32 : EE(TCP) 기준 상대 좌표 = inv(T_tcp) · p
  {o}_valid_pts    (1024,)  bool: padding 이 아닌 점
  {o}_relevance    (1024,)  f32 : task relevance (E6 전까지 1.0)
  {o}_valid        bool          : 이번 프레임 관측이 유효 (SAM2 logit > 0 이고 점 ≥ 50). 가려지면 False + 빈 cloud
  {o}_age_steps    int           : 마지막 유효 관측 이후 지난 step 수 (유효하면 0)
  {o}_confidence   f32           : sigmoid(SAM2 object_score_logit)
  peg_pose, hole_pose (7,) [x y z qw qx qy qz] world + *_pose_valid, *_pose_age_steps : pose_v1 추정 (없으면 마지막 값 유지 + age)
            peg 는 보이는 축 길이 ≥ 0.6 L 일 때만 유효 (화면 가장자리에 조금만 보이면 불확정), 그리고 시간 게이트: 직전 값에서 3 cm 넘게 튄 추정은 3 프레임 연속일 때만 수용
  tcp_pose (7,), frame_idx, timestamp
  gt_peg_pose, gt_box_hole_pose, gt_tcp_pose (7,) : 같은 시간축 GT (평가/학습 비교용, 모델 입력 아님)
Usage: python scripts/e9_export.py --run results/v1/E1/runs/sam2s_c_conf_e5/eval --set eval
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pointcloud as pcl  # noqa: E402
import pose_v1 as pv  # noqa: E402
import v1data  # noqa: E402
from e4_pose import erode  # noqa: E402
from eval_v1 import load_masks  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N = 1024


def pad(p, n=N):
    out = np.zeros((n, 3), np.float32)
    v = np.zeros(n, bool)
    out[:len(p)] = p[:n]
    v[:len(p)] = True
    return out, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", default="eval")
    ap.add_argument("--eps", nargs="*", default=None)
    ap.add_argument("--out", default=f"{ROOT}/data/v1/export")
    args = ap.parse_args()
    eps = args.eps or v1data.ep_list(args.set)
    for ep in eps:
        d = v1data.load(args.set, ep)
        masks, logits, _ = load_masks(f"{args.run}/{ep}_masks.npz")
        rgb, dep, K, c2w = (d[f"base_camera/{k}"] for k in ("rgb", "depth", "intrinsic", "cam2world"))
        L, r = float(d["peg_half_sizes"][0]), float(d["peg_half_sizes"][1])
        pe, he = pv.PegEstimator(L, r), pv.HoleEstimator(L, r)
        od = f"{args.out}/{args.set}/{ep}"
        os.makedirs(od, exist_ok=True)
        rng = np.random.default_rng(0)
        age = {"peg": 0, "box": 0}
        last = {"peg_pose": None, "hole_pose": None}
        page = {"peg_pose": 0, "hole_pose": 0}
        best_n = 0
        cand = []
        stats = dict(valid={"peg": 0, "box": 0}, pose_valid={"peg_pose": 0, "hole_pose": 0})
        T = len(rgb)
        for t in range(T):
            Ttcp = pv.T_from7(d["tcp_pose"][t])
            inv = np.linalg.inv(Ttcp)
            f = dict(frame_idx=int(d["frame_idx"][t]), timestamp=float(d["timestamp"][t]), tcp_pose=d["tcp_pose"][t].astype(np.float32),
                     gt_peg_pose=d["peg_pose"][t].astype(np.float32), gt_box_hole_pose=d["box_hole_pose"][t].astype(np.float32),
                     gt_tcp_pose=d["tcp_pose"][t].astype(np.float32))
            for o in ("peg", "box"):
                lg = logits[o][t]
                conf = float(1 / (1 + np.exp(-lg))) if np.isfinite(lg) else 0.0
                pts = np.zeros((0, 3), np.float32)
                if np.isfinite(lg) and lg > 0 and masks[o][t].sum() > 0:
                    _, pts, _ = pcl.extract(masks[o][t], dep[t], K[t], c2w[t], rng)
                valid = len(pts) >= 50
                age[o] = 0 if valid else age[o] + 1
                stats["valid"][o] += int(valid)
                pw, vp = pad(pts if valid else np.zeros((0, 3)))
                pe_ = (pw @ inv[:3, :3].T + inv[:3, 3]).astype(np.float32)
                pe_[~vp] = 0
                f.update({f"{o}_points_world": pw, f"{o}_points_ee": pe_, f"{o}_valid_pts": vp, f"{o}_relevance": vp.astype(np.float32),
                          f"{o}_valid": valid, f"{o}_age_steps": age[o], f"{o}_confidence": np.float32(conf)})
            # pose 추정
            new = {}
            if f["peg_valid"]:
                m = masks["peg"][t]
                P, v, u = pv.backproject_px(dep[t], K[t], erode(m) if m.sum() > 300 else m)
                if len(P) >= 30:
                    pe.set_view(dep[t], K[t], c2w[t], m)
                    Tp, pinfo = pe(pv.to_world(P, c2w[t]), pv.head_flag(rgb[t][v, u]))
                    if Tp is not None and pinfo["ok"]:  # 보이는 길이 < 0.6 L 이면 pose 불확정 → 무효
                        new["peg_pose"] = Tp
            if f["box_valid"] and masks["box"][t].sum() >= 200:
                P, _, _ = pv.backproject_px(dep[t], K[t], masks["box"][t])
                Th, hi = he(pv.to_world(P, c2w[t]), dep[t], K[t], c2w[t], masks["box"][t])
                if Th is not None and hi["n_plane"] >= 0.9 * best_n:  # 가장 덜 가려진 관측 유지 (E5 와 같은 규칙)
                    best_n = max(best_n, hi["n_plane"])
                    new["hole_pose"] = Th
            # 시간 게이트: 직전 유효값에서 3 cm 넘게 튄 peg 추정은 3 프레임 연속(서로 2 cm 이내)일 때만 수용
            if "peg_pose" in new and last["peg_pose"] is not None and np.linalg.norm(new["peg_pose"][:3, 3] - last["peg_pose"][:3, 3]) > 0.03:
                cand.append(new.pop("peg_pose"))
                if len(cand) >= 3 and all(np.linalg.norm(c[:3, 3] - cand[-1][:3, 3]) < 0.02 for c in cand[-3:]):
                    new["peg_pose"] = cand[-1]
                    cand.clear()
            elif "peg_pose" in new:
                cand.clear()
            for k in ("peg_pose", "hole_pose"):
                if k in new:
                    last[k], page[k] = new[k], 0
                    stats["pose_valid"][k] += 1
                else:
                    page[k] += 1
                f[k] = pv.pose7(last[k]).astype(np.float32) if last[k] is not None else np.zeros(7, np.float32)
                f[f"{k}_valid"] = k in new
                f[f"{k}_age_steps"] = page[k] if last[k] is not None else t + 1
            np.savez_compressed(f"{od}/f{t:04d}.npz", **f)
        json.dump(dict(ep=ep, set=args.set, run=args.run, frames=T, **stats), open(f"{od}/meta.json", "w"), indent=1)
        print(ep, stats, flush=True)


if __name__ == "__main__":
    main()
