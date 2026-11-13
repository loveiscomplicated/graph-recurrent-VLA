"""E7: RGB-only depth (Depth Anything V2) 를 sim depth 대신 쓸 때의 오차.

모델: metric = Depth-Anything-V2-Metric-Indoor-Small (m 단위 출력), rel = Depth-Anything-V2-Small (relative inverse depth)
scale 처리
  none    : metric 출력 그대로 (rel 은 불가)
  median  : 프레임마다 물체 mask 밖 장면 전체의 sim depth 중앙값으로 scale 맞춤 (oracle scale, 상한 참고용)
  table   : 첫 프레임에서 테이블 영역(물체/로봇 mask 밖 하단) 으로 scale(+shift) 1회 보정 후 고정 (실제 배포에서 가능한 방식)
지표 (GT mask 안, 물체별): AbsRel, δ1(<1.25), mask cloud centroid 오차 (sim depth cloud 대비, mm), peg pose 오차 (pose_v1, known size), ms/frame
Usage: python scripts/e7_depth.py --set eval --eps-n 10 --stride 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_v1 as pv  # noqa: E402
import v1data  # noqa: E402
from e4_pose import erode, peg_err  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = {"metric": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "rel": "depth-anything/Depth-Anything-V2-Small-hf"}


class YoloDepth:
    """Ultralytics YOLO26 depth (metric, m). 입력은 BGR numpy (ultralytics 규약), 출력은 입력 크기에 맞춘 (H, W)."""

    def __init__(self, name):
        from ultralytics import YOLO
        self.m = YOLO(f"{name}.pt")

    def __call__(self, rgb):
        r = self.m(np.ascontiguousarray(rgb[..., ::-1]), imgsz=768, verbose=False)[0]
        d = r.depth.data
        d = d.float().cpu().numpy() if hasattr(d, "cpu") else np.asarray(d, np.float32)
        if d.shape != rgb.shape[:2]:
            import cv2
            d = cv2.resize(d, rgb.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)
        return d


class DA:
    def __init__(self, name):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.proc = AutoImageProcessor.from_pretrained(MODELS[name])
        self.m = AutoModelForDepthEstimation.from_pretrained(MODELS[name]).cuda().eval()

    @torch.no_grad()
    def __call__(self, rgb):
        x = self.proc(images=rgb, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            y = self.m(**x).predicted_depth
        return F.interpolate(y[None].float(), size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def fit_scale(pred, gt, m, shift=False):
    p, g = pred[m], gt[m]
    ok = g > 0
    p, g = p[ok], g[ok]
    if shift:
        A = np.c_[p, np.ones_like(p)]
        s, b = np.linalg.lstsq(A, g, rcond=None)[0]
        return s, b
    return float(np.median(g) / np.median(p)), 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="eval")
    ap.add_argument("--eps-n", type=int, default=10)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default=f"{ROOT}/results/v1/E7")
    ap.add_argument("--models", nargs="+", default=["metric", "rel"], help="metric, rel (Depth Anything V2) 또는 yolo26{n,s,m,l,x}-depth")
    ap.add_argument("--name", default="depth")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    eps = v1data.ep_list(args.set)[:args.eps_n]
    res = {}
    for mname in args.models:
        da = YoloDepth(mname) if mname.startswith("yolo") else DA(mname)
        da(np.zeros((512, 512, 3), np.uint8))
        for scale in (("median", "table") if mname == "rel" else ("none", "median", "table")):
            key = f"{mname}_{scale}"
            rows, ms = [], []
            for ep in eps:
                d = v1data.load(args.set, ep)
                rgb, dep, K, c2w, seg = (d[f"base_camera/{k}"] for k in ("rgb", "depth", "intrinsic", "cam2world", "seg"))
                ids = {o: int(d[f"{o}_id"]) for o in ("peg", "box")}
                L, r = float(d["peg_half_sizes"][0]), float(d["peg_half_sizes"][1])
                sb = None
                pe = pv.PegEstimator(L, r)
                for t in range(0, len(rgb), args.stride):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    pr = da(rgb[t])
                    torch.cuda.synchronize()
                    ms.append((time.perf_counter() - t0) * 1000)
                    if mname == "rel":
                        pr = 1.0 / np.maximum(pr, 1e-3)  # inverse → depth (scale/shift 미정)
                    bg = (seg[t] != ids["peg"]) & (seg[t] != ids["box"]) & (dep[t] > 0)
                    if scale == "median":
                        s, b = fit_scale(pr, dep[t], bg)
                    elif scale == "table":
                        if sb is None:  # 첫 프레임 하단 절반 (테이블) 로 1회 보정
                            tb = bg.copy()
                            tb[: tb.shape[0] // 2] = False
                            sb = fit_scale(pr, dep[t], tb, shift=True)
                        s, b = sb
                    else:
                        s, b = 1.0, 0.0
                    pz = pr * s + b
                    r_ = dict(ep=ep, t=t)
                    for o in ("peg", "box"):
                        m = seg[t] == ids[o]
                        if m.sum() < 200:
                            continue
                        g, p = dep[t][m], pz[m]
                        ok = g > 0
                        r_[f"{o}_absrel"] = float(np.mean(np.abs(p[ok] - g[ok]) / g[ok]))
                        r_[f"{o}_d1"] = float(np.mean(np.maximum(p[ok] / g[ok], g[ok] / p[ok]) < 1.25))
                        Pg = pv.to_world(pv.backproject_px(dep[t], K[t], m)[0], c2w[t])
                        Pp = pv.to_world(pv.backproject_px(pz, K[t], m)[0], c2w[t])
                        r_[f"{o}_centroid_mm"] = float(np.linalg.norm(Pg.mean(0) - Pp.mean(0)) * 1000)
                        if o == "peg":
                            mp = erode(m) if m.sum() > 300 else m
                            P, v, u = pv.backproject_px(pz, K[t], mp)
                            pe.set_view(pz, K[t], c2w[t], m)
                            T, _ = pe(pv.to_world(P, c2w[t]), pv.head_flag(rgb[t][v, u]))
                            if T is not None:
                                r_["peg_pose_mm"], r_["peg_pose_deg"], _ = peg_err(T, d["peg_pose"][t], L)
                    rows.append(r_)
            agg = {}
            for k in ("peg_absrel", "peg_d1", "peg_centroid_mm", "box_absrel", "box_d1", "box_centroid_mm", "peg_pose_mm", "peg_pose_deg"):
                v = np.array([x[k] for x in rows if k in x])
                if len(v):
                    agg[k] = dict(median=round(float(np.median(v)), 4), mean=round(float(v.mean()), 4), p90=round(float(np.percentile(v, 90)), 4), n=len(v))
            agg["ms_p50"] = round(float(np.median(ms[1:])), 1)
            res[key] = agg
            print(key, json.dumps({k: (v["median"] if isinstance(v, dict) else v) for k, v in agg.items()}), flush=True)
        del da
        torch.cuda.empty_cache()
    json.dump(dict(eps=eps, stride=args.stride, results=res), open(f"{args.out}/{args.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
