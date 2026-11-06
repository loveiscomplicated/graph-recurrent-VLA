"""E3 실행기: ReanchorV1 설정별로 set(+교란)을 돌리고 track_v1 과 같은 형식으로 저장 (eval_v1.py 로 평가).
추가 저장: valid/age (출력 정책), 이벤트(후보별 특징 + GT bbox IoU: 임계값 보정용, 모델 경로 밖에서 계산).
Usage: python scripts/e3_run.py --set dev --cfg v1_full --perturb patch
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("TQDM_DISABLE", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1data  # noqa: E402
from grounding import gt_bbox  # noqa: E402
from reanchor_v1 import ReanchorV1, box_iou  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GATE_THR = {"peg": 0.8, "box": 0.5}  # dev shadow 시뮬레이션으로 고름 (e3_sim_gate.py)
CFGS = {
    "v1": dict(triggers=("lost",), reid="dino", lr="gate"),
    "v1_e5": dict(triggers=("lost",), reid="dino", lr="gate", every=5),
    "v1c": dict(triggers=("lost",), reid="dino", lr="gate", gates=("cont",)),
    "v1c_tcp": dict(triggers=("lost",), reid="dino", lr="gate", gates=("cont",), tcp=True),
    "v1_tcp": dict(triggers=("lost",), reid="dino", lr="gate", tcp=True),
    "v1_spike": dict(triggers=("lost", "spike"), reid="dino", lr="gate"),
    "none": dict(triggers=()),
    "v0conf": dict(triggers=("lost",), reid="sam", reid_thr=0.75),
    "v0conf_e5": dict(triggers=("lost",), reid="sam", reid_thr=0.75, every=5),
    "dino": dict(triggers=("lost",), reid="dino", reid_thr=0.6),
    "dino_gates": dict(triggers=("lost",), reid="dino", reid_thr=0.6, gates=("area", "size3d", "cont")),
    "trig": dict(triggers=("lost", "spike", "split", "jump", "small"), reid="dino", reid_thr=0.6, gates=("area", "size3d", "cont")),
    "trig_clean": dict(triggers=("lost", "spike", "split", "jump", "small"), reid="dino", reid_thr=0.6, gates=("area", "size3d", "cont"),
                       cleanup=True),
    "full": dict(triggers=("lost", "spike", "split", "jump", "small"), reid="dino", reid_thr=0.6, gates=("area", "size3d", "cont"),
                 cleanup=True, invalid_on_anomaly=True),
    "full_tcp": dict(triggers=("lost", "spike", "split", "jump", "small"), reid="dino", reid_thr=0.6, gates=("area", "size3d", "cont"),
                     cleanup=True, invalid_on_anomaly=True, tcp=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="dev")
    ap.add_argument("--perturb", default=None)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--extra", default="{}", help="cfg 덮어쓰기 json")
    ap.add_argument("--eps", nargs="*", default=None)
    ap.add_argument("--out", default=f"{ROOT}/results/v1/E3/runs")
    args = ap.parse_args()
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    cfg = dict(CFGS[args.cfg], **json.loads(args.extra))
    if cfg.get("lr") == "gate":
        g = json.load(open(f"{ROOT}/results/v1/E3/gate.json"))
        for o, t in GATE_THR.items():
            g[o]["thr_use"] = t
        cfg["lr"] = g
    tag = args.tag or args.cfg
    sub = args.set + (f"_{args.perturb}" if args.perturb else "")
    out = f"{args.out}/{tag}/{sub}"
    os.makedirs(out, exist_ok=True)
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from track_v1 import MODELS
    pred = build_sam2_video_predictor(*MODELS["sam2"]["small"], device="cuda")
    pred.add_all_frames_to_correct_as_cond = True
    imgp = SAM2ImagePredictor(pred)
    gdino = dino = None
    if cfg.get("triggers"):
        from grounding import GDino
        gdino = GDino()
    if cfg.get("reid") == "dino":
        from reanchor_v1 import Dino
        dino = Dino()
    ra = ReanchorV1(pred, imgp, gdino, dino, cfg)
    base_set = args.set
    init = json.load(open(f"{ROOT}/results/v1/init/init_boxes_{base_set}.json"))
    eps = args.eps or v1data.ep_list(args.set)
    res = dict(cfg=cfg, episodes={})
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for ep in eps:
            d = v1data.load(args.set, ep, args.perturb)
            rgb, dep, K, c2w = d["base_camera/rgb"], d["base_camera/depth"], d["base_camera/intrinsic"], d["base_camera/cam2world"]
            tcp = d["tcp_pose"]  # TCP prior ablation 에서만 사용 (로봇 상태)
            T, H, W = rgb.shape[:3]
            boxes = {n: (init[ep][n]["box"] if init[ep][n].get("frame", 0) == 0 else None) for n in ("peg", "box")}
            masks = {n: np.zeros((T, H, W), bool) for n in ("peg", "box")}
            logits = {n: np.full(T, np.nan) for n in ("peg", "box")}
            valid = {n: np.zeros(T, bool) for n in ("peg", "box")}
            age = {n: np.zeros(T, int) for n in ("peg", "box")}
            pf = np.zeros(T)
            for t in range(T):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if t == 0:
                    o = ra.start(rgb[0], dep[0], K[0], c2w[0], boxes)
                else:
                    o = ra.step(rgb[t], dep[t], c2w[t], tcp[t] if cfg.get("tcp") else None)
                torch.cuda.synchronize()
                pf[t] = (time.perf_counter() - t0) * 1000
                for n in o:
                    masks[n][t] = o[n]["mask"].cpu().numpy()
                    logits[n][t] = o[n]["logit"]
                    valid[n][t] = o[n]["valid"]
                    age[n][t] = o[n]["age"]
            ev = ra.events
            seg = d["base_camera/seg"]
            for e in ev:  # 평가용: 후보 bbox 의 GT IoU (모델 경로 밖)
                gb = gt_bbox(seg[e["t"]] == int(d[f"{e['obj']}_id"]))
                for c in e.get("cands", []):
                    c["gt_iou"] = round(box_iou(c["box"], gb), 3) if gb else 0.0
                e["gt_visible"] = gb is not None
            ra.reset()
            np.savez_compressed(f"{out}/{ep}_masks.npz", shape=np.array(masks["peg"].shape),
                                peg=np.packbits(masks["peg"]), box=np.packbits(masks["box"]),
                                peg_logit=logits["peg"], box_logit=logits["box"], per_frame_ms=pf,
                                peg_valid=valid["peg"], box_valid=valid["box"], peg_age=age["peg"], box_age=age["box"])
            acc = [e for e in ev if e.get("accepted") or e["kind"] == "cleanup"]
            res["episodes"][ep] = dict(frames=T, ms_mean=round(float(pf[1:].mean()), 2), ms_p50=round(float(np.median(pf[1:])), 2),
                                       ms_p95=round(float(np.percentile(pf[1:], 95)), 2), ms_max=round(float(pf[1:].max()), 1),
                                       gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3), events=ev)
            print(tag, sub, ep, f"ms p50 {np.median(pf[1:]):.1f} p95 {np.percentile(pf[1:], 95):.1f}", f"events {len(ev)} acc {len(acc)}",
                  flush=True)
    json.dump(res, open(f"{out}/run.json", "w"), default=float)


if __name__ == "__main__":
    main()
