"""Step 7: tracker 교체 비교용 tracking 실행기 (SAM2 / SAMURAI / EfficientTAM 공통).

각 tracker 는 서로 다른 venv 에 설치되어 있어서(패키지 이름 `sam2` 충돌 등) 이 스크립트는 numpy/cv2/torch 만 쓰고
tracking 결과(mask, logit, timing, GPU peak)만 저장한다. IoU 평가와 그림은 vision venv 의 `eval_step7.py` 가 한다.

- 입력: data/step2/ep00X.npz (base_camera RGB), results/step3/init_boxes.json (frame 0 GDINO bbox, Step 4 와 동일)
- 측정 방식은 Step 6 과 같다: CPU 스레드 8, bf16 autocast, 워밍업 1 ep(ep000) 후 결과 버림, 첫 propagate step 제외 평균.
- mode
    joint  : peg/box 2 물체를 한 inference state 에서 같이 추적 (Step 4/6 의 SAM2 방식)
    single : 물체마다 따로 추적 (init_state 1회 공유, 물체별로 reset_state 후 propagate). 프레임당 시간 = 두 물체 합.
             SAMURAI 는 Kalman filter 상태가 모델 전역(물체 1개 가정)이라 single 만 가능하다.

Usage (venv 별):
  source $VISION_ENVS/vision/bin/activate       && python scripts/track_compare.py --backend sam2 --model small --mode joint
  source $VISION_ENVS/samurai/bin/activate      && python scripts/track_compare.py --backend samurai --model small --mode single
  source $VISION_ENVS/efficienttam/bin/activate && python scripts/track_compare.py --backend efficienttam --model s --mode joint
"""
import argparse
import json
import os
import shutil
import time

import cv2
import numpy as np
import torch

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.environ.get("VISION_CACHE", f"{ROOT}/cache")
CK = f"{ROOT}/checkpoints"
CAM = "base_camera"
OBJS = {1: "peg", 2: "box"}
MODELS = {
    "sam2": {"small": ("configs/sam2.1/sam2.1_hiera_s.yaml", f"{CK}/sam2.1_hiera_small.pt"),
             "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", f"{CK}/sam2.1_hiera_base_plus.pt")},
    # SAMURAI 는 SAM2.1 weight 를 그대로 쓰고 config 만 다르다 (samurai_mode: true, Kalman filter + motion-aware memory)
    "samurai": {"small": ("configs/samurai/sam2.1_hiera_s.yaml", f"{CK}/sam2.1_hiera_small.pt"),
                "base_plus": ("configs/samurai/sam2.1_hiera_b+.yaml", f"{CK}/sam2.1_hiera_base_plus.pt")},
    "efficienttam": {m: (f"configs/efficienttam/efficienttam_{m}.yaml", f"{CK}/efficienttam_{m}.pt")
                     for m in ("s", "ti", "s_512x512", "ti_512x512")},
}


def build(backend, model, compile_mode):
    """compile_mode: default = 각 repo config 그대로 (SAM2/SAMURAI: image encoder compile OFF, EfficientTAM: ON max-autotune)."""
    cfg, ckpt = MODELS[backend][model]
    extra = [] if compile_mode == "default" else [f"++model.compile_image_encoder={compile_mode == 'on'}"]
    if backend in ("sam2", "samurai"):
        from sam2.build_sam import build_sam2_video_predictor
        return build_sam2_video_predictor(cfg, ckpt, device="cuda", hydra_overrides_extra=extra)
    from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
    return build_efficienttam_video_predictor(cfg, ckpt, device="cuda", hydra_overrides_extra=extra)


def reset_samurai(predictor):
    """SAMURAI Kalman filter 상태는 predictor(모델) 속성이라 reset_state 로 안 지워진다 → 물체/에피소드마다 직접 초기화."""
    if getattr(predictor, "samurai_mode", False):
        predictor.kf_mean, predictor.kf_covariance, predictor.stable_frames = None, None, 0


def propagate(predictor, state, obj_items, boxes, masks, logits):
    """obj_items 를 frame 0 bbox 로 넣고 끝까지 propagate. 프레임별 ms 리스트와 box 추가 ms 를 반환."""
    reset_samurai(predictor)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for oid, name in obj_items:
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=oid, box=np.array(boxes[name]["box"], np.float32))
    torch.cuda.synchronize()
    add_ms = (time.perf_counter() - t0) * 1000
    per_frame = []
    t_prev = time.perf_counter()
    for fi, obj_ids, res in predictor.propagate_in_video(state):
        for j, oid in enumerate(obj_ids):
            masks[OBJS[oid]][fi] = (res[j, 0] > 0).cpu().numpy()
        torch.cuda.synchronize()
        now = time.perf_counter()
        per_frame.append((now - t_prev) * 1000)
        t_prev = now
    for oid, name in obj_items:
        od = state["output_dict_per_obj"][predictor._obj_id_to_idx(state, oid)]
        for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for fi, out in od[key].items():
                logits[name][fi] = float(out["object_score_logits"].flatten()[0])
    predictor.reset_state(state)
    return per_frame, add_ms


def track_episode(predictor, rgb, boxes, tmp_dir, mode):
    T = len(rgb)
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    for t in range(T):
        cv2.imwrite(f"{tmp_dir}/{t:05d}.jpg", cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    state = predictor.init_state(video_path=tmp_dir)
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - t0) * 1000

    masks = {n: np.zeros((T,) + rgb.shape[1:3], bool) for n in OBJS.values()}
    logits = {n: np.full(T, np.nan) for n in OBJS.values()}
    groups = [list(OBJS.items())] if mode == "joint" else [[it] for it in OBJS.items()]
    per_frame = np.zeros(T)
    add_ms = 0.0
    for g in groups:
        pf, a = propagate(predictor, state, g, boxes, masks, logits)
        per_frame += np.array(pf)
        add_ms += a
    shutil.rmtree(tmp_dir)
    # per_frame[0] 은 propagate preflight(frame 0 memory 인코딩) 포함 → 제외 (Step 4/6 과 동일)
    first_ms = float(per_frame[0])
    timing = dict(load_frames_ms=load_ms, add_box_ms=add_ms, first_step_ms=first_ms,
                  per_frame_ms=float(per_frame[1:].mean()), per_frame_ms_p95=float(np.percentile(per_frame[1:], 95)))
    return masks, logits, timing, per_frame.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=list(MODELS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="joint", choices=["joint", "single"])
    ap.add_argument("--compile", default="default", choices=["default", "on", "off"],
                    help="image encoder torch.compile (default = repo config: EfficientTAM 만 ON)")
    ap.add_argument("--data", default=f"{ROOT}/data/step2")
    ap.add_argument("--boxes", default=f"{ROOT}/results/step3/init_boxes.json")
    ap.add_argument("--eps", nargs="+", default=["ep000", "ep001", "ep002"])
    ap.add_argument("--warmup-ep", default="ep000", help="측정 전에 1회 돌리고 버림 ('' = 안 함)")
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--out", default=f"{ROOT}/results/step7")
    args = ap.parse_args()
    if args.backend == "samurai" and args.mode != "single":
        ap.error("SAMURAI 는 Kalman filter 상태가 모델 전역이라 물체 1개씩만 추적 가능 → --mode single")
    tag = f"{args.backend}_{args.model}_{args.mode}" + ("" if args.compile == "default" else f"_compile{args.compile}")
    out = f"{args.out}/{tag}"
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.cpu_threads)
    init_boxes = json.load(open(args.boxes))
    gb = 1e9

    t0 = time.perf_counter()
    predictor = build(args.backend, args.model, args.compile)
    load_s = time.perf_counter() - t0
    weights_gb = torch.cuda.memory_allocated() / gb
    n_params = sum(p.numel() for p in predictor.parameters()) / 1e6
    print(f"[{tag}] model load {load_s:.1f}s, weights {weights_gb:.2f} GB, {n_params:.1f}M params")

    res = dict(tag=tag, backend=args.backend, model=args.model, mode=args.mode, compile=args.compile, cpu_threads=args.cpu_threads,
               model_load_s=round(load_s, 2), weights_gb=round(weights_gb, 3), params_m=round(n_params, 1),
               warmup_ep=args.warmup_ep or None, episodes={})
    tmp = f"{CACHE}/step7_frames/{tag}"
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if args.warmup_ep:
            d = np.load(f"{args.data}/{args.warmup_ep}.npz")
            track_episode(predictor, d[f"{CAM}/rgb"], init_boxes[args.warmup_ep], tmp, args.mode)
        for ep in args.eps:
            rgb = np.load(f"{args.data}/{ep}.npz")[f"{CAM}/rgb"]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            masks, logits, timing, per_frame = track_episode(predictor, rgb, init_boxes[ep], tmp, args.mode)
            timing = {k: round(v, 2) for k, v in timing.items()}
            mem = dict(peak_alloc_gb=round(torch.cuda.max_memory_allocated() / gb, 3),
                       peak_reserved_gb=round(torch.cuda.max_memory_reserved() / gb, 3))
            res["episodes"][ep] = dict(frames=len(rgb), timing=timing, mem=mem, per_frame_ms=[round(x, 2) for x in per_frame])
            np.savez_compressed(f"{out}/{ep}_masks.npz", peg=masks["peg"], box=masks["box"],
                                peg_logit=logits["peg"], box_logit=logits["box"])
            print(ep, json.dumps(timing), json.dumps(mem))
    json.dump(res, open(f"{out}/run.json", "w"), indent=1)


if __name__ == "__main__":
    main()
