"""Vision v0 전체 파이프라인을 한 번에 실행: RGB-D 에피소드 → Grounding DINO(frame 0) → SAM2 tracking → point cloud npz.

단계 (에피소드마다):
  1. (없으면) ManiSkill3 수집: scripts/collect.py 실행 → data/step2/ep00X.npz
  2. Grounding DINO tiny (HF): frame 0 에서 물체별 프롬프트 top-1 bbox. 후보가 하나도 없으면 GT bbox 로 대체(source 기록)
  3. SAM2.1 video predictor: frame 0 bbox 로 초기화 → 끝까지 propagate (re-anchor 없음, Step 4 와 동일)
  4. point cloud: mask erosion → 역투영 → SOR → FPS ≤1024 → <out>/<ep>/f{frame:04d}_{obj}.npz (Step 5 와 동일 포맷)
  5. 평가: GT mask IoU, centroid 오차 (GT mask cloud / GT pose), 단계별 latency, GPU peak → <out>/pipeline.json
  6. (--video 에 지정한 ep) 데모 영상: RGB+mask | 3D point cloud 2 view

Usage:
  source env.sh
  python scripts/run_pipeline.py --eps ep000 ep001 ep002 --video ep002 --out results/step5/pipeline
  # Step 6 벤치마크 (워밍업 1 ep 제외 후 3 ep 측정, 표 저장)
  python scripts/run_pipeline.py --warmup-ep ep000 --video ep002 --out results/step6 --bench-md results/step6/benchmark.md
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import imageio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pointcloud as pcl  # noqa: E402
from grounding import GDino, gt_bbox  # noqa: E402
from track_sam2 import MODELS, iou, stats, track_episode  # noqa: E402
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.environ.get("VISION_CACHE", f"{ROOT}/cache")
CAM = pcl.CAM
PROMPTS = {"peg": "red and white block", "box": "cardboard box"}  # Step 3 선택 프롬프트


def ensure_data(data_dir, eps):
    missing = [ep for ep in eps if not os.path.exists(f"{data_dir}/{ep}.npz")]
    if not missing:
        return
    seeds = [str(int(ep[2:])) for ep in missing]
    print(f"[collect] {missing} 없음 → scripts/collect.py --seeds {' '.join(seeds)}")
    subprocess.run([sys.executable, f"{ROOT}/scripts/collect.py", "--seeds", *seeds, "--out", data_dir], check=True)


def ground(gdino, d, t=0):
    boxes, ms, parts = {}, [], []
    for obj, prompt in PROMPTS.items():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b, s = gdino(d[f"{CAM}/rgb"][t], prompt)
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
        parts.append(gdino.last_ms)
        if b:
            boxes[obj] = dict(box=b[0], score=round(s[0], 3), source="gdino", prompt=prompt)
        else:
            boxes[obj] = dict(box=gt_bbox(d[f"{CAM}/seg"][t] == int(d[f"{obj}_id"])), score=None,
                              source="gt_fallback", prompt=prompt)
            print(f"  [warn] GDINO 후보 없음 ({obj}, '{prompt}') → GT bbox 대체")
    return boxes, ms, parts


def run_episode(gdino, predictor, d, ep, out_dir, rng, cluster):
    """grounding → SAM2 → point cloud 1 에피소드. 단계별 GPU peak 은 단계 직전에 reset 해서 따로 잰다(모델 weight 포함)."""
    gb = 1e9
    torch.cuda.reset_peak_memory_stats()
    boxes, g_ms, g_parts = ground(gdino, d)
    mem = dict(grounding_alloc=torch.cuda.max_memory_allocated() / gb, grounding_reserved=torch.cuda.max_memory_reserved() / gb)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        masks, logits, tr, tr_frames = track_episode(predictor, d[f"{CAM}/rgb"], boxes, f"{CACHE}/pipeline_frames/{ep}")
    mem.update(tracking_alloc=torch.cuda.max_memory_allocated() / gb, tracking_reserved=torch.cuda.max_memory_reserved() / gb)
    rows, clouds, confs, pc_ms = pcl.process_episode(d, masks, logits, out_dir, rng, cluster=cluster)
    return boxes, g_ms, g_parts, masks, logits, tr, tr_frames, rows, clouds, confs, pc_ms, mem


def hud(img, lines):
    """영상 아래에 흰 띠를 붙이고 latency 텍스트를 쓴다."""
    bar = np.full((10 + 20 * len(lines), img.shape[1], 3), 255, np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(bar, s, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([img, bar], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{ROOT}/data/step2")
    ap.add_argument("--eps", nargs="+", default=["ep000", "ep001", "ep002"])
    ap.add_argument("--model", default="small", choices=list(MODELS))
    ap.add_argument("--out", default=f"{ROOT}/results/step5/pipeline")
    ap.add_argument("--video", nargs="*", default=[], help="데모 영상 저장할 ep")
    ap.add_argument("--cluster", action="store_true", help="point cloud 떨어진 작은 cluster 제거 (기본 OFF)")
    ap.add_argument("--warmup-ep", default=None, help="측정 전에 이 ep 로 전체 파이프라인을 1회 돌리고 결과는 버림")
    ap.add_argument("--gdino-warmup", type=int, default=5, help="측정 전 GDINO 워밍업 호출 수 (실제 frame 0, 프롬프트별)")
    ap.add_argument("--bench-md", default=None, help="벤치마크 표(markdown) 저장 경로")
    ap.add_argument("--cpu-threads", type=int, default=8,
                    help="torch/OpenCV CPU 스레드 수 (0 = 라이브러리 기본값. Step 6: 8 이 128 보다 GDINO 전처리 11배, init_state 2배 빠름. 컨테이너 CPU quota ~31코어인데 torch 기본 128 스레드)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
        cv2.setNumThreads(args.cpu_threads)
    ensure_data(args.data, args.eps + ([args.warmup_ep] if args.warmup_ep else []))
    rng = np.random.default_rng(args.seed)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    gdino = GDino()
    cfg, ckpt = MODELS[args.model]
    predictor = build_sam2_video_predictor(cfg, ckpt, device="cuda")
    load_s = time.perf_counter() - t0
    print(f"models loaded {load_s:.1f}s (GDINO tiny + SAM2.1-{args.model})")
    weights_gb = torch.cuda.memory_allocated() / 1e9

    # 워밍업: GDINO 를 실제 크기/프롬프트로 N회 + (선택) 전체 파이프라인 1 에피소드. 결과는 버림.
    d0 = dict(np.load(f"{args.data}/{args.warmup_ep or args.eps[0]}.npz"))
    for _ in range(args.gdino_warmup):
        for prompt in PROMPTS.values():
            gdino(d0[f"{CAM}/rgb"][0], prompt)
    if args.warmup_ep:
        t0 = time.perf_counter()
        run_episode(gdino, predictor, d0, args.warmup_ep, f"{CACHE}/pipeline_warmup/{args.warmup_ep}",
                    np.random.default_rng(args.seed), args.cluster)
        print(f"warmup episode {args.warmup_ep} done ({time.perf_counter() - t0:.1f}s, 측정에서 제외)")
    del d0

    result = dict(model=args.model, prompts=PROMPTS, camera=CAM, cluster=args.cluster, model_load_s=round(load_s, 1),
                  warmup=dict(gdino_calls_per_prompt=args.gdino_warmup, episode=args.warmup_ep),
                  env=dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cpu_count=os.cpu_count(),
                           torch_threads=torch.get_num_threads(), cv2_threads=cv2.getNumThreads()),
                  model_weights_gb=round(weights_gb, 2), episodes={})
    for ep in args.eps:
        d = dict(np.load(f"{args.data}/{ep}.npz"))
        T = len(d["frame_idx"])
        t_ep = time.perf_counter()
        (boxes, g_ms, g_parts, masks, logits, tr, tr_frames, rows, clouds, confs, pc_ms,
         mem) = run_episode(gdino, predictor, d, ep, f"{args.out}/{ep}", rng, args.cluster)
        wall_s = time.perf_counter() - t_ep

        seg = d[f"{CAM}/seg"]
        r = dict(frames=T, init_boxes=boxes, iou={}, pointcloud=pcl.summarize(rows))
        for obj in pcl.OBJECTS:
            gt = seg == int(d[f"{obj}_id"])
            vis = gt.reshape(T, -1).sum(1) >= pcl.VIS_MIN
            r["iou"][obj] = stats([iou(masks[obj][t], gt[t]) if vis[t] else np.nan for t in range(T)])
        pc_frame = np.reshape(pc_ms, (T, -1)).sum(1)  # 프레임당 (물체 2개 합)
        pc_frame_ms = float(pc_frame.mean())
        r["latency_ms"] = dict(
            grounding_per_query=round(float(np.mean(g_ms)), 1), grounding_frame0_total=round(float(np.sum(g_ms)), 1),
            **{f"grounding_{k}": round(float(np.mean([p[k] for p in g_parts])), 1) for k in ("pre", "forward", "post")},
            sam2_write_jpeg=round(tr["write_jpeg_ms"], 1),
            sam2_init_load_frames=round(tr["load_frames_ms"], 1), sam2_init_add_boxes=round(tr["add_box_ms"], 1),
            sam2_first_step=round(tr["first_step_ms"], 1),
            track_per_frame=round(tr["per_frame_ms"], 1), track_per_frame_p95=round(tr["per_frame_ms_p95"], 1),
            pointcloud_per_frame=round(pc_frame_ms, 1),
            pointcloud_per_frame_p95=round(float(np.percentile(pc_frame, 95)), 1))
        # FPS 3가지:
        #  steady : 초기화 이후 프레임당 tracking + point cloud
        #  online : steady + init_state 의 프레임 로드/리사이즈를 프레임당으로 나눈 것 (스트리밍이면 매 프레임 내는 비용)
        #  e2e    : grounding + SAM2 초기화 전체 + 모든 프레임 tracking/point cloud 를 T 로 나눈 것 (JPEG 쓰기·npz 저장·평가 제외)
        compute_ms = (sum(g_ms) + tr["load_frames_ms"] + tr["add_box_ms"] + sum(tr_frames) + float(np.sum(pc_ms)))
        r["fps"] = dict(steady=round(1000 / (tr["per_frame_ms"] + pc_frame_ms), 2),
                        online=round(1000 / (tr["per_frame_ms"] + pc_frame_ms + tr["load_frames_ms"] / T), 2),
                        e2e=round(T * 1000 / compute_ms, 2),
                        wall_incl_io=round(T / wall_s, 2))
        r["gpu_gb"] = {k: round(v, 2) for k, v in mem.items()}
        result["episodes"][ep] = r
        print(ep, json.dumps(dict(iou={o: r["iou"][o].get("mean") for o in pcl.OBJECTS}, latency=r["latency_ms"],
                                  fps=r["fps"], gpu=r["gpu_gb"])))

        if ep in args.video:
            lim = pcl.scene_limits(d)
            path = f"{args.out}/{ep}_demo.mp4"
            with imageio.get_writer(path, fps=10, macro_block_size=1) as w:
                for t in range(T):
                    img = pcl.render_frame(d, t, masks, clouds[t], confs[t], lim, ep)
                    trk = tr_frames[t] if t > 0 else tr["add_box_ms"] + tr_frames[0]
                    w.append_data(hud(img, [f"SAM2-{args.model} track {trk:.0f} ms | point cloud {pc_frame[t]:.0f} ms"
                                            f" | steady {r['fps']['steady']:.1f} FPS",
                                            f"points: peg {len(clouds[t]['peg'])}  box {len(clouds[t]['box'])}  (max 1024, 0 = mask empty)"]))
            print("video:", path)

    result["gpu_peak_gb"] = round(max(max(e["gpu_gb"].values()) for e in result["episodes"].values()), 2)
    eps = list(result["episodes"].values())
    mean = lambda f: round(float(np.mean([f(e) for e in eps])), 2)  # noqa: E731
    result["all"] = dict(
        **{k: mean(lambda e, k=k: e["latency_ms"][k]) for k in eps[0]["latency_ms"]},
        **{f"fps_{k}": mean(lambda e, k=k: e["fps"][k]) for k in eps[0]["fps"]},
        **{f"gpu_{k}": round(max(e["gpu_gb"][k] for e in eps), 2) for k in eps[0]["gpu_gb"]})
    json.dump(result, open(f"{args.out}/pipeline.json", "w"), indent=1)
    print("ALL", json.dumps(result["all"]), "gpu_peak_gb", result["gpu_peak_gb"])
    if args.bench_md:
        write_bench_md(result, args.bench_md)


def write_bench_md(res, path):
    eps, a = res["episodes"], res["all"]
    cols = list(eps) + ["**평균**"]
    get = lambda f: [f(e) for e in eps.values()]  # noqa: E731
    rows = [
        ("frames", get(lambda e: e["frames"]), sum(get(lambda e: e["frames"]))),
        ("grounding / query (ms)", get(lambda e: e["latency_ms"]["grounding_per_query"]), a["grounding_per_query"]),
        ("  ├ 전처리 (HF processor, CPU)", get(lambda e: e["latency_ms"]["grounding_pre"]), a["grounding_pre"]),
        ("  ├ forward (GPU)", get(lambda e: e["latency_ms"]["grounding_forward"]), a["grounding_forward"]),
        ("  └ 후처리", get(lambda e: e["latency_ms"]["grounding_post"]), a["grounding_post"]),
        ("grounding frame 0 합계 (2 query)", get(lambda e: e["latency_ms"]["grounding_frame0_total"]),
         a["grounding_frame0_total"]),
        ("SAM2 init: JPEG 쓰기 (측정 외 참고)", get(lambda e: e["latency_ms"]["sam2_write_jpeg"]), a["sam2_write_jpeg"]),
        ("SAM2 init: `init_state` (전 프레임 로드)", get(lambda e: e["latency_ms"]["sam2_init_load_frames"]),
         a["sam2_init_load_frames"]),
        ("SAM2 init: box 2개 추가", get(lambda e: e["latency_ms"]["sam2_init_add_boxes"]), a["sam2_init_add_boxes"]),
        ("SAM2 첫 propagate step (frame 0 memory 인코딩)", get(lambda e: e["latency_ms"]["sam2_first_step"]),
         a["sam2_first_step"]),
        ("tracking / frame, 2 obj (ms)", get(lambda e: e["latency_ms"]["track_per_frame"]), a["track_per_frame"]),
        ("tracking / frame p95", get(lambda e: e["latency_ms"]["track_per_frame_p95"]), a["track_per_frame_p95"]),
        ("point cloud / frame, 2 obj, CPU (ms)", get(lambda e: e["latency_ms"]["pointcloud_per_frame"]),
         a["pointcloud_per_frame"]),
        ("point cloud / frame p95", get(lambda e: e["latency_ms"]["pointcloud_per_frame_p95"]),
         a["pointcloud_per_frame_p95"]),
        ("**FPS steady** (track + pc)", get(lambda e: e["fps"]["steady"]), a["fps_steady"]),
        ("**FPS online 추정** (+ 프레임 로드/T)", get(lambda e: e["fps"]["online"]), a["fps_online"]),
        ("**FPS e2e** (grounding·init 포함 / T)", get(lambda e: e["fps"]["e2e"]), a["fps_e2e"]),
        ("FPS wall (JPEG·npz 저장·평가 포함)", get(lambda e: e["fps"]["wall_incl_io"]), a["fps_wall_incl_io"]),
        ("GPU peak alloc: grounding (GB)", get(lambda e: e["gpu_gb"]["grounding_alloc"]), a["gpu_grounding_alloc"]),
        ("GPU peak alloc: tracking (GB)", get(lambda e: e["gpu_gb"]["tracking_alloc"]), a["gpu_tracking_alloc"]),
        ("GPU peak reserved: tracking (GB)", get(lambda e: e["gpu_gb"]["tracking_reserved"]), a["gpu_tracking_reserved"]),
        ("peg IoU mean", get(lambda e: e["iou"]["peg"]["mean"]), round(float(np.mean(get(lambda e: e["iou"]["peg"]["mean"]))), 3)),
        ("box IoU mean", get(lambda e: e["iou"]["box"]["mean"]), round(float(np.mean(get(lambda e: e["iou"]["box"]["mean"]))), 3)),
    ]
    env = res["env"]
    lines = [f"# Step 6 벤치마크 — GDINO tiny + SAM2.1-{res['model']} + point cloud ({res['camera']}, 512², 2 objects)", "",
             f"- {env['gpu']}, torch {env['torch']}, CPU {env['cpu_count']} 논리코어 (cgroup quota ≈31), torch {env['torch_threads']} / OpenCV {env['cv2_threads']} 스레드. SAM2 bf16 autocast, GDINO fp32.",
             f"- 워밍업 (측정 제외): GDINO {res['warmup']['gdino_calls_per_prompt']}회 × 프롬프트 2개 (실제 frame 0) + "
             f"전체 파이프라인 1 에피소드 (`{res['warmup']['episode']}`).",
             f"- 모델 로드 {res['model_load_s']} s (측정 제외), 모델 weight 상주 {res['model_weights_gb']} GB.",
             "- GPU 평균 열 = 3 ep 중 최댓값. 'GPU peak'는 `torch.cuda.max_memory_*` (모델 weight 포함, 단계 직전에 reset).", "",
             "| 항목 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, vals, avg in rows:
        lines.append(f"| {name} | " + " | ".join(str(v) for v in vals) + f" | {avg} |")
    open(path, "w").write("\n".join(lines) + "\n")
    print("bench md:", path)


if __name__ == "__main__":
    main()
