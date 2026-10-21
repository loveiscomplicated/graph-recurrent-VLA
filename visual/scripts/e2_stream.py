"""E2: causal streaming + 20 Hz 최적화 ablation.

설정 (누적 적용, --variant 로 고름)
  stream      : StreamTracker (PIL 전처리 = offline 과 같은 입력), point cloud 는 CPU (pointcloud.extract, v0 와 동일)
  +compile    : image encoder torch.compile
  +gpupc      : point cloud 를 torch GPU 로 (pc_gpu.GpuPointCloud, triton FPS)
  +gpupre     : 프레임 전처리(resize/정규화)를 GPU 로
  +bf16w      : 모델 weight 자체를 bf16 으로 (autocast 대신)
  +prune      : 오래된 비조건 memory 삭제 (keep 16 프레임)
  img768/img512 : SAM2 내부 입력 해상도 (기본 1024)
  cam384/cam320 : 카메라 입력을 384/320 으로 줄여서 넣음 (intrinsic 도 스케일)
측정: 프레임별 전처리 / tracking / point cloud / 합계 ms (CUDA sync), GPU peak, GT IoU,
     offline(track_v1 같은 모델 설정 결과) mask 와의 IoU (equivalence).
Usage: python scripts/e2_stream.py --set eval --variant stream+compile+gpupc --ref results/v1/E1/runs/sam2s_c_none/eval
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
import pc_gpu  # noqa: E402
import pointcloud as pcl  # noqa: E402
import v1data  # noqa: E402
from eval_v1 import iou_seq, load_masks  # noqa: E402
from stream_v1 import StreamTracker  # noqa: E402
from track_v1 import MODELS  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OBJS = {1: "peg", 2: "box"}


def build(model, compile_on, image_size):
    from sam2.build_sam import build_sam2_video_predictor
    cfg, ckpt = MODELS["sam2"][model]
    extra = [f"++model.compile_image_encoder={bool(compile_on)}"]
    if image_size != 1024:
        extra.append(f"++model.image_size={image_size}")
    return build_sam2_video_predictor(cfg, ckpt, device="cuda", hydra_overrides_extra=extra)


def run_ep(tr, d, init, flags, gpc, rng):
    rgb, dep, K, c2w = d["base_camera/rgb"], d["base_camera/depth"], d["base_camera/intrinsic"], d["base_camera/cam2world"]
    T, H, W = rgb.shape[:3]
    cam = flags.get("cam")
    if cam:
        s = cam / H
        rgb = np.stack([cv2.resize(x, (cam, cam), interpolation=cv2.INTER_AREA) for x in rgb])
        dep = np.stack([cv2.resize(x, (cam, cam), interpolation=cv2.INTER_NEAREST) for x in dep])
        K = K.copy()
        K[:, :2] *= s
        boxes = {oid: (None if init[n]["box"] is None or init[n].get("frame", 0) != 0 else [v * s for v in init[n]["box"]])
                 for oid, n in OBJS.items()}
        if gpc is not None:
            gpc = pc_gpu.GpuPointCloud(K[0], cam, cam)
    else:
        boxes = {oid: (init[n]["box"] if init[n].get("frame", 0) == 0 else None) for oid, n in OBJS.items()}
    Hc, Wc = rgb.shape[1:3]
    masks = {n: np.zeros((T, Hc, Wc), bool) for n in OBJS.values()}
    rec = []
    npts = []
    for t in range(T):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if t == 0:
            out = tr.start(rgb[0], boxes)
        else:
            _, out = tr.step(rgb[t])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mg = {OBJS[o]: m for o, m in out.items()}
        if gpc is not None:
            pcs = gpc({n: mg[n] for n in mg}, dep[t], c2w[t])
            torch.cuda.synchronize()
            np_ = {n: len(v[0]) for n, v in pcs.items()}
            for n, m in mg.items():
                masks[n][t] = m.cpu().numpy()
        else:
            np_ = {}
            for n, m in mg.items():
                mm = m.cpu().numpy()
                masks[n][t] = mm
                pc, pw, _ = pcl.extract(mm, dep[t], K[t], c2w[t], rng)
                np_[n] = len(pc)
        t2 = time.perf_counter()
        rec.append((t1 - t0, t2 - t1))
        npts.append(np_)
    tr.reset()
    ms = np.array(rec) * 1000
    return masks, ms, npts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="eval")
    ap.add_argument("--eps", nargs="*", default=None)
    ap.add_argument("--variant", required=True, help="'+' 로 이은 플래그: stream compile gpupc gpupre bf16w prune img768 img512 cam384 cam320 base_plus tiny")
    ap.add_argument("--ref", default=None, help="offline 결과 폴더 (equivalence 비교)")
    ap.add_argument("--out", default=f"{ROOT}/results/v1/E2/runs")
    ap.add_argument("--cpu-threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.cpu_threads)
    fl = set(args.variant.split("+"))
    flags = dict(cam=384 if "cam384" in fl else 320 if "cam320" in fl else None)
    img = 768 if "img768" in fl else 512 if "img512" in fl else 1024
    model = "base_plus" if "base_plus" in fl else "tiny" if "tiny" in fl else "small"
    pred = build(model, "compile" in fl, img)
    if "bf16w" in fl:
        pred = pred.to(torch.bfloat16)
    tr = StreamTracker(pred, preprocess="gpu" if "gpupre" in fl else "pil", prune="prune" in fl)
    init = json.load(open(f"{ROOT}/results/v1/init/init_boxes_{args.set}.json"))
    eps = args.eps or v1data.ep_list(args.set)
    out = f"{args.out}/{args.variant}/{args.set}"
    os.makedirs(out, exist_ok=True)
    K0 = v1data.load(args.set, eps[0])["base_camera/intrinsic"][0]
    gpc = pc_gpu.GpuPointCloud(K0, 512, 512) if "gpupc" in fl else None
    rng = np.random.default_rng(0)
    res = dict(variant=args.variant, image_size=img, model=model, episodes={})
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if "bf16w" not in fl else torch.autocast("cuda", enabled=False)
    with torch.inference_mode(), ctx:
        d = v1data.load(args.set, eps[0])
        run_ep(tr, d, init[eps[0]], flags, gpc, rng)  # warmup (compile 포함)
        for ep in eps:
            d = v1data.load(args.set, ep)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            masks, ms, npts = run_ep(tr, d, init[ep], flags, gpc, rng)
            gpu = torch.cuda.max_memory_allocated() / 1e9
            T = len(ms)
            seg = d["base_camera/seg"]
            r = dict(frames=T, gpu_peak_gb=round(gpu, 3), track_ms=ms[1:, 0].round(2).tolist(), pc_ms=ms[1:, 1].round(2).tolist())
            for n in OBJS.values():
                gt = seg == int(d[f"{n}_id"])
                m = masks[n]
                if m.shape[1] != gt.shape[1]:
                    m = np.stack([cv2.resize(x.astype(np.uint8), gt.shape[1:][::-1], interpolation=cv2.INTER_NEAREST) > 0 for x in m])
                io = iou_seq(m, gt)
                r[f"{n}_iou"] = io.round(4).tolist()
                if args.ref and os.path.exists(f"{args.ref}/{ep}_masks.npz"):
                    ref, _, _ = load_masks(f"{args.ref}/{ep}_masks.npz")
                    a, b = m.reshape(T, -1), ref[n].reshape(T, -1)
                    inter, uni = (a & b).sum(1), (a | b).sum(1)
                    eq = np.where(uni > 0, inter / np.maximum(uni, 1), 1.0)
                    r[f"{n}_eq_iou_mean"] = round(float(eq.mean()), 5)
                    r[f"{n}_eq_iou_min"] = round(float(eq.min()), 4)
                    r[f"{n}_eq_identical_frac"] = round(float((a != b).sum(1).__eq__(0).mean()), 4)
            tot = ms[1:].sum(1)
            print(args.variant, ep, f"track {np.median(ms[1:, 0]):.1f} pc {np.median(ms[1:, 1]):.1f} total p50 {np.median(tot):.1f} "
                  f"p95 {np.percentile(tot, 95):.1f} gpu {gpu:.2f}",
                  {k: v for k, v in r.items() if "eq_" in k}, flush=True)
            res["episodes"][ep] = r
    json.dump(res, open(f"{out}/run.json", "w"))


if __name__ == "__main__":
    main()
