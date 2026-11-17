"""E1/E3: v1 데이터셋 tracking 실행기 (SAM2 / SAMURAI / EfficientTAM 공통, 프레임 단위 진행 + re-anchor).

v0(track_compare.py / reanchor.py)와 다른 점
  - 입력은 data/v1/<set> (v1data.load, 교란 patch/dropout 적용 가능). JPEG 폴더를 쓰지 않는다:
    `load_video_frames` 를 메모리 배열 로더로 바꿔 끼운다(같은 PIL resize + 정규화. JPEG q95 손실만 없어짐).
  - 초기 bbox = results/v1/init/init_boxes_<set>.json (GDINO top-1, GT 대체 없음). 검출 실패 물체는 추적하지 않는다(빈 mask).
  - 프레임 단위로 진행한다 (`propagate_in_video(start_frame_idx=t, max_frame_num_to_track=0)`). re-anchor 가 없는 'none' 도 같은 루프.
  - re-anchor 설정은 v0 Step 8 과 같다 (conf / conf_e5, GDINO 재검출 + SAM2 embedding re-ID + 2프레임 확인).
    GDINO 는 transformers 가 있는 venv 에서만 쓸 수 있다.

출력: <out>/<tag>/<set>[_<perturb>]/<ep>_masks.npz (peg, box bool (T,H,W) packbits, *_logit), run.json (timing, events, GPU)
평가는 eval_v1.py.

Usage:
  python scripts/track_v1.py --backend sam2 --model small --reanchor none --set eval --tag sam2s
"""
import argparse
import json
import os
import sys
import time
from collections import deque

os.environ.setdefault("TQDM_DISABLE", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1data  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CK = f"{ROOT}/checkpoints"
CAM = "base_camera"
OBJS = {1: "peg", 2: "box"}
MODELS = {
    "sam2": {"small": ("configs/sam2.1/sam2.1_hiera_s.yaml", f"{CK}/sam2.1_hiera_small.pt"),
             "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", f"{CK}/sam2.1_hiera_tiny.pt"),
             "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", f"{CK}/sam2.1_hiera_base_plus.pt")},
    "samurai": {"small": ("configs/samurai/sam2.1_hiera_s.yaml", f"{CK}/sam2.1_hiera_small.pt"),
                "base_plus": ("configs/samurai/sam2.1_hiera_b+.yaml", f"{CK}/sam2.1_hiera_base_plus.pt")},
    "efficienttam": {m: (f"configs/efficienttam/efficienttam_{m}.yaml", f"{CK}/efficienttam_{m}.pt")
                     for m in ("s", "ti", "s_512x512", "ti_512x512")},
}
# re-anchor (v0 Step 8 과 같은 상수)
MAX_BOX_FRAC = 0.25
LOST_LOGIT = 0.0
LOST_K = 2
BANK_LOGIT = 2.0
BANK_N = 8
REID_THR = 0.75
CONFIRM_IOU = 0.5
REANCHOR = {"none": None, "conf": 1, "conf_e5": 5}  # 값 = 소실 중 재검출 간격
PROMPTS = {"peg": "red and white block", "box": "cardboard box"}

_ARRAY = {}


def _array_loader(video_path, image_size, offload_video_to_cpu=False, img_mean=(0.485, 0.456, 0.406),
                  img_std=(0.229, 0.224, 0.225), async_loading_frames=False, compute_device=torch.device("cuda")):
    """load_video_frames 대체: _ARRAY[video_path] (T,H,W,3) uint8 → 같은 전처리 (PIL resize → /255 → 정규화)."""
    rgb = _ARRAY[video_path]
    T, H, W = rgb.shape[:3]
    imgs = torch.zeros(T, 3, image_size, image_size, dtype=torch.float32)
    for t in range(T):
        im = np.array(Image.fromarray(rgb[t]).resize((image_size, image_size))) / 255.0
        imgs[t] = torch.from_numpy(im).permute(2, 0, 1)
    mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    if not offload_video_to_cpu:
        imgs, mean, std = imgs.to(compute_device), mean.to(compute_device), std.to(compute_device)
    imgs -= mean
    imgs /= std
    return imgs, H, W


def build(backend, model, compile_on):
    cfg, ckpt = MODELS[backend][model]
    extra = [f"++model.compile_image_encoder={bool(compile_on)}"]
    if backend in ("sam2", "samurai"):
        import sam2.sam2_video_predictor as vp
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor as IP
        pred = build_sam2_video_predictor(cfg, ckpt, device="cuda", hydra_overrides_extra=extra)
    else:
        import efficient_track_anything.efficienttam_video_predictor as vp
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
        from efficient_track_anything.efficienttam_image_predictor import EfficientTAMImagePredictor as IP
        pred = build_efficienttam_video_predictor(cfg, ckpt, device="cuda", hydra_overrides_extra=extra)
    vp.load_video_frames = _array_loader
    return pred, IP


def reset_samurai(p):
    if getattr(p, "samurai_mode", False):
        p.kf_mean, p.kf_covariance, p.stable_frames = None, None, 0


def set_image_features(imgp, backbone_out, hw):
    m = imgp.model
    _, vision_feats, _, _ = m._prepare_backbone_features(backbone_out)
    if m.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + m.no_mem_embed
    feats = [f.permute(1, 2, 0).view(1, -1, *s) for f, s in zip(vision_feats[::-1], imgp._bb_feat_sizes[::-1])][::-1]
    imgp._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
    imgp._orig_hw = [hw]
    imgp._is_image_set = True


def pool_embed(imgp, mask):
    feat = imgp._features["image_embed"][0].float()
    mm = F.interpolate(torch.as_tensor(mask, dtype=torch.float32, device=feat.device)[None, None],
                       size=feat.shape[-2:], mode="area")[0, 0]
    if mm.sum() < 0.5:
        return None
    return F.normalize((feat * mm).sum((1, 2)) / mm.sum(), dim=0)


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def reid_pick(gdino, imgp, rgb, obj, bank):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast("cuda", enabled=False):
        boxes, scores = gdino(rgb, PROMPTS[obj])
    torch.cuda.synchronize()
    g_ms = (time.perf_counter() - t0) * 1000
    H, W = rgb.shape[:2]
    cands = [(b, s) for b, s in zip(boxes, scores) if (b[2] - b[0]) * (b[3] - b[1]) < MAX_BOX_FRAC * H * W][:8]
    info = dict(gdino_ms=g_ms, reid_ms=0.0, n_cand=len(cands))
    if not cands or not bank:
        return None, info
    t0 = time.perf_counter()
    B = torch.stack(bank)
    best, best_sim = None, -1.0
    for b, s in cands:
        m, _, _ = imgp.predict(box=np.array(b, np.float32), multimask_output=False)
        e = pool_embed(imgp, m[0] > 0)
        if e is None:
            continue
        sim = float((B @ e).max())
        if sim > best_sim:
            best, best_sim = b, sim
    torch.cuda.synchronize()
    info.update(reid_ms=(time.perf_counter() - t0) * 1000, sim=round(best_sim, 3))
    return (best if best_sim >= REID_THR else None), info


def obj_logit(state, predictor, oid, t):
    od = state["output_dict_per_obj"][predictor._obj_id_to_idx(state, oid)]
    out = od["cond_frame_outputs"].get(t) or od["non_cond_frame_outputs"].get(t)
    return float(out["object_score_logits"].flatten()[0])


def run_group(predictor, imgp, gdino, rgb, objs, init, every, key, t0f=0):
    """objs: [(oid, name)] 를 한 state 에서 frame t0f 부터 프레임 단위로 추적 (+ conf re-anchor). per-frame ms 배열 반환."""
    T, H, W = rgb.shape[:3]
    reset_samurai(predictor)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    state = predictor.init_state(video_path=key)
    torch.cuda.synchronize()
    load_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    active = []
    for oid, name in objs:
        b = init[name]["box"]
        if b is not None:
            predictor.add_new_points_or_box(state, frame_idx=t0f, obj_id=oid, box=np.array(b, np.float32))
            active.append((oid, name))
    torch.cuda.synchronize()
    add_ms = (time.perf_counter() - t0) * 1000
    masks = {n: np.zeros((T, H, W), bool) for _, n in objs}
    logits = {n: np.full(T, np.nan) for _, n in objs}
    per_frame = np.zeros(T)
    events = []
    if not active:
        predictor.reset_state(state)
        return masks, logits, per_frame, events, dict(load_ms=load_ms, add_ms=add_ms)
    need_reid = every is not None
    banks = {n: deque(maxlen=BANK_N) for _, n in active}
    ref0 = {}
    lost = {n: 0 for _, n in active}
    pending = {n: None for _, n in active}
    for t in range(t0f, T):
        torch.cuda.synchronize()
        ts = time.perf_counter()
        for fi, oids, res in predictor.propagate_in_video(state, start_frame_idx=t, max_frame_num_to_track=0):
            for j, oid in enumerate(oids):
                masks[OBJS[oid]][fi] = (res[j, 0] > 0).cpu().numpy()
        if need_reid:
            set_image_features(imgp, state["cached_features"][t][1], (H, W))
        for oid, name in active:
            lg = obj_logit(state, predictor, oid, t)
            logits[name][t] = lg
            if not need_reid:
                continue
            if lg > BANK_LOGIT:
                e = pool_embed(imgp, masks[name][t])
                if e is not None:
                    if t == t0f:
                        ref0[name] = e
                    else:
                        banks[name].append(e)
            lost[name] = lost[name] + 1 if lg < LOST_LOGIT else 0
            trig = lost[name] >= LOST_K and (lost[name] - LOST_K) % every == 0
            if not trig and pending[name] is None:
                continue
            bank = ([ref0[name]] if name in ref0 else []) + list(banks[name])
            box, info = reid_pick(gdino, imgp, rgb[t], name, bank)
            cand = box
            if box is not None:
                prev = pending[name]
                ok = prev is not None and prev[0] == t - 1 and box_iou(box, prev[1]) >= CONFIRM_IOU
                pending[name] = None if ok else (t, box)
                box = box if ok else None
            else:
                pending[name] = None
            ev = dict(t=t, obj=name, logit=round(lg, 2), accepted=box is not None, cand=cand is not None,
                      **{k: (round(v, 1) if k.endswith("_ms") else v) for k, v in info.items()})
            if box is not None:
                ev["box"] = [round(float(v), 1) for v in box]
                _, o_ids, res = predictor.add_new_points_or_box(state, frame_idx=t, obj_id=oid, box=np.array(box, np.float32))
                masks[name][t] = (res[list(o_ids).index(oid), 0] > 0).cpu().numpy()
                lost[name] = 0
            events.append(ev)
        torch.cuda.synchronize()
        per_frame[t] = (time.perf_counter() - ts) * 1000
    predictor.propagate_in_video_preflight(state)
    for oid, name in active:
        for t in range(t0f, T):
            logits[name][t] = obj_logit(state, predictor, oid, t)
    predictor.reset_state(state)
    return masks, logits, per_frame, events, dict(load_ms=load_ms, add_ms=add_ms)


def track_episode(predictor, imgp, gdino, rgb, init, mode, every):
    _ARRAY["mem"] = rgb
    # frame 0 에서 초기화된 물체는 한 group (joint) 또는 물체별(single), 늦게 초기화된 물체는 그 프레임부터 따로
    f0 = [it for it in OBJS.items() if init[it[1]].get("frame", 0) == 0 and init[it[1]]["box"] is not None]
    groups = ([(f0, 0)] if f0 else []) if mode == "joint" else [([it], 0) for it in f0]
    if not getattr(predictor, "samurai_mode", False):  # SAMURAI 는 이전 전 프레임 memory 를 가정 → 늦은 초기화 불가 (미검출로 집계)
        groups += [([it], init[it[1]]["frame"]) for it in OBJS.items() if init[it[1]].get("frame") not in (0, None)]
    events = []
    per_frame = np.zeros(len(rgb))
    tm = dict(load_ms=0.0, add_ms=0.0)
    T, H, W = rgb.shape[:3]
    masks = {n: np.zeros((T, H, W), bool) for n in OBJS.values()}
    logits = {n: np.full(T, np.nan) for n in OBJS.values()}
    for g, t0f in groups:
        m, lg, pf, ev, t = run_group(predictor, imgp, gdino, rgb, g, init, every, "mem", t0f)
        masks.update(m)
        logits.update(lg)
        events += ev
        per_frame += pf
        for k in tm:
            tm[k] += t[k]
    _ARRAY.pop("mem")
    return masks, logits, per_frame, events, tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sam2", choices=list(MODELS))
    ap.add_argument("--model", default="small")
    ap.add_argument("--mode", default="joint", choices=["joint", "single"])
    ap.add_argument("--compile", type=int, default=0)
    ap.add_argument("--reanchor", default="none", choices=list(REANCHOR))
    ap.add_argument("--set", default="eval")
    ap.add_argument("--perturb", default=None, choices=[None, "patch", "dropout"])
    ap.add_argument("--eps", nargs="*", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--warmup", type=int, default=1, help="첫 ep 을 1회 돌리고 버림")
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--out", default=f"{ROOT}/results/v1/E1/runs")
    args = ap.parse_args()
    if args.backend == "samurai" and args.mode != "single":
        ap.error("SAMURAI 는 single 만 가능")
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.cpu_threads)
    sub = args.set + (f"_{args.perturb}" if args.perturb else "")
    out = f"{args.out}/{args.tag}/{sub}"
    os.makedirs(out, exist_ok=True)
    init = json.load(open(f"{ROOT}/results/v1/init/init_boxes_{args.set}.json"))
    eps = args.eps or v1data.ep_list(args.set)

    t0 = time.perf_counter()
    predictor, IP = build(args.backend, args.model, args.compile)
    load_s = time.perf_counter() - t0
    every = REANCHOR[args.reanchor]
    gdino = imgp = None
    if every is not None:
        from grounding import GDino
        gdino = GDino()
        imgp = IP(predictor)
    res = dict(args=vars(args), model_load_s=round(load_s, 1), episodes={})
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if args.warmup:
            d = v1data.load(args.set, eps[0], args.perturb)
            track_episode(predictor, imgp, gdino, d[f"{CAM}/rgb"], init[eps[0]], args.mode, every)
        for ep in eps:
            d = v1data.load(args.set, ep, args.perturb)
            rgb = d[f"{CAM}/rgb"]
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            masks, logits, pf, events, tm = track_episode(predictor, imgp, gdino, rgb, init[ep], args.mode, every)
            gpu = torch.cuda.max_memory_allocated() / 1e9
            np.savez_compressed(f"{out}/{ep}_masks.npz", shape=np.array(masks["peg"].shape),
                                peg=np.packbits(masks["peg"]), box=np.packbits(masks["box"]),
                                peg_logit=logits["peg"], box_logit=logits["box"], per_frame_ms=pf)
            res["episodes"][ep] = dict(frames=len(rgb), load_ms=round(tm["load_ms"], 1), add_ms=round(tm["add_ms"], 1),
                                       ms_mean=round(float(pf[1:].mean()), 2), ms_p50=round(float(np.median(pf[1:])), 2),
                                       ms_p95=round(float(np.percentile(pf[1:], 95)), 2), ms_max=round(float(pf[1:].max()), 1),
                                       gpu_peak_gb=round(gpu, 3), events=events)
            print(args.tag, sub, ep, json.dumps({k: v for k, v in res["episodes"][ep].items() if k != "events"}),
                  f"events={len(events)} acc={sum(e['accepted'] for e in events)}", flush=True)
    json.dump(res, open(f"{out}/run.json", "w"), indent=1)


if __name__ == "__main__":
    main()
