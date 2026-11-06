"""Step 8: 간단 Re-anchor — confidence 하락 / N 프레임 주기로 Grounding DINO 재검출 → SAM2 box 재프롬프트.

SAM2.1-small video predictor 를 프레임 단위로 진행시키면서(`propagate_in_video(start_frame_idx=t, max_frame_num_to_track=0)`)
트리거가 걸린 프레임에서 GDINO 로 재검출하고, 고른 후보 bbox 를 `add_new_points_or_box(frame_idx=t)` 로 넣는다.
`add_all_frames_to_correct_as_cond=True` 라서 재프롬프트한 프레임은 새 conditioning frame 이 된다(frame 0 memory 는 유지).

트리거 (물체별)
  conf   : SAM2 object_score_logit < 0 이 K(=2) 프레임 연속이면 "소실" → 소실 동안 `every` 프레임마다 재검출
  period : t % N == 0 마다 재검출 (tracking 이 정상이어도)
후보 선택 (Step 3: GDINO top-1 은 grasp 후 gripper 로 가므로 그대로 못 씀)
  reid   : 화면 1/4 이상 box 제외 → 후보마다 SAM2 image decoder 로 mask → SAM2 image embedding 을 mask 평균 pooling →
           re-ID bank(frame 0 + 최근 confident(logit>2) 8 프레임의 tracked mask 임베딩)와 cosine 최대값 ≥ 0.75 인 후보 중 최대.
           image feature 는 tracking 이 이미 계산한 frame t backbone feature 를 그대로 쓴다(encoder 재실행 없음).
  top1   : (ablation) 화면 1/4 이상 box 제외 후 GDINO score 최고 후보를 무조건 사용
  gt     : (oracle 상한) GT seg mask bbox (가시 px ≥ 200 일 때만). 검출기가 완벽할 때 re-anchor 정책 자체의 효과.

출력 (results/step8/): <config>/run.json, <config>/ep00X_masks.npz(git 제외), compare.md/json, iou_curves.png,
  events_<ep>.jpg (re-anchor 전/후), ep001_demo.mp4 (+ point cloud, git 제외)

Usage:
  source env.sh
  python scripts/reanchor.py --out results/step8      # 전체 설정 (~15 분)
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import deque

os.environ.setdefault("TQDM_DISABLE", "1")  # 프레임마다 propagate_in_video 를 부르므로 tqdm bar 끄기
import cv2  # noqa: E402
import imageio  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pointcloud as pcl  # noqa: E402
from grounding import GDino, gt_bbox  # noqa: E402
from grounding import iou as box_iou  # noqa: E402
from run_pipeline import PROMPTS, hud  # noqa: E402
from track_sam2 import (MODELS, OBJS, VIS_MIN, iou, label_txt, overlay, phase_labels,  # noqa: E402
                        stats)
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.environ.get("VISION_CACHE", f"{ROOT}/cache")
CAM = "base_camera"
IMG_AREA = 512 * 512
MAX_BOX_FRAC = 0.25  # 화면 1/4 이상 후보 제외 (Step 3: 팔 가림 시 화면 전체 box)
LOST_LOGIT = 0.0  # object_score_logit < 0 = SAM2 가 "물체 없음" 으로 판단
LOST_K = 2  # 연속 K 프레임 소실이면 conf 트리거
BANK_LOGIT = 2.0  # 이 이상일 때만 re-ID bank 에 넣음
BANK_N = 8  # 최근 confident 프레임 수 (+ frame 0 은 고정)
REID_THR = 0.75  # cosine. Step 8 offline probe: 0.75 에서 3 ep 오선택 0 / 가림 중 수락 0
CONFIRM_IOU = 0.5  # 검출기 후보는 다음 프레임 재검출에서 같은 자리(box IoU ≥ 0.5)가 다시 뽑혀야 수락 (1차 실행: ep002 f37 가림 중 오선택)

# name: (conf 트리거, conf 소실 중 재검출 간격, period N(0=off), 후보 선택)
CONFIGS = {
    "none": (False, 0, 0, None),
    "conf": (True, 1, 0, "reid"),
    "conf_e5": (True, 5, 0, "reid"),
    "period10": (False, 0, 10, "reid"),
    "period30": (False, 0, 30, "reid"),
    "conf+period30": (True, 1, 30, "reid"),
    "conf_top1": (True, 1, 0, "top1"),
    "period10_top1": (False, 0, 10, "top1"),
    "conf_gt": (True, 1, 0, "gt"),
    "period10_gt": (False, 0, 10, "gt"),
}


def set_image_features(imgp, backbone_out, hw):
    """tracking 이 계산한 frame t backbone feature 로 SAM2ImagePredictor 상태를 채운다 (set_image 와 같은 변환, encoder 재실행 없음)."""
    m = imgp.model
    _, vision_feats, _, _ = m._prepare_backbone_features(backbone_out)
    if m.directly_add_no_mem_embed:
        vision_feats[-1] = vision_feats[-1] + m.no_mem_embed
    feats = [f.permute(1, 2, 0).view(1, -1, *s) for f, s in zip(vision_feats[::-1], imgp._bb_feat_sizes[::-1])][::-1]
    imgp._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
    imgp._orig_hw = [hw]
    imgp._is_image_set = True


def pool_embed(imgp, mask):
    """image_embed (256,64,64) 를 mask 안에서 평균 → L2 정규화. mask 가 비면 None."""
    feat = imgp._features["image_embed"][0].float()
    mm = F.interpolate(torch.as_tensor(mask, dtype=torch.float32, device=feat.device)[None, None],
                       size=feat.shape[-2:], mode="area")[0, 0]
    if mm.sum() < 0.5:
        return None
    return F.normalize((feat * mm).sum((1, 2)) / mm.sum(), dim=0)


def pick_candidate(mode, gdino, imgp, rgb, obj, bank, gt_mask):
    """returns (box or None, info dict)."""
    if mode == "gt":
        b = gt_bbox(gt_mask)
        return b, dict(gdino_ms=0.0, reid_ms=0.0, n_cand=int(b is not None))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.autocast("cuda", enabled=False):  # GDINO 는 Step 3/6 과 같이 fp32
        boxes, scores = gdino(rgb, PROMPTS[obj])
    torch.cuda.synchronize()
    g_ms = (time.perf_counter() - t0) * 1000
    cands = [(b, s) for b, s in zip(boxes, scores) if (b[2] - b[0]) * (b[3] - b[1]) < MAX_BOX_FRAC * IMG_AREA][:8]
    info = dict(gdino_ms=g_ms, reid_ms=0.0, n_cand=len(cands))
    if not cands:
        return None, info
    if mode == "top1":
        info["score"] = round(cands[0][1], 3)
        return cands[0][0], info
    t0 = time.perf_counter()
    B = torch.stack(list(bank))
    best, best_sim, best_s = None, -1.0, None
    for b, s in cands:
        m, _, _ = imgp.predict(box=np.array(b, np.float32), multimask_output=False)
        e = pool_embed(imgp, m[0] > 0)
        if e is None:
            continue
        sim = float((B @ e).max())
        if sim > best_sim:
            best, best_sim, best_s = b, sim, s
    torch.cuda.synchronize()
    info.update(reid_ms=(time.perf_counter() - t0) * 1000, sim=round(best_sim, 3), score=None if best_s is None else round(best_s, 3))
    return (best if best_sim >= REID_THR else None), info


def obj_logit(state, predictor, oid, t):
    od = state["output_dict_per_obj"][predictor._obj_id_to_idx(state, oid)]
    out = od["cond_frame_outputs"].get(t) or od["non_cond_frame_outputs"].get(t)
    return float(out["object_score_logits"].flatten()[0])


def track_reanchor(predictor, imgp, gdino, d, init_boxes, cfg, tmp_dir):
    use_conf, every, period, mode = cfg
    rgb = d[f"{CAM}/rgb"]
    seg = d[f"{CAM}/seg"]
    ids = {"peg": int(d["peg_id"]), "box": int(d["box_id"])}
    T, H, W = rgb.shape[:3]
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
    t0 = time.perf_counter()
    for oid, name in OBJS.items():
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=oid, box=np.array(init_boxes[name]["box"], np.float32))
    torch.cuda.synchronize()
    add_ms = (time.perf_counter() - t0) * 1000

    need_reid = mode == "reid"
    masks = {n: np.zeros((T, H, W), bool) for n in OBJS.values()}
    logits = {n: np.full(T, np.nan) for n in OBJS.values()}
    ref0 = {}
    banks = {n: deque(maxlen=BANK_N) for n in OBJS.values()}
    lost_run = {n: 0 for n in OBJS.values()}
    pending = {n: None for n in OBJS.values()}  # (t, box): 확인 대기 중인 후보
    events, per_frame, parts = [], [], []
    for t in range(T):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        for fi, obj_ids, res in predictor.propagate_in_video(state, start_frame_idx=t, max_frame_num_to_track=0):
            for j, oid in enumerate(obj_ids):
                masks[OBJS[oid]][fi] = (res[j, 0] > 0).cpu().numpy()
        torch.cuda.synchronize()
        t_track = time.perf_counter()
        p = dict(track=(t_track - t_start) * 1000, bank=0.0, gdino=0.0, reid=0.0, prompt=0.0)
        if need_reid:
            set_image_features(imgp, state["cached_features"][t][1], (H, W))
        for oid, name in OBJS.items():
            lg = obj_logit(state, predictor, oid, t)
            logits[name][t] = lg
            if need_reid and lg > BANK_LOGIT:
                tb = time.perf_counter()
                e = pool_embed(imgp, masks[name][t])
                if e is not None:
                    if t == 0:
                        ref0[name] = e
                    else:
                        banks[name].append(e)
                p["bank"] += (time.perf_counter() - tb) * 1000
            lost_run[name] = lost_run[name] + 1 if lg < LOST_LOGIT else 0
            trig = []
            if use_conf and lost_run[name] >= LOST_K and (lost_run[name] - LOST_K) % every == 0:
                trig.append("conf")
            if period and t > 0 and t % period == 0:
                trig.append("period")
            if pending[name] is not None:
                trig.append("confirm")
            if not trig:
                continue
            bank = ([ref0[name]] if name in ref0 else []) + list(banks[name])
            box, info = pick_candidate(mode, gdino, imgp, rgb[t], name, bank, seg[t] == ids[name])
            p["gdino"] += info["gdino_ms"]
            p["reid"] += info["reid_ms"]
            gt_b = gt_bbox(seg[t] == ids[name])
            cand = box
            if mode != "gt" and box is not None:  # 2 프레임 연속 같은 자리일 때만 수락
                prev = pending[name]
                ok = prev is not None and prev[0] == t - 1 and box_iou(box, prev[1]) >= CONFIRM_IOU
                pending[name] = None if ok else (t, box)
                box = box if ok else None
            else:
                pending[name] = None
            ev = dict(t=t, obj=name, trigger="+".join(trig), logit=round(lg, 2), accepted=box is not None,
                      pending=box is None and cand is not None,
                      **{k: (round(v, 1) if k.endswith("_ms") else v) for k, v in info.items()},
                      gt_visible=gt_b is not None)
            if box is not None:
                ev["box"] = [round(float(v), 1) for v in box]
                ev["box_iou_gt"] = round(box_iou(box, gt_b), 3) if gt_b else 0.0
                ev["mask_before_px"] = int(masks[name][t].sum())
                ev["iou_before"] = round(iou(masks[name][t], seg[t] == ids[name]), 3) if gt_b else None
                tp = time.perf_counter()
                _, o_ids, res = predictor.add_new_points_or_box(state, frame_idx=t, obj_id=oid, box=np.array(box, np.float32))
                masks[name][t] = (res[list(o_ids).index(oid), 0] > 0).cpu().numpy()
                torch.cuda.synchronize()
                p["prompt"] += (time.perf_counter() - tp) * 1000
                ev["iou_after"] = round(iou(masks[name][t], seg[t] == ids[name]), 3) if gt_b else None
                ev["mask_after_px"] = int(masks[name][t].sum())
                lost_run[name] = 0
            events.append(ev)
        torch.cuda.synchronize()
        per_frame.append((time.perf_counter() - t_start) * 1000)
        parts.append(p)
    # 재프롬프트한 프레임의 최종 logit (cond output) 다시 읽기. 마지막 프레임 재프롬프트는 temp 에 남아 있으므로 consolidate
    predictor.propagate_in_video_preflight(state)
    for oid, name in OBJS.items():
        for t in range(T):
            logits[name][t] = obj_logit(state, predictor, oid, t)
    n_cond = {name: len(state["output_dict_per_obj"][predictor._obj_id_to_idx(state, oid)]["cond_frame_outputs"])
              for oid, name in OBJS.items()}
    predictor.reset_state(state)
    shutil.rmtree(tmp_dir)
    pf = np.array(per_frame)
    timing = dict(load_frames_ms=load_ms, add_box_ms=add_ms, first_step_ms=float(pf[0]),
                  per_frame_ms=float(pf[1:].mean()), per_frame_ms_p95=float(np.percentile(pf[1:], 95)),
                  per_frame_ms_max=float(pf[1:].max()),
                  **{f"{k}_ms_per_frame": float(np.mean([q[k] for q in parts[1:]])) for k in parts[0]})
    return masks, logits, timing, per_frame, events, n_cond


def evaluate(d, masks, logits):
    seg = d[f"{CAM}/seg"]
    T = len(seg)
    ph = phase_labels(d, int(d["peg_id"]))
    r, ious = {}, {}
    for n in OBJS.values():
        gt = seg == int(d[f"{n}_id"])
        gpx = gt.reshape(T, -1).sum(1)
        ious[n] = np.array([iou(masks[n][t], gt[t]) if gpx[t] >= VIS_MIN else np.nan for t in range(T)])
        r[n] = stats(ious[n])
        r[n]["fp_when_hidden"] = int(((gpx < VIS_MIN) & (masks[n].reshape(T, -1).sum(1) >= VIS_MIN)).sum())
    clean = ~(ph["arm_hidden"] | ph["gripper"] | ph["insertion"])
    r["peg_by_phase"] = {k: stats(ious["peg"][m]) for k, m in
                         (("clean", clean), ("gripper", ph["gripper"]), ("insertion", ph["insertion"]))}
    hid = np.nonzero(ph["arm_hidden"])[0]
    if len(hid):
        after = np.nonzero((np.arange(T) > hid[-1]) & (ious["peg"] >= 0.5))[0]
        r["peg_reacquire_after_hidden"] = int(after[0] - hid[-1]) if len(after) else None
    return r, ious, ph


def summarize_events(events):
    acc = [e for e in events if e["accepted"]]
    # 잘못된 re-anchor = GT 가 안 보이는데 수락 or 고른 box 가 GT bbox 와 IoU < 0.5
    wrong = [e for e in acc if (not e["gt_visible"]) or e["box_iou_gt"] < 0.5]
    gd = [e for e in events if e.get("gdino_ms", 0) > 0]
    return dict(triggers=len(events), gdino_calls=len(gd), accepted=len(acc), wrong=len(wrong),
                helped=sum(1 for e in acc if e.get("iou_before") is not None and e["iou_after"] - e["iou_before"] > 0.05),
                hurt=sum(1 for e in acc if e.get("iou_before") is not None and e["iou_before"] - e["iou_after"] > 0.05))


def event_panel(d, masks_before, masks_after, ev):
    t, obj = ev["t"], ev["obj"]
    rgb, seg = d[f"{CAM}/rgb"][t], d[f"{CAM}/seg"][t]
    gt = {obj: seg == int(d[f"{obj}_id"])}
    a = overlay(rgb, {obj: masks_before[obj][t]}, gt)
    b = overlay(rgb, {obj: masks_after[obj][t]}, gt)
    x1, y1, x2, y2 = [int(round(v)) for v in ev["box"]]
    cv2.rectangle(b, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 0), 2)
    label_txt(a, [f"t={t} {obj}: no re-anchor run (none)", f"this run pre-prompt: logit={ev['logit']} IoU={ev.get('iou_before')}"])
    label_txt(b, [f"AFTER re-anchor (yellow=GDINO box)", f"sim={ev.get('sim')} score={ev.get('score')} boxIoU_gt={ev['box_iou_gt']}",
                  f"IoU={ev.get('iou_after')} px={ev['mask_after_px']}"])
    return np.concatenate([a, b], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{ROOT}/data/step2")
    ap.add_argument("--boxes", default=f"{ROOT}/results/step3/init_boxes.json")
    ap.add_argument("--eps", nargs="+", default=["ep000", "ep001", "ep002"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--warmup-ep", default="ep000")
    ap.add_argument("--out", default=f"{ROOT}/results/step8")
    ap.add_argument("--demo", default="conf", help="이 설정의 --demo-ep 로 point cloud + 데모 영상")
    ap.add_argument("--demo-ep", default="ep001")
    ap.add_argument("--cpu-threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.cpu_threads)
    os.makedirs(args.out, exist_ok=True)
    init_boxes = json.load(open(args.boxes))

    gdino = GDino()
    predictor = build_sam2_video_predictor(*MODELS["small"], device="cuda")
    predictor.add_all_frames_to_correct_as_cond = True  # 재프롬프트 프레임을 conditioning frame 으로
    imgp = SAM2ImagePredictor(predictor)  # 같은 weight 공유 (prompt encoder / mask decoder 만 사용)
    D = {ep: dict(np.load(f"{args.data}/{ep}.npz")) for ep in sorted(set(args.eps + [args.warmup_ep]))}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.autocast("cuda", enabled=False):
            for _ in range(5):  # GDINO 워밍업 (Step 6 과 동일)
                for pr in PROMPTS.values():
                    gdino(D[args.warmup_ep][f"{CAM}/rgb"][0], pr)
        # 워밍업: 모든 코드 경로(conf+period, reid)를 1 ep 돌리고 버림
        track_reanchor(predictor, imgp, gdino, D[args.warmup_ep], init_boxes[args.warmup_ep],
                       (True, 1, 10, "reid"), f"{CACHE}/step8_frames/warmup")
        print("warmup done", flush=True)

        allres = {}
        for name in args.configs:
            cfg = CONFIGS[name]
            out = f"{args.out}/{name}"
            os.makedirs(out, exist_ok=True)
            res = dict(config=dict(zip(("conf", "conf_every", "period", "select"), cfg)), episodes={})
            for ep in args.eps:
                d = D[ep]
                torch.cuda.reset_peak_memory_stats()
                masks, logits, timing, per_frame, events, n_cond = track_reanchor(
                    predictor, imgp, gdino, d, init_boxes[ep], cfg, f"{CACHE}/step8_frames/{ep}")
                gpu = torch.cuda.max_memory_allocated() / 1e9
                r, ious, _ = evaluate(d, masks, logits)
                r.update(timing={k: round(v, 1) for k, v in timing.items()}, gpu_peak_alloc_gb=round(gpu, 2),
                         events=summarize_events(events), n_cond_frames=n_cond, event_list=events,
                         per_frame_ms=[round(v, 1) for v in per_frame])
                res["episodes"][ep] = r
                np.savez_compressed(f"{out}/{ep}_masks.npz", peg=masks["peg"], box=masks["box"],
                                    peg_logit=logits["peg"], box_logit=logits["box"], peg_iou=ious["peg"], box_iou=ious["box"])
                print(name, ep, json.dumps(dict(peg=r["peg"], box=r["box"], ev=r["events"],
                                                ms=r["timing"]["per_frame_ms"], p95=r["timing"]["per_frame_ms_p95"])), flush=True)
            eps = list(res["episodes"].values())
            allp = np.concatenate([np.load(f"{out}/{ep}_masks.npz")["peg_iou"] for ep in args.eps])
            allb = np.concatenate([np.load(f"{out}/{ep}_masks.npz")["box_iou"] for ep in args.eps])
            res["all"] = dict(peg=stats(allp), box=stats(allb),
                              **{k: round(float(np.mean([e["timing"][k] for e in eps])), 1) for k in eps[0]["timing"]},
                              gpu_peak_alloc_gb=max(e["gpu_peak_alloc_gb"] for e in eps),
                              **{k: sum(e["events"][k] for e in eps) for k in eps[0]["events"]},
                              fp_when_hidden={o: sum(e[o]["fp_when_hidden"] for e in eps) for o in OBJS.values()})
            json.dump(res, open(f"{out}/run.json", "w"), indent=1)
            allres[name] = res
    write_tables(allres, args)
    plot_curves(allres, args, D)
    event_images(allres, args, D)
    if args.demo in allres:
        demo(allres, args, D[args.demo_ep])


def write_tables(allres, args):
    base = allres.get("none")
    L = ["# Step 8 Re-anchor 비교 (SAM2.1-small, base_camera 512², 3 ep, 워밍업 1 ep 제외, CPU 8 스레드, bf16)", "",
         "| config | 트리거 | 선택 | peg IoU mean / min (hit@0.5) | box IoU mean / min | ΔIoU peg / box vs none | "
         "GDINO 호출 (3 ep) | re-anchor 수락 / 오선택 | 가림 중 오탐 peg/box | ms/frame mean | p95 | max | Δms vs none | FPS (tracking) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in allres.items():
        a, c = r["all"], r["config"]
        trig = "+".join([k for k, on in (("conf" + (f"/{c['conf_every']}f" if c["conf_every"] > 1 else ""), c["conf"]),
                                         (f"N={c['period']}", c["period"])) if on]) or "–"
        dp = f"{a['peg']['mean'] - base['all']['peg']['mean']:+.3f} / {a['box']['mean'] - base['all']['box']['mean']:+.3f}" if base else ""
        dm = f"{a['per_frame_ms'] - base['all']['per_frame_ms']:+.1f}" if base else ""
        L.append(f"| {name} | {trig} | {c['select'] or '–'} | {a['peg']['mean']:.3f} / {a['peg']['min']:.3f} ({a['peg']['hit50']:.2f}) | "
                 f"{a['box']['mean']:.3f} / {a['box']['min']:.3f} | {dp} | {a['gdino_calls']} | {a['accepted']} / {a['wrong']} | "
                 f"{a['fp_when_hidden']['peg']} / {a['fp_when_hidden']['box']} | {a['per_frame_ms']:.1f} | {a['per_frame_ms_p95']:.1f} | "
                 f"{a['per_frame_ms_max']:.0f} | {dm} | {1000 / a['per_frame_ms']:.1f} |")
    L += ["", "## 에피소드별 peg IoU mean (재획득 프레임 수: 완전 가림 해제 후 IoU ≥ 0.5 까지)", "",
          "| config | " + " | ".join(args.eps) + " | box " + " / ".join(args.eps) + " |", "|---|" + "---|" * (len(args.eps) + 1)]
    for name, r in allres.items():
        cells = []
        for ep in args.eps:
            e = r["episodes"][ep]
            ra = e.get("peg_reacquire_after_hidden", "–")
            cells.append(f"{e['peg']['mean']:.3f} (재획득 {ra if ra is not None else '실패'})")
        L.append(f"| {name} | " + " | ".join(cells) + " | " + " / ".join(f"{r['episodes'][ep]['box']['mean']:.3f}" for ep in args.eps) + " |")
    L += ["", "## 시간 분해 (3 ep 평균, ms/frame: 프레임 평균이라 트리거 없는 프레임 포함)", "",
          "| config | track | re-ID bank | GDINO | re-ID 후보 | 재프롬프트 | conditioning frame 수 peg/box (ep 평균) | GPU peak alloc GB |",
          "|---|---|---|---|---|---|---|---|"]
    for name, r in allres.items():
        a = r["all"]
        nc = [np.mean([r["episodes"][ep]["n_cond_frames"][o] for ep in args.eps]) for o in OBJS.values()]
        L.append(f"| {name} | {a['track_ms_per_frame']:.1f} | {a['bank_ms_per_frame']:.2f} | {a['gdino_ms_per_frame']:.1f} | "
                 f"{a['reid_ms_per_frame']:.1f} | {a['prompt_ms_per_frame']:.2f} | {nc[0]:.1f} / {nc[1]:.1f} | {a['gpu_peak_alloc_gb']} |")
    open(f"{args.out}/compare.md", "w").write("\n".join(L) + "\n")
    json.dump({k: dict(config=v["config"], all=v["all"]) for k, v in allres.items()}, open(f"{args.out}/compare.json", "w"), indent=1)
    print("\n".join(L))


def plot_curves(allres, args, D):
    show = [c for c in ("none", "conf", "period10", "conf_top1", "conf_gt") if c in allres]
    cols = {"none": "k", "conf": "r", "period10": "b", "conf_top1": "orange", "conf_gt": "g"}
    fig, axes = plt.subplots(len(args.eps), 2, figsize=(16, 3.3 * len(args.eps)), squeeze=False)
    for i, ep in enumerate(args.eps):
        ph = phase_labels(D[ep], int(D[ep]["peg_id"]))
        for j, obj in enumerate(OBJS.values()):
            ax = axes[i, j]
            x = np.arange(len(ph["arm_hidden"]))
            ax.fill_between(x, 0, 1.05, where=ph["arm_hidden"], color="0.85", step="mid", label="peg arm_hidden")
            for c in show:
                v = np.load(f"{args.out}/{c}/{ep}_masks.npz")[f"{obj}_iou"]
                ax.plot(x, v, ".-", ms=2, lw=1, color=cols[c], label=c, alpha=0.8)
                for e in allres[c]["episodes"][ep]["event_list"]:
                    if e["obj"] == obj and e["accepted"]:
                        ax.plot(e["t"], 1.02 if c != "none" else 1.0, "v", color=cols[c], ms=5)
            ax.set_ylim(0, 1.06)
            ax.set_title(f"{ep} {obj} IoU (▼ = accepted re-anchor)")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{args.out}/iou_curves.png", dpi=100)


def event_images(allres, args, D):
    """주요 설정의 수락된 re-anchor 이벤트: 전/후 mask. 설정·ep 당 최대 4개 (IoU 변화 큰 순)."""
    for c in ("conf", "period10", "conf_top1"):
        if c not in allres:
            continue
        for ep in args.eps:
            acc = [e for e in allres[c]["episodes"][ep]["event_list"] if e["accepted"]]
            if not acc:
                continue
            acc = sorted(acc, key=lambda e: -abs((e.get("iou_after") or 0) - (e.get("iou_before") or 0)))[:4]
            before = np.load(f"{args.out}/none/{ep}_masks.npz") if "none" in allres else None
            after = np.load(f"{args.out}/{c}/{ep}_masks.npz")
            rows = []
            for e in sorted(acc, key=lambda e: e["t"]):
                mb = {e["obj"]: before[e["obj"]] if before is not None else after[e["obj"]]}
                # "before" = 같은 프레임의 re-anchor 직전 mask 는 저장 안 하므로, none(재앵커 없음) 결과를 비교 대상으로 쓴다
                rows.append(event_panel(D[ep], mb, {e["obj"]: after[e["obj"]]}, e))
            cv2.imwrite(f"{args.out}/events_{c}_{ep}.jpg", cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])


def demo(allres, args, d):
    """demo 설정 masks 로 point cloud(Step 5 와 동일 처리) + 데모 영상 (none 과 나란히 비교)."""
    c, ep = args.demo, args.demo_ep
    T = len(d["frame_idx"])
    m = np.load(f"{args.out}/{c}/{ep}_masks.npz")
    m0 = np.load(f"{args.out}/none/{ep}_masks.npz")
    masks = {o: m[o] for o in OBJS.values()}
    logits = {o: m[f"{o}_logit"] for o in OBJS.values()}
    rows, clouds, confs, pc_ms = pcl.process_episode(d, masks, logits, f"{CACHE}/step8_pc/{ep}", np.random.default_rng(0))
    summ = pcl.summarize(rows)
    json.dump(dict(config=c, ep=ep, pointcloud=summ, pointcloud_ms_per_frame=float(np.sum(pc_ms) / T)),
              open(f"{args.out}/demo_{c}_{ep}_pointcloud.json", "w"), indent=1)
    lim = pcl.scene_limits(d)
    ev_t = {e["t"]: e for e in allres[c]["episodes"][ep]["event_list"] if e["accepted"]}
    pf = allres[c]["episodes"][ep]["per_frame_ms"]
    stills = []
    with imageio.get_writer(f"{args.out}/{ep}_demo.mp4", fps=10, macro_block_size=1) as w:
        for t in range(T):
            img = pcl.render_frame(d, t, masks, clouds[t], confs[t], lim, f"{ep} re-anchor={c}")
            base = overlay(d[f"{CAM}/rgb"][t], {o: m0[o][t] for o in OBJS.values()})
            base = cv2.resize(base, (img.shape[0], img.shape[0]))
            label_txt(base, ["no re-anchor (Step 4)", f"peg IoU={m0['peg_iou'][t]:.2f}"])
            img = np.concatenate([base, img], 1)
            ev = ev_t.get(t)
            img = hud(img, [f"t={t} frame {pf[t]:.0f} ms | peg IoU {m['peg_iou'][t]:.2f} (none {m0['peg_iou'][t]:.2f}) "
                            f"| points peg {len(clouds[t]['peg'])} box {len(clouds[t]['box'])}",
                            f"RE-ANCHOR {ev['obj']} sim={ev['sim']} ({ev['trigger']})" if ev else ""])
            w.append_data(img)
            if t in (20, 40, 66, 67, 80, T - 1):
                stills.append(img)
    cv2.imwrite(f"{args.out}/{ep}_demo_frames.jpg", cv2.cvtColor(np.concatenate(stills, 0), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 80])
    print("demo", json.dumps(summ))


if __name__ == "__main__":
    main()
