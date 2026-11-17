"""Step 4: SAM2 video predictor tracking (첫 프레임 GDINO bbox 초기화) + GT mask IoU + 가림 구간 분석.

- 입력: data/step2/ep00X.npz (base_camera), results/step3/init_boxes.json (frame 0 peg/box bbox)
- SAM2 video predictor 를 frame 0 bbox 2개(obj 1=peg, 2=box)로 초기화 → 에피소드 끝까지 forward propagate
- 프레임별 IoU (GT seg mask), object_score_logits (SAM2 자체 '물체 있음' confidence)
- 구간 라벨 (GT 기반, 휴리스틱):
    arm_hidden : GT peg 가시 px < VIS_MIN (팔이 카메라 앞을 지나며 완전히 가림)
    gripper    : peg GT mask 경계(5px 링) 안에 gripper(hand/finger, id 10-13) 픽셀이 있음 = gripper가 peg에 겹쳐 보임
    insertion  : peg head 가 hole 에서 INS_DIST 이내 (삽입 진행 중, peg 가 box 에 가려지기 시작)
- 출력 (results/step4/<model>/): tracking.json, iou_curve.png, ep00X_overlay.mp4, failure_*.jpg, masks ep00X.npz(git 제외)

Usage:
  source env.sh
  python scripts/track_sam2.py --model small --out results/step4
"""
import argparse
import json
import os
import shutil
import time

import cv2
import imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.environ.get("VISION_CACHE", f"{ROOT}/cache")
MODELS = {
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", f"{ROOT}/checkpoints/sam2.1_hiera_small.pt"),
    "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", f"{ROOT}/checkpoints/sam2.1_hiera_base_plus.pt"),
}
CAM = "base_camera"
OBJS = {1: "peg", 2: "box"}
COLORS = {"peg": np.array([255, 40, 40]), "box": np.array([40, 90, 255])}
GRIPPER_IDS = (10, 11, 12, 13)
VIS_MIN = 200  # Step 3 과 동일: GT 가시 px ≥ 200 일 때만 IoU 평가
INS_DIST = 0.05  # m


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else float("nan")


def phase_labels(d, peg_id):
    seg = d[f"{CAM}/seg"]
    T = len(seg)
    peg_px = (seg == peg_id).reshape(T, -1).sum(1)
    grip_ring = np.zeros(T, int)
    k = np.ones((11, 11), np.uint8)
    for t in range(T):
        m = (seg[t] == peg_id).astype(np.uint8)
        if m.sum() == 0:
            continue
        ring = cv2.dilate(m, k) > 0
        grip_ring[t] = np.isin(seg[t][ring], GRIPPER_IDS).sum()
    head_hole = np.linalg.norm(d["peg_head_pose"][:, :3] - d["box_hole_pose"][:, :3], axis=1)
    arm_hidden = peg_px < VIS_MIN
    gripper = (grip_ring > 0) & ~arm_hidden
    insertion = (head_hole < INS_DIST) & ~arm_hidden
    return dict(peg_px=peg_px, grip_ring=grip_ring, head_hole=head_hole,
                arm_hidden=arm_hidden, gripper=gripper, insertion=insertion)


def overlay(rgb, masks, gt=None, a=0.5):
    out = rgb.astype(np.float32).copy()
    for name, m in masks.items():
        out[m] = (1 - a) * out[m] + a * COLORS[name]
    out = out.astype(np.uint8)
    if gt is not None:  # GT 윤곽선 (초록)
        for m in gt.values():
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, (0, 255, 0), 1)
    return out


def track_episode(predictor, rgb, boxes, tmp_dir):
    """returns masks {name: (T,H,W) bool}, logits {name: (T,)}, timing dict."""
    T = len(rgb)
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    t0 = time.perf_counter()
    for t in range(T):
        cv2.imwrite(f"{tmp_dir}/{t:05d}.jpg", cv2.cvtColor(rgb[t], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    t_write = time.perf_counter() - t0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    state = predictor.init_state(video_path=tmp_dir)
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    for oid, name in OBJS.items():
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=oid, box=np.array(boxes[name]["box"], np.float32))
    torch.cuda.synchronize()
    t_prompt = time.perf_counter() - t0

    masks = {n: np.zeros((T,) + rgb.shape[1:3], bool) for n in OBJS.values()}
    per_frame = []
    t_prev = time.perf_counter()
    for fi, obj_ids, res in predictor.propagate_in_video(state):
        for j, oid in enumerate(obj_ids):
            masks[OBJS[oid]][fi] = (res[j, 0] > 0).cpu().numpy()
        torch.cuda.synchronize()
        now = time.perf_counter()
        per_frame.append((now - t_prev) * 1000)
        t_prev = now

    logits = {n: np.full(T, np.nan) for n in OBJS.values()}
    for oid, name in OBJS.items():
        od = state["output_dict_per_obj"][predictor._obj_id_to_idx(state, oid)]
        for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for fi, out in od[key].items():
                logits[name][fi] = float(out["object_score_logits"].flatten()[0])
    predictor.reset_state(state)
    shutil.rmtree(tmp_dir)
    # per_frame[0] 은 propagate preflight(frame 0 memory 인코딩) 포함 → 제외
    timing = dict(write_jpeg_ms=t_write * 1000, load_frames_ms=t_load * 1000, add_box_ms=t_prompt * 1000,
                  first_step_ms=per_frame[0], per_frame_ms=float(np.mean(per_frame[1:])),
                  per_frame_ms_p95=float(np.percentile(per_frame[1:], 95)))
    return masks, logits, timing, per_frame


def stats(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return dict(n=0)
    return dict(n=int(len(x)), mean=round(float(x.mean()), 4), min=round(float(x.min()), 4),
                hit50=round(float((x >= 0.5).mean()), 3))


def label_txt(img, lines, y0=20):
    for i, s in enumerate(lines):
        cv2.putText(img, s, (6, y0 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, s, (6, y0 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def frame_panel(d, t, masks, ious, logits, ph, peg_id, box_id):
    rgb, seg = d[f"{CAM}/rgb"][t], d[f"{CAM}/seg"][t]
    gt = {"peg": seg == peg_id, "box": seg == box_id}
    pred = {n: masks[n][t] for n in OBJS.values()}
    gt_ov = overlay(rgb, gt)
    pr_ov = overlay(rgb, pred, gt)
    tags = [k for k in ("arm_hidden", "gripper", "insertion") if ph[k][t]]
    label_txt(gt_ov, [f"GT t={t} {'/'.join(tags)}", f"peg_px={ph['peg_px'][t]} box_px={int(gt['box'].sum())}"])
    label_txt(pr_ov, [f"SAM2 (green=GT contour)",
                      f"peg IoU={ious['peg'][t]:.2f} logit={logits['peg'][t]:.1f} px={int(pred['peg'].sum())}",
                      f"box IoU={ious['box'][t]:.2f} logit={logits['box'][t]:.1f} px={int(pred['box'].sum())}"])
    return np.concatenate([gt_ov, pr_ov], 1)


def pick_failures(ious, logits, ph, masks, n=3):
    """구간별 대표 실패 프레임 (구간 시작 / IoU 최저 / 구간 종료 직후 회복 여부)."""
    out = {}
    T = len(ious["peg"])
    # 완전 가림 경계(±3 프레임)는 arm_hidden 쪽에서 보므로 gripper 구간에서는 제외
    near_hidden = np.convolve(ph["arm_hidden"].astype(int), np.ones(7, int), "same") > 0
    only = {"gripper": ph["gripper"] & ~ph["insertion"] & ~near_hidden, "insertion": ph["insertion"]}
    for key, m in only.items():
        idx = np.nonzero(m)[0]
        idx = idx[~np.isnan(ious["peg"][idx])]
        if len(idx) == 0:
            continue
        sel = [int(idx[0]), int(idx[np.argmin(ious["peg"][idx])]), int(idx[-1])]
        out[key] = sorted(set(sel))
    idx = np.nonzero(ph["arm_hidden"])[0]
    if len(idx):
        # 완전 가림: 가림 중 예측 mask 크기가 최대인 프레임(오탐) + 가림 해제 후 첫 3프레임(재획득 여부)
        fp = int(idx[np.argmax(masks["peg"][idx].reshape(len(idx), -1).sum(1))])
        after = [int(idx[-1]) + k for k in (1, 3, 10) if idx[-1] + k < T]
        out["arm_hidden"] = sorted(set([int(idx[0]), fp] + after))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{ROOT}/data/step2")
    ap.add_argument("--boxes", default=f"{ROOT}/results/step3/init_boxes.json")
    ap.add_argument("--model", default="small", choices=list(MODELS))
    ap.add_argument("--eps", nargs="+", default=["ep000", "ep001", "ep002"])
    ap.add_argument("--out", default=f"{ROOT}/results/step4")
    ap.add_argument("--video", nargs="*", default=None, help="영상 저장할 ep (기본: 전부)")
    args = ap.parse_args()
    out = f"{args.out}/{args.model}"
    os.makedirs(out, exist_ok=True)
    init_boxes = json.load(open(args.boxes))

    cfg, ckpt = MODELS[args.model]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    predictor = build_sam2_video_predictor(cfg, ckpt, device="cuda")
    t_model = time.perf_counter() - t0
    print(f"[{args.model}] model load {t_model:.1f}s")

    result = dict(model=args.model, camera=CAM, vis_min_px=VIS_MIN, model_load_s=round(t_model, 2), episodes={})
    fig, axes = plt.subplots(len(args.eps), 1, figsize=(12, 3.2 * len(args.eps)), squeeze=False)
    for ei, ep in enumerate(args.eps):
        d = dict(np.load(f"{args.data}/{ep}.npz"))
        peg_id, box_id = int(d["peg_id"]), int(d["box_id"])
        rgb, seg = d[f"{CAM}/rgb"], d[f"{CAM}/seg"]
        T = len(rgb)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            masks, logits, timing, _ = track_episode(predictor, rgb, init_boxes[ep], f"{CACHE}/step4_frames/{ep}")
        ph = phase_labels(d, peg_id)
        gt = {"peg": seg == peg_id, "box": seg == box_id}
        gt_px = {n: gt[n].reshape(T, -1).sum(1) for n in gt}
        ious = {n: np.array([iou(masks[n][t], gt[n][t]) if gt_px[n][t] >= VIS_MIN else np.nan for t in range(T)])
                for n in gt}
        # GT 가 (거의) 안 보이는데 SAM2 가 mask 를 내는 경우 = 오탐
        fp = {n: int(((gt_px[n] < VIS_MIN) & (masks[n].reshape(T, -1).sum(1) >= VIS_MIN)).sum()) for n in gt}
        hidden = {n: int((gt_px[n] < VIS_MIN).sum()) for n in gt}

        r = dict(frames=T, timing={k: round(v, 1) for k, v in timing.items()},
                 peg=stats(ious["peg"]), box=stats(ious["box"]),
                 hidden_frames=hidden, false_positive_when_hidden=fp,
                 peg_by_phase={}, logit_corr={})
        clean = ~(ph["arm_hidden"] | ph["gripper"] | ph["insertion"])
        for k, m in (("clean", clean), ("gripper", ph["gripper"]), ("insertion", ph["insertion"])):
            r["peg_by_phase"][k] = stats(ious["peg"][m])
            r["peg_by_phase"][k]["frames"] = [int(i) for i in np.nonzero(m)[0][[0, -1]]] if m.any() else []
        r["peg_by_phase"]["arm_hidden"] = dict(n=int(ph["arm_hidden"].sum()))
        for n in gt:
            v = ~np.isnan(ious[n])
            if v.sum() > 2:
                r["logit_corr"][n] = round(float(np.corrcoef(ious[n][v], logits[n][v])[0, 1]), 3)
        # 재획득: 완전 가림이 끝난 뒤 peg IoU ≥ 0.5 로 돌아오기까지 프레임 수
        hid = np.nonzero(ph["arm_hidden"])[0]
        if len(hid):
            after = np.nonzero((np.arange(T) > hid[-1]) & (ious["peg"] >= 0.5))[0]
            r["peg_reacquire_after_hidden"] = int(after[0] - hid[-1]) if len(after) else None
        fails = pick_failures(ious, logits, ph, masks)
        r["failure_frames"] = fails
        result["episodes"][ep] = r
        print(ep, json.dumps({k: r[k] for k in ("peg", "box", "timing", "false_positive_when_hidden")}))

        # 실패 프레임 이미지
        for key, frs in fails.items():
            rows = [frame_panel(d, t, masks, ious, logits, ph, peg_id, box_id) for t in frs]
            cv2.imwrite(f"{out}/failure_{ep}_{key}.jpg", cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        # 영상
        if args.video is None or ep in args.video:
            with imageio.get_writer(f"{out}/{ep}_overlay.mp4", fps=10, macro_block_size=1) as w:
                for t in range(T):
                    w.append_data(frame_panel(d, t, masks, ious, logits, ph, peg_id, box_id))
        np.savez_compressed(f"{out}/{ep}_masks.npz", peg=masks["peg"], box=masks["box"],
                            peg_logit=logits["peg"], box_logit=logits["box"], peg_iou=ious["peg"], box_iou=ious["box"])

        # IoU 곡선 + 구간 음영
        ax = axes[ei, 0]
        x = np.arange(T)
        for k, c in (("arm_hidden", "0.8"), ("gripper", "#ffd27f"), ("insertion", "#a8e6a1")):
            ax.fill_between(x, 0, 1.05, where=ph[k], color=c, alpha=0.6, step="mid", label=k)
        ax.plot(x, ious["peg"], "r.-", ms=3, label="peg IoU")
        ax.plot(x, ious["box"], "b.-", ms=3, label="box IoU")
        ax2 = ax.twinx()
        ax2.plot(x, logits["peg"], "r:", lw=1, label="peg obj logit")
        ax2.plot(x, logits["box"], "b:", lw=1, label="box obj logit")
        ax2.set_ylabel("object_score_logit")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("IoU")
        ax.set_title(f"{ep} SAM2.1-{args.model}: peg mean {r['peg'].get('mean')} min {r['peg'].get('min')} | "
                     f"box mean {r['box'].get('mean')} min {r['box'].get('min')}")
        if ei == 0:
            ax.legend(loc="lower left", fontsize=7, ncol=5)
            ax2.legend(loc="lower right", fontsize=7)
    axes[-1, 0].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(f"{out}/iou_curve.png", dpi=110)

    result["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    allp = np.concatenate([np.load(f"{out}/{ep}_masks.npz")["peg_iou"] for ep in args.eps])
    allb = np.concatenate([np.load(f"{out}/{ep}_masks.npz")["box_iou"] for ep in args.eps])
    result["all"] = dict(peg=stats(allp), box=stats(allb),
                         per_frame_ms=round(float(np.mean([result["episodes"][e]["timing"]["per_frame_ms"]
                                                           for e in args.eps])), 1))
    json.dump(result, open(f"{out}/tracking.json", "w"), indent=1)
    print("ALL", json.dumps(result["all"]), "gpu_peak_gb", result["gpu_peak_gb"])


if __name__ == "__main__":
    main()
