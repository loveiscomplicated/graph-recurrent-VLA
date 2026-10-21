"""Step 3 Grounding: Grounding DINO(HF)로 peg / box 검출 + GT mask bbox 대비 IoU, 프롬프트 비교.

- 물체별로 프롬프트를 따로 질의 (text = "<prompt>."), 예측 = 최고 score box (top-1).
- 평가 프레임
  * init  : 각 에피소드 base_camera frame 0 (Step 4 SAM2 초기화에 쓸 프레임)
  * sample: base_camera 10 프레임 간격, GT 가시 픽셀 >= MIN_PX 인 프레임 (가림/자세 변화 견고성)
- 지표: top-1 IoU, hit(IoU>=0.5), oracle IoU(score>=0.1 후보 중 최대 IoU; 랭킹 문제인지 판별용)
- 최적 프롬프트 = sample 프레임 hit rate 최대 (동률 시 mean IoU), init 프레임 3개 모두 hit 이어야 함.
- 최적 프롬프트로 전 프레임(base_camera) IoU 곡선, 오버레이 저장, Step 4용 초기 bbox(init_boxes.json) 저장.
  init 프레임에서 hit 실패 시 해당 물체는 GT mask bbox 로 대체(source="gt_fallback").

Usage:
  source env.sh
  python scripts/grounding.py --data data/step2 --out results/step3
"""
import argparse
import glob
import json
import os
import time

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GDINO_ID = "IDEA-Research/grounding-dino-tiny"
CAM = "base_camera"
MIN_PX = 200
STRIDE = 10
CAND_THR = 0.1  # 후보 수집용 낮은 threshold (top-1 은 score 와 무관하게 최고점)
HIT_IOU = 0.5

PROMPTS = {
    "peg": [
        "peg",
        "white peg with red tip",
        "red and white block",
        "white block",
        "white rectangular block",
        "white stick",
        "white bar",
        "cuboid",
    ],
    "box": [
        "box with hole",
        "box",
        "wooden box",
        "brown box",
        "cardboard box",
        "brown block with a hole",
        "brown block",
        "hole",
    ],
}
COLORS = {"peg": (255, 60, 60), "box": (60, 120, 255)}


def gt_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_PX:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class GDino:
    def __init__(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.proc = AutoProcessor.from_pretrained(GDINO_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).cuda().eval()

    @torch.no_grad()
    def __call__(self, rgb, prompt):
        t0 = time.perf_counter()
        img = Image.fromarray(rgb)
        inputs = self.proc(images=img, text=prompt.lower().strip() + ".", return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = self.model(**inputs)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        det = self.proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=CAND_THR, text_threshold=CAND_THR,
            target_sizes=[img.size[::-1]],
        )[0]
        boxes, scores = det["boxes"].cpu().numpy(), det["scores"].cpu().numpy()
        # 마지막 호출의 구간별 시간 (ms): 전처리(HF processor, CPU) / forward(GPU) / 후처리
        self.last_ms = dict(pre=(t1 - t0) * 1000, forward=(t2 - t1) * 1000, post=(time.perf_counter() - t2) * 1000)
        order = np.argsort(-scores)
        return boxes[order].tolist(), scores[order].tolist()


def evaluate(pred_boxes, pred_scores, gt):
    if not pred_boxes:
        return dict(iou=0.0, score=0.0, oracle=0.0, box=None)
    ious = [iou(b, gt) for b in pred_boxes]
    return dict(iou=ious[0], score=pred_scores[0], oracle=max(ious), box=pred_boxes[0])


def draw(rgb, items, title=None):
    """items: list of (box, color, text, thickness)."""
    im = rgb.copy()
    for box, color, text, th in items:
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(im, (x1, y1), (x2 - 1, y2 - 1), color, th)
        if text:
            cv2.putText(im, text, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(im, text, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if title:
        cv2.putText(im, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(im, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return im


def grid(imgs, ncol):
    h, w = imgs[0].shape[:2]
    nrow = (len(imgs) + ncol - 1) // ncol
    canvas = np.zeros((nrow * h, ncol * w, 3), np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, ncol)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return canvas


def save_jpg(path, rgb):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/step2")
    ap.add_argument("--out", default="results/step3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    gd = GDino()
    load_s = time.time() - t0

    eps = {}
    for f in sorted(glob.glob(f"{args.data}/ep*.npz")):
        d = np.load(f)
        name = os.path.basename(f)[:-4]
        eps[name] = dict(rgb=d[f"{CAM}/rgb"], seg=d[f"{CAM}/seg"],
                         ids={"peg": int(d["peg_id"]), "box": int(d["box_id"])})

    # ---- 1) 프롬프트 비교: init(frame 0) + sample(stride) 프레임
    rows = []  # per (obj, prompt, ep, frame)
    for name, e in eps.items():
        T = len(e["rgb"])
        frames = sorted(set([0] + list(range(0, T, STRIDE))))
        for t in frames:
            for obj, prompts in PROMPTS.items():
                gt = gt_bbox(e["seg"][t] == e["ids"][obj])
                if gt is None:
                    continue
                for p in prompts:
                    boxes, scores = gd(e["rgb"][t], p)
                    r = evaluate(boxes, scores, gt)
                    rows.append(dict(obj=obj, prompt=p, ep=name, frame=t, gt=gt, **r))
        print(f"[compare] {name}: {len(frames)} frames done ({time.time() - t0:.0f}s)")

    summary = {}
    for obj, prompts in PROMPTS.items():
        summary[obj] = []
        for p in prompts:
            rs = [r for r in rows if r["obj"] == obj and r["prompt"] == p]
            init = [r for r in rs if r["frame"] == 0]
            summary[obj].append(dict(
                prompt=p,
                init_iou=[round(r["iou"], 3) for r in init],
                init_score=[round(r["score"], 3) for r in init],
                init_hits=sum(r["iou"] >= HIT_IOU for r in init),
                n=len(rs),
                hit_rate=float(np.mean([r["iou"] >= HIT_IOU for r in rs])),
                mean_iou=float(np.mean([r["iou"] for r in rs])),
                min_iou=float(np.min([r["iou"] for r in rs])),
                mean_score=float(np.mean([r["score"] for r in rs])),
                oracle_mean_iou=float(np.mean([r["oracle"] for r in rs])),
            ))
    best = {}
    for obj, lst in summary.items():
        ok = [s for s in lst if s["init_hits"] == len(eps)] or lst
        best[obj] = max(ok, key=lambda s: (s["hit_rate"], s["mean_iou"]))["prompt"]
    print("best prompts:", best)

    # ---- 2) 프롬프트 비교 그리드 (ep000 frame 0, 물체별)
    e0 = eps[sorted(eps)[0]]
    for obj, prompts in PROMPTS.items():
        gt = gt_bbox(e0["seg"][0] == e0["ids"][obj])
        tiles = []
        for p in prompts:
            boxes, scores = gd(e0["rgb"][0], p)
            items = [(gt, (0, 255, 0), None, 2)]
            for b, s in list(zip(boxes, scores))[:3][::-1]:  # top-3, top-1 을 마지막에(위에) 그림
                top = b == boxes[0]
                items.append((b, COLORS[obj] if top else (255, 255, 0), f"{s:.2f}" + (f" IoU {iou(b, gt):.2f}" if top else ""),
                              2 if top else 1))
            tiles.append(draw(e0["rgb"][0], items, title=f'"{p}"'))
        save_jpg(f"{args.out}/prompts_{obj}_{sorted(eps)[0]}_f0.jpg", grid(tiles, 4))

    # ---- 3) 최적 프롬프트로 전 프레임 평가 + latency
    full = {}
    lat = []
    for name, e in eps.items():
        full[name] = {}
        for obj in PROMPTS:
            seq = []
            for t in range(len(e["rgb"])):
                torch.cuda.synchronize()
                t1 = time.time()
                boxes, scores = gd(e["rgb"][t], best[obj])
                torch.cuda.synchronize()
                lat.append((time.time() - t1) * 1000)
                gt = gt_bbox(e["seg"][t] == e["ids"][obj])
                r = evaluate(boxes, scores, gt) if gt is not None else dict(iou=None, score=scores[0] if scores else 0.0,
                                                                              oracle=None, box=boxes[0] if boxes else None)
                seq.append(dict(frame=t, gt=gt, **r))
            full[name][obj] = seq
        print(f"[full] {name} done")
    lat_ms = float(np.mean(lat[5:]))

    # IoU 곡선
    fig, axes = plt.subplots(len(eps), 1, figsize=(9, 2.4 * len(eps)), sharex=True)
    for ax, (name, fo) in zip(np.atleast_1d(axes), full.items()):
        for obj, seq in fo.items():
            c = np.array(COLORS[obj]) / 255
            fr = [s["frame"] for s in seq]
            ax.plot(fr, [np.nan if s["iou"] is None else s["iou"] for s in seq], color=c, label=f'{obj} IoU ("{best[obj]}")')
            ax.plot(fr, [s["score"] for s in seq], color=c, ls=":", label=f"{obj} top-1 score")
            vis = [s["gt"] is not None for s in seq]
            ax.fill_between(fr, -0.05, 0, where=[not v for v in vis], color=c, alpha=0.3, step="mid")
        ax.set_ylim(-0.08, 1.02)
        ax.set_ylabel(name)
        ax.axhline(HIT_IOU, color="gray", lw=0.5)
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=2, loc="lower left")
    np.atleast_1d(axes)[-1].set_xlabel(f"frame ({CAM}); shaded bar below 0 = GT not visible (<{MIN_PX}px)")
    plt.tight_layout()
    plt.savefig(f"{args.out}/iou_curve_best.png", dpi=110)
    plt.close()

    # ---- 4) init 박스 결정 (Step 4 입력) + 오버레이
    init_boxes = {}
    tiles = []
    for name, e in eps.items():
        init_boxes[name] = {"camera": CAM, "frame": 0}
        items = []
        for obj in PROMPTS:
            s = full[name][obj][0]
            ok = s["gt"] is not None and s["iou"] >= HIT_IOU
            src = "gdino" if ok else "gt_fallback"
            box = s["box"] if ok else s["gt"]
            init_boxes[name][obj] = dict(box=[round(v, 1) for v in box], source=src, prompt=best[obj],
                                         score=round(s["score"], 3), iou_vs_gt=round(s["iou"] or 0.0, 3))
            items += [(s["gt"], (0, 255, 0), None, 2),
                      (box, COLORS[obj], f'{obj} "{best[obj]}" {s["score"]:.2f} IoU {s["iou"] or 0:.2f}', 2)]
        tiles.append(draw(e["rgb"][0], items, title=f"{name} f0  green=GT bbox"))
        # 대표 프레임: 0 / 가림 구간 / 삽입 후
        T = len(e["rgb"])
        rep = []
        for t in [0, T // 4, T // 2, (3 * T) // 4, T - 1]:
            its = []
            for obj in PROMPTS:
                s = full[name][obj][t]
                txt = f'{obj} {s["score"]:.2f}' + ("" if s["iou"] is None else f' IoU {s["iou"]:.2f}')
                its += [(s["gt"], (0, 255, 0), None, 1), (s["box"], COLORS[obj], txt, 2)]
            rep.append(draw(e["rgb"][t], its, title=f"{name} t={t}"))
        save_jpg(f"{args.out}/{name}_frames_best.jpg", grid(rep, 5))
    save_jpg(f"{args.out}/init_overlay.jpg", grid(tiles, len(tiles)))
    with open(f"{args.out}/init_boxes.json", "w") as f:
        json.dump(init_boxes, f, indent=1)

    # 전 프레임 통계 (GT 가시 프레임만)
    full_stats = {}
    for obj in PROMPTS:
        st = {}
        for name in eps:
            v = [s for s in full[name][obj] if s["gt"] is not None]
            ious = np.array([s["iou"] for s in v])
            st[name] = dict(n_visible=len(v), mean_iou=round(float(ious.mean()), 3), min_iou=round(float(ious.min()), 3),
                            hit_rate=round(float((ious >= HIT_IOU).mean()), 3))
        full_stats[obj] = st

    res = dict(model=GDINO_ID, camera=CAM, min_px=MIN_PX, stride=STRIDE, hit_iou=HIT_IOU, load_s=round(load_s, 2),
               latency_ms_per_query_512=round(lat_ms, 1), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
               best_prompt=best, prompt_summary=summary, full_frame_stats=full_stats, init_boxes=init_boxes)
    with open(f"{args.out}/grounding.json", "w") as f:
        json.dump(res, f, indent=1)

    # markdown 표
    md = [f"# Step 3 prompt comparison ({GDINO_ID}, {CAM}, top-1 box)\n"]
    for obj, lst in summary.items():
        md.append(f"\n## {obj}\n\n| prompt | init IoU (ep0/1/2) | init score | sample hit@0.5 | mean IoU | min IoU | oracle IoU | mean score |")
        md.append("|---|---|---|---|---|---|---|---|")
        for s in sorted(lst, key=lambda s: (-s["hit_rate"], -s["mean_iou"])):
            mark = " **(best)**" if s["prompt"] == best[obj] else ""
            md.append(f'| "{s["prompt"]}"{mark} | {" / ".join(f"{x:.2f}" for x in s["init_iou"])} | '
                      f'{" / ".join(f"{x:.2f}" for x in s["init_score"])} | {s["hit_rate"]:.2f} (n={s["n"]}) | '
                      f'{s["mean_iou"]:.3f} | {s["min_iou"]:.3f} | {s["oracle_mean_iou"]:.3f} | {s["mean_score"]:.2f} |')
    with open(f"{args.out}/prompt_table.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(json.dumps(full_stats, indent=1))
    print(json.dumps(init_boxes, indent=1))
    print(f"latency {lat_ms:.1f} ms/query, total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
