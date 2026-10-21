"""Step 7: tracker 비교 평가. track_compare.py 결과(results/step7/<tag>/)를 읽어 GT IoU / 속도 / 메모리 표와 그림을 만든다.

- IoU 기준은 Step 4 와 같다: GT 가시 px ≥ 200 프레임만, 구간 라벨(clean / gripper / insertion / arm_hidden)도 Step 4 함수 그대로.
- 출력: results/step7/compare.json, compare.md, iou_curves.png, speed_vs_iou.png, cmp_<ep>_<case>.jpg

Usage:
  source env.sh && python scripts/eval_step7.py
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_sam2 import VIS_MIN, iou, label_txt, overlay, phase_labels, stats  # noqa: E402

ROOT = os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAM = "base_camera"
# 그림에 올릴 대표 설정 (색 = 계열, dataviz 기본 팔레트 slot 1-3)
FAMILY_COLOR = {"sam2": "#2a78d6", "samurai": "#eb6834", "efficienttam": "#1baf7a"}
NAME = {"sam2": "SAM2.1", "samurai": "SAMURAI", "efficienttam": "EfficientTAM"}
CURVE_TAGS = ["sam2_small_joint", "samurai_small_single", "efficienttam_s_joint", "efficienttam_ti_joint"]
CURVE_STYLE = {"sam2_small_joint": "-", "samurai_small_single": "-", "efficienttam_s_joint": "-", "efficienttam_ti_joint": "--"}
# 실패 사례 비교 프레임 (Step 4 에서 본 장면): ep001 가림 해제 후 재획득, ep002 finger 가림(tail 만 잡힘), ep000 삽입
CASES = {"ep001_reacquire": ("ep001", [22, 67, 76, 100, 135]),
         "ep002_gripper": ("ep002", [50, 90, 110, 127, 145]),
         "ep000_insertion": ("ep000", [100, 112, 121])}


def compiled(r):
    """image encoder torch.compile 실제 적용 여부 (repo 기본값: EfficientTAM 만 ON)."""
    return r["compile"] == "on" or (r["compile"] == "default" and r["backend"] == "efficienttam")


def label(r):
    m = r["model"].replace("_plus", "+").replace("_512x512", "@512")
    return f"{NAME[r['backend']]}-{m}" + (" (per-obj)" if r["mode"] == "single" else "") + (" +compile" if compiled(r) else "")


def evaluate(tag_dir, data, eps):
    run = json.load(open(f"{tag_dir}/run.json"))
    r = dict(tag=run["tag"], backend=run["backend"], model=run["model"], mode=run["mode"], compile=run.get("compile", "default"), params_m=run["params_m"],
             weights_gb=run["weights_gb"], episodes={}, ious={})
    allv = {"peg": [], "box": []}
    phase_all = {k: [] for k in ("clean", "gripper", "insertion")}
    for ep in eps:
        d = data[ep]
        m = dict(np.load(f"{tag_dir}/{ep}_masks.npz"))
        if run["tag"] in CURVE_TAGS:  # 비교 이미지용으로 mask 보관 (npz 를 프레임마다 다시 풀지 않게)
            r.setdefault("masks", {})[ep] = {n: m[n] for n in ("peg", "box")}
        seg = d["seg"]
        T = len(seg)
        gt = {"peg": seg == d["peg_id"], "box": seg == d["box_id"]}
        gt_px = {n: gt[n].reshape(T, -1).sum(1) for n in gt}
        ious = {n: np.array([iou(m[n][t], gt[n][t]) if gt_px[n][t] >= VIS_MIN else np.nan for t in range(T)]) for n in gt}
        fp = {n: int(((gt_px[n] < VIS_MIN) & (m[n].reshape(T, -1).sum(1) >= VIS_MIN)).sum()) for n in gt}
        ph = d["ph"]
        clean = ~(ph["arm_hidden"] | ph["gripper"] | ph["insertion"])
        phases = {}
        for k, msk in (("clean", clean), ("gripper", ph["gripper"]), ("insertion", ph["insertion"])):
            phases[k] = stats(ious["peg"][msk])
            phase_all[k].append(ious["peg"][msk])
        hid = np.nonzero(ph["arm_hidden"])[0]
        reacq = None
        if len(hid):
            after = np.nonzero((np.arange(T) > hid[-1]) & (ious["peg"] >= 0.5))[0]
            reacq = int(after[0] - hid[-1]) if len(after) else None
        e = run["episodes"][ep]
        r["episodes"][ep] = dict(peg=stats(ious["peg"]), box=stats(ious["box"]), peg_by_phase=phases,
                                 false_positive_when_hidden=fp, peg_reacquire_after_hidden=reacq,
                                 timing=e["timing"], mem=e["mem"])
        r["ious"][ep] = {n: ious[n] for n in ious}
        r.setdefault("logits", {})[ep] = {n: m[f"{n}_logit"] for n in ("peg", "box")}
        for n in allv:
            allv[n].append(ious[n])
    E = r["episodes"].values()
    r["all"] = dict(peg=stats(np.concatenate(allv["peg"])), box=stats(np.concatenate(allv["box"])),
                    peg_by_phase={k: stats(np.concatenate(v)) for k, v in phase_all.items()},
                    per_frame_ms=round(float(np.mean([e["timing"]["per_frame_ms"] for e in E])), 2),
                    per_frame_ms_p95=round(float(np.mean([e["timing"]["per_frame_ms_p95"] for e in E])), 2),
                    load_frames_ms=round(float(np.mean([e["timing"]["load_frames_ms"] for e in E])), 1),
                    add_box_ms=round(float(np.mean([e["timing"]["add_box_ms"] for e in E])), 1),
                    peak_alloc_gb=round(max(e["mem"]["peak_alloc_gb"] for e in E), 2),
                    peak_reserved_gb=round(max(e["mem"]["peak_reserved_gb"] for e in E), 2))
    r["all"]["fps"] = round(1000 / r["all"]["per_frame_ms"], 1)
    T_all = sum(len(data[ep]["seg"]) for ep in eps)
    # init_state(전 프레임 로드)를 프레임 수로 나눠 더한 값 = 스트리밍이었다면 낼 비용 (Step 6 'online 추정' 과 같은 정의)
    load_per_frame = sum(e["timing"]["load_frames_ms"] for e in E) / T_all
    r["all"]["fps_online"] = round(1000 / (r["all"]["per_frame_ms"] + load_per_frame), 1)
    return r


def fmt(s, key="mean"):
    return "–" if not s.get("n") else f"{s[key]:.3f}"


def write_md(rs, eps, path):
    L = ["# Step 7: tracker 비교 (같은 3 ep, 같은 frame 0 GDINO bbox, base_camera 512², bf16, CPU 8 스레드, 워밍업 1 ep 제외)", "",
         "## 요약", "",
         "| model | mode | params (M) | peg IoU mean / min | peg hit@0.5 | box IoU mean / min | ep001 peg 재획득 | ms/frame (2 obj) | FPS | FPS online* | GPU peak alloc / reserved (GB) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        a = r["all"]
        re1 = r["episodes"]["ep001"]["peg_reacquire_after_hidden"] if "ep001" in r["episodes"] else None
        L.append(f"| {label(r)} | {r['mode']} | {r['params_m']} | {fmt(a['peg'])} / {fmt(a['peg'], 'min')} | {a['peg'].get('hit50', '–')} | "
                 f"{fmt(a['box'])} / {fmt(a['box'], 'min')} | {'실패' if re1 is None else f'{re1} 프레임 후'} | "
                 f"{a['per_frame_ms']} (p95 {a['per_frame_ms_p95']}) | {a['fps']} | {a['fps_online']} | {a['peak_alloc_gb']} / {a['peak_reserved_gb']} |")
    L += ["", "\\* FPS online = 1000 / (tracking ms + init_state 전 프레임 로드 시간 ÷ 프레임 수). tracking 만의 값 (point cloud 27 ms 미포함).", "",
          "## 에피소드별 peg IoU mean (min)", "", "| model | " + " | ".join(eps) + " | box " + " | box ".join(eps) + " |",
          "|---|" + "---|" * (2 * len(eps))]
    for r in rs:
        E = r["episodes"]
        L.append(f"| {label(r)} | " + " | ".join(f"{fmt(E[ep]['peg'])} ({fmt(E[ep]['peg'], 'min')})" for ep in eps) + " | "
                 + " | ".join(f"{fmt(E[ep]['box'])} ({fmt(E[ep]['box'], 'min')})" for ep in eps) + " |")
    L += ["", "## 구간별 peg IoU (3 ep 합산)", "", "| model | clean | gripper | insertion | 가림 중 오탐 (peg/box, 3 ep 합) | init_state ms | box 추가 ms |",
          "|---|---|---|---|---|---|---|"]
    for r in rs:
        p = r["all"]["peg_by_phase"]
        fp = [sum(e["false_positive_when_hidden"][n] for e in r["episodes"].values()) for n in ("peg", "box")]
        L.append(f"| {label(r)} | {fmt(p['clean'])} | {fmt(p['gripper'])} (min {fmt(p['gripper'], 'min')}) | "
                 f"{fmt(p['insertion'])} (min {fmt(p['insertion'], 'min')}) | {fp[0]} / {fp[1]} | {r['all']['load_frames_ms']} | {r['all']['add_box_ms']} |")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def plot_curves(rs, eps, data, path):
    by = {r["tag"]: r for r in rs}
    tags = [t for t in CURVE_TAGS if t in by]
    fig, axes = plt.subplots(len(eps), 2, figsize=(15, 3.0 * len(eps)), squeeze=False, sharex="row")
    for i, ep in enumerate(eps):
        ph = data[ep]["ph"]
        for j, n in enumerate(("peg", "box")):
            ax = axes[i, j]
            x = np.arange(len(ph["arm_hidden"]))
            if n == "peg":
                for k, c in (("arm_hidden", "0.85"), ("gripper", "#fbe8c4"), ("insertion", "#d3efd0")):
                    ax.fill_between(x, 0, 1.02, where=ph[k], color=c, step="mid", lw=0, label=k if i == 0 else None)
            for t in tags:
                r = by[t]
                ax.plot(x, r["ious"][ep][n], CURVE_STYLE[t], color=FAMILY_COLOR[r["backend"]], lw=1.6, label=label(r) if i == 0 else None)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{ep} {n} IoU vs GT (gray = peg fully hidden by arm)", fontsize=10, loc="left")
            ax.grid(axis="y", color="0.9")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    axes[-1, 0].set_xlabel("frame")
    axes[-1, 1].set_xlabel("frame")
    axes[0, 0].legend(loc="lower left", fontsize=7, ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def plot_speed(rs, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, n in zip(axes, ("peg", "box")):
        # 라벨 겹침 방지: IoU 순으로 정렬해 위/아래 번갈아 배치
        for i, r in enumerate(sorted(rs, key=lambda r: (r["all"][n]["mean"], r["all"]["fps"]))):
            a = r["all"]
            mk = "s" if r["mode"] == "single" else "o"
            ax.scatter(a["fps"], a[n]["mean"], s=70, marker=mk, color=FAMILY_COLOR[r["backend"]], edgecolor="white", lw=2, zorder=3)
            ax.annotate(label(r), (a["fps"], a[n]["mean"]), xytext=(7, 5 if i % 2 else -11), textcoords="offset points",
                        fontsize=7, color="0.2")
        ax.set_xlabel("tracking FPS (2 objects, excl. init)")
        ax.set_ylabel(f"{n} mean IoU (3 ep)")
        ax.set_title(f"{n}: speed vs accuracy (circle = joint 2-obj, square = per-object)", fontsize=10, loc="left")
        ax.grid(color="0.92")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    hs = [plt.Line2D([], [], marker="o", ls="", color=c, label=NAME[b]) for b, c in FAMILY_COLOR.items()]
    axes[0].legend(handles=hs, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def compare_frames(rs, data, out):
    by = {r["tag"]: r for r in rs}
    tags = [t for t in CURVE_TAGS if t in by]
    for case, (ep, frames) in CASES.items():
        d = data[ep]
        rows = []
        for t in frames:
            if t >= len(d["seg"]):
                continue
            gt = {"peg": d["seg"][t] == d["peg_id"], "box": d["seg"][t] == d["box_id"]}
            gt_img = overlay(d["rgb"][t], gt)
            tags_on = [k for k in ("arm_hidden", "gripper", "insertion") if d["ph"][k][t]]
            label_txt(gt_img, [f"GT {ep} t={t}", "/".join(tags_on), f"peg px={int(gt['peg'].sum())}"])
            row = [gt_img]
            for tg in tags:
                m = by[tg]["masks"][ep]
                img = overlay(d["rgb"][t], {"peg": m["peg"][t], "box": m["box"][t]}, gt)
                r = by[tg]
                pi, bi = r["ious"][ep]["peg"][t], r["ious"][ep]["box"][t]
                label_txt(img, [label(r), f"peg IoU={pi:.2f} logit={r['logits'][ep]['peg'][t]:.1f}", f"box IoU={bi:.2f}"])
                row.append(img)
            rows.append(np.concatenate(row, 1))
        img = np.concatenate(rows, 0)
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{out}/cmp_{case}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{ROOT}/data/step2")
    ap.add_argument("--out", default=f"{ROOT}/results/step7")
    ap.add_argument("--eps", nargs="+", default=["ep000", "ep001", "ep002"])
    args = ap.parse_args()
    data = {}
    for ep in args.eps:
        d = dict(np.load(f"{args.data}/{ep}.npz"))
        data[ep] = dict(seg=d[f"{CAM}/seg"], rgb=d[f"{CAM}/rgb"], peg_id=int(d["peg_id"]), box_id=int(d["box_id"]),
                        ph=phase_labels(d, int(d["peg_id"])))
    order = ["sam2_small_joint", "sam2_small_single", "sam2_base_plus_joint", "samurai_small_single", "samurai_base_plus_single",
             "efficienttam_s_joint", "efficienttam_ti_joint", "efficienttam_s_512x512_joint", "efficienttam_ti_512x512_joint",
             "efficienttam_s_single", "efficienttam_s_joint_compileoff", "efficienttam_ti_joint_compileoff", "sam2_small_joint_compileon"]
    dirs = {os.path.basename(os.path.dirname(p)): os.path.dirname(p) for p in glob.glob(f"{args.out}/*/run.json")}
    tags = [t for t in order if t in dirs] + sorted(t for t in dirs if t not in order)
    rs = []
    for t in tags:
        t0 = time.perf_counter()
        rs.append(evaluate(dirs[t], data, args.eps))
        print(f"eval {t} {time.perf_counter() - t0:.1f}s", flush=True)
    write_md(rs, args.eps, f"{args.out}/compare.md")
    json.dump([{k: v for k, v in r.items() if k not in ("ious", "logits", "masks")} for r in rs], open(f"{args.out}/compare.json", "w"), indent=1)
    plot_curves(rs, args.eps, data, f"{args.out}/iou_curves.png")
    plot_speed(rs, f"{args.out}/speed_vs_iou.png")
    compare_frames(rs, data, args.out)


if __name__ == "__main__":
    main()
