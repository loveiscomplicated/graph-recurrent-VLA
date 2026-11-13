"""E1/E3: track_v1.py 결과 평가 (GT 는 여기서만 읽는다).

지표 (물체별, GT 가시 px ≥ 200 프레임)
  iou      : 프레임 pooled 평균 + episode bootstrap 95% CI, 에피소드 평균의 평균, hit@0.5, 최저 에피소드 평균
  reacq    : 가림 이벤트(GT 가시 < 200 px 가 3 프레임 이상, 앞뒤로 보이는 구간 존재) 후 재등장 구간 안에서 IoU ≥ 0.5 회복 비율,
             회복까지 프레임 수 (중앙값 / 평균). 회복 못 하면 미획득.
  fp_hidden: GT 가 안 보이는 프레임 중 예측 mask ≥ 200 px 인 비율 (가림 중 FP)
  patch_fp : (patch 교란) 판 영역 안 예측 mask px ≥ 200 인 교란 프레임 비율 = occluder FP
  phase    : peg 의 clean / gripper / insertion 구간 IoU (track_sam2.phase_labels)
  time     : 프레임별 ms p50 / p95 / max (frame 0 제외), GPU peak
Usage: python scripts/eval_v1.py --runs results/v1/E1/runs --set eval [--perturb patch] --tags a b c --base a --out results/v1/E1
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1data  # noqa: E402
from track_sam2 import phase_labels  # noqa: E402

VIS_MIN = 200
OBJ = ("peg", "box")


def load_masks(path):
    z = np.load(path)
    shp = tuple(z["shape"])
    n = int(np.prod(shp))
    return {o: np.unpackbits(z[o])[:n].reshape(shp).astype(bool) for o in OBJ}, {o: z[f"{o}_logit"] for o in OBJ}, z["per_frame_ms"]


def iou_seq(pred, gt):
    T = len(gt)
    inter = (pred & gt).reshape(T, -1).sum(1)
    uni = (pred | gt).reshape(T, -1).sum(1)
    gpx = gt.reshape(T, -1).sum(1)
    out = np.where(uni > 0, inter / np.maximum(uni, 1), np.nan)
    out[gpx < VIS_MIN] = np.nan
    return out


def occlusion_events(vis):
    """vis (T,) bool → [(hide_start, reappear, stretch_end)]"""
    T = len(vis)
    ev, t = [], 0
    while t < T:
        if not vis[t] and t > 0 and vis[:t].any():
            s = t
            while t < T and not vis[t]:
                t += 1
            if t >= T:
                break
            if t - s >= 3:
                e = t
                while e < T and vis[e]:
                    e += 1
                ev.append((s, t, e))
        else:
            t += 1
    return ev


def eval_ep(d, masks, pf, perturb_spec=None):
    seg = d[f"{v1data.CAM}/seg"]
    T = len(seg)
    r = {}
    for o in OBJ:
        gt = seg == int(d[f"{o}_id"])
        gpx = gt.reshape(T, -1).sum(1)
        ppx = masks[o].reshape(T, -1).sum(1)
        io = iou_seq(masks[o], gt)
        vis = gpx >= VIS_MIN
        evs = []
        for s, a, e in occlusion_events(vis):
            ok = np.nonzero(io[a:e] >= 0.5)[0]
            evs.append(dict(hide=[int(s), int(a)], stretch=int(e - a), reacq=int(ok[0]) if len(ok) else None))
        r[o] = dict(iou=io, n_hidden=int((~vis).sum()), fp_hidden=int(((~vis) & (ppx >= VIS_MIN)).sum()), occ=evs)
        if perturb_spec is not None and "rect" in perturb_spec:
            x1, y1, x2, y2 = perturb_spec["rect"]
            s, e = perturb_spec["start"], perturb_spec["end"]
            inside = masks[o][s:e, y1:y2, x1:x2].reshape(e - s, -1).sum(1)
            r[o]["patch_fp"] = int((inside >= VIS_MIN).sum())
            r[o]["patch_n"] = e - s
    ph = phase_labels(d, int(d["peg_id"]))
    clean = ~(ph["arm_hidden"] | ph["gripper"] | ph["insertion"])
    r["phase"] = {k: m for k, m in (("clean", clean), ("gripper", ph["gripper"]), ("insertion", ph["insertion"]))}
    r["ms"] = pf[1:]
    return r


def summarize(per_ep, run):
    S = {}
    for o in OBJ:
        ious = [per_ep[e][o]["iou"] for e in per_ep]
        epm = {e: float(np.nanmean(per_ep[e][o]["iou"])) if np.isfinite(per_ep[e][o]["iou"]).any() else np.nan for e in per_ep}
        allv = np.concatenate(ious)
        allv = allv[~np.isnan(allv)]
        occ = [x for e in per_ep for x in per_ep[e][o]["occ"]]
        got = [x["reacq"] for x in occ if x["reacq"] is not None]
        S[o] = dict(iou=v1data.boot_ci(ious), iou_epmean=v1data.boot_ci(ious, stat="epmean"),
                    hit50=round(float((allv >= 0.5).mean()), 3), ep_min=round(float(np.nanmin(list(epm.values()))), 3),
                    n_frames=int(len(allv)), ep_means={e: round(v, 4) for e, v in epm.items()},
                    occ_events=len(occ), reacq_rate=round(len(got) / len(occ), 3) if occ else None,
                    reacq_frames_median=float(np.median(got)) if got else None,
                    reacq_frames_mean=round(float(np.mean(got)), 2) if got else None,
                    fp_hidden_rate=round(sum(per_ep[e][o]["fp_hidden"] for e in per_ep) /
                                         max(1, sum(per_ep[e][o]["n_hidden"] for e in per_ep)), 4))
        if any("patch_fp" in per_ep[e][o] for e in per_ep):
            S[o]["patch_fp_rate"] = round(sum(per_ep[e][o].get("patch_fp", 0) for e in per_ep) /
                                          max(1, sum(per_ep[e][o].get("patch_n", 0) for e in per_ep)), 4)
    S["peg_phase"] = {}
    for k in ("clean", "gripper", "insertion"):
        S["peg_phase"][k] = v1data.boot_ci([per_ep[e]["peg"]["iou"][per_ep[e]["phase"][k]] for e in per_ep])
    ms = np.concatenate([per_ep[e]["ms"] for e in per_ep])
    S["time"] = dict(ms_mean=round(float(ms.mean()), 2), ms_p50=round(float(np.median(ms)), 2),
                     ms_p95=round(float(np.percentile(ms, 95)), 2), ms_max=round(float(ms.max()), 1),
                     gpu_peak_gb=max(v["gpu_peak_gb"] for v in run["episodes"].values()))
    evs = [x for v in run["episodes"].values() for x in v["events"]]
    S["reanchor"] = dict(gdino_calls=len(evs), accepted=sum(x["accepted"] for x in evs))
    S["n_ep"] = len(per_ep)
    return S


def fmt_ci(c):
    return f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(v1data.ROOT, "results/v1/E1/runs"))
    ap.add_argument("--set", default="eval")
    ap.add_argument("--perturb", default=None)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--out", default=os.path.join(v1data.ROOT, "results/v1/E1"))
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    sub = args.set + (f"_{args.perturb}" if args.perturb else "")
    spec = v1data.make_spec(args.perturb, args.set) if args.perturb else None
    D = {}
    allS, per_tag_ep = {}, {}
    for tag in args.tags:
        rd = f"{args.runs}/{tag}/{sub}"
        if not os.path.exists(f"{rd}/run.json"):
            print("skip (no run.json)", tag)
            continue
        run = json.load(open(f"{rd}/run.json"))
        per_ep = {}
        for ep in run["episodes"]:
            if ep not in D:
                D[ep] = v1data.load(args.set, ep, args.perturb)
            masks, _, pf = load_masks(f"{rd}/{ep}_masks.npz")
            per_ep[ep] = eval_ep(D[ep], masks, pf, spec[ep] if spec else None)
        allS[tag] = summarize(per_ep, run)
        per_tag_ep[tag] = per_ep
        print(tag, sub, "peg", fmt_ci(allS[tag]["peg"]["iou"]), "box", fmt_ci(allS[tag]["box"]["iou"]), flush=True)
    base = args.base or args.tags[0]
    for tag in allS:
        if tag != base and base in allS:
            allS[tag]["diff_vs_base"] = {o: v1data.paired_diff_ci(allS[tag][o]["ep_means"], allS[base][o]["ep_means"]) for o in OBJ}
    name = args.name or sub
    os.makedirs(args.out, exist_ok=True)
    json.dump(allS, open(f"{args.out}/summary_{name}.json", "w"), indent=1)
    L = [f"# {name}: {len(D)} ep, 95% CI = episode bootstrap (2000회). Δ = 같은 ep 대응 차이(ep 평균), 기준 `{base}`", "",
         "| config | peg IoU pooled [CI] | Δpeg vs base [CI] (승/패) | peg hit@0.5 | peg 최저 ep | box IoU pooled [CI] | Δbox [CI] | "
         "peg 재획득 (이벤트, 중앙 프레임) | box 재획득 | 가림 중 FP peg/box | ms p50 / p95 / max | GPU GB | GDINO 호출 / 수락 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, S in allS.items():
        dv = S.get("diff_vs_base")
        dp = f"{fmt_ci(dv['peg'])} ({dv['peg']['wins']}/{dv['peg']['losses']})" if dv else "–"
        db = fmt_ci(dv["box"]) if dv else "–"
        ra = lambda o: (f"{S[o]['reacq_rate']:.2f} ({S[o]['occ_events']}, {S[o]['reacq_frames_median']})"
                        if S[o]["reacq_rate"] is not None else "–")
        L.append(f"| {tag} | {fmt_ci(S['peg']['iou'])} | {dp} | {S['peg']['hit50']:.2f} | {S['peg']['ep_min']:.2f} | "
                 f"{fmt_ci(S['box']['iou'])} | {db} | {ra('peg')} | {ra('box')} | "
                 f"{S['peg']['fp_hidden_rate']:.3f} / {S['box']['fp_hidden_rate']:.3f} | "
                 f"{S['time']['ms_p50']:.1f} / {S['time']['ms_p95']:.1f} / {S['time']['ms_max']:.0f} | {S['time']['gpu_peak_gb']:.2f} | "
                 f"{S['reanchor']['gdino_calls']} / {S['reanchor']['accepted']} |")
    if any("patch_fp_rate" in S["peg"] or "patch_fp_rate" in S["box"] for S in allS.values()):
        L += ["", "| config | occluder FP 율 (판 안 mask ≥200px 프레임 비율) peg / box |", "|---|---|"]
        for tag, S in allS.items():
            L.append(f"| {tag} | {S['peg'].get('patch_fp_rate', float('nan')):.3f} / {S['box'].get('patch_fp_rate', float('nan')):.3f} |")
    L += ["", "| config | peg clean | peg gripper | peg insertion |", "|---|---|---|---|"]
    for tag, S in allS.items():
        L.append(f"| {tag} | " + " | ".join(fmt_ci(S["peg_phase"][k]) for k in ("clean", "gripper", "insertion")) + " |")
    open(f"{args.out}/table_{name}.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
