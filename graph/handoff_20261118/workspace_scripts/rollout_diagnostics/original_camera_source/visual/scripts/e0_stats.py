"""E0: v1 수집 통계 (ep 길이, 정책별 성공률, 가림 구간 길이 분포, 교란 spec) → results/v1/E0/"""
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1data  # noqa: E402

OUT = os.path.join(os.environ.get("VISION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results/v1/E0")


def runs(vis):
    out, t = [], 0
    while t < len(vis):
        if not vis[t]:
            s = t
            while t < len(vis) and not vis[t]:
                t += 1
            out.append((s, t - s))
        else:
            t += 1
    return out


def main(sets):
    os.makedirs(OUT, exist_ok=True)
    S = {}
    occ_all = {}
    for s in sets:
        if not os.path.isdir(f"{v1data.DATA}/{s}"):
            continue
        summ = json.load(open(f"{v1data.DATA}/{s}/summary.json"))
        rows = []
        for e in summ:
            z = np.load(f"{v1data.DATA}/{s}/ep{e['seed']:03d}.npz")
            seg = z["base_camera/seg"]
            T = len(seg)
            r = dict(seed=e["seed"], policy=e["policy"], T=T, success=e["success_final"], first_success=e["first_success_frame"])
            for o in ("peg", "box"):
                px = (seg == int(z[f"{o}_id"])).reshape(T, -1).sum(1)
                rr = runs(px >= 200)
                r[f"{o}_hidden_frames"] = int((px < 200).sum())
                r[f"{o}_occ_runs"] = [n for st, n in rr if st > 0]  # frame 0 부터 안 보이는 것 제외
                r[f"{o}_hidden_from_f0"] = rr[0][1] if rr and rr[0][0] == 0 else 0
                occ_all.setdefault((s, o), []).extend(r[f"{o}_occ_runs"])
            rows.append(r)
        pol = {}
        for p in ("expert", "noisy", "retry", "fail"):
            x = [r for r in rows if r["policy"] == p]
            if x:
                pol[p] = dict(n=len(x), success=round(np.mean([r["success"] for r in x]), 3),
                              T_mean=round(np.mean([r["T"] for r in x]), 1),
                              succ_within_100=round(np.mean([r["first_success"] is not None and r["first_success"] <= 100 for r in x]), 3))
        S[s] = dict(n=len(rows), T_mean=round(np.mean([r["T"] for r in rows]), 1), T_min=min(r["T"] for r in rows),
                    T_max=max(r["T"] for r in rows), success=round(np.mean([r["success"] for r in rows]), 3), policy=pol,
                    **{f"{o}_occ": dict(n_runs=len(occ_all[(s, o)]), mean_len=round(float(np.mean(occ_all[(s, o)])), 1) if occ_all[(s, o)] else 0,
                                        max_len=max(occ_all[(s, o)]) if occ_all[(s, o)] else 0,
                                        eps_with_occ=sum(1 for r in rows if r[f"{o}_occ_runs"]),
                                        eps_hidden_at_f0=sum(1 for r in rows if r[f"{o}_hidden_from_f0"] > 0))
                       for o in ("peg", "box")}, episodes=rows)
        if "box_move" in s:
            S[s]["box_move"] = [dict(seed=e["seed"], frame=e.get("box_move_frame"), offset_cm=round(100 * float(np.hypot(*e["box_move_offset"])), 2))
                                for e in summ if "box_move_offset" in e]
    for kind in ("patch", "dropout"):
        for s in ("dev", "eval"):
            if os.path.isdir(f"{v1data.DATA}/{s}"):
                S[f"spec_{kind}_{s}"] = v1data.make_spec(kind, s)
    json.dump(S, open(f"{OUT}/stats.json", "w"), indent=1)
    L = ["| set | ep | 길이 평균 (min–max) | 성공률 | 정책별 (n, 성공률, 100 step 안 성공률) | peg 가림 구간 (수, 평균/최대 길이, 가림 있는 ep) | box 가림 구간 | frame 0 에 안 보임 peg/box |",
         "|---|---|---|---|---|---|---|---|"]
    for s, x in S.items():
        if s.startswith("spec_"):
            continue
        pol = "; ".join(f"{p} {v['n']}, {v['success']:.2f}, {v['succ_within_100']:.2f}" for p, v in x["policy"].items())
        L.append(f"| {s} | {x['n']} | {x['T_mean']} ({x['T_min']}–{x['T_max']}) | {x['success']:.2f} | {pol} | "
                 f"{x['peg_occ']['n_runs']}, {x['peg_occ']['mean_len']}/{x['peg_occ']['max_len']}, {x['peg_occ']['eps_with_occ']} | "
                 f"{x['box_occ']['n_runs']}, {x['box_occ']['mean_len']}/{x['box_occ']['max_len']}, {x['box_occ']['eps_with_occ']} | "
                 f"{x['peg_occ']['eps_hidden_at_f0']} / {x['box_occ']['eps_hidden_at_f0']} |")
    open(f"{OUT}/stats.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    for j, o in enumerate(("peg", "box")):
        for s in sets:
            v = occ_all.get((s, o), [])
            if v:
                ax[j].hist(v, bins=np.arange(0, 80, 4), alpha=0.5, label=f"{s} (n={len(v)})")
        ax[j].set_title(f"{o}: base_camera 가림 구간 길이 (frames)".replace("가림 구간 길이", "occlusion run length"))
        ax[j].set_xlabel("frames (20 Hz)")
        ax[j].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/occlusion_hist.png", dpi=100)


if __name__ == "__main__":
    main(sys.argv[1:] or ["dev", "eval", "dev_box_move", "eval_box_move", "eval_lighting"])
