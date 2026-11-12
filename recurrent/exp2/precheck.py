import glob
import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from exp2_train import features
from kin_filter import filter_episodes, q_conj, q_rot

W = Path("/workspace")
EV = W / "eval"


def rows(d):
    return [json.loads(l) for f in glob.glob(f"{EV / d}/results_*.jsonl") for l in open(f)]


def tci(rs, cand):
    by = {}
    for r in rs:
        if r["cand"] == cand and "none" in r:
            by.setdefault(r["train_seed"], []).append(r["none"]["success"])
    p = np.array([np.mean(v) for v in by.values()]) * 100
    n = len(p)
    h = stats.t.ppf(0.975, n - 1) * p.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return f"{p.mean():.1f}+-{h:.1f} seeds {n} eps {sum(len(v) for v in by.values())} errors {sum('error' in r for r in rs if r['cand'] == cand)}"


def sec(t):
    print(f"\n## {t}", flush=True)


def check1():
    sec("1 oracle fxyz (filter+prior distortion of GT)")
    for d in ("diag_oracle", "gtin_oracle"):
        rs = rows(d)
        if not rs:
            print(d, "missing")
            continue
        for cand in sorted({r["cand"] for r in rs}):
            eps = [r["none"] for r in rs if r["cand"] == cand and "none" in r]
            for tag, grp in (("success", [e for e in eps if e["success"]]), ("fail", [e for e in eps if not e["success"]])):
                prof, sy, sz, resets = [], [], [], []
                for e in grp:
                    f = np.array(e["fxyz_mm"])
                    lat = np.linalg.norm(f[:, 1:], axis=1)
                    prof.append([lat[a:b].mean() if len(lat) > a else np.nan for a, b in ((0, 40), (40, 80), (80, 120), (120, 250))])
                    late = f[80:] if len(f) > 80 else f[-20:]
                    sy.append(late[:, 1].mean())
                    sz.append(late[:, 2].mean())
                    on = np.where(np.abs(f).max(1) > 0.05)[0]
                    if len(on):
                        seg = np.abs(f[on[0]:]).max(1)
                        resets.append(int(np.sum((seg[1:] < 0.01) & (seg[:-1] >= 0.05))))
                prof = np.nanmean(np.array(prof), 0)
                print(f"{d} {cand} {tag} n {len(grp)} lat_by_steps {np.round(prof, 2).tolist()} late_signed_y {np.mean(sy):.2f} z {np.mean(sz):.2f} "
                      f"held_reset_per_ep {np.mean(resets) if resets else 0:.2f}")


def check2():
    sec("2 paired diag vs diag_oracle")
    a, b = rows("diag"), rows("diag_oracle")
    for cand in sorted({r["cand"] for r in a}):
        ka = {(r["train_seed"], r["env_seed"]): r["none"]["success"] for r in a if r["cand"] == cand and "none" in r}
        kb = {(r["train_seed"], r["env_seed"]): r["none"]["success"] for r in b if r["cand"] == cand and "none" in r}
        keys = sorted(set(ka) & set(kb))
        n10 = sum(ka[k] and not kb[k] for k in keys)
        n01 = sum(kb[k] and not ka[k] for k in keys)
        p = stats.binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
        print(f"{cand} pairs {len(keys)} graph_only {n10} gt_only {n01} mcnemar_p {p:.3f}")
        env = sorted({k[1] for k in keys})
        both = {e: (sum(ka[k] for k in keys if k[1] == e), sum(kb[k] for k in keys if k[1] == e)) for e in env}
        hard = [e for e, (x, y) in both.items() if x + y <= 2]
        easy = [e for e, (x, y) in both.items() if x + y >= 8]
        print(f"  env seeds {len(env)} hard(<=2/10 succ) {len(hard)} easy(>=8/10) {len(easy)}")
        print("  per_env graph/gt " + " ".join(f"{e % 1000}:{x}/{y}" for e, (x, y) in both.items()))
    sec("2b t-interval for existing cells")
    for d in ("diag", "diag_oracle", "gtin", "gtin_oracle"):
        rs = rows(d)
        for cand in sorted({r["cand"] for r in rs}):
            print(f"{d} {cand} {tci(rs, cand)}")


def check3():
    sec("3 identity of reused cells")
    s = {}
    for d in ("diag", "diag_oracle", "gtin", "gtin_oracle"):
        p = EV / d / "env_seeds.json"
        s[d] = json.loads(p.read_text())[:50] if p.is_file() else None
    ref = s["diag"]
    for d, v in s.items():
        print(f"env_seeds {d} {'missing' if v is None else ('same' if v == ref else 'DIFFERENT')}")
    for r in ("runs_final", "runs_gtin"):
        p = W / "exp2" / r / "mamba_step_s0" / "last.pt"
        if not p.is_file():
            print(r, "missing")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        meta = {k: ck.get(k) for k in ("steps", "batch", "lr", "splits", "episodes", "input", "chunk")}
        print(f"{r} {json.dumps(meta, default=str)} kin {json.dumps(ck.get('kin'))}")
    kb = json.loads((W / "kin_best.json").read_text())
    kf = json.loads((W / "kin_final.json").read_text())
    kf.pop("prior", None)
    kf.pop("hole_corr", None)
    print(f"kin_best {kb} kin_final_minus_prior {kf} {'same' if kb == kf else 'DIFFERENT'}")


def flips(q, off):
    n = 0
    for a, b in zip(off[:-1], off[1:]):
        n += int(np.sum(np.sum(q[a + 1:b] * q[a:b - 1], 1) < 0))
    return n


def check4():
    sec("4 train.npz")
    with np.load(W / "data_kin/train.npz", allow_pickle=False) as d:
        g, gt, off, counts = d["state36_transformer"], d["state36_gt"], d["episode_offsets"], d["counts"]
        rseed = d["reset_seed"] if "reset_seed" in d else None
    n = len(off) - 1
    print(f"episodes {n} frames {len(gt)}")
    w = gt[:, 3]
    print(f"tcp quat w abs: median {np.median(np.abs(w)):.3f} p05 {np.percentile(np.abs(w), 5):.3f} frames_w_under_0.05 {100 * np.mean(np.abs(w) < 0.05):.1f}%")
    for name, sl in (("tcp_q", slice(3, 7)), ("peg_q", slice(18, 22)), ("hole_q", slice(28, 32))):
        print(f"sign flips within episode {name}: gt {flips(gt[:, sl], off)} graph {flips(g[:, sl], off)}")
    hw = gt[:, 25:28] + gt[:, 0:3]
    mv = [np.abs(hw[a:b] - hw[a]).max() * 1000 for a, b in zip(off[:-1], off[1:])]
    print(f"hole world max move per episode mm: median {np.median(mv):.2f} max {np.max(mv):.2f} episodes_over_1mm {int(np.sum(np.array(mv) > 1))}")
    if rseed is not None:
        es = set(json.loads((EV / "diag" / "env_seeds.json").read_text()))
        print(f"reset_seed range {rseed.min()}-{rseed.max()} overlap_with_eval {len(es & set(rseed.tolist()))}")
    m = min(n, 300)
    sub_g, sub_gt, sub_c, sub_off = g[:off[m]], gt[:off[m]], counts[:off[m]], off[:m + 1]
    kf = json.loads((W / "kin_final.json").read_text())
    kb = json.loads((W / "kin_best.json").read_text())
    step = np.concatenate([np.arange(b - a) for a, b in zip(sub_off[:-1], sub_off[1:])])
    for name, cfg in (("filter+prior", kf), ("filter", kb)):
        f = filter_episodes(sub_gt, sub_off, sub_c, cfg)
        e = np.linalg.norm(f[:, 22:25] - sub_gt[:, 22:25], axis=1) * 1000
        print(f"GT through {name}: tip_hole change mm mean {e.mean():.2f} p90 {np.percentile(e, 90):.2f} late(80+) {e[step >= 80].mean():.2f}")
        fg = filter_episodes(sub_g, sub_off, sub_c, cfg)
        late = step >= 80
        hq = sub_gt[:, 28:32]
        d = np.array([q_rot(q_conj(hq[t]), fg[t, 22:25] - sub_gt[t, 22:25]) for t in np.where(late)[0]]) * 1000
        print(f"Graph through {name}: late signed tip_hole err in hole frame axial {d[:, 0].mean():.2f} y {d[:, 1].mean():.2f} z {d[:, 2].mean():.2f} (sd {d.std(0).round(2).tolist()})")
        rg = features(fg)[:, -3:]
        rt = features(f)[:, -3:]
        de = np.degrees(np.linalg.norm(rg - rt, axis=1))
        print(f"Graph through {name}: rel_rot feature err deg mean {de.mean():.2f} late {de[late].mean():.2f}")
    grip = sub_gt[:, 13] + sub_gt[:, 14]
    tr = 0
    for a, b in zip(sub_off[:-1], sub_off[1:]):
        gm, prev, h = 0.0, None, []
        for t in range(a, b):
            gm = max(gm, grip[t])
            h.append(prev is not None and grip[t] < 0.75 * gm and abs(grip[t] - prev) < 5e-4)
            prev = grip[t]
        h = np.array(h)
        tr += int(np.sum(h[:-1] & ~h[1:]))
    print(f"held on->off transitions per episode {tr / m:.2f}")


for fn in (check3, check4, check1, check2):
    try:
        fn()
    except Exception as e:
        print(f"{fn.__name__} FAIL {e!r}", flush=True)
