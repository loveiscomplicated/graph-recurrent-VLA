from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.ticker import FuncFormatter, NullFormatter

from recurrent.common import OUTPUT_ROOT, load_config, resolve
from recurrent.run_experiments import experiment_jobs, validate_results
from recurrent.evaluate import wilson
from recurrent.stages import FAILURE_ORDER

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
MODEL_COLORS = {
    "mlp": "#2a78d6",
    "stack_mlp": "#eb6834",
    "gru": "#1baf7a",
    "lstm": "#eda100",
    "cfc": "#e87ba4",
    "rkn": "#008300",
    "mamba": "#6250d6",
}
LEVEL_COLORS = {"none": "#86b6ef", "low": "#3987e5", "mid": "#1c5cab", "high": "#0d366b"}
STAGE_COLORS = {
    "success": "#0ca30c",
    "insertion": "#104281",
    "alignment": "#1c5cab",
    "transport": "#2a78d6",
    "retention": "#5598e7",
    "acquisition": "#86b6ef",
}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CONDITION_LABEL = {"normal": "normal", "hole_shift": "hole shift"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "text.color": INK,
        "font.size": 9,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def load_rows(exp: Path, cfg):
    names = [n for n, _, _, _ in experiment_jobs(cfg)]
    if cfg["eval"].get("with_expert", True):
        names.append("expert")
    return [json.loads(line) for name in names
            for line in (exp / "eval" / f"{name}.jsonl").read_text().splitlines()]


def load_histories(exp: Path, cfg):
    out = []
    which = cfg["eval"].get("checkpoint", "best")
    for name, _, _, _ in experiment_jobs(cfg):
        run = exp / "runs" / name
        h = json.loads((run / "history.json").read_text())
        c = h["config"]
        checkpoint = torch.load(run / f"{which}.pt", map_location="cpu", weights_only=False)
        # last.pt stores the history row; best.pt stores unprefixed validation metrics.
        metrics = checkpoint["val"]
        mse = metrics.get("action_mse_z", metrics.get("val_action_mse_z"))
        out.append({"name": name, "model": c["model"], "level": c["train_level"], "seed": c["seed"],
                    "history": h["history"], **h["summary"], "evaluated_checkpoint": which,
                    "evaluated_step": checkpoint["step"], "evaluated_val_action_mse_z": mse,
                    "training_device": c["device"]})
    return out


def group_rate(rows):
    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r.get("train_seed")].append(r["success"])
    rates = [np.mean(v) for v in by_seed.values()]
    k, n = sum(sum(v) for v in by_seed.values()), sum(len(v) for v in by_seed.values())
    lo, hi = wilson(k, n)
    return {"mean": float(np.mean(rates)) if rates else None, "std": float(np.std(rates)) if len(rates) > 1 else 0.0,
            "k": k, "n": n, "lo": lo, "hi": hi, "seeds": len(rates)}


def index(rows):
    idx = defaultdict(list)
    for r in rows:
        idx[(r["model"], r["train_level"], r["eval_level"], r["condition"])].append(r)
    return idx


def paired_diff(a_rows, b_rows, n_boot=2000, seed=0):
    a = {(r.get("train_seed"), r["episode_seed"]): float(r["success"]) for r in a_rows}
    b = {(r.get("train_seed"), r["episode_seed"]): float(r["success"]) for r in b_rows}
    keys = sorted(set(a) & set(b), key=str)
    if not keys:
        return None
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)), "n": len(d)}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def plain_log(ax, axis="y"):
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_formatter(NullFormatter())


def luminance(color):
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_success_by_level(idx, models, levels, conditions, expert, path):
    fig, axes = plt.subplots(len(conditions), len(models), figsize=(2.3 * len(models), 2.6 * len(conditions)), sharey=True, squeeze=False)
    x = np.arange(len(levels))
    for ci, cond in enumerate(conditions):
        means = {m: [group_rate(idx.get((m, l, l, cond), []))["mean"] for l in levels] for m in models}
        for mi, m in enumerate(models):
            ax = axes[ci][mi]
            for other in models:
                if other != m:
                    ys = [np.nan if v is None else v for v in means[other]]
                    ax.plot(x, ys, color=BASE, lw=1, zorder=1)
            stats = [group_rate(idx.get((m, l, l, cond), [])) for l in levels]
            ys = np.array([np.nan if s["mean"] is None else s["mean"] for s in stats])
            sd = np.array([s["std"] for s in stats])
            ax.fill_between(x, ys - sd, ys + sd, color=MODEL_COLORS.get(m, INK2), alpha=0.1, lw=0, zorder=2)
            ax.plot(x, ys, color=MODEL_COLORS.get(m, INK2), lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, zorder=3)
            if expert.get(cond) is not None:
                ax.axhline(expert[cond], color=INK2, lw=1, zorder=1)
                if mi == 0:
                    ax.text(len(levels) - 1, expert[cond], "expert", color=INK2, fontsize=7.5, ha="right", va="bottom")
            last = np.where(~np.isnan(ys))[0]
            if len(last):
                j = last[-1]
                ax.annotate(f"{ys[j]:.2f}", (x[j], ys[j]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK, va="center")
            ax.set_xticks(x, levels)
            ax.set_ylim(-0.03, 1.05)
            ax.set_xlim(-0.3, len(levels) - 0.4)
            if ci == 0:
                ax.set_title(m)
            if mi == 0:
                ax.set_ylabel(f"success rate\n{CONDITION_LABEL.get(cond, cond)}")
            ax.grid(axis="x", visible=False)
    fig.suptitle("Closed-loop success by corruption level (train level = eval level, mean ± std over seeds)", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, path)


def fig_heatmap(idx, models, levels, conditions, path, cross=False):
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, len(conditions), figsize=(3.6 * len(conditions) + 0.6, 0.42 * len(models) + 1.3), squeeze=False)
    for ci, cond in enumerate(conditions):
        ax = axes[0][ci]
        grid = np.full((len(models), len(levels)), np.nan)
        text = [["" for _ in levels] for _ in models]
        for mi, m in enumerate(models):
            for li, l in enumerate(levels):
                key = (m, "none", l, cond) if cross else (m, l, l, cond)
                s = group_rate(idx.get(key, []))
                if s["mean"] is not None:
                    grid[mi, li] = s["mean"]
                    text[mi][li] = f"{s['mean']:.2f}" + (f"\n±{s['std']:.2f}" if s["seeds"] > 1 else "")
        ax.pcolormesh(np.arange(len(levels) + 1) - 0.5, np.arange(len(models) + 1) - 0.5, np.nan_to_num(grid, nan=0.0),
                      cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
        ax.set_ylim(len(models) - 0.5, -0.5)
        for mi in range(len(models)):
            for li in range(len(levels)):
                if text[mi][li]:
                    color = "white" if luminance(cmap(grid[mi, li])) < 0.5 else INK
                    ax.text(li, mi, text[mi][li], ha="center", va="center", fontsize=7.5, color=color)
                else:
                    ax.text(li, mi, "–", ha="center", va="center", fontsize=8, color=MUTED)
        ax.set_xticks(range(len(levels)), levels)
        ax.set_yticks(range(len(models)), models if ci == 0 else [""] * len(models))
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(CONDITION_LABEL.get(cond, cond))
    title = "Success rate, models trained clean (none) and evaluated at each level" if cross else "Success rate table (train level = eval level)"
    fig.suptitle(title, x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, path)


def fig_paired(idx, models, levels, conditions, baseline, path):
    others = [m for m in models if m != baseline]
    fig, axes = plt.subplots(len(conditions), len(levels), figsize=(2.6 * len(levels) + 0.8, 0.38 * len(others) * len(conditions) + 1.6), sharey=True, sharex=True, squeeze=False)
    table = []
    for ci, cond in enumerate(conditions):
        for li, l in enumerate(levels):
            ax = axes[ci][li]
            ax.axvline(0, color=BASE, lw=1, zorder=1)
            for k, m in enumerate(others):
                r = paired_diff(idx.get((m, l, l, cond), []), idx.get((baseline, l, l, cond), []))
                if r is None:
                    continue
                y = len(others) - 1 - k
                ax.plot([100 * r["lo"], 100 * r["hi"]], [y, y], color=MODEL_COLORS.get(m, INK2), lw=2, zorder=2)
                ax.plot(100 * r["diff"], y, "o", color=MODEL_COLORS.get(m, INK2), ms=6, mec=SURFACE, mew=1.5, zorder=3)
                table.append({"model": m, "baseline": baseline, "level": l, "condition": cond, **r})
            ax.set_yticks(range(len(others)), list(reversed(others)))
            ax.grid(axis="y", visible=False)
            if ci == 0:
                ax.set_title(l)
            if li == 0:
                ax.set_ylabel(CONDITION_LABEL.get(cond, cond))
            if ci == len(conditions) - 1:
                ax.set_xlabel(f"success − {baseline} (pp)")
    fig.suptitle(f"Paired success difference vs {baseline} (same episodes and corruption, 95% bootstrap CI)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, path)
    return table


def fig_drop(idx, models, levels, cond, path):
    lo, hi = levels[0], levels[-1]
    panels = [("matched training", lambda m, l: (m, l, l, cond)), ("trained clean", lambda m, l: (m, "none", l, cond))]
    table = []
    for label, key in panels:
        vals = []
        for m in models:
            a, b = group_rate(idx.get(key(m, lo), []))["mean"], group_rate(idx.get(key(m, hi), []))["mean"]
            vals.append(None if a is None or b is None else 100 * (a - b))
        table.append((label, vals))
    known = [abs(v) for _, vals in table for v in vals if v is not None]
    span = max([10.0] + known)
    neg = any(v is not None and v < 0 for _, vals in table for v in vals)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 0.36 * len(models) + 1.4), sharey=True, sharex=True)
    y = np.arange(len(models))[::-1]
    for ax, (label, vals) in zip(axes, table):
        for yi, v in zip(y, vals):
            if v is None:
                ax.text(0, yi, "–", va="center", color=MUTED)
                continue
            ax.barh(yi, v, height=0.55, color=MODEL_COLORS["mlp"], zorder=2)
            off = span * 0.03 * (1 if v >= 0 else -1)
            ax.text(v + off, yi, f"{v:.0f}", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
        ax.set_xlim(-span * 1.3 if neg else -span * 0.08, span * 1.3)
        ax.axvline(0, color=BASE, lw=1)
        ax.set_yticks(y, models)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"success drop {lo} → {hi} (pp)")
        ax.set_title(label)
    fig.suptitle(f"Robustness: how much success is lost at the highest corruption ({CONDITION_LABEL.get(cond, cond)})", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, path)


def fig_failures(idx, models, levels, cond, path):
    order = ["success", "insertion", "alignment", "transport", "retention", "acquisition"]
    fig, axes = plt.subplots(1, len(levels), figsize=(2.9 * len(levels), 0.36 * len(models) + 1.7), sharey=True, squeeze=False)
    for li, l in enumerate(levels):
        ax = axes[0][li]
        y = np.arange(len(models))[::-1]
        for yi, m in zip(y, models):
            rs = idx.get((m, l, l, cond), [])
            if not rs:
                continue
            counts = {"success": sum(r["success"] for r in rs)}
            for s in FAILURE_ORDER:
                counts[s] = sum(r.get("failure_stage") == s for r in rs)
            left = 0.0
            for s in order:
                w = counts[s] / len(rs)
                if w > 0:
                    ax.barh(yi, w, left=left, height=0.6, color=STAGE_COLORS[s], edgecolor=SURFACE, linewidth=1.5, zorder=2)
                left += w
        ax.set_xlim(0, 1)
        ax.set_yticks(y, models)
        ax.grid(False)
        ax.set_title(l)
        ax.set_xlabel("fraction of episodes")
    handles = [plt.Rectangle((0, 0), 1, 1, color=STAGE_COLORS[s]) for s in order]
    labels = ["success"] + [f"failed: {s}" for s in order[1:]]
    fig.legend(handles, labels, loc="lower center", ncol=len(order), bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Where episodes end (Graph v2 stage definitions, {CONDITION_LABEL.get(cond, cond)})", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, path)


def fig_blackout(idx, models, stress_level, lengths, path):
    fig, axes = plt.subplots(1, len(models), figsize=(2.3 * len(models), 2.7), sharey=True, squeeze=False)
    x = np.arange(len(lengths))
    means = {m: [group_rate(idx.get((m, stress_level, "none", f"blackout_{b}"), []))["mean"] for b in lengths] for m in models}
    for mi, m in enumerate(models):
        ax = axes[0][mi]
        for other in models:
            if other != m:
                ax.plot(x, [np.nan if v is None else v for v in means[other]], color=BASE, lw=1, zorder=1)
        stats = [group_rate(idx.get((m, stress_level, "none", f"blackout_{b}"), [])) for b in lengths]
        ys = np.array([np.nan if s["mean"] is None else s["mean"] for s in stats])
        sd = np.array([s["std"] for s in stats])
        ax.fill_between(x, ys - sd, ys + sd, color=MODEL_COLORS.get(m, INK2), alpha=0.1, lw=0, zorder=2)
        ax.plot(x, ys, color=MODEL_COLORS.get(m, INK2), lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5, zorder=3)
        ok = np.where(~np.isnan(ys))[0]
        if len(ok):
            j = ok[-1]
            ax.annotate(f"{ys[j]:.2f}", (x[j], ys[j]), xytext=(4, 0), textcoords="offset points", fontsize=8, va="center")
        ax.set_xticks(x, [str(b) for b in lengths])
        ax.set_xlim(-0.3, len(lengths) - 0.4)
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(m)
        ax.set_xlabel("blackout steps")
        ax.grid(axis="x", visible=False)
        if mi == 0:
            ax.set_ylabel("success rate")
    fig.suptitle(f"Memory stress test: object observations frozen right after grasp (models trained at '{stress_level}')", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, path)


def fig_curves(hist, models, levels, path):
    fig, axes = plt.subplots(1, len(models), figsize=(2.3 * len(models), 2.7), sharey=True, squeeze=False)
    for mi, m in enumerate(models):
        ax = axes[0][mi]
        for l in levels:
            hs = [h for h in hist if h["model"] == m and h["level"] == l]
            if not hs:
                continue
            steps = [r["step"] for r in hs[0]["history"]]
            vals = np.array([[r["val_action_mse_z"] for r in h["history"]][: len(steps)] for h in hs if len(h["history"]) >= len(steps)])
            if not len(vals):
                continue
            ax.plot(steps, vals.mean(0), color=LEVEL_COLORS.get(l, INK2), lw=2, label=l)
        ax.set_yscale("log")
        plain_log(ax)
        ax.set_title(m)
        ax.set_xlabel("train step")
        if mi == 0:
            ax.set_ylabel("val action MSE (z)")
    handles = [plt.Line2D([0], [0], color=LEVEL_COLORS[l], lw=2) for l in levels]
    fig.legend(handles, levels, loc="lower center", ncol=len(levels), bbox_to_anchor=(0.5, -0.05), title="train level")
    fig.suptitle("Validation action error during training (mean over seeds)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, path)


def fig_offline_online(hist, idx, path):
    xs, ys = [], []
    for h in hist:
        rs = idx.get((h["model"], h["level"], h["level"], "normal"), [])
        rs = [r for r in rs if r.get("train_seed") == h["seed"]]
        if rs:
            xs.append(h["evaluated_val_action_mse_z"])
            ys.append(np.mean([r["success"] for r in rs]))
    if len(xs) < 2:
        return None
    rho = spearman(xs, ys)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.scatter(xs, ys, s=36, color=MODEL_COLORS["mlp"], edgecolor=SURFACE, linewidth=1.5, zorder=2)
    ax.set_xscale("log")
    plain_log(ax, "x")
    ax.set_xlabel("final val action MSE (z, log)")
    ax.set_ylabel("closed-loop success (normal)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"Offline error vs closed-loop success, one dot per run (Spearman ρ = {rho:.2f})" if rho is not None else "Offline error vs closed-loop success", loc="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, path)
    return rho


def fig_cost(hist, latency, models, path):
    params = {m: next((h["params"] for h in hist if h["model"] == m), None) for m in models}
    ttime = {m: np.mean([h["train_time_s"] for h in hist if h["model"] == m]) / 60 if any(h["model"] == m for h in hist) else None for m in models}
    lat = {}
    for r in (latency or {}).get("rows", []):
        if r["device"] == "cpu" and r["threads"] == 1:
            lat[r["model"]] = r["median_ms"]
    panels = [("parameters (K)", {m: (v / 1000 if v else None) for m, v in params.items()}, "{:.0f}"),
              ("train time per run (min)", ttime, "{:.1f}"),
              ("inference per step, CPU 1 thread (ms)", lat, "{:.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 0.36 * len(models) + 1.4), sharey=True)
    y = np.arange(len(models))[::-1]
    for ax, (label, vals, fmt) in zip(axes, panels):
        top = max([v for v in vals.values() if v is not None and v > 0] or [1])
        for yi, m in zip(y, models):
            v = vals.get(m)
            if v is None:
                ax.text(0, yi, "–", va="center", color=MUTED)
                continue
            ax.barh(yi, v, height=0.55, color=MODEL_COLORS["mlp"], zorder=2)
            ax.text(v + top * 0.02, yi, fmt.format(v), va="center", fontsize=8)
        ax.set_xlim(0, top * 1.22)
        ax.set_yticks(y, models)
        ax.grid(axis="y", visible=False)
        ax.set_title(label)
    fig.suptitle("Cost of each controller", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, path)


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def fmt(v, p=2):
    return "–" if v is None else f"{v:.{p}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiment.yaml")
    p.add_argument("--sim-backend", choices=["physx_cpu", "physx_cuda"], default=None)
    p.add_argument("--initial-states", default=None)
    p.add_argument("--latency", default=str(OUTPUT_ROOT / "latency.json"))
    args = p.parse_args()
    cfg = load_config(args.config)
    exp = resolve(cfg["out_dir"])
    names = experiment_jobs(cfg)
    first = exp / "eval" / f"{names[0][0]}.identity.json"
    backend = args.sim_backend or json.loads(first.read_text())["backend"]
    initial = str(resolve(args.initial_states)) if args.initial_states else None
    if initial is None and (exp / "initial_states.pt").exists():
        initial = str(exp / "initial_states.pt")
    validate_results(cfg, backend, initial)
    out = exp / "report"
    out.mkdir(parents=True, exist_ok=True)
    style()
    rows = load_rows(exp, cfg)
    hist = load_histories(exp, cfg)
    lat_path = resolve(args.latency)
    latency = json.loads(lat_path.read_text()) if lat_path.exists() else None
    present = {r["model"] for r in rows} | {h["model"] for h in hist}
    models = [m for m in cfg["models"] if m in present]
    levels = list(cfg["levels"])
    names = list(cfg["eval"].get("conditions", {"normal": {}}))
    names = sorted(names, key=lambda c: (c != "normal", c != "hole_shift", c))
    conditions = [c for c in names if any(r["condition"] == c for r in rows)]
    idx = index(rows)
    expert = {c: group_rate(idx.get(("expert", "none", "none", c), []))["mean"] for c in conditions}
    stress = cfg["eval"].get("stress") or {}
    stress_levels = [l for l in stress.get("train_levels", []) if l in levels]
    lengths = stress.get("blackout_lengths", [])
    figures = []
    if models and conditions:
        fig_success_by_level(idx, models, levels, conditions, expert, out / "success_by_level.png")
        fig_heatmap(idx, models, levels, conditions, out / "success_table.png")
        figures += ["success_by_level.png", "success_table.png"]
        if any(k[1] == "none" and k[2] != "none" for k in idx):
            fig_heatmap(idx, models, levels, conditions, out / "success_table_clean_trained.png", cross=True)
            figures.append("success_table_clean_trained.png")
    paired = []
    for base in ("mlp", "stack_mlp"):
        if base in models and len(models) > 1 and conditions:
            paired += fig_paired(idx, models, levels, conditions, base, out / f"paired_vs_{base}.png")
            figures.append(f"paired_vs_{base}.png")
    if models and "normal" in conditions and len(levels) > 1:
        fig_drop(idx, models, levels, "normal", out / "robustness_drop.png")
        fig_failures(idx, models, levels, "normal", out / "failure_stages.png")
        figures += ["robustness_drop.png", "failure_stages.png"]
    if stress_levels and lengths and any(r["condition"].startswith("blackout_") for r in rows):
        fig_blackout(idx, models, stress_levels[0], lengths, out / "blackout_stress.png")
        figures.append("blackout_stress.png")
    rho = None
    if hist:
        fig_curves(hist, models, levels, out / "learning_curves.png")
        rho = fig_offline_online(hist, idx, out / "offline_vs_closed_loop.png")
        fig_cost(hist, latency, models, out / "cost.png")
        figures += ["learning_curves.png", "cost.png"] + (["offline_vs_closed_loop.png"] if rho is not None else [])

    summary_rows = []
    for (m, tl, el, c), rs in sorted(idx.items()):
        s = group_rate(rs)
        fails = {f"fail_{x}": sum(r.get("failure_stage") == x for r in rs) for x in FAILURE_ORDER}
        steps = [r["steps"] for r in rs if r["success"]]
        summary_rows.append({"model": m, "train_level": tl, "eval_level": el, "condition": c, "success_mean": s["mean"],
                             "success_std_over_seeds": s["std"], "successes": s["k"], "episodes": s["n"], "ci_low": s["lo"],
                             "ci_high": s["hi"], "mean_steps_success": float(np.mean(steps)) if steps else None,
                             "mean_action_delta": float(np.mean([r["mean_action_delta"] for r in rs])), **fails})
    write_csv(out / "summary.csv", summary_rows)
    write_csv(out / "episodes.csv", rows)
    write_csv(out / "paired.csv", paired)
    (out / "report.md").write_text(render_md(cfg, models, levels, conditions, idx, hist, latency, expert, paired, rho, figures, stress_levels, lengths))
    print(f"report {out / 'report.md'} figures {len(figures)}")


def best_per_level(idx, models, levels, cond):
    out = []
    for l in levels:
        cands = [(group_rate(idx.get((m, l, l, cond), []))["mean"], m) for m in models]
        cands = [c for c in cands if c[0] is not None]
        if cands:
            v, m = max(cands)
            ties = [mm for vv, mm in cands if abs(vv - v) < 1e-9]
            out.append((l, ties, v))
    return out


def render_md(cfg, models, levels, conditions, idx, hist, latency, expert, paired, rho, figures, stress_levels, lengths):
    ev = cfg["eval"]
    seeds = cfg["seeds"]
    lines = ["# Recurrent controller 비교 결과", ""]
    lines += ["이 문서는 `python -m recurrent.report`가 결과 파일에서 자동으로 만든 것입니다. 수치는 모두 실제 rollout 기록에서 계산했고, 해석 문장은 직접 작성해야 합니다.", ""]
    lines += ["## 실험 설정", ""]
    lines += [f"- 데모: `{cfg['data']}`",
              f"- 모델: {', '.join(models)}",
              f"- 관측 손상 단계: {', '.join(levels)} (학습과 평가에 같은 단계를 적용)",
              f"- 학습 seed: {', '.join(map(str, seeds))}",
              f"- 실제 학습 device: {', '.join(sorted({h['training_device'] for h in hist}))}",
              f"- 정책 평가 physics backend: {', '.join(sorted({r['sim_backend'] for rs in idx.values() for r in rs if r['model'] != 'expert'}))}",
              f"- Expert 평가 physics backend: {', '.join(sorted({r['sim_backend'] for rs in idx.values() for r in rs if r['model'] == 'expert'})) or '미실행'} (GPU 정책과 CPU expert는 물리 backend가 다를 수 있음)",
              f"- 초기 상태 출처 (CPU reset fixture의 SHA-256 또는 legacy reset): {', '.join(sorted({r['initial_state_source'] for rs in idx.values() for r in rs}))} (fixture SHA-256 또는 reset 출처)",
              f"- Offline 오차: 평가에 사용한 `{ev.get('checkpoint', 'best')}.pt`의 validation action MSE",
              f"- 평가: seed {ev['start_seed']}부터 {ev['episodes']}개 배치, horizon {ev.get('horizon', 250)} step, 조건 {', '.join(conditions)}",
              "- 성공 판정: rollout 중 `info['success']`가 한 번이라도 참이면 성공 (Graph 팀 v2 이후 평가와 같은 기준)",
              ""]
    if any(v is not None for v in expert.values()):
        lines += ["expert 성공률 (손상 없음, 상한 기준): " + ", ".join(f"{CONDITION_LABEL.get(c, c)} {fmt(v)}" for c, v in expert.items() if v is not None), ""]

    lines += ["## 1. 손상 단계별 성공률", "",
              "무엇을 보는가: 같은 단계의 손상으로 학습하고 평가했을 때 각 모델의 성공률입니다. 칸마다 seed 평균이고 ± 값은 seed 간 표준편차입니다.", ""]
    if "success_by_level.png" in figures:
        lines += ["![](success_by_level.png)", "", "![](success_table.png)", ""]
    for cond in conditions:
        header = "| model | " + " | ".join(levels) + " |"
        lines += [f"**{CONDITION_LABEL.get(cond, cond)}**", "", header, "|" + " --- |" * (len(levels) + 1)]
        for m in models:
            cells = []
            for l in levels:
                s = group_rate(idx.get((m, l, l, cond), []))
                cells.append("–" if s["mean"] is None else f"{s['mean']:.2f} ± {s['std']:.2f} ({s['k']}/{s['n']})")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
        for l, ties, v in best_per_level(idx, models, levels, cond):
            if v > 0:
                lines.append(f"- {l}: 가장 높은 성공률 {', '.join(ties)} ({v:.2f})")
            else:
                lines.append(f"- {l}: 모든 모델 성공 0")
        lines.append("")
    if "success_table_clean_trained.png" in figures:
        lines += ["### 손상 없이 학습한 모델을 손상 환경에서 평가", "",
                  "무엇을 보는가: 학습 때 손상을 한 번도 보지 않은 모델이 손상된 관측을 받으면 얼마나 무너지는지입니다. 위 표와 비교하면 손상을 학습에 넣는 것의 효과가 보입니다.", "",
                  "![](success_table_clean_trained.png)", ""]

    lines += ["## 2. 과거를 보는 것의 효과 (질문 1)", "",
              "무엇을 보는가: 각 모델의 성공률에서 바닐라 MLP의 성공률을 뺀 값입니다. 모든 모델이 같은 배치와 같은 손상 패턴을 받으므로 episode 단위로 짝지어 비교했고, 막대는 bootstrap 95% 신뢰구간입니다. 구간이 0을 포함하면 차이가 있다고 말할 수 없습니다.", ""]
    if "paired_vs_mlp.png" in figures:
        lines += ["![](paired_vs_mlp.png)", ""]
    lines += paired_lines(paired, "mlp")

    lines += ["## 3. 단순히 과거를 보는 것과 recurrent 기억의 차이 (질문 2)", "",
              "무엇을 보는가: 최근 8 step을 이어 붙여 보는 stack MLP를 기준으로 한 같은 비교입니다. recurrent 계열이 stack MLP보다 좋다면, 고정 길이 window로는 부족하고 기억이 필요하다는 근거가 됩니다.", ""]
    if "paired_vs_stack_mlp.png" in figures:
        lines += ["![](paired_vs_stack_mlp.png)", ""]
    lines += paired_lines(paired, "stack_mlp")
    if "blackout_stress.png" in figures:
        lines += ["### 기억 스트레스 테스트", "",
                  f"무엇을 보는가: peg를 잡아 든 직후부터 peg와 hole 관측을 일정 step 동안 끊고 마지막 값만 유지합니다(다른 손상 없음, '{stress_levels[0]}' 단계로 학습한 모델). 그동안 peg는 손과 함께 움직이므로, 끊긴 동안의 상태를 기억하거나 추정해야 성공할 수 있습니다. stack MLP의 window는 8 step이므로 그보다 긴 끊김에서 recurrent 계열과 차이가 나는지 봅니다.", "",
                  "![](blackout_stress.png)", ""]
        header = "| model | " + " | ".join(f"{b} steps" for b in lengths) + " |"
        lines += [header, "|" + " --- |" * (len(lengths) + 1)]
        for m in models:
            cells = []
            for b in lengths:
                s = group_rate(idx.get((m, stress_levels[0], "none", f"blackout_{b}"), []))
                cells.append("–" if s["mean"] is None else f"{s['mean']:.2f} ({s['k']}/{s['n']})")
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## 4. 어떤 모델이 어디서 실패하는가 (질문 3)", ""]
    if "robustness_drop.png" in figures:
        lines += ["무엇을 보는가: 가장 약한 손상에서 가장 강한 손상으로 갈 때 잃는 성공률(%p)입니다. 왼쪽은 같은 단계로 학습한 경우, 오른쪽은 손상 없이 학습한 경우입니다.", "", "![](robustness_drop.png)", ""]
    if "failure_stages.png" in figures:
        lines += ["무엇을 보는가: episode가 어느 단계에서 끝났는지의 비율입니다. 단계 정의는 Graph 팀 v2 평가와 같습니다(acquisition: 5 step 연속 grasp 실패, retention: 잡았다가 놓침, transport: 들어 올리지 못함, alignment: hole 입구 정렬 실패, insertion: 정렬 후 삽입 실패).", "", "![](failure_stages.png)", ""]

    lines += ["## 5. 학습 과정과 비용", ""]
    if "learning_curves.png" in figures:
        lines += ["무엇을 보는가: 학습 중 검증 데이터의 action 오차입니다. 손상 단계별로 색을 진하게 했습니다.", "", "![](learning_curves.png)", ""]
    if "offline_vs_closed_loop.png" in figures:
        lines += [f"무엇을 보는가: 검증 action 오차(offline)가 낮을수록 실제 성공률(closed-loop)도 높은지입니다. Spearman ρ = {fmt(rho)}. Graph 팀 v4에서도 offline 오차 감소가 성공률로 이어지지 않았으므로, 모델 선택은 성공률 기준으로 해야 합니다.", "", "![](offline_vs_closed_loop.png)", ""]
    if "cost.png" in figures:
        lines += ["무엇을 보는가: 파라미터 수, run당 학습 시간, 제어 1 step당 추론 시간(CPU 1 thread, batch 1)입니다.", "", "![](cost.png)", ""]
    lines += model_table(models, hist, latency, idx, levels)
    lines += ["## 파일", "", "- `summary.csv`: 모델, 학습 단계, 평가 단계, 조건별 집계",
              "- `episodes.csv`: episode 단위 원자료 (성공, 실패 단계, step 수, action 변화량, 추론 시간)",
              "- `paired.csv`: 짝지은 비교의 차이와 신뢰구간", ""]
    return "\n".join(lines)


def paired_lines(paired, base):
    rows = [p for p in paired if p["baseline"] == base and p["lo"] > 0]
    worse = [p for p in paired if p["baseline"] == base and p["hi"] < 0]
    out = []
    if rows:
        out.append(f"신뢰구간이 0보다 큰 경우 ({base}보다 높음):")
        out += [f"- {p['model']}, {p['level']}, {CONDITION_LABEL.get(p['condition'], p['condition'])}: {100 * p['diff']:+.0f}%p [{100 * p['lo']:+.0f}, {100 * p['hi']:+.0f}]" for p in rows]
    if worse:
        out.append(f"신뢰구간이 0보다 작은 경우 ({base}보다 낮음):")
        out += [f"- {p['model']}, {p['level']}, {CONDITION_LABEL.get(p['condition'], p['condition'])}: {100 * p['diff']:+.0f}%p [{100 * p['lo']:+.0f}, {100 * p['hi']:+.0f}]" for p in worse]
    if paired and not rows and not worse:
        out.append(f"모든 비교에서 신뢰구간이 0을 포함했습니다 ({base}와 차이를 확인하지 못함).")
    return out + [""]


def model_table(models, hist, latency, idx, levels):
    lat = {r["model"]: r["median_ms"] for r in (latency or {}).get("rows", []) if r["device"] == "cpu" and r["threads"] == 1}
    gpu = {r["model"]: r["median_ms"] for r in (latency or {}).get("rows", []) if r["device"] == "cuda"}
    lines = ["| model | params | train min | CPU ms/step | GPU ms/step | success none | success " + levels[-1] + " | shift " + levels[-1] + " |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m in models:
        hs = [h for h in hist if h["model"] == m]
        params = hs[0]["params"] if hs else None
        tmin = np.mean([h["train_time_s"] for h in hs]) / 60 if hs else None
        a = group_rate(idx.get((m, levels[0], levels[0], "normal"), []))["mean"]
        b = group_rate(idx.get((m, levels[-1], levels[-1], "normal"), []))["mean"]
        c = group_rate(idx.get((m, levels[-1], levels[-1], "hole_shift"), []))["mean"]
        lines.append(f"| {m} | {params if params else '–'} | {fmt(tmin, 1)} | {fmt(lat.get(m), 3)} | {fmt(gpu.get(m), 3)} | {fmt(a)} | {fmt(b)} | {fmt(c)} |")
    return lines + [""]


if __name__ == "__main__":
    main()
