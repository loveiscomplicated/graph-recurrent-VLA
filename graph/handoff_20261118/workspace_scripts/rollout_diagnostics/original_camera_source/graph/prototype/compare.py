"""Compare paired results; reject mismatched data, seeds and rollout states."""
import argparse
import json
from pathlib import Path

from .common import write_json
from .visualization import plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, nargs=3, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/graph/comparison"))
    args = p.parse_args()
    reports = [json.loads(path.read_text()) for path in args.results]
    if {r["model"] for r in reports} != {"flat", "center", "surface"}:
        raise ValueError("need exactly flat, center, surface results")
    reports.sort(key=lambda r: ["flat", "center", "surface"].index(r["model"]))
    first = reports[0]
    for r in reports[1:]:
        if (r["dataset_sha256"], r["test_episode_ids"], r["config"]) != (first["dataset_sha256"], first["test_episode_ids"], first["config"]):
            raise ValueError("dataset/split/training configuration differs")
        for setting in ("rollout", "alignment_rollout"):
            a, b = first[setting], r[setting]
            if (a is None) != (b is None):
                raise ValueError(f"{setting} execution differs")
            if a:
                signature = lambda result: [(e["episode_id"], e["seed"], e.get("start_timestep"), e.get("initial_geometry_sha256")) for e in result["episodes"]]
                if a["horizon"] != b["horizon"] or signature(a) != signature(b):
                    raise ValueError(f"{setting} paired states/horizon mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    lines = ["# Graph prototype comparison", "", "Single training seed; held-out expert episodes. Errors use normalized controller actions.", "",
             "| Model | Action MSE | Translation RMSE | Rotation RMSE | Rollout SR | Params | Latency (ms) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    names = {"flat": "Flat MLP", "center": "Center GNN", "surface": "Surface GNN"}
    for r in reports:
        m, sr = r["offline"], r["rollout"]
        lines.append(f"| {names[r['model']]} | {m['action_mse']:.6f} | {m['translation_rmse']:.6f} | {m['rotation_rmse']:.6f} | {sr['success_rate']:.1%}"+f" ({sr['evaluated_episodes']} eps) | {r['parameters']:,} | {r['latency']['total_ms']:.3f} |" if sr else
                     f"| {names[r['model']]} | {m['action_mse']:.6f} | {m['translation_rmse']:.6f} | {m['rotation_rmse']:.6f} | N/A | {r['parameters']:,} | {r['latency']['total_ms']:.3f} |")
    lines += ["", "Latency = feature/graph construction + network inference, batch size 1 on CPU.",
              "Translation/rotation RMSE is per normalized action component, not realised TCP motion error.", ""]
    if first["rollout"]:
        lines.append(f"Full rollout horizon: {first['rollout']['horizon']} steps; task's registered limit is 100.")
    else:
        lines.append("Rollout SR is N/A because simulator rollout was not requested.")
    if first["alignment_rollout"]:
        lines += ["", "| Model | Alignment start SR | Final transverse error (m) |", "|---|---:|---:|"]
        for r in reports:
            a = r["alignment_rollout"]
            sr = f"{a['success_rate']:.1%}" if a['success_rate'] is not None else "N/A"
            error = f"{a['mean_final_transverse_error_m']:.5f}" if a['mean_final_transverse_error_m'] is not None else "N/A"
            lines.append(f"| {names[r['model']]} | {sr} ({a['evaluated_episodes']} eps) | {error} |")
    lines += ["", "Parameter counts differ; this run does not isolate graph structure from encoder capacity.",
              "Offline action error and rollout success measure different outcomes; neither establishes superiority from one seed."]
    (args.output/"results.md").write_text("\n".join(lines)+"\n")
    write_json(args.output/"results.json", reports)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ("action_mse", "translation_mse", "rotation_mse")):
        ax.bar([names[r["model"]] for r in reports], [r["offline"][metric] for r in reports])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(args.output/"errors.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    phases = list(first["phase_offline"])
    for i, r in enumerate(reports):
        ax.bar([j+(i-1)*.25 for j in range(len(phases))],
               [r["phase_offline"][phase].get("action_mse", 0) for phase in phases],
               width=.25, label=names[r["model"]])
    ax.set_xticks(range(len(phases)), phases)
    ax.set_ylabel("Normalized action MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output/"phase_errors.png", dpi=150)
    plt.close(fig)
    print((args.output/"results.md").read_text())


if __name__ == "__main__":
    main()
