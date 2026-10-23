"""Summarize paired validation rows and isolated measurements without uploads."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("outputs/recurrent/gpu_validation")
FIELDS = (
    "success",
    "failure_stage",
    "steps",
    "transport_step",
    "shift_applied",
    "shift_step",
    "blackout_applied",
    "blackout_step",
)


def read_rows(paths):
    rows = {}
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (row["condition"], row["eval_level"], row["episode_seed"])
            if key in rows:
                raise ValueError(f"Duplicate episode: {key}")
            rows[key] = row
    return rows


def main():
    cpu = {
        "mlp": read_rows([ROOT / "mlp_physx_cpu_1.jsonl"]),
        "gru": read_rows(sorted(ROOT.glob("cpu_gru_*.jsonl"))),
    }
    comparison = {}
    aggregates = {}
    for model in ("mlp", "gru"):
        variants = {"cpu1": cpu[model]}
        for batch in (16, 30):
            variants[f"gpu{batch}"] = read_rows(
                [ROOT / f"{model}_physx_cuda_{batch}.jsonl"]
            )
        for variant, rows in variants.items():
            for condition, level, _ in rows:
                group = [
                    r for (c, l, _), r in rows.items() if (c, l) == (condition, level)
                ]
                aggregates[f"{model}/{variant}/{condition}/{level}"] = {
                    "episodes": len(group),
                    "successes": sum(r["success"] for r in group),
                    "failures": dict(Counter(r["failure_stage"] for r in group)),
                    "shift_events": sum(r["shift_applied"] for r in group),
                    "blackout_events": sum(r["blackout_applied"] for r in group),
                    "termination_steps": dict(Counter(r["steps"] for r in group)),
                }
        for variant in ("gpu16", "gpu30"):
            if variants[variant].keys() != cpu[model].keys():
                raise ValueError(f"Unpaired episode keys for {model}/{variant}")
            mismatches = {}
            for key, reference in cpu[model].items():
                actual = variants[variant][key]
                different = {
                    f: {"cpu": reference[f], "gpu": actual[f]}
                    for f in FIELDS
                    if reference[f] != actual[f]
                }
                if different:
                    mismatches["/".join(map(str, key))] = different
            comparison[f"{model}/{variant}"] = {
                "episodes": len(cpu[model]),
                "field_mismatches": {
                    f: sum(f in m for m in mismatches.values()) for f in FIELDS
                },
                "episode_differences": mismatches,
            }
        differences = {}
        for key, reference in variants["gpu16"].items():
            actual = variants["gpu30"][key]
            fields = {
                f: {"gpu16": reference[f], "gpu30": actual[f]}
                for f in FIELDS
                if reference[f] != actual[f]
            }
            if fields:
                differences["/".join(map(str, key))] = fields
        comparison[f"{model}/gpu16_vs_gpu30"] = {
            "episodes": len(variants["gpu16"]),
            "field_mismatches": {
                f: sum(f in m for m in differences.values()) for f in FIELDS
            },
            "episode_differences": differences,
        }
    performance = {}
    for backend, batch in [("physx_cpu", 1), ("physx_cuda", 16), ("physx_cuda", 30)]:
        repeats = []
        for repeat in range(3):
            stem = ROOT / f"perf_{backend}_{batch}_{repeat}"
            perf = json.loads(stem.with_suffix(".summary.json").read_text())[
                "performance"
            ]
            process = json.loads(stem.with_suffix(".process.json").read_text())
            if (
                process.get("completed_episodes_before") != 0
                or process.get("new_episodes") != 30
            ):
                raise ValueError(
                    f"Performance must be a fresh 30-episode invocation: {stem}"
                )
            if backend == "physx_cpu":
                reset, rollout = perf["reset_s"], perf["rollout_s"]
                active = physical = perf["active_env_steps"]
                infer = None
            else:
                reset = sum(b["reset_s"] for b in perf["batches"])
                rollout = sum(b["rollout_s"] for b in perf["batches"])
                active = sum(b["active_env_steps"] for b in perf["batches"])
                physical = sum(b["physical_env_steps"] for b in perf["batches"])
                infer = float(np.mean([b["batch_infer_ms"] for b in perf["batches"]]))
            repeats.append(
                {
                    "process_s": process["process_s"],
                    "evaluation_s": perf["elapsed_s"],
                    "creation_s": perf["environment_creation_s"],
                    "reset_s": reset,
                    "rollout_s": rollout,
                    "save_s": perf["save_s"],
                    "episodes_per_s": 30 / process["process_s"],
                    "active_env_steps_per_s": active / rollout,
                    "overall_env_steps_per_s": active / process["process_s"],
                    "physical_env_steps_per_s": physical / rollout,
                    "batch_infer_ms": infer,
                    "peak_device_vram_gib": process["peak_device_vram_bytes"] / 2**30,
                }
            )
        means = {
            k: float(np.mean([r[k] for r in repeats]))
            for k in repeats[0]
            if repeats[0][k] is not None
        }
        performance[f"{backend}/{batch}"] = {"repeats": repeats, "means": means}
    cpu_rate = performance["physx_cpu/1"]["means"]["episodes_per_s"]
    for value in performance.values():
        value["speedup_vs_cpu"] = value["means"]["episodes_per_s"] / cpu_rate
    result = {
        "aggregates": aggregates,
        "paired_comparison": comparison,
        "performance": performance,
    }
    (ROOT / "comparison.json").write_text(json.dumps(result, indent=2))
    lines = [
        "| Backend | Process s | Create s | Reset s | Rollout s | Save s | Episode/s | Rollout env-step/s | Peak VRAM GiB | Speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in performance.items():
        m = value["means"]
        lines.append(
            f"| {name} | {m['process_s']:.2f} | {m['creation_s']:.2f} | {m['reset_s']:.2f} | {m['rollout_s']:.2f} | {m['save_s']:.4f} | {m['episodes_per_s']:.3f} | {m['active_env_steps_per_s']:.1f} | {m['peak_device_vram_gib']:.2f} | {value['speedup_vs_cpu']:.2f}× |"
        )
    (ROOT / "performance.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
