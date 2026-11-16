"""Run matched B2 comparisons for the endpoint and spatial vertex objectives."""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import canonical_hash, file_sha256
from data_collection.state21 import audit_splits
from data_collection.state21_batches import PreparedWindows
from data_collection.state21_curriculum import STAGES, frame_limit, read_graph_episode
from data_collection.state21_shared_tapnext import load_shared_tapnext
from data_collection.state21_vertex_archive import FORMAT
from data_collection.state21_vertex_targets import (
    annotate_graph_vertices,
    annotate_vertices,
)
from main import load_config
from training.state21 import TrainingData, effective_config, fit, prepare_dataset
from training.state21_curriculum import _training_bundle
from utils.experiment_logger import ExperimentLogger, write


def _source_entries(settings, report):
    """Pair each loaded episode with its raw root, mode and TAPNext++ anchor."""
    by_name = {}
    for name, source in settings["sources"].items():
        manifest = Path(source["source_manifest"])
        with np.load(source["cache"], allow_pickle=False) as archive:
            anchors = dict(
                zip(
                    archive["episode_ids"].tolist(),
                    archive["surface_anchor_frame"].tolist(),
                    strict=True,
                )
            )
        selected = {
            row["episode_id"] for row in report["sources"][name]["selected_windows"]
        }
        for row in json.loads(manifest.read_text())["episodes"]:
            identity = row["path"]
            if identity in selected:
                by_name[identity] = manifest.parent, row["mode"], anchors[identity]
    return by_name


def _annotate_episode(episode, *, entries, random_frames, gate_m, margin_m):
    raw, mode, anchor = entries[episode.episode_id]
    if mode == "random":
        episode = replace(episode, samples=episode.samples[:random_frames])
    return annotate_vertices(episode, raw, anchor, gate_m=gate_m, margin_m=margin_m)


def _coverage():
    return {
        split: {kind: Counter() for kind in ("peg", "hole_assembly", "families")}
        for split in ("train", "validation")
    }


def _record_coverage(coverage, split, counts, families):
    for kind, values in counts.items():
        coverage[split][kind].update(values)
    coverage[split]["families"].update(families)


def _verify_coverage(coverage):
    for split, counts in coverage.items():
        if any(not counts[kind]["matched"] for kind in ("peg", "hole_assembly")):
            raise ValueError(
                f"No certified Peg or Hole assembly labels in {split}: {counts}"
            )
    print(f"Vertex alignment coverage: {json.dumps(coverage)}", flush=True)


def _prepare(config, matching, gate, margin):
    """Reuse the original split, matching once for all four training variants."""
    settings = config["data"]
    episodes, report = load_shared_tapnext(settings, splits=("train", "validation"))
    entries = _source_entries(settings, report)
    annotate = partial(
        _annotate_episode,
        entries=entries,
        random_frames=config["curriculum"]["auxiliary_random_frames"],
        gate_m=gate,
        margin_m=margin,
    )
    coverage = _coverage()
    labeled = []
    with ThreadPoolExecutor(max_workers=matching["workers"]) as workers:
        for episode, counts, families in workers.map(annotate, episodes):
            labeled.append(episode)
            _record_coverage(coverage, episode.split, counts, families)
    _verify_coverage(coverage)
    report["vertex_supervision"] = coverage
    report["manifest_sha256"] = canonical_hash(
        {
            "source": report["manifest_sha256"],
            "windows": [
                (episode.episode_id, len(episode.samples)) for episode in labeled
            ],
            "split": report["selected_reset_seeds"],
            "matching": {
                "gate_mm": matching["gate_mm"],
                "margin_mm": matching["margin_mm"],
            },
            "label_recipe": "anchor_and_current_same_object_nearest_with_margin",
        }
    )
    return TrainingData(
        settings,
        prepare_dataset(labeled, config, "train"),
        prepare_dataset(labeled, config, "validation"),
        report,
    )


def _prepare_cached(config, matching, gate, margin, reference):
    """Reuse the original graph bundle with a compact verified GT pose archive."""
    settings = config["data"]
    manifest_path, bundle = _training_bundle(settings)
    path = Path(reference["path"])
    digest = file_sha256(path)
    if digest != reference["sha256"]:
        raise ValueError(f"Vertex reference archive SHA-256 differs: {path}")
    if settings.get("smoke"):
        raise ValueError(
            "The original graph and vertex reference archives cover the full split"
        )
    coverage = _coverage()
    prepared = {
        split: PreparedWindows([], settings["history"], settings["max_edge_m"])
        for split in coverage
    }
    labeled = []
    random_frames = config["curriculum"]["auxiliary_random_frames"]
    with np.load(path, allow_pickle=False) as archive:
        receipt = json.loads(archive["metadata_json"].item())
        if receipt["format"] != FORMAT or any(
            receipt[key] != bundle[key]
            for key in ("full_split", "split_reference_sha256", "source_caches")
        ):
            raise ValueError(
                "Vertex reference archive and graph bundle sources or split differ"
            )
        if set(receipt["episodes"]) != {
            row["episode_id"] for row in bundle["episodes"]
        }:
            raise ValueError(
                "Vertex reference archive and graph bundle episodes differ"
            )
        for row in bundle["episodes"]:
            identity = row["episode_id"]
            count = frame_limit(row, STAGES[-1], 0, random_frames)
            stored = receipt["episodes"][identity]
            poses = archive[identity]
            if (
                poses.shape != (count, 25)
                or stored["frames"] != count
                or not np.isfinite(poses).all()
            ):
                raise ValueError(f"Unexpected GT pose rows for {identity}")
            episode, graphs = read_graph_episode(manifest_path.parent, row)
            episode = replace(episode, samples=episode.samples[:count])
            graphs = graphs[:count]
            episode, counts, families = annotate_graph_vertices(
                episode, graphs, poses, stored["anchor"], gate_m=gate, margin_m=margin
            )
            _record_coverage(coverage, episode.split, counts, families)
            prepared[episode.split].add_episode(episode, graphs)
            labeled.append(episode)
            if len(labeled) % 60 == 0:
                print(
                    f"Matched {len(labeled)}/{len(bundle['episodes'])} cached episodes",
                    flush=True,
                )
    _verify_coverage(coverage)
    report = audit_splits(labeled)
    report["verified_visible_vertex_rows"] = {
        split: counts["peg"]["visible"] + counts["hole_assembly"]["visible"]
        for split, counts in coverage.items()
    }
    report["selected_reset_seeds"] = {
        split: bundle["full_split"]["reset_seeds"][split] for split in prepared
    }
    report["vertex_supervision"] = coverage
    report["manifest_sha256"] = canonical_hash(
        {
            "graph_bundle": file_sha256(manifest_path),
            "gt_pose_archive": digest,
            "matching": {
                "gate_mm": matching["gate_mm"],
                "margin_mm": matching["margin_mm"],
            },
            "label_recipe": "anchor_and_current_same_object_nearest_with_margin",
            "windows": [
                (episode.episode_id, len(episode.samples)) for episode in labeled
            ],
        }
    )
    return TrainingData(settings, prepared["train"], prepared["validation"], report)


def _comparison_table(results):
    """Report final-epoch changes relative to the matched baseline."""
    names = {
        "baseline": "기준선",
        "vertex_only": "꼭짓점 손실만",
        "endpoint_only": "3 mm 끝점 손실만",
        "vertex_and_endpoint": "두 손실 모두",
    }
    baseline = results["baseline"]["last_validation"]
    rows = [
        f"## B2 · Epoch {baseline['epoch']} 검증",
        "",
        f"동일 검증 {results['baseline']['validation_frames_per_epoch']:,}프레임 기준입니다.",
        "",
        "| 설정 | 꼭짓점 RMSE | 기준선 대비 | 끝점 3 mm 이내 | 기준선 대비 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        metrics = result["last_validation"]
        rmse = metrics["spatial_vertex_rmse_mm"]
        rate = metrics["endpoint_within_3mm"]
        reference_rmse = baseline["spatial_vertex_rmse_mm"]
        rmse_change = (
            f"{(rmse / reference_rmse - 1) * 100:+.2f}%" if reference_rmse else "—"
        )
        rate_change = (rate - baseline["endpoint_within_3mm"]) * 100
        rows.append(
            f"| {names[name]} | {rmse:.3f} mm | {rmse_change} | "
            f"{rate:.2%} | {rate_change:+.2f}%p |"
        )
    return "\n".join(rows) + "\n"


def compare(config):
    """Fit all combinations using the same preparation, seed and epoch budget."""
    source = effective_config(load_config(config["source_config"]))
    options = {
        **source,
        "trainer": config["trainer"],
        "device": config["device"],
        "model": {**source["model"], "name": "spatial_first", "vertex_aux": True},
        "data": {**source["data"], **config.get("data", {})},
        "training": {
            **source["training"],
            "epochs": config["epochs"],
            "checkpoint_metric": "endpoint_miss_rate",
        },
        "comparison": config,
    }
    options = effective_config(options)
    torch.set_num_threads(options["training"]["cpu_threads"])
    matching = config["matching"]
    gate = matching["gate_mm"] / 1000
    margin = matching["margin_mm"] / 1000
    if not (np.isfinite(gate) and np.isfinite(margin) and gate > 0 and margin > 0):
        raise ValueError("Vertex matching needs positive finite gate_mm and margin_mm")
    reference = config.get("vertex_reference_archive")
    if reference is None:
        prepared = _prepare(options, matching, gate, margin)
    else:
        print("Preparing pinned graph and GT reference archives", flush=True)
        prepared = _prepare_cached(options, matching, gate, margin, reference)
    print(
        f"Prepared {len(prepared.train.windows)} train and "
        f"{len(prepared.validation.windows)} validation frames",
        flush=True,
    )
    endpoint = {
        "endpoint_transition_m": config["endpoint"]["transition_mm"] / 1000,
        "endpoint_weight": config["endpoint"]["weight"],
    }
    vertex = {"vertex_weight": config["vertex_weight"]}
    variants = {
        "baseline": {},
        "vertex_only": vertex,
        "endpoint_only": endpoint,
        "vertex_and_endpoint": {**vertex, **endpoint},
    }
    with ExperimentLogger(
        team="graph",
        model="spatial_first_loss_comparison",
        config=options,
        seed=options["seed"],
    ) as experiment:
        root = Path(config["output"]) / experiment.run_id
        print(f"Comparison output: {root}", flush=True)
        result = {}
        for name, task in variants.items():
            print(f"Training variant {name}", flush=True)
            output = fit(
                {
                    **options,
                    "task": {**options["task"], **task},
                    "output": {"root": str(root / name)},
                },
                prepared=prepared,
            )
            result[name] = json.loads((output / "training.json").read_text())
            result[name]["checkpoint"] = str(output / "last.pt")
            experiment.log(result[name]["best_validation"], prefix=name)
        summary = root / "comparison.json"
        write(
            summary,
            {
                "run_id": experiment.run_id,
                "reset_seeds": prepared.report["selected_reset_seeds"],
                "coverage": prepared.report["vertex_supervision"],
                "variants": result,
            },
        )
        experiment.link_output(summary, kind="metrics")
        table = root / "comparison.md"
        table.write_text(_comparison_table(result), encoding="utf-8")
        experiment.link_output(table, kind="metrics")
        print(table, flush=True)
        print(summary, flush=True)
        return summary
