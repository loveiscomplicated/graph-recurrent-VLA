"""Measure fixed-ID forecasts at naturally reappearing corner detections."""

from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.fixed_id_corner_sequences import (
    load_source_episode, read_manifest, read_source_manifest, sequences_from_arrays,
    validate_packed_episode,
)
from training.fixed_id_gnn import load_temporal
from utils.experiment_logger import ExperimentLogger, write


def reappearance_events(sequence, source):
    """Find real missing-to-detected transitions and all same-type competitors."""
    events = []
    for index in range(1, len(sequence.observed)):
        if not sequence.observed[index] or sequence.observed[index - 1]:
            continue
        start = index - 1
        while start >= 0 and not sequence.observed[start]:
            start -= 1
        gap = index - 1 - start
        frame = sequence.start_frame + index
        kind = 0 if sequence.vertex_id < 8 else 1
        candidates = np.flatnonzero(source["valid"][frame] & (source["kind"][frame] == kind))
        positions = source["xyz"][frame, candidates]
        target = sequence.targets[index]
        # The prepared dataset resolves rare duplicate certificates using GT;
        # identify that exact measured coordinate without replacing it by GT.
        same_id = source["physical_ids"][frame, candidates] == sequence.vertex_id
        matching = np.flatnonzero(same_id & np.all(positions == target, axis=1))
        if not len(matching):
            raise ValueError("Prepared reappearance is missing its certified detector point")
        selected = int(matching[0])
        keep = np.arange(len(candidates)) != selected
        others = positions[keep]
        other_ids = source["physical_ids"][frame, candidates[keep]]
        events.append(dict(
            episode=sequence.episode, vertex_id=sequence.vertex_id, frame=frame,
            gap_frames=gap, inputs=sequence.inputs[:index + 1],
            last=sequence.last_observed_xyz_world[index], target=target,
            competitors=others, competitor_ids=other_ids,
        ))
    return events


def forecast_events(model, events, device, batch_size):
    """Forecast each event from its own right-padded, causal prefix only."""
    predictions = []
    with torch.inference_mode():
        for left in range(0, len(events), batch_size):
            batch = events[left:left + batch_size]
            maximum = max(len(event["inputs"]) for event in batch)
            inputs = np.zeros((len(batch), maximum, 13), dtype=np.float32)
            mask = np.zeros((len(batch), maximum), dtype=bool)
            for index, event in enumerate(batch):
                size = len(event["inputs"])
                inputs[index, :size] = event["inputs"]
                mask[index, :size] = True
            context = torch.as_tensor(mask, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                delta = model(torch.as_tensor(inputs, device=device), context).float()
            lengths = context.sum(dim=1)
            final = delta[torch.arange(len(batch), device=device), lengths - 1].cpu().numpy()
            predictions.extend(event["last"] + change for event, change in zip(batch, final))
    return np.asarray(predictions, dtype=np.float32)


def event_measurement(event, prediction):
    """Compare the forecast to the true detection and same-kind alternatives."""
    true_distance = float(np.linalg.norm(prediction - event["target"]) * 1000)
    distances = np.linalg.norm(event["competitors"] - prediction, axis=1) * 1000
    nearest_index = int(np.argmin(distances)) if len(distances) else None
    competitor_distance = float(distances[nearest_index]) if nearest_index is not None else None
    return dict(
        true_detection_distance_mm=true_distance,
        nearest_competitor_distance_mm=competitor_distance,
        nearest_competitor_id=int(event["competitor_ids"][nearest_index]) if nearest_index is not None else None,
        competitor_count=len(distances),
        unknown_competitor_count=int((event["competitor_ids"] < 0).sum()),
        closer_competitor=competitor_distance is not None and competitor_distance < true_distance,
    )


def summary(records):
    """Summarize each seed independently, including threshold/crowding tradeoffs."""
    result = {}
    groups = {"all": lambda gap: True, "1": lambda gap: gap == 1,
              "2-5": lambda gap: 2 <= gap <= 5, "6-10": lambda gap: 6 <= gap <= 10,
              "11-20": lambda gap: 11 <= gap <= 20, "21+": lambda gap: gap >= 21}
    for group, select in groups.items():
        selected = [record for record in records if select(record["gap_frames"])]
        if not selected:
            result[group] = dict(events=0)
            continue
        error = np.asarray([record["true_detection_distance_mm"] for record in selected])
        closer = np.asarray([record["closer_competitor"] for record in selected])
        result[group] = dict(
            events=len(selected), mean_mm=float(error.mean()), median_mm=float(np.median(error)),
            p90_mm=float(np.percentile(error, 90)), p95_mm=float(np.percentile(error, 95)),
            p99_mm=float(np.percentile(error, 99)), max_mm=float(error.max()),
            competitor_closer_count=int(closer.sum()), competitor_closer_fraction=float(closer.mean()),
            thresholds_mm={str(limit): dict(
                true_within_fraction=float((error <= limit).mean()),
                true_within_and_nearest_fraction=float(((error <= limit) & ~closer).mean()),
            ) for limit in (5, 10, 20, 50)},
        )
    return result


def analyze(config):
    """Evaluate six frozen models on identical natural-validation reappearances."""
    root = Path(config["data"]["corner_dataset"]).resolve()
    manifest = read_manifest(root)
    if manifest["dataset_sha256"] != config["data"]["dataset_sha256"]:
        raise ValueError("Fixed-ID dataset hash mismatch")
    source_root = Path(manifest["config"]["source"]["corner_dataset"]).resolve()
    source_manifest = read_source_manifest(source_root)
    if source_manifest["dataset_sha256"] != manifest["source_dataset_sha256"]:
        raise ValueError("Source dataset lineage mismatch")
    source_rows = {row["episode"]: row for row in source_manifest["episodes"]}
    events = []
    episodes = [row for row in manifest["episodes"] if row["split"] == "validation"]
    for row in episodes:
        path = (root / row["file"]).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != row["sha256"]:
            raise ValueError(f"Prepared episode checksum mismatch: {row['episode']}")
        with np.load(path, allow_pickle=False) as archive:
            packed = {key: archive[key].copy() for key in archive.files}
        validate_packed_episode(packed)
        source = load_source_episode(source_root, source_rows[row["episode"]])
        for sequence in sequences_from_arrays(packed, row):
            events.extend(reappearance_events(sequence, source))
    if not events:
        raise ValueError("Validation contains no reappearance events")
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA and official mamba_ssm are required for matched evaluation")
    with ExperimentLogger(team="graph", model="fixed_id_reappearance", config=config,
                          seed=0, job_type="evaluation") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        event_rows = [dict(episode=e["episode"], vertex_id=e["vertex_id"], frame=e["frame"],
                           gap_frames=e["gap_frames"]) for e in events]
        summaries = {}
        for architecture in ("mamba", "transformer"):
            summaries[architecture] = {}
            for seed in (0, 1, 2):
                setting = config["models"][architecture][str(seed)]
                model_config = dict(seed=seed, data=dict(dataset_sha256=manifest["dataset_sha256"]),
                                    temporal=dict(architecture=architecture, **setting))
                model, checkpoint = load_temporal(model_config, device)
                positions = forecast_events(model, events, device, config["batch_size"])
                records = []
                for row, event, position in zip(event_rows, events, positions):
                    value = event_measurement(event, position)
                    row.setdefault("models", {}).setdefault(architecture, {})[str(seed)] = value
                    records.append(dict(gap_frames=event["gap_frames"], **value))
                summaries[architecture][str(seed)] = summary(records)
                logger.link_output(setting["checkpoint"], checkpoint=True)
                print(f"Evaluated {architecture} seed {seed}: {len(records)} reappearances", flush=True)
                del model
        result = dict(dataset_sha256=manifest["dataset_sha256"], episodes=len(episodes),
                      event_count=len(events), gap_max=max(event["gap_frames"] for event in events),
                      artificial_occlusion=False, matching_performed=False,
                      identity="offline GT-assisted certificates, including GT duplicate resolution",
                      competitor_scope="all other actual valid same-detector-type points, including unknown IDs",
                      summaries=summaries)
        write(output / "summary.json", result)
        write(output / "events.json", event_rows)
        logger.link_output(output / "summary.json")
        logger.link_output(output / "events.json")
        logger.record(dataset_sha256=manifest["dataset_sha256"], event_count=len(events))
        print(f"Reappearance analysis: {output}", flush=True)
        return str(output)
