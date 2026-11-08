"""Render all previously detected fixed-ID corners on original camera frames."""

from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.fixed_id_corner_sequences import (
    load_source_episode, read_manifest, read_source_manifest, sequences_from_arrays, validate_packed_episode,
)
from data_collection.vertex_family_manifest import read_manifest as read_family_manifest
from evaluation.fixed_id_corner_video import load_models, marker, predict, project_world, put_text, read_camera
from utils.experiment_logger import ExperimentLogger, write


def prepare_tracks(packed, row, models, device):
    """Predict every ID with independent causal prefixes and shared model weights."""
    tracks = []
    for index, sequence in enumerate(sequences_from_arrays(packed, row)):
        left, right = packed["offsets"][index:index + 2]
        tracks.append(dict(sequence=sequence, gt=packed["gt_xyz_world"][left:right],
                           predictions=predict(models, sequence, device)))
        print(f"Prepared corner {index + 1}/{len(packed['vertex_ids'])}: ID {sequence.vertex_id}", flush=True)
    return tracks


def snapshot_for_frame(tracks, frame):
    """Return only IDs already observed, with actual detection or model output."""
    result = []
    for track in tracks:
        sequence = track["sequence"]
        index = frame - sequence.start_frame
        if index < 0 or index >= len(sequence.inputs):
            continue
        observed = bool(sequence.observed[index])
        if not observed and not sequence.prediction_valid[index]:
            raise ValueError("A missing corner cannot precede its first detection")
        measurement = sequence.targets[index] if observed else None
        result.append(dict(
            vertex_id=sequence.vertex_id,
            observed=observed,
            measurement=None if measurement is None else measurement.tolist(),
            gt=track["gt"][index].tolist(),
            predictions={name: values[index].tolist() for name, values in track["predictions"].items()}
            if sequence.prediction_valid[index] else {},
        ))
    return result


def draw_all_frame(image, intrinsic, extrinsic, frame, snapshot):
    """Show observed outputs and missing outputs for all seen IDs side by side."""
    canvas = np.full((704, 1024, 3), (28, 25, 23), dtype=np.uint8)
    put_text(canvas, "ALL PREVIOUSLY SEEN CORNERS  |  VALIDATION  |  GT-ASSISTED IDS", (18, 25), 0.6, thickness=2)
    put_text(canvas, "Yellow: actual detection    Pink: prediction when missing    Green: missing-point GT", (18, 50), 0.52)
    put_text(canvas, "Small points may overlap; unknown IDs are unassigned. No artificial masking.", (18, 73), 0.47)
    observed_count = sum(item["observed"] for item in snapshot)
    missing_count = len(snapshot) - observed_count
    errors = {}
    for panel, name in enumerate(("mamba", "transformer")):
        x_offset = panel * 512
        view = image.copy()
        errors[name] = []
        for item in snapshot:
            if not item["observed"]:
                marker(view, project_world(item["gt"], intrinsic, extrinsic), "gt")
        for item in snapshot:
            if item["observed"]:
                marker(view, project_world(item["measurement"], intrinsic, extrinsic), "detection")
            else:
                point = item["predictions"][name]
                marker(view, project_world(point, intrinsic, extrinsic), "prediction")
                errors[name].append(float(np.linalg.norm(np.asarray(point) - item["gt"]) * 1000))
        canvas[100:612, x_offset:x_offset + 512] = view
        put_text(canvas, f"{name.upper()}  |  seed 0", (x_offset + 12, 93), 0.58, thickness=2)
        put_text(canvas, f"Frame {frame:03d}  |  Seen {len(snapshot)}  |  Detected {observed_count}  |  Missing {missing_count}",
                 (x_offset + 12, 638), 0.45, thickness=2)
        if errors[name]:
            rmse = float(np.sqrt(np.mean(np.square(errors[name]))))
            put_text(canvas, f"Missing vs GT: {rmse:.1f} mm RMSE (3D)", (x_offset + 12, 662), 0.51)
        else:
            put_text(canvas, "No missing previously seen corners", (x_offset + 12, 662), 0.48)
    cv2.line(canvas, (512, 82), (512, 690), (95, 95, 95), 1)
    return canvas, errors


def render_episode(output, item, packed, row, source_arrays, frame_rows, family_root, models, fps):
    """Render one complete real-camera episode without artificial occlusion."""
    known = source_arrays["valid"] & (source_arrays["physical_ids"] >= 0)
    seen_ids = set(map(int, source_arrays["physical_ids"][known]))
    packed_ids = set(map(int, packed["vertex_ids"]))
    if seen_ids != packed_ids:
        raise ValueError(f"Episode contains IDs omitted from prediction samples: {sorted(seen_ids - packed_ids)}")
    tracks = prepare_tracks(packed, row, models, next(iter(models.values())).input_mean.device)
    path = output / f"{item['name']}.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
               "-video_size", "1024x704", "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    records = []
    try:
        for frame, sample_id in enumerate(packed["sample_ids"]):
            source_row = frame_rows[str(sample_id)]
            if source_row["split"] != "validation":
                raise ValueError("Video source frames must belong to development validation")
            image, intrinsic, extrinsic = read_camera(family_root, source_row)
            snapshot = snapshot_for_frame(tracks, frame)
            current_ids = set(map(int, source_arrays["physical_ids"][frame][known[frame]]))
            snapshot_ids = {item["vertex_id"] for item in snapshot if item["observed"]}
            if current_ids != snapshot_ids:
                raise ValueError(f"Frame {frame}: certified detections were lost or invented")
            canvas, errors = draw_all_frame(image, intrinsic, extrinsic, frame, snapshot)
            process.stdin.write(canvas.tobytes())
            records.append(dict(frame=frame, corners=snapshot, missing_errors_mm=errors))
    finally:
        process.stdin.close()
        exit_code = process.wait()
    if exit_code:
        raise RuntimeError(f"FFmpeg exited {exit_code} creating {path}")
    missing_errors = {name: [error for frame in records for error in frame["missing_errors_mm"][name]]
                      for name in ("mamba", "transformer")}
    result = dict(episode=item["episode"], frames=len(records), fps=fps, vertex_ids=sorted(seen_ids),
                  video_sha256=file_sha256(path), natural_visibility=True, online_id_matching=False,
                  unknown_detections_unassigned=int((source_arrays["valid"] & (source_arrays["physical_ids"] < 0)).sum()),
                  missing_corner_frames=len(missing_errors["mamba"]),
                  missing_3d_rmse_mm={name: float(np.sqrt(np.mean(np.square(errors)))) if errors else None
                                      for name, errors in missing_errors.items()},
                  frame_records=records)
    write(output / f"{item['name']}.json", result)
    return path, result


def render(config):
    """Create complete validation-episode videos using every previously seen ID."""
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Video inference requires CUDA and the official Mamba runtime")
    dataset_root = Path(config["data"]["corner_dataset"]).resolve()
    family_root = Path(config["data"]["family_dataset"]).resolve()
    manifest = read_manifest(dataset_root)
    if manifest["dataset_sha256"] != config["data"]["dataset_sha256"]:
        raise ValueError("Fixed-ID dataset hash mismatch")
    source_root = Path(manifest["config"]["source"]["corner_dataset"]).resolve()
    source = read_source_manifest(source_root)
    family = read_family_manifest(family_root)
    if (source["dataset_sha256"] != manifest["source_dataset_sha256"]
            or source["provenance"]["source_dataset_sha256"] != family["dataset_sha256"]):
        raise ValueError("Raw family dataset does not match the fixed-ID source lineage")
    models, model_info, seed = load_models(config, manifest["dataset_sha256"], device)
    rows = {row["episode"]: row for row in manifest["episodes"] if row["split"] == "validation"}
    source_rows = {row["episode"]: row for row in source["episodes"] if row["split"] == "validation"}
    selected = {item["episode"] for item in config["episodes"]}
    frame_rows = {row["sample_id"]: row for row in family["samples"] if row["episode"] in selected}
    with ExperimentLogger(team="graph", model="fixed_id_all_corners_video", config=config,
                          seed=seed, job_type="visualization") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        results = []
        for item in config["episodes"]:
            row = rows[item["episode"]]
            path = (dataset_root / row["file"]).resolve()
            if not path.is_relative_to(dataset_root) or file_sha256(path) != row["sha256"]:
                raise ValueError(f"Prepared episode checksum mismatch: {item['episode']}")
            with np.load(path, allow_pickle=False) as archive:
                packed = {name: archive[name].copy() for name in archive.files}
            validate_packed_episode(packed)
            source_arrays = load_source_episode(source_root, source_rows[item["episode"]])
            video, result = render_episode(output, item, packed, row, source_arrays, frame_rows, family_root, models, config["fps"])
            results.append({key: value for key, value in result.items() if key != "frame_records"})
            logger.link_output(video, kind="rollouts")
            logger.link_output(output / f"{item['name']}.json")
            print(f"Rendered {video}: seen IDs={len(result['vertex_ids'])}, missing RMSE={result['missing_3d_rmse_mm']}", flush=True)
        write(output / "manifest.json", dict(dataset_sha256=manifest["dataset_sha256"], models=model_info,
                                            config=config, episodes=results))
        logger.link_output(output / "manifest.json")
        logger.record(dataset_sha256=manifest["dataset_sha256"], source_checkpoints=model_info,
                      id_matching="offline GT-assisted fixed IDs", artificial_occlusion=False,
                      selected_examples_not_representative=True)
        print(f"All-corner video output: {output}", flush=True)
        return str(output)
