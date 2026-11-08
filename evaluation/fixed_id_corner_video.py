"""Render real-camera videos of fixed-ID per-corner temporal predictions."""

import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.fixed_id_corner_sequences import (
    read_manifest, read_source_manifest, sequences_from_arrays, validate_packed_episode,
)
from data_collection.vertex_family_manifest import read_manifest as read_family_manifest
from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from utils.experiment_logger import ExperimentLogger, write


def project_world(point, intrinsic, extrinsic, width=512, height=512):
    """Project a world XYZ point into the raw CV camera; return None off screen."""
    camera = extrinsic[:, :3] @ np.asarray(point) + extrinsic[:, 3]
    if not np.isfinite(camera).all() or camera[2] <= 0:
        return None
    pixel = intrinsic @ camera
    xy = pixel[:2] / pixel[2]
    if not (0 <= xy[0] < width and 0 <= xy[1] < height):
        return None
    rounded = np.rint(xy).astype(int)
    return int(min(rounded[0], width - 1)), int(min(rounded[1], height - 1))


def load_models(config, dataset_hash, device):
    """Load the same-seed best checkpoints and verify their dataset and identity."""
    models, provenance = {}, {}
    seeds = set()
    for name in ("mamba", "transformer"):
        settings = config["models"][name]
        path = Path(settings["checkpoint"]).resolve()
        digest = file_sha256(path)
        if digest != settings["sha256"]:
            raise ValueError(f"{name} checkpoint checksum mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (checkpoint["dataset_sha256"] != dataset_hash
                or checkpoint["config"]["model"]["architecture"] != name):
            raise ValueError(f"{name} checkpoint architecture/dataset mismatch")
        source_root = Path(__file__).resolve().parents[1]
        for source_name in ("graph/temporal/fixed_id_corner.py", "graph/temporal/causal_transformer.py",
                            "data_collection/temporal/fixed_id_corner_sequences.py"):
            expected = checkpoint["config"]["implementation"]["files_sha256"][source_name]
            if file_sha256(source_root / source_name) != expected:
                raise ValueError(f"{name} inference source differs from checkpoint: {source_name}")
        model = FixedIdCornerPredictor(**checkpoint["config"]["model"])
        model.load_state_dict(checkpoint["model"], strict=True)
        models[name] = model.to(device).eval()
        seeds.add(checkpoint["config"]["seed"])
        provenance[name] = dict(path=str(path), sha256=digest, epoch=checkpoint["epoch"],
                                run_id=checkpoint["run_id"], seed=checkpoint["config"]["seed"])
    if len(seeds) != 1:
        raise ValueError("The two checkpoints must use the same seed")
    if (not torch.equal(models["mamba"].input_mean, models["transformer"].input_mean)
            or not torch.equal(models["mamba"].input_std, models["transformer"].input_std)):
        raise ValueError("The two checkpoints must use matching input normalization")
    return models, provenance, seeds.pop()


def predict(models, sequence, device):
    """Run each prediction using only the sequence prefix available at that frame."""
    inputs = torch.as_tensor(sequence.inputs[None], device=device)
    predictions = {}
    use_amp = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        for name, model in models.items():
            positions = sequence.last_observed_xyz_world.copy()
            for index in range(1, inputs.shape[1]):
                prefix = inputs[:, :index + 1]
                context = torch.ones(prefix.shape[:2], device=device, dtype=torch.bool)
                displacement = model(prefix, context)[0, -1].cpu().numpy()
                positions[index] += displacement
            predictions[name] = positions
    return predictions


def read_camera(root, row):
    """Load the original RGB and paired camera calibration after hash verification."""
    path = (root / row["raw_path"]).resolve()
    if not path.is_relative_to(root) or file_sha256(path) != row["source_sha256"]:
        raise ValueError(f"Original RGB-D source path/hash mismatch: {row['sample_id']}")
    with np.load(path, allow_pickle=False) as archive:
        rgb = archive["observation/sensor_data/base_camera/rgb"][0].copy()
        intrinsic = archive["observation/sensor_param/base_camera/intrinsic_cv"][0].copy()
        extrinsic = archive["observation/sensor_param/base_camera/extrinsic_cv"][0].copy()
    if rgb.shape != (512, 512, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected original base-camera RGB uint8 [512,512,3]")
    if intrinsic.shape != (3, 3) or extrinsic.shape != (3, 4):
        raise ValueError("Expected paired CV intrinsic [3,3] and extrinsic [3,4]")
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), intrinsic, extrinsic


def put_text(canvas, text, position, scale=0.52, color=(235, 235, 235), thickness=1):
    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def marker(image, xy, kind):
    """Draw a small color-coded point at the projected image coordinate."""
    if xy is None:
        return
    colors = dict(gt=(70, 235, 90), prediction=(235, 80, 245), detection=(0, 230, 255))
    if kind not in colors:
        raise ValueError(f"Unknown point type: {kind}")
    cv2.circle(image, xy, 4, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, xy, 3, colors[kind], -1, cv2.LINE_AA)


def draw_frame(image, intrinsic, extrinsic, frame, selection, sequence, gt, predictions):
    """Overlay current detection, model forecast and GT without changing observations."""
    canvas = np.full((704, 1024, 3), (28, 25, 23), dtype=np.uint8)
    put_text(canvas, f"PEG CORNER ID {selection['vertex_id']}  |  VALIDATION  |  GT-ASSISTED ID", (18, 25), 0.61, thickness=2)
    put_text(canvas, "Yellow: detection    Pink: model prediction    Green: GT projection", (18, 50), 0.55)
    put_text(canvas, "Natural detection gap; fixed ID supplied offline. Predictions use past observations only.", (18, 73), 0.47)
    index = frame - sequence.start_frame
    active = 0 <= index < len(sequence.inputs)
    visible = active and bool(sequence.observed[index])
    can_predict = active and bool(sequence.prediction_valid[index])
    state = "DETECTED" if visible else "NO CERTIFIED DETECTION" if active else "NOT YET SEEN"
    record = dict(frame=frame, state=state, gt_world=None, detection_world=None, predictions={})
    for panel, name in enumerate(("mamba", "transformer")):
        x_offset = panel * 512
        view = image.copy()
        if active:
            gt_point = gt[index]
            gt_xy = project_world(gt_point, intrinsic, extrinsic)
            marker(view, gt_xy, "gt")
            record["gt_world"] = gt_point.tolist()
            if visible:
                measurement = sequence.targets[index]
                marker(view, project_world(measurement, intrinsic, extrinsic), "detection")
                record["detection_world"] = measurement.tolist()
            if can_predict:
                predicted = predictions[name][index]
                pred_xy = project_world(predicted, intrinsic, extrinsic)
                marker(view, pred_xy, "prediction")
                error_mm = float(np.linalg.norm(predicted - gt_point) * 1000)
                record["predictions"][name] = dict(xyz_world=predicted.tolist(), gt_error_mm=error_mm,
                                                   prediction_in_frame=pred_xy is not None,
                                                   gt_in_frame=gt_xy is not None)
        canvas[100:612, x_offset:x_offset + 512] = view
        put_text(canvas, f"{name.upper()}  |  seed 0", (x_offset + 12, 93), 0.58, thickness=2)
        put_text(canvas, f"Frame {frame:03d}  |  {state}", (x_offset + 12, 638), 0.48,
                 (0, 230, 255) if visible else (235, 80, 245), 2)
        if can_predict:
            item = record["predictions"][name]
            put_text(canvas, f"Prediction vs GT: {item['gt_error_mm']:.1f} mm (3D)", (x_offset + 12, 662), 0.52)
            if not item["prediction_in_frame"] or not item["gt_in_frame"]:
                put_text(canvas, "Prediction or GT is outside camera view", (x_offset + 12, 683), 0.42)
        else:
            put_text(canvas, "No prediction until after first detection", (x_offset + 12, 662), 0.48)
    cv2.line(canvas, (512, 82), (512, 690), (95, 95, 95), 1)
    return canvas, record


def render_clip(output, item, packed, sequence, frame_rows, family_root, models, fps):
    """Stream the selected real frames into an H.264 MP4 and record 3-D errors."""
    start, end = item["gap_start"], item["gap_end"]
    first = sequence.start_frame
    length = len(packed["sample_ids"])
    left, right = start - first, end - first + 1
    if (left < 1 or right >= len(sequence.observed) or sequence.observed[left:right].any()
            or not sequence.observed[left - 1] or not sequence.observed[right]):
        raise ValueError(f"Selected interval is not a real detection gap with reappearance: {item['episode']}")
    predictions = predict(models, sequence, next(iter(models.values())).input_mean.device)
    index = int(np.flatnonzero(packed["vertex_ids"] == item["vertex_id"])[0])
    a, b = packed["offsets"][index:index + 2]
    gt = packed["gt_xyz_world"][a:b]
    begin = max(0, start - item["context_before"])
    finish = min(length, end + item["context_after"] + 1)
    path = output / f"{item['name']}.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
               "-video_size", "1024x704", "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    records = []
    try:
        for frame in range(begin, finish):
            sample_id = str(packed["sample_ids"][frame])
            row = frame_rows[sample_id]
            if row["split"] != "validation":
                raise ValueError("Video source frames must belong to development validation")
            image, intrinsic, extrinsic = read_camera(family_root, row)
            canvas, record = draw_frame(image, intrinsic, extrinsic, frame, item, sequence, gt, predictions)
            process.stdin.write(canvas.tobytes())
            records.append(record)
    finally:
        process.stdin.close()
        exit_code = process.wait()
    if exit_code:
        raise RuntimeError(f"FFmpeg exited {exit_code} creating {path}")
    gaps = [row for row in records if start <= row["frame"] <= end]
    summary = dict(selection=item, frame_range=[begin, finish - 1], fps=fps, video_sha256=file_sha256(path),
                   gap_frames=len(gaps), natural_gap=True, online_id_matching=False,
                   ground_truth_usage="offline ID certification, clip selection, labels and visualization; never model input",
                   missing_gap_3d_rmse_mm={name: float(np.sqrt(np.mean([
                       row["predictions"][name]["gt_error_mm"] ** 2 for row in gaps])))
                       for name in ("mamba", "transformer")}, frames=records)
    write(output / f"{item['name']}.json", summary)
    return path, summary


def render(config):
    """Create two real-validation video examples using the fixed-ID dataset."""
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Video inference requires CUDA and the official Mamba runtime")
    dataset_root = Path(config["data"]["corner_dataset"]).resolve()
    family_root = Path(config["data"]["family_dataset"]).resolve()
    manifest = read_manifest(dataset_root)
    if manifest["dataset_sha256"] != config["data"]["dataset_sha256"]:
        raise ValueError("Fixed-ID dataset hash mismatch")
    source = read_source_manifest(Path(manifest["config"]["source"]["corner_dataset"]))
    family = read_family_manifest(family_root)
    if (source["dataset_sha256"] != manifest["source_dataset_sha256"]
            or source["provenance"]["source_dataset_sha256"] != family["dataset_sha256"]):
        raise ValueError("Raw family dataset does not match the fixed-ID source lineage")
    models, model_info, seed = load_models(config, manifest["dataset_sha256"], device)
    rows = {row["episode"]: row for row in manifest["episodes"] if row["split"] == "validation"}
    frame_rows = {row["sample_id"]: row for row in family["samples"]
                  if row["episode"] in {item["episode"] for item in config["clips"]}}
    with ExperimentLogger(team="graph", model="fixed_id_corner_video", config=config,
                          seed=seed, job_type="visualization") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        results = []
        for item in config["clips"]:
            row = rows[item["episode"]]
            path = (dataset_root / row["file"]).resolve()
            if not path.is_relative_to(dataset_root) or file_sha256(path) != row["sha256"]:
                raise ValueError(f"Prepared episode checksum mismatch: {item['episode']}")
            with np.load(path, allow_pickle=False) as archive:
                packed = {name: archive[name].copy() for name in archive.files}
            validate_packed_episode(packed)
            sequence, = [s for s in sequences_from_arrays(packed, row) if s.vertex_id == item["vertex_id"]]
            video, summary = render_clip(output, item, packed, sequence, frame_rows, family_root, models, config["fps"])
            results.append({key: value for key, value in summary.items() if key != "frames"})
            logger.link_output(video, kind="rollouts")
            logger.link_output(output / f"{item['name']}.json")
            print(f"Rendered {video}: gap RMSE {summary['missing_gap_3d_rmse_mm']}", flush=True)
        write(output / "manifest.json", dict(dataset_sha256=manifest["dataset_sha256"], models=model_info,
                                            config=config, clips=results))
        logger.link_output(output / "manifest.json")
        logger.record(dataset_sha256=manifest["dataset_sha256"], source_checkpoints=model_info,
                      id_matching="offline GT-assisted fixed IDs", artificial_occlusion=False,
                      selected_examples_not_representative=True)
        print(f"Video output: {output}", flush=True)
        return str(output)
