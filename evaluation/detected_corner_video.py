"""Render detector-only corner association on the original camera video."""

import colorsys
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.temporal.fixed_id_corner_sequences import load_source_episode, read_manifest, read_source_manifest
from data_collection.vertex_family_manifest import read_manifest as read_family_manifest
from evaluation.fixed_id_corner_video import load_models, project_world, put_text, read_camera
from graph.temporal.detected_corner_tracking import DetectedCornerTracker
from utils.experiment_logger import ExperimentLogger, write


def track_color(track_id):
    """Assign a stable bright color to each track, independently per panel."""
    hue = (track_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 0.95)
    return int(blue * 255), int(green * 255), int(red * 255)


def draw_track(view, point, track_id, observed):
    """Draw a small colored point with a black or white one-pixel rim."""
    if point is None:
        return
    rim = (0, 0, 0) if observed else (245, 245, 245)
    cv2.circle(view, point, 4, rim, -1, cv2.LINE_AA)
    cv2.circle(view, point, 3, track_color(track_id), -1, cv2.LINE_AA)


def draw_frame(image, intrinsic, extrinsic, frame, outputs):
    """Display every online track with stable colors and detection provenance."""
    canvas = np.full((704, 1024, 3), (28, 25, 23), dtype=np.uint8)
    put_text(canvas, "ONLINE CORNER TRACKING  |  VALIDATION  |  NO GT IDS IN MATCHING", (18, 25), .57, thickness=2)
    put_text(canvas, "Dot color = track ID    Black rim = detected    White rim = prediction", (18, 50), .51)
    put_text(canvas, "Colors are independent across panels. Predicted points may project over occluders.", (18, 73), .45)
    for panel, name in enumerate(("mamba", "transformer")):
        x_offset = panel * 512
        view = image.copy()
        value = outputs[name]
        for track_id, (position, observed) in enumerate(zip(value["positions"], value["observed"])):
            draw_track(view, project_world(position, intrinsic, extrinsic), track_id, bool(observed))
        canvas[100:612, x_offset:x_offset + 512] = view
        put_text(canvas, f"{name.upper()}  |  seed 0  |  distance gate 30 mm", (x_offset + 12, 93), .51, thickness=2)
        visible = int(value["observed"].sum())
        total = len(value["positions"])
        put_text(canvas, f"Frame {frame:03d}  |  Tracks {total}  |  Detected {visible}  |  Predicted {total-visible}",
                 (x_offset + 12, 638), .43, thickness=2)
        put_text(canvas, f"Current frame: {value['matches']} matches, {value['new_tracks']} new tracks",
                 (x_offset + 12, 662), .46)
    cv2.line(canvas, (512, 82), (512, 690), (95, 95, 95), 1)
    return canvas


def validation_records(config):
    """Verify each source evaluation uses the selected model and threshold."""
    records = {}
    for name in ("mamba", "transformer"):
        path = Path(config["validation"][name]["path"])
        if file_sha256(path) != config["validation"][name]["sha256"]:
            raise ValueError(f"{name} validation result hash mismatch")
        data = json.loads(path.read_text())
        if (data["config"]["seed"] != 0 or data["config"]["temporal"]["architecture"] != name
                or data["config"]["temporal"]["sha256"] != config["models"][name]["sha256"]
                or data["selection"]["threshold_mm"] != config["threshold_mm"]):
            raise ValueError(f"{name} source validation identity differs from this rendering")
        records[name] = {row["episode"]: row["counts"] for row in data["result"]["per_episode"]}
    return records


def render_episode(output, item, source, frame_rows, family_root, models, threshold, fps, expected):
    """Replay raw detections with no fixed IDs and write every frame's track state."""
    device = next(iter(models.values())).input_mean.device
    trackers = {name: DetectedCornerTracker(model, threshold / 1000, device) for name, model in models.items()}
    path = output / f"{item['name']}.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
               "-video_size", "1024x704", "-framerate", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    records = []
    totals = {name: dict(matches=0, new_tracks=0) for name in models}
    try:
        for frame, sample_id in enumerate(source["sample_ids"]):
            row = frame_rows[str(sample_id)]
            if row["split"] != "validation":
                raise ValueError("Video may only use development-validation frames")
            image, intrinsic, extrinsic = read_camera(family_root, row)
            valid = source["valid"][frame]
            xyz, kinds = source["xyz"][frame, valid], source["kind"][frame, valid]
            outputs = {name: tracker.step(xyz, kinds, source["robot"][frame]) for name, tracker in trackers.items()}
            frame_record = {"frame": frame, "sample_id": str(sample_id), "models": {}}
            for name, result in outputs.items():
                totals[name]["matches"] += result["matches"]
                totals[name]["new_tracks"] += result["new_tracks"]
                # Every actual detector point is represented exactly, including
                # points whose physical identity cannot be certified offline.
                np.testing.assert_array_equal(result["positions"][result["detection_tracks"]], xyz)
                frame_record["models"][name] = dict(
                    positions=result["positions"].tolist(), observed=result["observed"].tolist(),
                    detection_tracks=result["detection_tracks"].tolist(), matches=result["matches"],
                    new_tracks=result["new_tracks"],
                    in_image=[project_world(pos, intrinsic, extrinsic) is not None for pos in result["positions"]],
                )
            canvas = draw_frame(image, intrinsic, extrinsic, frame, outputs)
            process.stdin.write(canvas.tobytes())
            records.append(frame_record)
    finally:
        process.stdin.close()
        exit_code = process.wait()
    if exit_code:
        raise RuntimeError(f"FFmpeg exited {exit_code} creating {path}")
    for name in models:
        expected_counts = expected[name][item["episode"]]
        if (totals[name]["matches"] != expected_counts["matches"]
                or totals[name]["new_tracks"] != expected_counts["new_tracks"]
                or len(trackers[name].tracks) != expected_counts["final_tracks"]
                or len(records) != expected_counts["frames"]):
            raise ValueError(f"Replay differs from recorded {name} validation counts")
    result = dict(episode=item["episode"], frames=len(records), fps=fps, threshold_mm=threshold,
                  validation_counts={name: expected[name][item["episode"]] for name in models},
                  video_sha256=file_sha256(path), frame_records=records)
    write(output / f"{item['name']}.json", result)
    return path, result


def render(config):
    """Render matched validation examples using actual original camera frames."""
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Rendering requires CUDA and official mamba_ssm")
    dataset_root = Path(config["data"]["corner_dataset"]).resolve()
    family_root = Path(config["data"]["family_dataset"]).resolve()
    manifest = read_manifest(dataset_root)
    if manifest["dataset_sha256"] != config["data"]["dataset_sha256"]:
        raise ValueError("Fixed-ID dataset hash mismatch")
    source_root = Path(manifest["config"]["source"]["corner_dataset"]).resolve()
    source_manifest = read_source_manifest(source_root)
    family = read_family_manifest(family_root)
    if (source_manifest["dataset_sha256"] != manifest["source_dataset_sha256"]
            or source_manifest["provenance"]["source_dataset_sha256"] != family["dataset_sha256"]):
        raise ValueError("Camera, detector and fixed-ID datasets have incompatible lineage")
    models, model_info, seed = load_models(config, manifest["dataset_sha256"], device)
    if seed != 0 or config["threshold_mm"] != 30:
        raise ValueError("This visualization is configured for seed 0 and the calibrated 30 mm gate")
    expected = validation_records(config)
    source_rows = {row["episode"]: row for row in source_manifest["episodes"] if row["split"] == "validation"}
    selected = {item["episode"] for item in config["episodes"]}
    frame_rows = {row["sample_id"]: row for row in family["samples"] if row["episode"] in selected}
    with ExperimentLogger(team="graph", model="detected_corner_video", config=config,
                          seed=seed, job_type="visualization") as logger:
        output = Path(config["output"]["root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        episodes = []
        for item in config["episodes"]:
            source = load_source_episode(source_root, source_rows[item["episode"]])
            video, data = render_episode(output, item, source, frame_rows, family_root, models,
                                         config["threshold_mm"], config["fps"], expected)
            episodes.append({key: value for key, value in data.items() if key != "frame_records"})
            logger.link_output(video, kind="rollouts")
            logger.link_output(output / f"{item['name']}.json")
            print(f"Rendered {video}: {data['frames']} frames", flush=True)
        write(output / "manifest.json", dict(config=config, models=model_info, episodes=episodes,
              gt_in_matching=False, natural_visibility=True, selected_examples_not_representative=True))
        logger.link_output(output / "manifest.json")
        logger.record(dataset_sha256=manifest["dataset_sha256"], source_checkpoints=model_info,
                      gt_in_matching=False)
        print(f"Online tracking videos: {output}", flush=True)
        return str(output)
