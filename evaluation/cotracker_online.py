"""Replay a saved ManiSkill RGB sequence with CoTracker3 Online."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from cotracker.predictor import CoTrackerOnlinePredictor

from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import ExperimentLogger


def load_rgb_sequence(source, episode):
    """Load RGB alone and check frame identities against the episode receipt."""
    images = []
    for index, checksum in enumerate(episode["frame_sha256"]):
        path = source / f"frame_{index:06d}.npz"
        if file_sha256(path) != checksum:
            raise ValueError(f"Frame checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as frame:
            images.append(frame["observation/sensor_data/base_camera/rgb"][0])
    return np.stack(images)


@torch.inference_mode()
def track(predictor, frames, query_xy, device, add_support_grid):
    """Keep the first estimate the online predictor makes for each source frame."""
    frame_count = len(frames)
    query_count = len(query_xy)
    window = predictor.step * 2
    end_frames = list(range(window, frame_count, predictor.step)) + [frame_count]
    xy = np.full((frame_count, query_count, 2), np.nan, dtype=np.float32)
    visible = np.zeros((frame_count, query_count), dtype=bool)
    emitted_at = np.full(frame_count, -1, dtype=np.int64)
    frame_ms = np.zeros(frame_count, dtype=np.float64)
    batches = []

    queries = np.column_stack((np.zeros(query_count, dtype=np.float32), query_xy))
    queries = torch.from_numpy(queries).to(device)[None]
    # This call only sets the online query coordinates and empty model state.
    first_image = torch.from_numpy(frames[:1]).to(device).permute(0, 3, 1, 2)[None]
    predictor(first_image, is_first_step=True, queries=queries, add_support_grid=add_support_grid)
    torch.cuda.synchronize(device)

    previous_end = 0
    for batch_index, end in enumerate(end_frames):
        window_start = batch_index * predictor.step
        start = perf_counter()
        chunk = torch.from_numpy(frames[window_start:end]).to(device)
        chunk = chunk.permute(0, 3, 1, 2)[None].float()
        tracks, visibilities = predictor(chunk, add_support_grid=add_support_grid)
        new_xy = tracks[0, previous_end:end].cpu().numpy()
        new_visibility = visibilities[0, previous_end:end].cpu().numpy()
        torch.cuda.synchronize(device)
        elapsed = (perf_counter() - start) * 1000

        count = end - previous_end
        if new_xy.shape != (count, query_count, 2) or not np.isfinite(new_xy).all():
            raise RuntimeError(f"Invalid online prediction for frames {previous_end}:{end}")
        xy[previous_end:end] = new_xy
        visible[previous_end:end] = new_visibility
        emitted_at[previous_end:end] = end - 1
        frame_ms[previous_end:end] = elapsed / count
        batches.append(dict(window_start=window_start, end=end, new_frames=count, elapsed_ms=elapsed))
        previous_end = end
    return xy, visible, emitted_at, frame_ms, batches


def evaluate(config):
    """Record online CUDA inference from the exact saved TAPNext++ query points."""
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CoTracker3 comparison requires CUDA")
    torch.set_num_threads(2)
    source = Path(config["source"])
    receipt = source / "episode.json"
    episode = json.loads(receipt.read_text())
    checkpoint = Path(config["checkpoint"])
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != config["checkpoint_sha256"]:
        raise ValueError(f"Unexpected CoTracker3 checkpoint checksum: {checkpoint}")

    reference = Path(config["reference_run"])
    reference_summary = json.loads((reference / "summary.json").read_text())
    with np.load(reference / "tracks.npz", allow_pickle=False) as data:
        query_xy = data["query_xy"]
        supported_gt_xy = data["supported_gt_xy"]
        phases = data["phases"]
    frames = load_rgb_sequence(source, episode)
    if len(frames) != len(supported_gt_xy) or len(frames) != len(phases):
        raise ValueError("The reference run and source have different frame counts")

    with ExperimentLogger(team="visual", model="cotracker3_online", config=config,
                          seed=episode["seed"], job_type="evaluation") as logger:
        output = Path(config["output_root"]) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        logger.record(
            gpu=torch.cuda.get_device_name(device), torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda, episode_receipt_sha256=file_sha256(receipt),
            reference_tracks_sha256=file_sha256(reference / "tracks.npz"),
        )
        print(f"Output directory: {output}", flush=True)
        predictor = CoTrackerOnlinePredictor(checkpoint=checkpoint, v2=False).to(device).eval()
        xy, visible, availability, timings, batches = track(
            predictor, frames, query_xy, device, config["add_support_grid"]
        )
        np.savez_compressed(
            output / "tracks.npz", query_xy=query_xy, xy=xy, visible=visible,
            supported_gt_xy=supported_gt_xy, phases=phases,
            tracker_ms=timings, emitted_at_frame=availability,
        )
        summary = dict(
            frames=len(frames), identities=reference_summary["identities"],
            query_count=len(query_xy), window=2 * predictor.step, stride=predictor.step,
            batches=batches,
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        for path in sorted(output.iterdir()):
            logger.link_output(path)
        logger.log({"mean_amortized_ms": float(timings.mean())})
        print(f"Finished {len(frames)} frames in {len(batches)} windows", flush=True)
