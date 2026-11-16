"""Train B2-2 with loss on missing, previously observed corner IDs."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from graph.state_estimation.graph import Frame, Node, Source, batch_graphs, build_graph
from graph.state_estimation.spatial_corner_tracker import SpatialCornerTracker
from utils.experiment_logger import ExperimentLogger


def episode_windows(root, row, length):
    """Build one complete episode or non-overlapping blocks with shared anchors."""
    path = root / row["file"]
    if file_sha256(path) != row["sha256"]:
        raise ValueError(f"Invalid episode checksum: {path}")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    target = arrays["targets"]
    count, slots, _ = target.shape
    if count != row["frames"]:
        raise ValueError(f"Incorrect episode length: {path}")
    if length is None:
        length = count
    graphs, windows = [], []
    rows = np.full((count, slots), -1, dtype=np.int64)
    anchors = np.zeros_like(target)
    seen = np.zeros(slots, dtype=bool)
    missing = np.zeros((count, slots), dtype=bool)
    position = np.zeros((slots, 3), dtype=np.float32)
    for index in range(count):
        first, last = arrays["offsets"][index:index + 2]
        nodes = []
        for local, offset in enumerate(range(first, last)):
            xyz = arrays["xyz"][offset]
            vertex = int(arrays["vertex_ids"][offset])
            if vertex >= 0:
                rows[index, vertex] = local
                position[vertex] = xyz
                seen[vertex] = True
            is_corner = bool(arrays["is_corner"][offset])
            nodes.append(Node(tuple(xyz), int(arrays["kinds"][offset]), is_corner,
                              vertex if vertex >= 0 else None, is_corner, None,
                              Source.OBSERVED if is_corner else Source.TRACKER))
        anchors[index] = position
        missing[index] = seen & (rows[index] < 0)
        frame = Frame(row["episode"], index, tuple(arrays["tcp"][index]), tuple(nodes))
        graphs.append(build_graph(frame))
    for start in range(0, count, length):
        end = min(start + length, count)
        if missing[start:end].any():
            windows.append((graphs[start:end], rows[start:end], anchors[start:end],
                            target[start:end], missing[start:end]))
    return windows


def collate(windows, length, device):
    """Pad only the ends of episodes and collect their spatial graphs."""
    slots = windows[0][1].shape[-1]
    shape = (len(windows), length, slots)
    rows = np.full(shape, -1, dtype=np.int64)
    anchors, targets = (np.zeros((*shape, 3), dtype=np.float32) for _ in range(2))
    mask = np.zeros(shape, dtype=bool)
    graphs = []
    for batch, (frames, indices, last, target, missing) in enumerate(windows):
        size = len(frames)
        graphs.extend(frames + [frames[-1]] * (length - size))
        rows[batch, :size] = indices
        anchors[batch, :size] = last
        targets[batch, :size] = target
        mask[batch, :size] = missing
    return batch_graphs(graphs).to(device), tuple(torch.from_numpy(array).to(device) for array in (
        rows, anchors, targets, mask,
    ))


def epoch_metrics(model, windows, settings, device, optimizer=None):
    """Report target-weighted XYZ loss and Euclidean RMSE against last observation."""
    model.train(optimizer is not None)
    total_error = baseline_error = 0.0
    count = 0
    batch_size = settings["batch_size"]
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            length = settings["sequence_length"]
            if length is None:
                length = max(len(episode[0]) for episode in batch)
            graph, (rows, anchors, targets, mask) = collate(
                batch, length, device
            )
            pred = model(graph, rows, rows >= 0, anchors)
            error = (pred[mask] - targets[mask]).square()
            loss = error.mean() / settings["position_scale_m"] ** 2
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite B2-2 loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            total_error += float(error.detach().sum())
            baseline_error += float((anchors[mask] - targets[mask]).square().sum())
            count += len(error)
    if not count:
        raise ValueError("No previously observed, currently missing vertex targets")
    return {"loss": total_error / count / 3 / settings["position_scale_m"] ** 2,
            "missing_rmse_mm": 1000 * (total_error / count) ** 0.5,
            "hold_rmse_mm": 1000 * (baseline_error / count) ** 0.5, "missing_targets": count}


def run(config):
    """Train and select a checkpoint by validation missing-corner loss."""
    seed = config["seed"]
    torch.manual_seed(seed)
    torch.set_num_threads(config["cpu_threads"])
    device = torch.device(config["device"])
    settings = config["training"]
    length = settings["sequence_length"]
    if length is not None and (type(length) is not int or length < 1):
        raise ValueError("training.sequence_length must be null or a positive integer")
    manifest = Path(config["data"]["manifest"])
    rows = json.loads(manifest.read_text())["episodes"]
    split_seeds = {split: {row["seed"] for row in rows if row["split"] == split}
                   for split in ("train", "validation")}
    if split_seeds["train"] & split_seeds["validation"]:
        raise ValueError("Train and validation reset seeds overlap")
    with ExperimentLogger(team="graph", model="b2_2", seed=seed, config=config, job_type="train") as logger:
        directory = Path(config["output_root"]) / logger.run_id
        directory.mkdir(parents=True)
        logger.record(dataset_sha256=file_sha256(manifest), device=str(device))
        windows = {"train": [], "validation": []}
        for number, row in enumerate(rows, 1):
            windows[row["split"]].extend(episode_windows(manifest.parent, row, length))
            if number % 50 == 0:
                print(f"Prepared {number}/{len(rows)} episodes", flush=True)
        model = SpatialCornerTracker(**config["model"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])
        rng = np.random.default_rng(seed)
        history = []
        best = float("inf")
        for epoch in range(1, settings["epochs"] + 1):
            train = windows["train"]
            shuffled = [train[i] for i in rng.permutation(len(train))]
            metrics = {"train": epoch_metrics(model, shuffled, settings, device, optimizer),
                       "validation": epoch_metrics(model, windows["validation"], settings, device)}
            history.append({"epoch": epoch, **metrics})
            logger.log(metrics, step=epoch)
            payload = {"epoch": epoch, "config": config, "model": model.state_dict(),
                       "optimizer": optimizer.state_dict(), "history": history}
            if metrics["validation"]["loss"] < best:
                best = metrics["validation"]["loss"]
                torch.save(payload, directory / "best.pt")
            torch.save(payload, directory / "last.pt")
            write_json(directory / "history.json", history)
            print(f"Epoch {epoch}/{settings['epochs']}: {metrics}", flush=True)
        for name in ("best.pt", "last.pt", "history.json"):
            logger.link_output(directory / name)
