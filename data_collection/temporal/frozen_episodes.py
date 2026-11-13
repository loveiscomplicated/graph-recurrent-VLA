"""Aligned frozen-frame observations and full episode sequences, without GT inputs."""

import numpy as np
import torch
from torch.utils.data import Dataset

from evaluation.gnn.temporal_memory import sample_order
from graph.temporal.observations import observation_features, temporal_inputs


def infer_observations(model, loader, device, include_pose_quality=False):
    """Capture frozen outputs/observations; keep labels separate from rollout inputs."""
    arrays = {name: [] for name in ("prediction", "target", "tcp_world", "features", "quality")}
    if include_pose_quality:
        arrays["pose_quality"] = []
    identifiers = []
    with torch.no_grad():
        for batch in loader:
            graph = batch.graph
            count = len(batch.sample_ids)
            if graph.tcp_xyz_world.shape != (count, 3) or batch.targets["object21"].shape != (
                count,
                21,
            ):
                raise ValueError("Frozen graph, target and sample-ID batch sizes must agree")
            features = observation_features(graph)
            box_count, hole_count = features[:, 0], features[:, 8]
            known = box_count * features[:, 3] + hole_count * features[:, 11]
            confidence_sum = (
                box_count * features[:, 3] * features[:, 5]
                + hole_count * features[:, 11] * features[:, 13]
            )
            count = box_count + hole_count
            quality = torch.stack(
                (count, confidence_sum / known.clamp_min(1), known / count.clamp_min(1)), dim=1
            )
            values = [
                model(graph.to(device))["object21"].cpu(),
                batch.targets["object21"],
                graph.tcp_xyz_world,
                features,
                quality,
            ]
            if include_pose_quality:
                values.append(observation_features(graph, include_peg=True))
            identifiers.extend(batch.sample_ids)
            for name, value in zip(arrays, values):
                arrays[name].append(value.numpy())
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Selected frames must be nonempty and unique")
    order = sorted(range(len(identifiers)), key=lambda i: sample_order(identifiers[i]))
    data = {name: np.concatenate(values)[order] for name, values in arrays.items()}
    if not all(np.isfinite(value).all() for value in data.values()):
        raise ValueError("Nonfinite frozen predictions, observations or targets")
    data["sample_ids"] = np.asarray(identifiers)[order]
    data["episode"] = np.asarray([sample_order(value)[0] for value in data["sample_ids"]])
    data["frame"] = np.asarray([sample_order(value)[1] for value in data["sample_ids"]])
    for episode in np.unique(data["episode"]):
        frames = data["frame"][data["episode"] == episode]
        if np.any(np.diff(frames) != 1):
            raise ValueError(f"Temporal episodes require contiguous cached frames: {episode}")
    return data


def episode_tensors(data, episode, device):
    """Transfer one complete episode; labels remain outside the gate input tuple."""
    mask = data["episode"] == episode
    inputs = tuple(
        torch.as_tensor(data[name][mask], device=device, dtype=torch.float32)
        for name in ("prediction", "tcp_world", "features")
    )
    target_world = torch.as_tensor(
        data["target"][mask, 10:13] + data["tcp_world"][mask], device=device, dtype=torch.float32
    )
    return inputs, target_world


class FrozenEpisodeDataset(Dataset):
    """Validate row alignment and expose complete causal episodes, including frame 0.

    Frozen inference sorts all arrays by the same sample-ID permutation. Cache
    loading verifies each graph/target pair against its published sample identity
    and byte checksum before it reaches this class. Array metadata is checked
    again here to reject shifted, duplicate, truncated or interleaved sequences.
    """

    def __init__(self, data):
        count = len(data["sample_ids"])
        if count == 0 or len(set(data["sample_ids"])) != count:
            raise ValueError("Episode data needs nonempty, unique sample IDs")
        shapes = {
            "prediction": (count, 21),
            "target": (count, 21),
            "tcp_world": (count, 3),
            "features": (count, 17),
            "quality": (count, 3),
            "episode": (count,),
            "frame": (count,),
        }
        for name, expected in shapes.items():
            if data[name].shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            if name != "episode" and not np.isfinite(data[name]).all():
                raise ValueError(f"{name} contains nonfinite values")
        orders = [sample_order(value) for value in data["sample_ids"]]
        if orders != sorted(orders) or len(set(orders)) != count:
            raise ValueError("Episode frames must be unique and sorted by episode/time")
        if any(
            episode != data["episode"][index] or frame != data["frame"][index]
            for index, (episode, frame) in enumerate(orders)
        ):
            raise ValueError("sample_ids, episode and frame arrays are misaligned")
        self.data = data
        self.episodes = np.unique(data["episode"])
        self.indices = []
        for episode in self.episodes:
            indices = np.flatnonzero(data["episode"] == episode)
            if not np.array_equal(data["frame"][indices], np.arange(len(indices))):
                raise ValueError(f"Expected complete episode starting at frame 0: {episode}")
            self.indices.append(indices)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        """Return inputs[T,23], target_world[T,3]; never concatenate labels to inputs."""
        rows = self.indices[index]
        prediction, tcp, quality = (
            torch.as_tensor(self.data[name][rows], dtype=torch.float32)
            for name in ("prediction", "tcp_world", "features")
        )
        target = torch.as_tensor(self.data["target"][rows, 10:13], dtype=torch.float32) + tcp
        return temporal_inputs(prediction, tcp, quality), target
