"""Replay frozen hybrid branches and cache causal Peg-head training inputs."""

import numpy as np
import torch

from data_collection.temporal.insertion_sequences import InsertionEpisodeDataset
from data_collection.temporal.pose_sequences import stream_arguments
from graph.temporal.hybrid import StreamingHybridEstimator, world_object21


@torch.inference_mode()
def replay_hybrid(mamba, gate, data, device):
    """Replay shared frames once, placing each output back by validated row identity."""
    dataset = InsertionEpisodeDataset(data)
    estimator = StreamingHybridEstimator(mamba, gate)
    predictions = {}
    alpha = np.empty(len(data["frame"]), dtype=np.float32)
    predicted_hole_frame = np.empty((len(data["frame"]), 3), dtype=np.float32)
    for episode, rows in zip(dataset.episodes, dataset.indices):
        estimator.reset(str(episode))
        for row in rows:
            observation = stream_arguments(data, row, device)
            observation["quality17"] = torch.as_tensor(
                data["features"][row : row + 1], device=device
            )
            result = estimator.step(
                observation, episode_id=str(episode), frame_index=int(data["frame"][row])
            )
            for name, value in result["predictions"].items():
                if name not in predictions:
                    predictions[name] = np.empty_like(data["prediction"])
                predictions[name][row] = value[0].cpu().numpy()
            alpha[row] = float(result["gate_alpha"])
            predicted_hole_frame[row] = result["relative_hole_frame"][0].cpu().numpy()
        estimator.discard(str(episode))
    return predictions, alpha, predicted_hole_frame


@torch.no_grad()
def cache_frozen_features(mamba, gate, data, device):
    """Capture Peg-head inputs from actual full-episode streaming, without GT inputs.

    No reset-window approximation is used. This deliberately differs from the
    original pretrained reset-at-64-frame training, matching H1 inference instead.
    The head is pointwise, so shuffled valid frames retain the same causal context.
    There is no padded frame to supervise or a backbone gradient to retain.
    """
    dataset = InsertionEpisodeDataset(data)
    captured = []

    def capture_head_input(module, arguments):
        captured.append(arguments[0].detach().cpu().clone())
        if len(captured) % 5000 == 0:
            print(f"Frozen H1 streaming: {len(captured)}/{len(data['frame'])} frames", flush=True)

    hook = mamba.heads["peg"].register_forward_pre_hook(capture_head_input)
    try:
        predictions, _, _ = replay_hybrid(mamba, gate, data, device)
    finally:
        hook.remove()
    representation = torch.cat(captured)
    if len(representation) != len(data["frame"]):
        raise AssertionError("Streaming head activations and input frame identities differ")
    # replay_hybrid validates sorted identities and resets/discards every episode.
    assert len(dataset) > 0
    tcp = torch.from_numpy(data["tcp_world"])
    cache = {
        "representation": representation,
        "original_h1": torch.from_numpy(predictions["hybrid_h1_a"].copy()),
    }
    for name, array in (
        ("frozen_world", data["prediction"]),
        ("mamba_world", predictions["mamba"]),
        ("gate_world", predictions["learned_gate"]),
        ("target", data["target"]),
    ):
        cache[name] = world_object21(torch.from_numpy(array), tcp)
    return cache, predictions["hybrid_h1_a"]


@torch.no_grad()
def predict_cached(adapter, cache, data, device, batch_size):
    """Decode all valid cached frames and restore the scorer's TCP-relative frame."""
    outputs = []
    adapter.eval()
    for start in range(0, len(data["frame"]), batch_size):
        inputs = [
            cache[name][start : start + batch_size].to(device)
            for name in ("representation", "frozen_world", "mamba_world", "gate_world")
        ]
        outputs.append(adapter(*inputs).cpu())
    world = torch.cat(outputs)
    world[:, :3] -= torch.from_numpy(data["tcp_world"])
    world[:, 10:13] -= torch.from_numpy(data["tcp_world"])
    return world.numpy()
