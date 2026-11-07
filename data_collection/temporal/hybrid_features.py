"""Replay frozen hybrid branches over complete episodes."""

import numpy as np
import torch

from data_collection.temporal.insertion_sequences import InsertionEpisodeDataset
from data_collection.temporal.pose_sequences import stream_arguments
from graph.temporal.hybrid import StreamingHybridEstimator


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
