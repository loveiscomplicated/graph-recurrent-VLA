"""Extract deterministic pre-pooling node tokens without changing GNN outputs."""

from contextlib import contextmanager

import numpy as np
import torch


@contextmanager
def capture_pre_pool_features(model, on_capture):
    """Observe the existing peg-pool input and always remove the temporary hook.

    The callback receives the original tensor. Frozen callers detach it, while
    joint training retains its autograd graph so temporal loss reaches the GNN.
    """
    def capture(module, arguments):
        on_capture(arguments[0])

    handle = model.peg_pool.register_forward_pre_hook(capture)
    try:
        yield
    finally:
        handle.remove()


def ordered_node_features(graph, features):
    """Return valid tokens per frame, ordered by object type then world X/Y/Z.

    Equal keys retain detector order. Coordinates supply ordering only; feature
    selection preserves gradients and excludes padded nodes, including in an
    entirely empty frame. Physical IDs and GT labels are never consulted.
    """
    if features.ndim != 3 or features.shape[:2] != graph.node_valid.shape:
        raise ValueError("Node features must have shape [B,N,H] matching graph.node_valid")

    coordinates = graph.xyz_world.detach().cpu().numpy()
    kinds = graph.object_type.detach().cpu().numpy()
    valid = graph.node_valid.detach().cpu().numpy()
    ordered = []
    for frame in range(len(features)):
        indices = np.flatnonzero(valid[frame])
        xyz = coordinates[frame, indices]
        order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0], kinds[frame, indices]))
        selected = torch.as_tensor(indices[order].copy(), device=features.device)
        ordered.append(features[frame].index_select(0, selected))
    return ordered
