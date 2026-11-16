"""Measure GNN inference with Delaunay and complete edges on the same nodes."""

from dataclasses import replace
from itertools import combinations
from time import perf_counter_ns

import numpy as np
import torch

from graph.build_graph.surface_graph import collate_surface_graphs
from graph.gated_mpnn import VertexEdgeGatedMPNN


def distribution(values):
    """Summarize measurements across frames and repetitions."""
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def prepared_graph(frame, device):
    """Move the graph and features before the timed forward pass."""
    graph, features, _ = collate_surface_graphs([frame])
    return graph.to(device), features.to(device)


@torch.inference_mode()
def time_forwards(pairs, model, device, repeats):
    """Warm all graph shapes and alternate topology order for the timed forwards."""
    def synchronize():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)

    for pair in pairs:
        for graph in pair:
            model(*graph)
    synchronize()

    measurements = np.empty((repeats, len(pairs), 2))
    for repeat in range(repeats):
        for index, pair in enumerate(pairs):
            order = (0, 1) if (index + repeat) % 2 == 0 else (1, 0)
            for topology in order:
                start = perf_counter_ns()
                model(*pair[topology])
                synchronize()
                measurements[repeat, index, topology] = (perf_counter_ns() - start) / 1e6
    return measurements


def benchmark(frames, triangulation_ms, model_config, settings):
    """Compare edge counts and single-frame forward times with fixed model weights."""
    complete = [
        replace(frame, edges=np.asarray(list(combinations(range(len(frame.xyz)), 2)),
                                        dtype=np.int64).reshape(-1, 2))
        for frame in frames
    ]
    model = VertexEdgeGatedMPNN(**model_config).eval()
    result = {
        "frames": len(frames),
        "nodes_per_frame": distribution([len(frame.xyz) for frame in frames]),
        "overlapping_pixel_nodes": sum(len(frame.uv) - len(np.unique(np.rint(frame.uv), axis=0))
                                       for frame in frames),
        "delaunay_construction_ms": distribution(triangulation_ms),
        "delaunay": {},
        "complete": {},
    }
    for name, graphs in (("delaunay", frames), ("complete", complete)):
        edges = [len(frame.edges) for frame in graphs]
        result[name] = {
            "total_undirected_edges": sum(edges),
            "total_directed_edges": 2 * sum(edges),
            "undirected_edges_per_frame": distribution(edges),
            "forward_ms": {},
        }

    repeats = settings["repeats"]
    for name in settings["devices"]:
        device = torch.device(name)
        model.to(device)
        pairs = [(prepared_graph(sparse, device), prepared_graph(dense, device))
                 for sparse, dense in zip(frames, complete, strict=True)]
        measurements = time_forwards(pairs, model, device, repeats)
        for index, topology in enumerate(("delaunay", "complete")):
            result[topology]["forward_ms"][name] = distribution(measurements[:, :, index])
    result["repeats"] = repeats
    result["cpu_threads"] = torch.get_num_threads()
    return result
