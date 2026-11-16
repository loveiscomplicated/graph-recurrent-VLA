"""Prepare graph windows once and batch windows of equal history length."""

from collections import defaultdict
from dataclasses import dataclass

import torch

from data_collection.state21 import Sample
from graph.state_estimation.graph import Graph, batch_graphs, build_graph


@dataclass(frozen=True)
class Window:
    graphs: tuple[Graph, ...]
    sample: Sample
    previous: Sample | None


class PreparedWindows:
    """Cache each graph once and index independent episode-local causal windows."""

    def __init__(self, episodes, history, max_edge_m):
        if type(history) is not int or history < 1:
            raise ValueError("history must be a positive number of saved frames")
        self.history = history
        self.windows = []
        self.graph_counts = defaultdict(int)
        self.rejected_edges = 0
        for episode in episodes:
            self.add_episode(
                episode,
                (
                    build_graph(sample.observation, max_edge_m=max_edge_m)
                    for sample in episode.samples
                ),
            )

    def add_episode(self, episode, frame_graphs):
        """Index prebuilt graphs using the same episode-local history as raw observations."""
        graphs = []
        for index, (sample, graph) in enumerate(
            zip(episode.samples, frame_graphs, strict=True)
        ):
            if (graph.episode_id, graph.frame_index) != (
                episode.episode_id,
                sample.observation.index,
            ):
                raise ValueError(
                    f"Graph and observation frame differ in {episode.episode_id}"
                )
            for method in graph.methods:
                self.graph_counts[method] += 1
            self.rejected_edges += graph.rejected_edges
            graphs.append(graph)
            start = max(0, len(graphs) - self.history)
            previous = episode.samples[index - 1] if index else None
            self.windows.append(Window(tuple(graphs[start:]), sample, previous))

    def batches(self, batch_size, device, generator=None):
        """Batch equal-length windows without padding time or crossing episodes."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        order = range(len(self.windows))
        if generator is not None:
            order = torch.randperm(len(self.windows), generator=generator).tolist()
        buckets = defaultdict(list)
        for index in order:
            window = self.windows[index]
            buckets[len(window.graphs)].append(window)
        chunks = [
            group[start : start + batch_size]
            for group in buckets.values()
            for start in range(0, len(group), batch_size)
        ]
        if generator is not None:
            indices = torch.randperm(len(chunks), generator=generator).tolist()
            chunks = [chunks[index] for index in indices]
        for windows in chunks:
            frames = [
                batch_graphs(graphs).to(device)
                for graphs in zip(*(item.graphs for item in windows))
            ]
            targets = torch.stack([item.sample.target for item in windows]).to(device)
            yield frames, targets, windows
