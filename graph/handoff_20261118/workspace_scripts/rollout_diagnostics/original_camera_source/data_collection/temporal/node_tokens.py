"""Align cached node tokens with insertion windows and right-pad their batches."""

import torch


def collate_node_windows(items):
    """Right-pad frames and nodes; node_mask excludes every artificial token."""
    observations, targets, nodes = zip(*items)
    lengths = torch.tensor([len(value) for value in observations], dtype=torch.long)
    frames = int(lengths.max())
    frame_mask = torch.arange(frames)[None] < lengths[:, None]
    node_count = max(1, max((len(node) for episode in nodes for node in episode), default=0))
    width = next(node.shape[1] for episode in nodes for node in episode)
    node_tensor = torch.zeros(len(items), frames, node_count, width)
    node_mask = torch.zeros(len(items), frames, node_count, dtype=torch.bool)
    for batch, episode in enumerate(nodes):
        for frame, node in enumerate(episode):
            count = len(node)
            node_tensor[batch, frame, :count] = torch.as_tensor(node)
            node_mask[batch, frame, :count] = True
    return (torch.nn.utils.rnn.pad_sequence(observations, batch_first=True),
            torch.nn.utils.rnn.pad_sequence(targets, batch_first=True),
            node_tensor, node_mask, frame_mask)


class NodeWindows:
    """Add aligned node lists to the existing nonoverlapping insertion windows."""

    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        observations, target = self.windows[index]
        episode, start, stop = self.windows.windows[index]
        rows = self.windows.episodes.indices[episode][start:stop]
        nodes = self.windows.episodes.data['node_features']
        return observations, target, [nodes[row] for row in rows]
