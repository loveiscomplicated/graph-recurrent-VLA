"""Causal insertion estimator consuming individual frozen GNN node features."""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from graph.temporal.insertion import TemporalInsertionEstimator


class NodeTokenMamba(TemporalInsertionEstimator):
    """Process valid node tokens followed by one observation token per frame.

    Nodes must already be sorted deterministically within each frame. No padded
    node is passed to the mixer: otherwise it would alter the selective state.
    The frame token contains the original 64D observation and supplies the pose
    heads, keeping the original residual decoder and supervision unchanged.
    """

    def __init__(self, node_width, **kwargs):
        super().__init__(**kwargs)
        if self.architecture != 'mamba' or node_width < 1:
            raise ValueError('Node tokens require Mamba and a positive node width')
        self.node_projection = nn.Sequential(nn.Linear(node_width, self.d_model), nn.SiLU())

    def forward_nodes(self, observations, nodes, node_mask, frame_mask=None):
        """Return Object21 for each valid frame using packed causal tokens."""
        frame_mask = self._validate(observations, frame_mask)
        if nodes.ndim != 4 or nodes.shape[:2] != observations.shape[:2]:
            raise ValueError('Expected nodes[B,T,N,H] aligned with observations')
        if node_mask.dtype != torch.bool or node_mask.shape != nodes.shape[:3]:
            raise ValueError('Expected boolean node_mask[B,T,N]')
        if nodes.shape[-1] != self.node_projection[0].in_features:
            raise ValueError('Node feature width differs from the configured GNN')
        if not torch.isfinite(nodes).all():
            raise ValueError('Node features must be finite')
        frame_tokens = self.projection((observations - self.input_mean) / self.input_std)
        node_tokens = self.node_projection(nodes)
        sequences, positions = [], []
        for batch in range(len(observations)):
            tokens, frame_positions = [], []
            offset = 0
            for frame in range(int(frame_mask[batch].sum())):
                valid_nodes = node_tokens[batch, frame][node_mask[batch, frame]]
                tokens.append(valid_nodes)
                offset += len(valid_nodes)
                tokens.append(frame_tokens[batch, frame:frame + 1])
                frame_positions.append(offset)
                offset += 1
            sequences.append(torch.cat(tokens, dim=0))
            positions.append(frame_positions)
        representation = pad_sequence(sequences, batch_first=True)
        for norm, mixer in zip(self.norms, self.mixers):
            representation = representation + mixer(norm(representation))
        frame_representation = observations.new_zeros((*observations.shape[:2], self.d_model))
        for batch, indices in enumerate(positions):
            frame_representation[batch, :len(indices)] = representation[batch, indices]
        return self._decode(observations, frame_representation, frame_mask)
