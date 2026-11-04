"""Replaceable graph encoders; the default is the existing live encoder."""
from .edge_mlp import GraphEncoder, MessageLayer, structured_readout

__all__ = ['GraphEncoder', 'MessageLayer', 'structured_readout']
