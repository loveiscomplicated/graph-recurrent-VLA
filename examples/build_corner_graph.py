"""Smoke-check native RGB-D -> corners -> GraphBatch -> an untrained GNN."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from graph.basic_gnn import BasicGNN
from graph.build_graph import GraphBuilder, GraphSettings
from graph.gated_mpnn import VertexEdgeGatedMPNN
from visual.corner_detection import detect_tensors
from utils.device_set import set_device


def main():
    """Use a saved frame and explicit TCP position; no simulator or GT is read."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='NPZ with rgb, depth, intrinsic_cv, extrinsic_cv, optional masks')
    parser.add_argument('--tcp-xyz', type=float, nargs=3, required=True,
                        help='Observed TCP world position in metres')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model', choices=('basic', 'gated'), default='gated')
    parser.add_argument('--intra-k', type=int, default=None)
    parser.add_argument('--cross-k', type=int, default=None)
    parser.add_argument('--use-confidence', action='store_true')
    args = parser.parse_args()
    device = set_device(args.device)
    settings = GraphSettings(intra_k=args.intra_k, cross_k=args.cross_k)
    with np.load(args.input, allow_pickle=False) as frame:
        result = detect_tensors(
            frame['rgb'], frame['depth'], frame['intrinsic_cv'], frame['extrinsic_cv'],
            masks=frame['masks'] if 'masks' in frame else None, device=device,
        )
    tcp = torch.tensor([args.tcp_xyz], dtype=torch.float32, device=device)
    graph = GraphBuilder(settings)([result], tcp)
    if args.model == 'gated':
        model = VertexEdgeGatedMPNN(num_semantic_ids=1, use_confidence=args.use_confidence)
    else:
        model = BasicGNN(2, 8, 64, 16, 8, use_confidence=args.use_confidence)
    model = model.to(device).eval()
    with torch.no_grad():
        output = model(graph)
    shape = output['object21'].shape if isinstance(output, dict) else output.shape
    print(f'Device: {device}; nodes: {graph.node_valid.sum().item()}; edges: {graph.edge_index.shape[1]}')
    print(f'{type(model).__name__} output shape: {tuple(shape)}')
    print('Untrained model smoke check only; no training or accuracy evaluation.')


if __name__ == '__main__':
    main()
