"""Instantiate the previously evaluated frontend with full-insertion checkpoints."""
from pathlib import Path

import torch

from data_collection.episode_dataset import file_sha256
from graph.gated_mpnn import VertexEdgeGatedMPNN
from hybrid_frontend import HybridFrontend, KINDS, StreamingTracker
from tracker import CornerTransformer
from visual.corner_detection import TorchCornerDetector

ROOT = Path('/workspace/full_insertion_frontend')
TRACKER = ROOT/'trained_until_plateau/tracker/best.pt'
GNN = ROOT/'trained_until_plateau/gnn/best.pt'


class FullInsertionFrontend(HybridFrontend):
    def __init__(self, temporal_path, gnn_path, device):
        self.device=device
        tracker=torch.load(temporal_path,map_location='cpu',weights_only=True)
        gnn=torch.load(gnn_path,map_location='cpu',weights_only=True)
        assert tracker['epoch']==430 and gnn['epoch']==28
        assert tracker['config']['tracker_sha256']==file_sha256(Path(__import__('tracker').__file__))
        assert gnn['config']['tracker_sha256']==file_sha256(temporal_path)
        model=CornerTransformer(spatial=True).to(device).eval().requires_grad_(False)
        model.load_state_dict(tracker['model'],strict=True)
        self.tracker=StreamingTracker(model,tracker['fallback'].numpy(),device)
        self.gnn=VertexEdgeGatedMPNN(**gnn['config']['gnn']).to(device).eval().requires_grad_(False)
        self.gnn.load_state_dict(gnn['model'],strict=True)
        self.detector=TorchCornerDetector.main(device=device,plane_fit_method='ransac')
        self.kinds=torch.as_tensor(KINDS[None],device=device)
        self.valid=torch.ones_like(self.kinds,dtype=torch.bool)
