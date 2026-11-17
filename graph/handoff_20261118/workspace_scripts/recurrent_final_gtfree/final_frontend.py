"""Instantiate the frozen GT-free frontend selected in the 3D comparison."""
import json
from pathlib import Path

import torch

import box_matcher
import match
from data_collection.episode_dataset import file_sha256
from evaluate_gtfree_ids import CameraOnlyFrontend
from graph.gated_mpnn import VertexEdgeGatedMPNN
from hybrid_frontend import KINDS, StreamingTracker
from tracker import CornerTransformer
import tracker as tracker_code
from visual.corner_detection import TorchCornerDetector

ROOT = Path('/workspace/gtfree_frontend_retrain')
FRONT = ROOT / 'finetuned'


class FinalFrontend(CameraOnlyFrontend):
    def __init__(self, temporal_path, gnn_path, device):
        tracker = torch.load(temporal_path, map_location='cpu', weights_only=True)
        gnn = torch.load(gnn_path, map_location='cpu', weights_only=True)
        dataset = json.loads((temporal_path.parents[1] / 'dataset.json').read_text())
        self.code_hashes = dict(
            frontend=file_sha256(__file__), matcher=file_sha256(match.__file__),
            box_matcher=file_sha256(box_matcher.__file__))
        assert dataset['matcher_sha256'] == self.code_hashes['matcher'] == gnn['config']['matcher_sha256']
        assert dataset['box_matcher_sha256'] == self.code_hashes['box_matcher']
        assert tracker['config']['tracker_sha256'] == file_sha256(tracker_code.__file__)
        assert gnn['config']['selected_tracker_sha256'] == file_sha256(temporal_path)
        self.device = device
        temporal = CornerTransformer(spatial=True).to(device).eval().requires_grad_(False)
        temporal.load_state_dict(tracker['model'], strict=True)
        self.tracker = StreamingTracker(temporal, tracker['fallback'].numpy(), device)
        self.gnn = VertexEdgeGatedMPNN(**gnn['config']['gnn']).to(device).eval().requires_grad_(False)
        self.gnn.load_state_dict(gnn['model'], strict=True)
        self.detector = TorchCornerDetector.main(device=device, plane_fit_method='ransac')
        self.kinds = torch.as_tensor(KINDS[None], device=device)
        self.valid = torch.ones_like(self.kinds, dtype=torch.bool)
