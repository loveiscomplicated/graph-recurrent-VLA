"""Run the 954-epoch fixed-ID pipeline causally during a closed-loop rollout."""
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from graph.gated_mpnn import VertexEdgeGatedMPNN
from graph.temporal.causal_transformer import sinusoidal_positions
from graph.temporal.detected_corner_tracking import match_positions
from rigid_box import RigidBoxTranslation
from tracker import CornerTransformer
from training.detected_corner_gnn import graph_batch
from visual.corner_detection import TorchCornerDetector
from visual.corner_detection.evaluation import visible_reference

KINDS = np.array([0] * 8 + [1] * 12, dtype=np.int64)


def identity(row):
    return {'peg': 0, 'box_outer': 8, 'hole_rim': 16}[row['kind']] + row['corner_id']


class StreamingTracker:
    def __init__(self, model, fallback, device):
        self.model = model
        self.fallback = fallback
        self.device = device
        self.reset()

    def reset(self):
        self.frame = 0
        self.last_seen = np.full(20, -1, dtype=np.int64)
        self.anchor = self.fallback.copy()
        width = self.model.ids.embedding_dim
        self.history = [torch.empty((20, 0, width), device=self.device) for _ in self.model.temporal]

    @torch.inference_mode()
    def step(self, detected, observed, tcp):
        frame = self.frame
        self.last_seen[observed] = frame
        self.anchor[observed] = detected[observed]
        features = np.concatenate(((self.anchor - self.fallback) / .1, (tcp - self.anchor) / .1,
                                   observed[:, None], (self.last_seen >= 0)[:, None],
                                   (np.minimum(frame - self.last_seen, 40) / 40)[:, None],
                                   np.full((20, 1), frame / 200)), axis=-1).astype(np.float32)
        model = self.model
        hidden = model.input(torch.as_tensor(features, device=self.device)) + model.ids.weight
        hidden = hidden + sinusoidal_positions(1, hidden.shape[-1], offset=frame,
                                              device=self.device, dtype=hidden.dtype)
        hidden, self.history[0] = model.temporal[0].step(hidden[:, None], self.history[0])
        hidden = hidden.transpose(0, 1)
        norm = model.spatial_norm(hidden)
        update, _ = model.spatial(norm, norm, norm, attn_mask=model.spatial_mask, need_weights=False)
        hidden = (hidden + update).transpose(0, 1)
        hidden, self.history[1] = model.temporal[1].step(hidden, self.history[1])
        coords = self.anchor + model.output(hidden[:, 0]).float().cpu().numpy() * .1
        coords[observed] = detected[observed]
        self.frame += 1
        return coords


class HybridFrontend:
    def __init__(self, temporal_path, gnn_path, device):
        self.device = device
        tracker = torch.load(temporal_path, map_location='cpu', weights_only=True)
        assert tracker['epoch'] == 954
        assert tracker['config']['tracker_sha256'] == file_sha256(Path(__import__('tracker').__file__))
        temporal = CornerTransformer(spatial=True).to(device).eval().requires_grad_(False)
        temporal.load_state_dict(tracker['model'], strict=True)
        self.tracker = StreamingTracker(temporal, tracker['fallback'].numpy(), device)
        checkpoint = torch.load(gnn_path, map_location='cpu', weights_only=True)
        assert checkpoint['epoch'] == 54
        self.gnn = VertexEdgeGatedMPNN(**checkpoint['config']['gnn']).to(device).eval().requires_grad_(False)
        self.gnn.load_state_dict(checkpoint['model'], strict=True)
        self.detector = TorchCornerDetector.main(device=device, plane_fit_method='ransac')
        self.kinds = torch.as_tensor(KINDS[None], device=device)
        self.valid = torch.ones_like(self.kinds, dtype=torch.bool)

    def reset(self, actor_ids):
        self.tracker.reset()
        self.rigid = RigidBoxTranslation(use_velocity=True)
        self.actor_ids = actor_ids

    def detect(self, obs, gt_match_only):
        images = {key: val[0].cpu().numpy() for key, val in obs['sensor_data']['base_camera'].items()}
        camera = {key: val[0].cpu().numpy() for key, val in obs['sensor_param']['base_camera'].items()}
        raw = self.detector(images['rgb'], images['depth'], camera['intrinsic_cv'], camera['extrinsic_cv'])
        xyz, kind = raw.xyz.cpu().numpy(), raw.object_id.cpu().numpy()
        visible = visible_reference(gt_match_only, {**images, **camera}, self.actor_ids)
        gt_xyz = np.asarray([r['xyz'] for r in visible]).reshape(-1, 3)
        gt_kind = np.asarray([r['object_id'] for r in visible], dtype=np.int64)
        coordinates = np.zeros((20, 3), dtype=np.float32)
        seen = np.zeros(20, dtype=bool)
        for i, j in match_positions(xyz, kind, gt_xyz, gt_kind, .005):
            slot = identity(visible[j])
            if slot < 20:
                seen[slot], coordinates[slot] = True, xyz[i]
        return coordinates, seen

    @torch.inference_mode()
    def step(self, obs, tcp, gt_match_only):
        start = perf_counter()
        detected, observed = self.detect(obs, gt_match_only)
        after_detect = perf_counter()
        filled = self.tracker.step(detected, observed, tcp)
        exterior = set(np.flatnonzero(observed[8:16]) + 8)
        centers, fit = self.rigid.update(filled, observed, KINDS, exterior)
        np.testing.assert_array_equal(centers[observed], detected[observed])
        after_track = perf_counter()
        graph, flags = graph_batch(torch.as_tensor(centers[None], device=self.device), self.kinds,
                                   self.valid, torch.as_tensor(observed[None], device=self.device),
                                   torch.as_tensor(tcp[None], device=self.device))
        object21 = self.gnn(graph, flags)['object21'][0].cpu().numpy()
        after_gnn = perf_counter()
        counts = (int(observed[:8].sum()), int(observed[8:].sum()))
        elapsed = (after_detect - start, after_track - after_detect, after_gnn - after_track)
        return object21, counts, elapsed, fit['support']
