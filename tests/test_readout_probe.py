"""Controlled readout parity, initialization and encoder isolation contracts."""

from copy import deepcopy
import tempfile
import unittest
from pathlib import Path

import torch

from tests.test_vertex_ablation import observed_graph
from training.gnn.data import TrainingBatch
from training.gnn.readout_probe import (
    COMPONENTS, ReadoutProbe, capture_features, check_parents, feature_batch,
    make_probe, state_digest, verify_pretrained_replay, train_probe, train,
)
from graph.gated_mpnn import VertexEdgeGatedMPNN


class ReadoutProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.source = VertexEdgeGatedMPNN(16, hidden=16, layers=1, use_confidence=True).requires_grad_(False).eval()
        graph = observed_graph()
        with torch.no_grad():
            target = self.source(graph)['object21']
        self.batch = TrainingBatch(graph, {'object21': target}, [f'episode_{i}/frame_0' for i in range(4)])

    def test_cached_features_replay_original_masks_anchors_and_outputs(self):
        before = state_digest(self.source)
        data = capture_features(self.source, [self.batch], 'cpu')
        self.assertLess(verify_pretrained_replay(self.source, data, 'cpu'), 1e-6)
        model = ReadoutProbe(self.source)
        model.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        output = model(feature_batch(data, slice(None), 'cpu'))
        loss = (output['object21'] - data['target'] + .01).square().mean()
        loss.backward()
        optimizer.step()
        self.assertTrue(all(parameter.grad is None for parameter in self.source.parameters()))
        self.assertEqual(state_digest(self.source), before)
        self.assertFalse(data['h'].requires_grad)
        self.assertFalse(any('node_encoder' in key for key in model.state_dict()))

    def test_new_heads_identical_for_each_readout_and_not_pretrained(self):
        config = dict(name='vertex_edge_gated_mpnn', num_semantic_ids=16, hidden=16, layers=1)
        attention = make_probe(config, 'attention', 7, torch.device('cpu'))
        multiseed = make_probe(config, 'multiseed', 7, torch.device('cpu'))
        for name in ('peg_head', 'hole_head', 'endpoint_residual'):
            for key, value in getattr(attention, name).state_dict().items():
                torch.testing.assert_close(value, getattr(multiseed, name).state_dict()[key], atol=0, rtol=0)
        self.assertFalse(torch.equal(attention.peg_head[0].weight, self.source.peg_head[0].weight))

    def test_parent_leakage_rejected(self):
        identity = {'splits': {'train': ['a'], 'validation': ['b'], 'test': ['c']}}
        manifest = {'parent_by_episode': {'a': 'same', 'b': 'same', 'c': 'other'}}
        with self.assertRaisesRegex(ValueError, 'Parent leakage'):
            check_parents(identity, manifest)

    def test_tiny_optimizer_workflow_saves_predictions_and_fixed_budget(self):
        data = capture_features(self.source, [self.batch], 'cpu')
        source_config = dict(model=dict(name='vertex_edge_gated_mpnn', num_semantic_ids=16, hidden=16, layers=1),
                             task={'settings': dict(pos_unit_m=.01, consistency_weight=.05,
                                                    size_units_m=[.04,.01,.01,.02])},
                             training=dict(epochs=1,batch_size=2,gradient_clip_norm=1.),
                             optimizer=dict(name='adamw',lr=.001,weight_decay=.0001),
                             scheduler=dict(name='step',step_size=10,gamma=.5))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root/'source.pt'
            torch.save({'model': self.source.state_dict()}, checkpoint)
            config = dict(device='cpu', checkpoint=str(checkpoint), feature_sha256='fixture', epochs=1)
            result = train_probe(config, source_config, {'train':data,'validation':data}, 'attention', 0, root)
            self.assertEqual(result['updates'], 2)
            self.assertTrue((Path(result['path'])/'predictions.npz').exists())

    def test_full_cache_entrypoint_and_reuse_checksum(self):
        import json
        from tests.test_temporal_hole_gru import write_smoke_fixture

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = write_smoke_fixture(root, pose_targets=True)
            config = dict(device='cpu', checkpoint=fixture['checkpoint'],
                          parent_manifest=fixture['split_manifest'], cache_dir=fixture['data']['cache_dir'],
                          output={'root': str(root/'probe')}, epochs=1, batch_size=2, seeds=[0],
                          variants=['attention'], capacity_control=False, seed_count_controls=False,
                          num_workers=0, extraction_batch_size=4, prepare_only=True)
            train(config)
            audit = json.loads((root/'probe/feature_audit.json').read_text())
            self.assertEqual(audit['pretrained_replay_max_difference'], 0)
            config.update(prepare_only=False,prepared_feature_sha256=audit['feature_sha256'])
            train(config)
            result = json.loads((root/'probe/completion.json').read_text())
            self.assertTrue(result['frozen_source_unchanged'])
            self.assertEqual(result['runs'][0]['optimizer_step_counts'], [5])
            with self.assertRaises(FileExistsError):
                train(config)
            config['prepared_feature_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                train(config)


if __name__ == '__main__':
    unittest.main()
