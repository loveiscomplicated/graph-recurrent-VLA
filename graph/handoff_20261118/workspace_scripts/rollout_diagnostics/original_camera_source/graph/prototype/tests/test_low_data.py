"""Low-data split, pairing and custom-dataset evaluation contracts; no simulator."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from graph.prototype.common import config, seed_all
from graph.prototype.data import PHASES, sha256, split_indices
from graph.prototype.low_data import subset, summarize
from graph.prototype.models import parameter_count
from graph.prototype.normalization import FeatureNormalizer
from graph.prototype.tests.fixtures import geometry
from graph.prototype.v2.analyze import offline
from graph.prototype.v2.models import CommonPolicy
from graph.prototype.v2.state import StateNormalizer


def source():
    n = 18
    return dict(geometry=geometry(n).as_dict(), actions=torch.linspace(-.5, .5, n * 7).reshape(n, 7),
                episode_id=torch.arange(9).repeat_interleave(2), timestep=torch.tensor([0, 1] * 9),
                phase=torch.zeros(n, dtype=torch.long),
                manifest=dict(episodes=[dict(episode_id=i, episode_seed=i + 100) for i in range(9)],
                              splits=dict(train=list(range(5)), validation=[5, 6], test=[7, 8]),
                              phase_counts={name: n if i == 0 else 0 for i, name in enumerate(PHASES)},
                              config=config(), samples=n))


class SubsetTests(unittest.TestCase):
    def test_fixed_selection_preserves_alignment_and_all_heldout_values(self):
        data = source()
        reduced = subset(data, 2, 12)
        self.assertEqual(reduced['manifest']['splits'], subset(data, 2, 12)['manifest']['splits'])
        self.assertEqual(len(reduced['manifest']['splits']['train']), 2)
        self.assertEqual(len(data['manifest']['splits']['train']), 5)
        for split in ('validation', 'test'):
            self.assertEqual(data['manifest']['splits'][split], reduced['manifest']['splits'][split])
            a, b = split_indices(data, split), split_indices(reduced, split)
            for field in ('actions', 'episode_id', 'timestep', 'phase'):
                self.assertTrue(torch.equal(data[field][a], reduced[field][b]))
            for field in data['geometry']:
                self.assertTrue(torch.equal(data['geometry'][field][a], reduced['geometry'][field][b]))
        for eid in reduced['manifest']['splits']['train']:
            a, b = data['episode_id'] == eid, reduced['episode_id'] == eid
            self.assertTrue(torch.equal(data['actions'][a], reduced['actions'][b]))
            self.assertEqual(reduced['timestep'][b].tolist(), [0, 1])

    def test_selection_ignores_heldout_values_and_normalizers_use_selected_train(self):
        data = source()
        changed = copy.deepcopy(data)
        held = torch.cat([split_indices(changed, s) for s in ('validation', 'test')])
        for value in changed['geometry'].values():
            value[held] = 10000
        changed['actions'][held] = -10000
        a, b = subset(data, 2, 8), subset(changed, 2, 8)
        self.assertEqual(a['manifest']['splits'], b['manifest']['splits'])
        from graph.prototype.geometry import GeometryState
        stats = []
        for reduced in (a, b):
            ids = split_indices(reduced, 'train')
            stats.append(FeatureNormalizer.fit(GeometryState(**reduced['geometry']).select(ids)).as_dict())
        for key in stats[0]:
            self.assertTrue(torch.equal(stats[0][key], stats[1][key]))

    def test_same_reset_group_is_retained_whole_and_cross_split_groups_rejected(self):
        data = source()
        data['manifest']['episodes'][1]['episode_seed'] = 100
        # Seed 11 shuffles the group shared by episodes 0/1 first.
        group_seed = next(s for s in range(50) if subset(data, 1, s)['manifest']['splits']['train'] == [0, 1])
        reduced = subset(data, 1, group_seed)
        self.assertEqual(reduced['manifest']['low_data']['actual_train_episodes'], 2)
        data['manifest']['episodes'][7]['episode_seed'] = 100
        with self.assertRaisesRegex(ValueError, 'reset group'):
            subset(data, 1)

    def test_invalid_budget_and_split_overlap_are_rejected(self):
        data = source()
        for count in (0, 6):
            with self.assertRaises(ValueError):
                subset(data, count)
        data['manifest']['splits']['test'][0] = 0
        with self.assertRaises(ValueError):
            subset(data, 2)


class ComparisonTests(unittest.TestCase):
    def test_seed_pairs_and_missing_or_duplicate_runs(self):
        rows = []
        for seed in (0, 1, 2):
            for model, factor in [('flat', 1.), ('surface', .8)]:
                rows.append(dict(model=model, seed=seed, action_mse=(seed + 1) * factor,
                                 translation_rmse=.1, rotation_rmse=.2, gripper_mae=.3, latency_ms=1.))
        result = summarize(list(reversed(rows)), [0, 1, 2])
        self.assertEqual(result['surface_lower_seeds'], 3)
        self.assertAlmostEqual(result['mean_mse_reduction_percent'], 20)
        for invalid in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                summarize(invalid, [0, 1, 2])

    def test_custom_dataset_identity_episode_metrics_and_actual_seed(self):
        seed_all(7, 1)
        data = subset(source(), 2)
        from graph.prototype.geometry import GeometryState
        g = GeometryState(**data['geometry'])
        qvel = torch.zeros(len(data['actions']), 9)
        cfg = dict(config(), seed=7, flat_width=192, threads=1)
        ids = split_indices(data, 'train')
        norm = FeatureNormalizer.fit(g.select(ids))
        sn = StateNormalizer.fit(g.select(ids), qvel[ids], 'goal')
        model = CommonPolicy('flat', cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, checkpoint, output = root / 'small.pt', root / 'best.pt', root / 'eval.json'
            torch.save(data, dataset)
            ck = dict(schema_version=2, model_name='flat', config=cfg, state_mode='goal', gripper_mode='regression',
                      model_state=model.state_dict(), normalizer=norm.as_dict(), state_normalizer=sn.as_dict(),
                      dataset_sha256=sha256(dataset), splits=data['manifest']['splits'],
                      parameters=parameter_count(model), epoch=1)
            torch.save(ck, checkpoint)
            with patch.dict(os.environ, WANDB_MODE='disabled', EXPERIMENTS_DIR=str(root / 'experiments')):
                report = offline(checkpoint, data, g, qvel, dataset, per_episode=True, output=output)
            self.assertEqual(report['seed'], 7)
            self.assertEqual(report['dataset_sha256'], sha256(dataset))
            self.assertEqual(report['checkpoint_sha256'], sha256(checkpoint))
            self.assertEqual([r['episode_id'] for r in report['episode_offline']], [7, 8])
            expected = sum(r['samples'] * r['action_mse'] for r in report['episode_offline']) / 4
            self.assertAlmostEqual(expected, report['offline']['action_mse'], places=6)
            self.assertEqual(report['phase_offline']['insertion'], dict(samples=0))
            self.assertEqual(report, json.loads(output.read_text()))
            meta = next((root / 'experiments').glob('*/metadata.json'))
            self.assertEqual(json.loads(meta.read_text())['seed'], 7)
            changed = root / 'changed.pt'
            torch.save(dict(data, different_identity=True), changed)
            with self.assertRaisesRegex(ValueError, 'dataset/split mismatch'):
                offline(checkpoint, data, g, qvel, changed)


if __name__ == '__main__':
    unittest.main()
