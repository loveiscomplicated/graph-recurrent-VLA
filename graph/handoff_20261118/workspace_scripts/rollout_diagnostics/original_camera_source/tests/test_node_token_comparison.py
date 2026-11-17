"""Check frame alignment and paired aggregation for node-token comparisons."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from evaluation.gnn.temporal.node_token_comparison import METRICS, compare


class NodeTokenComparisonTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ids = [f'episode/frame_{i:04d}' for i in range(960)]
        self.target = np.zeros((960, 21), dtype=np.float32)
        rows = [{'sample_id': value, 'both_pass': i < 151} for i, value in enumerate(self.ids)]
        selection = self.root / 'selection.jsonl'
        selection.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.config = {'evaluation_selection': str(selection), 'output_root': str(self.root / 'comparison'),
                       'methods': {name: [] for name in ('node_early', 'mamba_early', 'node_120')}}
        for seed in range(3):
            for name in self.config['methods']:
                root = self.root / name / str(seed)
                run = root / 'run'
                run.mkdir(parents=True)
                self.config['methods'][name].append(str(root))
                (run / 'effective_config.json').write_text(json.dumps({'seed': seed,
                    'source_checkpoint_sha256': f'gnn_{seed}'}))
                amount = {'node_early': 2.0, 'mamba_early': 1.0, 'node_120': 3.0}[name]
                errors = {key: np.full(960, amount, dtype=np.float32) for key in METRICS}
                if name == 'mamba_early':
                    np.savez(run / 'mamba_selected_errors.npz', **errors)
                    np.savez(run / 'predictions.npz', sample_ids=np.array(self.ids), target=self.target,
                             tcp_world=np.zeros((960, 3), dtype=np.float32))
                else:
                    np.savez(run / 'selected_errors.npz', **errors)
                    np.savez(run / 'selected_predictions.npz', sample_ids=np.array(self.ids), target=self.target)

    def test_paired_comparison(self):
        with patch.dict(os.environ, {'WANDB_MODE': 'disabled', 'EXPERIMENTS_DIR': str(self.root / 'experiments')}):
            output = compare(self.config)
        result = json.loads((output / 'comparison.json').read_text())
        group = result['groups']['physical_both_pass']
        self.assertEqual(group['count'], 151)
        self.assertEqual(group['paired_node_early_minus_mamba_early']['peg_position_error_mm']['mean'], 1.0)
        self.assertEqual(group['paired_node_early_minus_node_120']['peg_position_error_mm']['mean'], -1.0)

    def test_rejects_wrong_frozen_gnn(self):
        path = Path(self.config['methods']['node_early'][1]) / 'run' / 'effective_config.json'
        path.write_text(json.dumps({'seed': 1, 'source_checkpoint_sha256': 'wrong'}))
        with self.assertRaisesRegex(ValueError, 'GNN checkpoint differs'):
            compare(self.config)

    def test_rejects_changed_frame_order(self):
        path = Path(self.config['methods']['node_early'][2]) / 'run' / 'selected_predictions.npz'
        np.savez(path, sample_ids=np.array(self.ids[::-1]), target=self.target)
        with self.assertRaisesRegex(ValueError, 'frame order differs'):
            compare(self.config)


if __name__ == '__main__':
    unittest.main()
