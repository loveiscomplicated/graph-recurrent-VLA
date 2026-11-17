"""Deterministic progress checks without training or generating frame caches."""

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from training.gnn.progress import TrainingProgress
from training.gnn.tasks import StepResult
from main import load_config
from training.gnn.trainer import resolve_config, run_epoch, train


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.experiment = Mock()
        self.output = io.StringIO()
        self.stdout = redirect_stdout(self.output)
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def progress(self, train_loader=None):
        return TrainingProgress(
            self.experiment, 5, [0] * 4 if train_loader is None else train_loader,
            [0] * 2, interval_seconds=10, clock=lambda: self.now,
        )

    def latest(self):
        return self.experiment.log.call_args.args[0]

    def test_cadence_phase_rates_and_epoch_overhead(self):
        progress = self.progress()
        progress.begin_epoch(1)
        progress.begin_phase('train')
        self.assertNotIn('total_remaining_seconds', self.latest())
        self.now = 5
        progress.batch_completed()
        self.assertEqual(self.experiment.log.call_count, 1)
        self.now = 10
        progress.batch_completed()
        self.assertEqual(self.latest()['epoch_elapsed_seconds'], 10)
        self.assertEqual(self.latest()['epoch_remaining_seconds'], 20)
        self.assertEqual(self.latest()['total_remaining_seconds'], 140)
        self.now = 20
        progress.batch_completed()
        progress.batch_completed()
        progress.finish_phase()
        progress.begin_phase('validation')
        self.now = 22
        progress.batch_completed()
        progress.report(force=True)
        self.assertEqual(self.latest()['epoch_remaining_seconds'], 2)
        self.assertEqual(self.latest()['total_remaining_seconds'], 98)
        self.now = 24
        progress.batch_completed()
        progress.finish_phase()
        self.now = 30  # checkpoint writes take six seconds
        self.assertEqual(progress.finish_epoch(), 30)
        progress.begin_epoch(2)
        progress.begin_phase('train')
        self.assertEqual(self.latest()['epoch_remaining_seconds'], 24)
        self.assertEqual(self.latest()['total_remaining_seconds'], 114)
        self.assertIn('Epoch 2/5 [train]', self.output.getvalue())

    def test_resume_does_not_count_past_epochs_as_remaining(self):
        progress = self.progress()
        progress.begin_epoch(5)
        progress.begin_phase('train')
        self.now = 10
        progress.batch_completed()
        self.assertEqual(self.latest()['epoch_remaining_seconds'], 50)
        self.assertEqual(self.latest()['total_remaining_seconds'], 50)

    def test_unsized_iterables_keep_elapsed_logging_without_fabricated_eta(self):
        progress = self.progress(iter([0, 0]))
        progress.begin_epoch(1)
        progress.begin_phase('train')
        self.now = 10
        progress.batch_completed()
        self.assertEqual(self.latest()['epoch_elapsed_seconds'], 10)
        self.assertNotIn('epoch_remaining_seconds', self.latest())
        self.assertIn('batch 1/?', self.output.getvalue())

    def test_run_epoch_counts_filtered_batches_without_changing_metrics(self):
        progress = Mock()
        batch = SimpleNamespace(to=lambda device: None)
        task = Mock()
        task.compute.return_value = StepResult(torch.tensor(2.), {'loss': 2.}, {'loss': 1})
        metrics = run_epoch(torch.nn.Linear(1, 1), [None, batch], task,
                            torch.device('cpu'), progress=progress)
        self.assertEqual(metrics, {'loss': 2.})
        self.assertEqual(progress.batch_completed.call_count, 2)

    def test_invalid_interval_is_rejected(self):
        for interval in (0, -1, True, float('nan'), '10'):
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, 'log_interval_seconds'):
                resolve_config({
                    'model': {'name': 'vertex_edge_gated_mpnn', 'num_semantic_ids': 4},
                    'training': {'log_interval_seconds': interval},
                })

    def test_training_orchestration_persists_progress_without_optimizer_execution(self):
        def fake_epoch(model, loader, task, device, optimizer=None,
                       gradient_clip_norm=None, progress=None):
            for _ in loader:
                progress.batch_completed()
            return {'loss': 1.0}

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            'WANDB_MODE': 'disabled', 'EXPERIMENTS_DIR': directory + '/experiments',
        }), patch('training.gnn.trainer.run_epoch', side_effect=fake_epoch):
            config = load_config('graph/configs/train_gated_mpnn.yaml')
            config['device'] = 'cpu'
            config['data'] = {'factory': 'tests.test_gnn_training:fixture_data'}
            config['training'] = {'epochs': 2, 'batch_size': 1}
            config['scheduler'] = {'name': 'none'}
            config['output'] = {'root': directory + '/outputs'}
            output = train(config)
            history = json.loads((output / 'history.json').read_text())
            self.assertEqual([row['epoch'] for row in history], [1, 2])
            self.assertTrue(all(row['seconds'] >= 0 for row in history))
            metrics_path = Path(directory) / 'experiments' / output.name / 'metrics.json'
            events = json.loads(metrics_path.read_text())
            progress_events = [event for event in events if 'progress/epoch' in event['metrics']]
            self.assertEqual({event['step'] for event in progress_events}, {1, 2})
            last = progress_events[-1]['metrics']
            self.assertEqual(last['progress/epoch_remaining_seconds'], 0)
            self.assertEqual(last['progress/total_remaining_seconds'], 0)
            self.assertTrue((output / 'last.pt').exists())
            self.assertTrue((output / 'best.pt').exists())


if __name__ == '__main__':
    unittest.main()
