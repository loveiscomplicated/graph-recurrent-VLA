"""Physical-unit metrics, subset aggregation and training diagnostics."""

import io
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from training.gnn.loss import object21_loss
from training.gnn.progress import TrainingProgress
from training.gnn.tasks import Object21Task, StepResult, geometry_errors
from training.gnn.trainer import run_epoch


def identity_targets(count):
    target = torch.zeros(count, 21)
    target[:, 3] = 1
    target[:, 13] = 1
    return target


class PredictionModel(torch.nn.Module):
    def forward(self, graph):
        return graph.output


def metric_batch(distances, empty):
    target = identity_targets(len(distances))
    prediction = target.clone()
    prediction[:, 0] = torch.tensor(distances)
    graph = SimpleNamespace(
        node_valid=~torch.tensor(empty)[:, None],
        output={'object21': prediction, 'endpoint_residual': torch.zeros(len(distances), 3)},
    )
    batch = SimpleNamespace(graph=graph, targets={'object21': target},
                            sample_ids=[str(index) for index in range(len(distances))])
    batch.to = lambda device: batch
    return batch


class MetricTests(unittest.TestCase):
    def test_quaternion_sign_angles_and_size_units(self):
        target = identity_targets(3)
        prediction = target.clone()
        prediction[0, 3] = -1  # Same rotation, opposite quaternion sign.
        prediction[1, 3:7] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
        prediction[2, 3:7] = torch.tensor([0., 1, 0, 0])
        prediction[:, 17:21] = torch.tensor([0.001, -0.002, 0.003, -0.004])
        errors = geometry_errors(prediction.requires_grad_(), target)
        torch.testing.assert_close(errors['peg_rotation_error_deg'], torch.tensor([0., 90., 180.]))
        self.assertTrue(all(not value.requires_grad for value in errors.values()))
        for name, expected in (
            ('peg_half_size_x_error_mm', 1), ('peg_half_size_y_error_mm', 2),
            ('peg_half_size_z_error_mm', 3), ('hole_radius_error_mm', 4),
        ):
            torch.testing.assert_close(errors[name], torch.full((3,), float(expected)))

    def test_weighted_contributions_match_total_and_existing_objective(self):
        target = identity_targets(2)
        prediction = target.clone()
        prediction[:, 0] = 0.02
        terms = object21_loss({'object21': prediction, 'endpoint_residual': torch.ones(2, 3) * 0.01},
                              target, consistency_weight=0.2)
        torch.testing.assert_close(terms['total'], sum(value for key, value in terms.items()
                                                     if key.startswith('weighted/')))
        self.assertAlmostEqual(terms['weighted/endpoint_consistency'].item(), 0.2)
        torch.testing.assert_close(terms['total'], terms['peg_position'] + 0.2)

    def test_validation_subsets_use_their_own_denominators_and_omit_absent_groups(self):
        model = PredictionModel()
        batches = [metric_batch([1., 2., 4.], [True, False, False]),
                   metric_batch([9.], [True])]
        metrics = run_epoch(model, batches, Object21Task(), torch.device('cpu'))
        self.assertAlmostEqual(metrics['peg_position_error_m'], 4., places=6)
        self.assertEqual(metrics['empty_input/peg_position_error_m'], 5.)
        self.assertEqual(metrics['nonempty_input/peg_position_error_m'], 3.)
        self.assertAlmostEqual(metrics['empty_input_fraction'], 0.5)
        only_empty = run_epoch(model, [batches[1]], Object21Task(), torch.device('cpu'))
        self.assertFalse(any(name.startswith('nonempty_input/') for name in only_empty))
        model.train()
        training_result = Object21Task().compute(model, batches[0])
        self.assertFalse(any(name.startswith('empty_input/') for name in training_result.metrics))

    def test_running_loss_is_weighted_and_resets_between_phases(self):
        now = [0.]
        logger = Mock()
        progress = TrainingProgress(logger, 1, [0, 0], [0], clock=lambda: now[0])
        with redirect_stdout(io.StringIO()) as output:
            progress.begin_epoch(1)
            progress.begin_phase('train')
            now[0] = 10
            progress.batch_completed({'loss': torch.tensor(6.)}, {'loss': 3})
            now[0] = 20
            progress.batch_completed({'loss': torch.tensor(14.)}, {'loss': 4})
            self.assertEqual(logger.log.call_args.args[0]['train/loss'], 3.5)
            progress.finish_phase()
            progress.begin_phase('validation')
            self.assertNotIn('train/loss', logger.log.call_args.args[0])
        self.assertIn('loss=3.500000', output.getvalue())

    def test_gradient_metrics_do_not_change_updates_and_use_step_denominators(self):
        for clip_norm in (None, 3.):
            with self.subTest(clip_norm=clip_norm):
                model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad():
                    model.weight.fill_(1.)
                steps = iter([(1., 7), (2., 1)])
                def compute(model, batch):
                    scale, count = next(steps)
                    loss = scale * model.weight.square().sum()
                    return StepResult(loss, {'loss': loss.detach()}, {'loss': count})
                optimizer = Mock()
                optimizer.zero_grad.side_effect = lambda **kwargs: model.zero_grad(**kwargs)
                batch = SimpleNamespace(to=lambda device: None)
                metrics = run_epoch(model, [batch, batch], SimpleNamespace(compute=compute),
                                    torch.device('cpu'), optimizer, clip_norm)
                self.assertEqual(metrics['gradient_norm'], 3.)
                expected_gradient = 4. if clip_norm is None else 4. * 3. / (4. + 1e-6)
                self.assertAlmostEqual(model.weight.grad.item(), expected_gradient)
                if clip_norm is None:
                    self.assertNotIn('gradient_clip_fraction', metrics)
                else:
                    self.assertEqual(metrics['gradient_clip_fraction'], 0.5)
                self.assertEqual(optimizer.step.call_count, 2)


if __name__ == '__main__':
    unittest.main()
