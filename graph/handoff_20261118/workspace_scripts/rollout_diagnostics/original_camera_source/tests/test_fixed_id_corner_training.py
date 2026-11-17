"""Causality, padding, supervision and shared initialization for corner prediction."""

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from graph.temporal.fixed_id_corner import FixedIdCornerPredictor
from training.fixed_id_corners import (
    error_totals, input_statistics, load_resume_checkpoint, metrics_from_totals,
    restore_training_state, save_checkpoint, update_early_stopping,
)


class FixedIdCornerTrainingTests(unittest.TestCase):
    def test_early_stopping_counts_since_global_best_and_resets_on_improvement(self):
        best, best_epoch = 0.1, 36
        for epoch in range(41, 46):
            best, best_epoch, stale, improved, stop = update_early_stopping(
                0.2, epoch, best, best_epoch, patience=10,
            )
            self.assertEqual(stale, epoch - 36)
            self.assertFalse(improved)
            self.assertFalse(stop)
        best, best_epoch, stale, improved, stop = update_early_stopping(
            0.1, 46, best, best_epoch, patience=10,
        )
        self.assertEqual(stale, 10)
        self.assertTrue(stop)  # An equal loss is not an improvement.
        self.assertFalse(improved)
        best, best_epoch, stale, improved, stop = update_early_stopping(
            0.09, 46, best, best_epoch, patience=10,
        )
        self.assertEqual((best, best_epoch, stale, improved, stop), (0.09, 46, 0, True, False))
        self.assertFalse(update_early_stopping(0.2, 100, best, best_epoch, patience=None)[-1])
        with self.assertRaisesRegex(ValueError, "finite"):
            update_early_stopping(float("nan"), 47, best, best_epoch, patience=10)

    def predictor(self):
        torch.manual_seed(2)
        model = FixedIdCornerPredictor(architecture="transformer", width=16, layers=2, heads=2)
        # A zero head would make the causality/padding checks vacuously pass.
        torch.nn.init.normal_(model.delta_head.weight, std=0.1)
        return model.eval()

    def test_future_and_padding_do_not_change_valid_prefixes(self):
        model = self.predictor()
        values = torch.randn(2, 8, 13)
        context = torch.ones(2, 8, dtype=torch.bool)
        with torch.no_grad():
            expected = model(values, context)
            changed = values.clone()
            changed[:, 4:] += 10
            actual = model(changed, context)
            torch.testing.assert_close(actual[:, :4], expected[:, :4])
            short = model(values[:, :4], context[:, :4])
            torch.testing.assert_close(short, expected[:, :4], atol=1e-7, rtol=1e-5)
            context[:, 4:] = False
            padded = model(changed, context)
            torch.testing.assert_close(padded[:, :4], short, atol=1e-7, rtol=1e-5)

    def test_sample_permutation_does_not_change_predictions(self):
        model = self.predictor()
        values = torch.randn(3, 7, 13)
        context = torch.ones(3, 7, dtype=torch.bool)
        with torch.no_grad():
            original = model(values, context)
            order = torch.tensor([2, 0, 1])
            reordered = model(values[order], context[order])
        torch.testing.assert_close(reordered, original[order])

    def test_supervision_excludes_initialization_and_padding_but_keeps_visible(self):
        prediction = torch.zeros(1, 5, 3, requires_grad=True)
        target = torch.zeros_like(prediction)
        target[0, 0] = 99  # Initialization must not affect the loss.
        target[0, 1, 0] = 0.003
        target[0, 2, 1] = 0.004
        target[0, 3, 2] = 0.012
        target[0, 4] = 99  # Padding must not affect the loss.
        batch = dict(
            targets=target,
            supervised=torch.tensor([[False, True, True, True, False]]),
            observed=torch.tensor([[True, True, False, True, False]]),
        )
        totals = error_totals(prediction, batch)
        metrics = metrics_from_totals(totals, displacement_scale_m=0.01)
        self.assertEqual(metrics["all_targets"], 3)
        self.assertEqual(metrics["visible_targets"], 2)
        self.assertEqual(metrics["missing_targets"], 1)
        self.assertAlmostEqual(metrics["all_rmse_mm"], (169 / 3) ** 0.5, places=5)
        self.assertAlmostEqual(metrics["missing_rmse_mm"], 4, places=5)
        totals[0].backward()
        self.assertEqual(float(prediction.grad[:, 0].abs().sum()), 0)
        self.assertEqual(float(prediction.grad[:, 4].abs().sum()), 0)
        self.assertGreater(float(prediction.grad[:, 1].abs().sum()), 0)
        self.assertGreater(float(prediction.grad[:, 2].abs().sum()), 0)

    def test_normalization_ignores_padded_context(self):
        values = torch.ones(2, 3, 13)
        values[:, 2] = 999
        data = dict(inputs=values, context=torch.tensor([[True, True, False]] * 2))
        mean, std = input_statistics(data)
        torch.testing.assert_close(mean, torch.ones(13))
        torch.testing.assert_close(std, torch.full((13,), 0.001))

    def test_backward_is_finite_and_updates_predictor(self):
        model = self.predictor().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        values = torch.randn(2, 6, 13)
        context = torch.ones(2, 6, dtype=torch.bool)
        before = model.delta_head.weight.detach().clone()
        prediction = model(values, context)
        loss = (prediction - 0.002).square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.delta_head.weight))

    def test_resume_reproduces_next_shuffle_optimizer_and_update(self):
        model_config = dict(architecture="transformer", width=16, layers=1, heads=2)
        settings = dict(device="cuda:0", epochs=2, batch_size=4, learning_rate=0.001,
                        weight_decay=0.01, gradient_clip=1.0, precision="bfloat16")
        config = dict(seed=7, model=model_config, training=settings)
        torch.manual_seed(7)
        model = FixedIdCornerPredictor(**model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        shuffle = np.random.default_rng(7)
        inputs = torch.randn(8, 4, 13)
        target = torch.randn(8, 4, 3) * 0.01
        context = torch.ones(8, 4, dtype=torch.bool)

        def update(network, optim, order):
            optim.zero_grad()
            indices = torch.as_tensor(order)
            loss = (network(inputs[indices], context[indices]) - target[indices]).square().mean()
            loss.backward()
            optim.step()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = []
            for epoch in (1, 2):
                order = shuffle.permutation(8)
                update(model, optimizer, order)
                history.append(dict(epoch=epoch, validation=dict(loss=float(epoch)),
                                    batch_order_sha256=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest(),
                                    elapsed_seconds=float(epoch)))
                save_checkpoint(root / "last.pt", model, optimizer, config, "dataset", "old_run", epoch, history, shuffle)
                if epoch == 1:
                    save_checkpoint(root / "best.pt", model, optimizer, config, "dataset", "old_run", epoch, history, shuffle)
            resumed_config = copy.deepcopy(config)
            resumed_config["training"]["epochs"] = 4
            resumed_config["resume"] = dict(checkpoint=str(root / "last.pt"))
            checkpoint, _, best_path = load_resume_checkpoint(resumed_config, "dataset")
            self.assertEqual(best_path, (root / "best.pt").resolve())

            expected_random = torch.rand(3)
            expected_order = shuffle.permutation(8)
            update(model, optimizer, expected_order)
            expected_state = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())

            resumed_model = FixedIdCornerPredictor(**model_config)
            resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.001, weight_decay=0.01)
            resumed_shuffle = np.random.default_rng(7)
            restore_training_state(checkpoint, resumed_model, resumed_optimizer, resumed_shuffle, 8)
            torch.testing.assert_close(torch.rand(3), expected_random, rtol=0, atol=0)
            resumed_order = resumed_shuffle.permutation(8)
            np.testing.assert_array_equal(resumed_order, expected_order)
            update(resumed_model, resumed_optimizer, resumed_order)
            for key, value in expected_state.items():
                torch.testing.assert_close(resumed_model.state_dict()[key], value, rtol=0, atol=0)
            for index, expected in expected_optimizer["state"].items():
                actual = resumed_optimizer.state_dict()["state"][index]
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

            bad_config = copy.deepcopy(resumed_config)
            bad_config["training"]["batch_size"] = 5
            with self.assertRaisesRegex(ValueError, "batch_size"):
                load_resume_checkpoint(bad_config, "dataset")
            with self.assertRaisesRegex(ValueError, "dataset hash"):
                load_resume_checkpoint(resumed_config, "another_dataset")
            altered = copy.deepcopy(checkpoint)
            altered["history"][0]["batch_order_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "batch order"):
                restore_training_state(altered, resumed_model, resumed_optimizer, np.random.default_rng(7), 8)


if __name__ == "__main__":
    unittest.main()
