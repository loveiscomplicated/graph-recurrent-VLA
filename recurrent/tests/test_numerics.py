"""Numerical regressions; no saved experiment or simulator rollout is needed."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from recurrent.models import RKNPolicy, build
from recurrent.numerics import numerics_context, require_state


class RKNNumericsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def model(self, device="cpu"):
        return RKNPolicy(3, 7, latent_obs=1, num_basis=1,
                         enc_hidden=4, head_hidden=4).to(device)

    def test_rounded_unit_gain_keeps_positive_variance(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device), torch.no_grad():
                model = self.model(device)
                model.basis.copy_(torch.eye(2, device=device)[None])
                model.w_var.weight.zero_()
                model.w_var.bias.fill_(-9)
                x = torch.zeros(1, 1, 3, device=device)
                state = (torch.zeros(1, 2, device=device),
                         torch.tensor([[1e6, 1.0]], device=device))
                noise = torch.nn.functional.softplus(model.w_var(model.encoder(x)))[:, 0] + 1e-4
                prior = state[1][:, :1] + model.log_q.exp()[:1]
                gain = prior / (prior + noise)
                self.assertEqual(gain.item(), 1.0)
                self.assertEqual(((1 - gain) * prior).item(), 0.0)
                out, (_, var) = model(x, state)
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue((var > 0).all())
                expected = (prior.double() * noise.double()
                            / (prior.double() + noise.double())).float()
                torch.testing.assert_close(var[:, :1], expected)

    def test_float64_reference_and_stepwise_state(self):
        model = self.model().eval()
        reference = copy.deepcopy(model).double()
        x = torch.randn(3, 30, 3)
        with torch.no_grad():
            full, state = model(x)
            expected, expected_state = reference(x.double())
            torch.testing.assert_close(full.double(), expected, atol=2e-5, rtol=1e-4)
            for actual, target in zip(state, expected_state):
                torch.testing.assert_close(actual.double(), target, atol=2e-5, rtol=1e-4)
            previous, outputs = None, []
            for step in range(x.shape[1]):
                out, previous = model(x[:, step:step + 1], previous)
                outputs.append(out)
            torch.testing.assert_close(full, torch.cat(outputs, 1))
            for actual, target in zip(previous, state):
                torch.testing.assert_close(actual, target)

    def test_backward_gradients_remain_finite(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model = self.model(device)
                x = torch.randn(3, 30, 3, device=device, requires_grad=True)
                out, (_, var) = model(x)
                self.assertTrue((var > 0).all())
                out.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)


def checkpoint(path, kind="mlp"):
    cfg = {"latent_obs": 2, "num_basis": 1, "enc_hidden": 4, "head_hidden": 4} if kind == "rkn" else {"hidden": 8}
    model = build(kind, 38, 7, cfg)
    stats = {"obs_mean": [0.0] * 38, "obs_std": [1.0] * 38,
             "act_mean": [0.0] * 6, "act_std": [1.0] * 6}
    torch.save(dict(kind=kind, obs_dim=38, cfg=cfg, stats=stats,
                    state_dict=model.state_dict(), train_level="none", seed=0), path)
    return path


class EvaluationNumericsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.ckpt = checkpoint(self.root / "best.pt")

    def test_state_batch_dimensions_and_active_mask(self):
        active = torch.tensor([True, False, True])
        for kind in ("gru", "lstm", "rkn", "mamba"):
            with self.subTest(kind=kind):
                if kind in ("gru", "lstm"):
                    state = torch.ones(2, 3, 4)
                    state[:, 1] = float("nan")
                    if kind == "lstm":
                        state = (state, state.clone())
                else:
                    state = torch.ones(3, 4)
                    state[1] = float("nan")
                    state = (state, state.clone()) if kind == "rkn" else [(state, state[:, :, None])]
                require_state(state, kind, active=active)
                with self.assertRaisesRegex(FloatingPointError, "slots=\\[1\\]"):
                    require_state(state, kind, active=torch.ones(3, dtype=torch.bool))
        with self.assertRaisesRegex(FloatingPointError, "positive"):
            require_state((torch.zeros(3, 4), torch.zeros(3, 4)), "rkn", active=active)

    def test_nonfinite_model_output_cannot_be_hidden_by_clipping_or_gripper(self):
        from recurrent.evaluate import LearnedPolicy
        from recurrent.gpu_eval import TensorPolicy
        from recurrent.tests.test_recurrent import fake_raw

        raw = fake_raw(2, np.random.default_rng(5))
        for column, value in ((0, float("inf")), (6, float("nan"))):
            with self.subTest(column=column):
                class BadModel(torch.nn.Module):
                    def forward(self, x, state=None):
                        out = x.new_zeros(len(x), 1, 7)
                        out[:, :, column] = value
                        return out, None

                learned = LearnedPolicy(self.ckpt)
                learned.model = BadModel()
                with self.assertRaisesRegex(FloatingPointError, "model_output"):
                    learned.act(None, {k: v[0] for k, v in raw.items()}, 1, 0)
                tensor = TensorPolicy(learned, "cpu")
                with self.assertRaisesRegex(FloatingPointError, "model_output.*slots=\\[0, 1\\]"):
                    tensor.act({k: torch.tensor(v) for k, v in raw.items()},
                               torch.ones(2), torch.zeros(2), torch.ones(2, dtype=torch.bool))

    def test_inactive_policy_slot_is_ignored_and_its_state_stays_frozen(self):
        from recurrent.evaluate import LearnedPolicy
        from recurrent.gpu_eval import TensorPolicy
        from recurrent.tests.test_recurrent import fake_raw

        class BadInactiveModel(torch.nn.Module):
            def forward(self, x, state=None):
                out = x.new_zeros(len(x), 1, 7)
                mu, var = x.new_ones(len(x), 4), x.new_ones(len(x), 4)
                out[1] = mu[1] = var[1] = float("nan")
                return out, (mu, var)

        learned = LearnedPolicy(self.ckpt)
        learned.model, learned.name = BadInactiveModel(), "rkn"
        raw = {k: torch.tensor(v) for k, v in fake_raw(2, np.random.default_rng(6)).items()}
        raw["peg_pose"][1] = float("nan")
        tensor = TensorPolicy(learned, "cpu")
        for _ in range(2):
            action = tensor.act(raw, torch.ones(2), torch.zeros(2), torch.tensor([True, False]))
            self.assertTrue(torch.isfinite(action[0]).all())
            for state in tensor.state:
                torch.testing.assert_close(state[1], torch.zeros(4))

    def test_invalid_input_and_state_fail_before_model(self):
        from recurrent.evaluate import LearnedPolicy
        from recurrent.tests.test_recurrent import fake_raw

        learned = LearnedPolicy(self.ckpt)
        learned.model = Mock(side_effect=AssertionError("model must not run"))
        raw = {k: v[0] for k, v in fake_raw(1, np.random.default_rng(3)).items()}
        raw["tcp_pose"][0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "observation.tcp_pose"):
            learned.act(None, raw, 1, 0)
        raw["tcp_pose"][0] = 0.0
        learned.stats["obs_std"][0] = 0.0
        with np.errstate(divide="ignore", invalid="ignore"), self.assertRaisesRegex(FloatingPointError, "normalized_input"):
            learned.act(None, raw, 1, 0)
        learned.stats["obs_std"][0] = 1.0
        learned.state = torch.full((1, 1, 4), float("inf"))
        with self.assertRaisesRegex(FloatingPointError, "state"):
            learned.act(None, raw, 1, 0)
        learned.model.assert_not_called()

    def test_gpu_batch_reports_seed_step_and_stops_before_simulator_step(self):
        from recurrent.corruption import level_config
        from recurrent.gpu_eval import run_batch
        from recurrent.tests.test_recurrent import fake_raw

        env = SimpleNamespace(device="cpu", step=Mock(), box=SimpleNamespace(
            pose=SimpleNamespace(raw_pose=torch.tensor([[0., 0., 0., 1., 0., 0., 0.]] * 2))))
        env.unwrapped = env
        policy = SimpleNamespace(state=None, act=lambda *args: torch.full((2, 7), float("nan")))
        raw = {k: torch.tensor(v) for k, v in fake_raw(2, np.random.default_rng(9)).items()}
        with patch("recurrent.initial_states.restore", return_value={}), \
             patch("recurrent.gpu_eval.raw_tensor", return_value=raw), \
             patch("recurrent.gpu_eval.stage_snapshots", return_value=[]), \
             patch("torch.cuda.synchronize"), \
             self.assertRaisesRegex(FloatingPointError, "checkpoint=.*condition=normal.*level=none.*episode_seeds=\\[2010, 2011\\].*step=1.*action"):
            with numerics_context(checkpoint=str(self.ckpt), condition="normal", level="none"):
                run_batch(env, policy, [2010, 2011], level_config("none"), {}, 250, {})
        env.step.assert_not_called()

    def test_cpu_episode_rejects_action_before_simulator_step(self):
        from recurrent.corruption import level_config
        from recurrent.evaluate import run_episode
        from recurrent.tests.test_recurrent import fake_raw

        env = SimpleNamespace(reset=Mock(), step=Mock())
        policy = SimpleNamespace(reset=Mock(), act=Mock(return_value=np.full(7, np.inf)))
        raw = {k: v[0] for k, v in fake_raw(1, np.random.default_rng(3)).items()}
        with patch("recurrent.evaluate.raw_gt", return_value=raw), \
             patch("recurrent.evaluate.StageTracker") as tracker, \
             self.assertRaisesRegex(FloatingPointError, "episode_seed=2010.*step=1.*action"):
            tracker.return_value.transport_step = None
            run_episode(env, policy, 2010, level_config("none"), {}, 250)
        env.step.assert_not_called()

    def test_rkn_identity_refuses_legacy_results_without_changing_other_models(self):
        from recurrent.evaluate import evaluation_identity, verify_resume

        jobs = [("normal", "none", {})]
        for kind in ("mlp", "rkn", "expert"):
            with self.subTest(kind=kind):
                ckpt = "expert" if kind == "expert" else checkpoint(self.ckpt, kind)
                identity = evaluation_identity(ckpt, jobs, 1, 2010, 250, "corruption.yaml", "physx_cpu", None)
                legacy = copy.deepcopy(identity)
                legacy["settings"].pop("rkn_covariance_update", None)
                out = self.root / f"{kind}.jsonl"
                out.with_suffix(".identity.json").write_text(json.dumps(legacy))
                if kind == "rkn":
                    self.assertEqual(identity["settings"]["rkn_covariance_update"], "stable_v1")
                    with self.assertRaisesRegex(ValueError, "mismatch"):
                        verify_resume(out, ckpt, jobs, 1, 2010, 250, "corruption.yaml", "physx_cpu", None)
                else:
                    self.assertEqual(identity, legacy)
                    verify_resume(out, ckpt, jobs, 1, 2010, 250, "corruption.yaml", "physx_cpu", None)

    def test_saved_invalid_metric_reports_location_and_is_not_repaired(self):
        from recurrent.evaluate import evaluation_identity
        from recurrent.integrity import read_eval_rows
        from recurrent.tests.test_main_execution import row

        identity = evaluation_identity(self.ckpt, [("normal", "none", {})], 1, 2010,
                                       250, "corruption.yaml", "physx_cpu", None)
        path = self.root / "bad.jsonl"
        for value in (float("nan"), float("inf"), -1.0, "bad"):
            with self.subTest(value=value):
                record = row(identity, seed=2010)
                record["mean_action_delta"] = value
                original = json.dumps(record) + "\n"
                path.write_text(original)
                with self.assertRaisesRegex(ValueError, r"mean_action_delta=.*bad.jsonl:1.*condition=normal.*level=none.*episode_seed=2010"):
                    read_eval_rows(path, identity, repair=True)
                self.assertEqual(path.read_text(), original)

    def test_nonfinite_row_is_never_written_or_marked_complete(self):
        from recurrent.common import NullLogger
        from recurrent.evaluate import evaluate
        from recurrent.tests.test_main_execution import fixture

        bad = dict(success=False, episode_seed=2010, mean_action_delta=float("nan"))
        for backend in ("physx_cpu", "physx_cuda"):
            with self.subTest(backend=backend):
                initial = self.root / "fixture.pt"
                fixture(initial, range(2010, 2011))
                path = self.root / f"{backend}.jsonl"
                env = SimpleNamespace(device="cpu" if backend == "physx_cpu" else "cuda",
                                      backend=SimpleNamespace(sim_backend=backend, render_backend="none"), close=Mock())
                env.unwrapped = env
                with patch("recurrent.evaluate.make_env", return_value=env), \
                     patch("envs.maniskill.make_peg_insertion_env", return_value=env), \
                     patch("mani_skill.utils.registration.TimeLimitWrapper", side_effect=lambda e, h: e), \
                     patch("recurrent.evaluate.run_episode", return_value=bad), \
                     patch("recurrent.gpu_eval.run_batch", return_value=([bad], {})), \
                     patch("recurrent.gpu_eval.TensorPolicy", return_value=object()), \
                     patch("recurrent.evaluate.experiment_logger", return_value=NullLogger()), \
                     patch("torch.cuda.is_available", return_value=True), \
                     patch("torch.cuda.reset_peak_memory_stats"), \
                     patch("torch.cuda.max_memory_allocated", return_value=0), \
                     self.assertRaisesRegex(ValueError, "Out of range float values"):
                    evaluate(self.ckpt, [("normal", "none", {})], 1, 2010, 250, path,
                             log=lambda _: None, sim_backend=backend,
                             initial_states=str(initial) if backend == "physx_cuda" else None)
                self.assertEqual(path.read_text(), "")
                self.assertFalse(path.with_suffix(".complete.json").exists())
                self.assertFalse(path.with_suffix(".summary.json").exists())
                env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
