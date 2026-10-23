import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from recurrent.corruption import Corruptor, level_config
from recurrent.evaluate import verify_resume
from recurrent.features import OBS_DIM, build_features
from recurrent.gpu_eval import TensorCorruptor, features, freeze_state
from recurrent.models import REGISTRY, build
from recurrent.tests.test_recurrent import fake_raw


class TensorEvaluationTest(unittest.TestCase):
    def test_features_and_rng(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        raw = fake_raw(100, np.random.default_rng(4))
        for level in ("none", "low", "mid", "high"):
            cfg = level_config(level)
            a = Corruptor(cfg, np.random.default_rng(42))
            b = TensorCorruptor(cfg, np.random.default_rng(42))
            for t in range(100):
                if t == 10:
                    a.force_drop(20)
                    b.force_drop(20)
                r = {k: v[t] for k, v in raw.items()}
                ra, f, age = a.step(r)
                rb, fb, ab = b.step(
                    {k: torch.tensor(v, device=device) for k, v in r.items()}
                )
                self.assertEqual((f, age), (fb, ab))
                actual = (
                    features(
                        rb,
                        torch.tensor(fb, device=device),
                        torch.tensor(ab, device=device),
                    )
                    .cpu()
                    .numpy()
                )
                np.testing.assert_allclose(
                    actual,
                    build_features(ra, np.asarray(f), np.asarray(age)),
                    atol=1e-4,
                    rtol=1e-4,
                )

    def test_all_models_independent_and_freeze(self):
        import copy

        torch.set_num_threads(1)
        torch.manual_seed(8)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for kind in REGISTRY:
            model = build(kind, OBS_DIM, 7).to(device).eval()
            x = torch.randn(3, 6, OBS_DIM, device=device)
            with torch.no_grad():
                batched, _ = model(x)
                reference, _ = copy.deepcopy(model).cpu()(x.cpu())
                torch.testing.assert_close(
                    batched.cpu(), reference, atol=1e-4, rtol=1e-4
                )
                separate = torch.cat([model(x[i : i + 1])[0] for i in range(3)])
                torch.testing.assert_close(batched, separate, atol=1e-4, rtol=1e-4)
                _, old = model(x[:, :1])
                _, new = model(x[:, 1:2], old)
                active = torch.tensor([True, False, True], device=device)
                frozen = freeze_state(new, old, active, kind)

                def check(a, b, c, root=True, kind=kind):
                    if a is None:
                        return
                    if isinstance(a, torch.Tensor):
                        dim = 1 if root and kind in ("gru", "lstm") else 0
                        torch.testing.assert_close(a.select(dim, 1), b.select(dim, 1))
                        torch.testing.assert_close(a.select(dim, 0), c.select(dim, 0))
                    else:
                        for aa, bb, cc in zip(a, b, c):
                            check(aa, bb, cc, root and kind == "lstm")

                check(frozen, old, new)

    def test_checkpoint_normalization_actions(self):
        from recurrent.evaluate import LearnedPolicy
        from recurrent.gpu_eval import TensorPolicy
        from recurrent.train import normalize_obs

        device = "cuda" if torch.cuda.is_available() else "cpu"
        seq = fake_raw(10, np.random.default_rng(31))
        with tempfile.TemporaryDirectory() as tmp:
            for kind in ("mlp", "gru"):
                path = Path(tmp) / f"{kind}.pt"
                cfg = {"hidden": 8}
                torch.manual_seed(31)
                model = build(kind, OBS_DIM, 7, cfg)
                rng = np.random.default_rng(31)
                stats = {
                    "obs_mean": rng.normal(0, 0.1, OBS_DIM).astype(np.float32),
                    "obs_std": rng.uniform(0.1, 2, OBS_DIM).astype(np.float32),
                    "obs_keep": (np.arange(OBS_DIM) % 3 != 0).astype(np.float32),
                    "act_mean": rng.normal(0, 0.1, 6).astype(np.float32),
                    "act_std": rng.uniform(0.1, 2, 6).astype(np.float32),
                }
                torch.save({"kind": kind, "obs_dim": OBS_DIM, "cfg": cfg,
                            "state_dict": model.state_dict(), "stats": stats}, path)
                cpu = LearnedPolicy(path)
                gpu = TensorPolicy(LearnedPolicy(path), device)
                for t in range(10):
                    r = {k: v[t] for k, v in seq.items()}
                    fresh = np.float32(t % 2)
                    age = np.float32(t)
                    x = normalize_obs(build_features(r, fresh, age), cpu.stats)
                    rg = {k: torch.tensor(v[None], device=device) for k, v in r.items()}
                    fg = torch.tensor([fresh], device=device)
                    ag = torch.tensor([age], device=device)
                    z = (
                        (features(rg, fg, ag) - gpu.stats["obs_mean"])
                        / gpu.stats["obs_std"]
                        * gpu.stats["obs_keep"]
                    )
                    np.testing.assert_allclose(z.cpu().numpy()[0], x, atol=1e-4, rtol=1e-4)
                    action = (
                        gpu.act(rg, fg, ag, torch.tensor([True], device=device))
                        .cpu()
                        .numpy()[0]
                    )
                    np.testing.assert_allclose(
                        action, cpu.act(None, r, fresh, age), atol=1e-4, rtol=1e-4
                    )

    def test_invalid_backend_requests_fail_before_checkpoint_load(self):
        from recurrent.evaluate import evaluate

        for backend, batch, ckpt, message in [
            ("physx_cpu", 2, "missing.pt", "CPU evaluation"),
            ("physx_cuda", 1, "expert", "expert"),
            ("unknown", 1, "missing.pt", "Unknown"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                evaluate(
                    ckpt,
                    [],
                    1,
                    2000,
                    250,
                    Path("/tmp/unused_eval.jsonl"),
                    sim_backend=backend,
                    num_envs=batch,
                )

    def test_resume_identity(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "best.pt"
            torch.save({"kind": "mlp", "version": 1}, ckpt)
            out = Path(d) / "eval.jsonl"
            args = (
                out,
                ckpt,
                [("normal", "none", {})],
                30,
                2000,
                250,
                "corruption.yaml",
                "physx_cuda",
                None,
            )
            verify_resume(*args)
            verify_resume(*args)
            torch.save({"kind": "mlp", "version": 2}, ckpt)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_resume(*args)


class BatchLifecycleTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_resume_changes_batch_size_without_duplicates(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch

        from recurrent.common import NullLogger
        from recurrent.evaluate import evaluate_gpu, verify_resume

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            path = d / "best.pt"
            fixture = d / "fixture.pt"
            out = d / "eval.jsonl"
            stats = {
                "obs_mean": np.zeros(OBS_DIM, np.float32),
                "obs_std": np.ones(OBS_DIM, np.float32),
                "act_mean": np.zeros(6, np.float32),
                "act_std": np.ones(6, np.float32),
            }
            model = build("mlp", OBS_DIM, 7, {"hidden": 8, "depth": 2})
            torch.save(
                {
                    "kind": "mlp",
                    "obs_dim": OBS_DIM,
                    "cfg": {"hidden": 8, "depth": 2},
                    "state_dict": model.state_dict(),
                    "stats": stats,
                },
                path,
            )
            from recurrent.tests.test_main_execution import fixture as make_fixture
            make_fixture(fixture, range(2000, 2005))
            job = [("normal", "none", {})]
            row = {
                "condition": "normal",
                "eval_level": "none",
                "episode_seed": 2000,
                "success": False,
                "steps": 250,
                "shift_applied": False,
                "shift_step": None,
                "blackout_applied": False,
                "blackout_step": None,
                "mean_action_delta": 0.0,
                "mean_infer_ms": None,
                "acquisition": False,
                "transport": False,
                "alignment": False,
                "failure_stage": "acquisition",
            }
            identity = verify_resume(
                out, path, job, 5, 2000, 250, "corruption.yaml", "physx_cuda", fixture
            )
            from recurrent.integrity import identity_hash
            row.update(ckpt=path.parent.name, model="mlp", train_level=None, train_seed=None,
                       evaluation_identity=identity_hash(identity), sim_backend="physx_cuda",
                       initial_state_source=identity["initial_states"])
            out.write_text(json.dumps(row) + "\n")
            env = SimpleNamespace(
                device="cuda",
                backend=SimpleNamespace(
                    sim_backend="physx_cuda", render_backend="none"
                ),
                close=lambda: None,
            )
            env.unwrapped = env

            def batch(e, p, seeds, *args):
                return [{**row, "episode_seed": seed} for seed in seeds], {
                    "reset_s": 0,
                    "rollout_s": 0,
                }

            with (
                patch(
                    "envs.maniskill.make_peg_insertion_env", return_value=env
                ) as factory,
                patch(
                    "mani_skill.utils.registration.TimeLimitWrapper",
                    side_effect=lambda e, h: e,
                ),
                patch("recurrent.gpu_eval.run_batch", side_effect=batch) as rollout,
                patch(
                    "recurrent.evaluate.experiment_logger", return_value=NullLogger()
                ),
            ):
                evaluate_gpu(
                    path,
                    job,
                    5,
                    2000,
                    250,
                    out,
                    None,
                    "corruption.yaml",
                    lambda x: None,
                    3,
                    fixture,
                )
                self.assertEqual(
                    [c.kwargs["num_envs"] for c in factory.call_args_list], [3, 1]
                )
                self.assertEqual(
                    [c.args[2] for c in rollout.call_args_list],
                    [[2001, 2002, 2003], [2004]],
                )
                before = out.read_bytes()
                evaluate_gpu(
                    path,
                    job,
                    5,
                    2000,
                    250,
                    out,
                    None,
                    "corruption.yaml",
                    lambda x: None,
                    2,
                    fixture,
                )
                self.assertEqual(out.read_bytes(), before)
                self.assertEqual(rollout.call_count, 2)

    def test_events_early_end_partial_batch_and_horizon(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from mani_skill.utils.structs.pose import Pose

        from recurrent.gpu_eval import run_batch

        class Box:
            def __init__(self):
                self.pose = Pose.create(
                    torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 3)
                )

            def set_pose(self, p):
                self.pose = p

        class Env:
            device = "cpu"

            def __init__(self):
                self.unwrapped = self
                self.t = 0
                self.box = Box()
                self.scene = SimpleNamespace(
                    _gpu_apply_all=lambda: None, _gpu_fetch_all=lambda: None
                )

            def step(self, a):
                self.t += 1
                return (
                    None,
                    None,
                    torch.tensor([False, self.t == 8, False]),
                    torch.zeros(3, dtype=torch.bool),
                    {"success": torch.tensor([self.t == 2, False, False])},
                )

        class Policy:
            state = None
            t = 0

            def act(self, r, f, a, active):
                self.t += 1
                return torch.full((3, 7), float(self.t**2))

        env = Env()
        raw = {
            k: torch.tensor(v) for k, v in fake_raw(3, np.random.default_rng(9)).items()
        }

        def snaps(e, r):
            return [
                {
                    "grasped": True,
                    "lifted": e.t >= 4,
                    "transverse": 0.02,
                    "progress": 0.0,
                    "alignment": False,
                    "insertion": False,
                }
                for _ in range(3)
            ]

        with (
            patch("recurrent.initial_states.restore", return_value={}),
            patch("recurrent.gpu_eval.raw_tensor", return_value=raw),
            patch("recurrent.gpu_eval.stage_snapshots", side_effect=snaps),
            patch("torch.cuda.synchronize"),
        ):
            rows, timing = run_batch(
                env,
                Policy(),
                [2000, 2001, 2002],
                level_config("none"),
                {"shift_dx": 0.02, "shift_delay": 2, "blackout": 10},
                12,
                {},
            )
        self.assertEqual([r["steps"] for r in rows], [2, 8, 12])
        np.testing.assert_allclose(
            [r["mean_action_delta"] for r in rows],
            np.sqrt(6) * np.array([3.0, 9.0, 13.0]),
            rtol=1e-6,
        )
        self.assertEqual([r["blackout_step"] for r in rows], [None, 4, 4])
        self.assertEqual([r["shift_step"] for r in rows], [None, 6, 6])
        self.assertEqual([r["transport_step"] for r in rows], [None, 4, 4])
        self.assertEqual(timing["active_env_steps"], 22)
        self.assertEqual(env.t, 12)
        torch.testing.assert_close(
            env.box.pose.p[:, 0], torch.tensor([0.0, -0.02, 0.02])
        )
