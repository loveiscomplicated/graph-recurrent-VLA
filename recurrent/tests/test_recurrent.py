import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from recurrent.common import RAW_KEYS
from recurrent.corruption import Corruptor, level_config
from recurrent.features import AGE_IDX, FRESH_IDX, OBS_DIM, build_features
from recurrent.models import REGISTRY, build, num_params
from recurrent.run_experiments import experiment_jobs
from recurrent.train import make_stats, normalize_obs


def fake_raw(n, rng):
    raw = {}
    for k, d in RAW_KEYS.items():
        v = rng.normal(0, 0.05, (n, d)).astype(np.float32)
        if d == 7:
            q = rng.normal(0, 1, (n, 4))
            v[:, 3:] = q / np.linalg.norm(q, axis=1, keepdims=True)
        raw[k] = v
    raw["peg_half"][:] = [0.1, 0.02, 0.02]
    raw["hole_radius"][:] = 0.023
    return raw


def env_or_skip():
    try:
        from recurrent.common import make_env
        return make_env()
    except Exception as exc:
        raise unittest.SkipTest(f"simulator unavailable: {type(exc).__name__}")


class ModelTest(unittest.TestCase):
    def test_full_sequence_matches_step_by_step(self):
        torch.manual_seed(0)
        x = torch.randn(3, 20, OBS_DIM)
        for kind in REGISTRY:
            model = build(kind, OBS_DIM, 7).eval()
            with torch.no_grad():
                full, _ = model(x)
                state, outs = None, []
                for t in range(x.shape[1]):
                    o, state = model(x[:, t:t + 1], state)
                    outs.append(o)
            self.assertTrue(torch.allclose(full, torch.cat(outs, 1), atol=1e-4), kind)

    def test_parameter_counts_are_matched(self):
        counts = [num_params(build(k, OBS_DIM, 7)) for k in REGISTRY]
        self.assertLess(max(counts) / min(counts), 1.15)


class ShardTest(unittest.TestCase):
    def test_main_grid_is_split_without_overlap(self):
        cfg = {"levels": ["none", "low", "mid", "high"],
               "models": ["mlp", "stack_mlp", "gru", "lstm", "cfc", "rkn", "mamba"],
               "seeds": [0, 1, 2]}
        whole = experiment_jobs(cfg)
        shards = [experiment_jobs(cfg, i, 4) for i in range(4)]
        self.assertEqual(len(whole), 84)
        self.assertEqual([len(shard) for shard in shards], [21] * 4)
        self.assertEqual(set(whole), set().union(*(set(shard) for shard in shards)))
        self.assertEqual(sum(map(len, shards)), len(set(whole)))


class CorruptionTest(unittest.TestCase):
    def test_none_level_is_identity(self):
        rng = np.random.default_rng(0)
        raw = {k: v[0] for k, v in fake_raw(1, rng).items()}
        c = Corruptor(level_config("none"), np.random.default_rng(1))
        out, fresh, age = c.step(raw)
        for k in RAW_KEYS:
            np.testing.assert_array_equal(out[k], raw[k])
        self.assertEqual((fresh, age), (1.0, 0.0))

    def test_same_seed_same_corruption(self):
        seq = fake_raw(60, np.random.default_rng(0))
        runs = []
        for _ in range(2):
            c = Corruptor(level_config("high"), np.random.default_rng(5))
            runs.append([c.step({k: v[t] for k, v in seq.items()}) for t in range(60)])
        for (a, fa, ga), (b, fb, gb) in zip(*runs):
            np.testing.assert_array_equal(a["hole_pose"], b["hole_pose"])
            self.assertEqual((fa, ga), (fb, gb))

    def test_forced_blackout_holds_last_observation(self):
        seq = fake_raw(12, np.random.default_rng(0))
        c = Corruptor(level_config("none"), np.random.default_rng(0))
        first, _, _ = c.step({k: v[0] for k, v in seq.items()})
        c.force_drop(5)
        steps = [c.step({k: v[t] for k, v in seq.items()}) for t in range(1, 8)]
        self.assertEqual([s[1] for s in steps], [0, 0, 0, 0, 0, 1, 1])
        self.assertEqual([s[2] for s in steps[:5]], [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(steps[3][0]["peg_pose"], first["peg_pose"])
        np.testing.assert_array_equal(steps[3][0]["tcp_pose"], seq["tcp_pose"][4])


class FeatureTest(unittest.TestCase):
    def test_feature_dimension_and_flags(self):
        raw = fake_raw(5, np.random.default_rng(0))
        f = build_features(raw, np.zeros(5, np.float32), np.full(5, 40, np.float32))
        self.assertEqual(f.shape, (5, OBS_DIM))
        self.assertTrue(np.all(f[:, FRESH_IDX] == 0))
        self.assertTrue(np.all(f[:, AGE_IDX] == 1))

    def test_constant_channels_are_masked(self):
        rng = np.random.default_rng(0)
        eps = []
        for _ in range(3):
            raw = fake_raw(30, rng)
            raw["hole_pose"][:, 3:] = [1, 0, 0, 0]
            eps.append((raw, rng.normal(0, 1, (30, 7)).astype(np.float32), 0))
        stats = make_stats(eps)
        obs = build_features(eps[0][0], np.ones(30, np.float32), np.zeros(30, np.float32))
        obs[:, 32:35] += 0.01
        z = normalize_obs(obs, stats)
        self.assertTrue(np.all(np.abs(z) < 50))
        self.assertEqual(stats["obs_keep"][FRESH_IDX], 1.0)


class SimulatorTest(unittest.TestCase):
    def test_horizon_and_expert_stages(self):
        from recurrent.common import is_success, to_action_tensor
        from recurrent.expert import ScriptedExpert
        from recurrent.stages import StageTracker

        env = env_or_skip()
        try:
            self.assertEqual(env._max_episode_steps, 250)
            env.reset(seed=0)
            expert, tracker = ScriptedExpert(env), StageTracker()
            tracker.update(env)
            for _ in range(250):
                _, _, _, _, info = env.step(to_action_tensor(expert.act(), env))
                tracker.update(env, info)
                if is_success(info):
                    break
            s = tracker.summary()
            self.assertTrue(tracker.success)
            self.assertTrue(s["acquisition"] and s["transport"])
            self.assertIsNone(s["failure_stage"])
        finally:
            env.close()

    def test_official_format_extraction(self):
        import h5py
        from mani_skill.utils.wrappers.record import RecordEpisode
        from recurrent.common import is_success, make_env, to_action_tensor
        from recurrent.expert import ScriptedExpert
        from recurrent.official_demos import extract

        env_or_skip().close()
        with tempfile.TemporaryDirectory() as d:
            env = RecordEpisode(make_env(), output_dir=d, trajectory_name="demo", save_video=False, save_on_reset=False, record_env_state=True)
            try:
                env.reset(seed=3)
                expert = ScriptedExpert(env)
                for _ in range(250):
                    _, _, _, _, info = env.step(to_action_tensor(expert.act(), env))
                    if is_success(info):
                        break
                env.flush_trajectory()
            finally:
                env.close()
            path = Path(d) / "demo.h5"
            meta = json.loads(path.with_suffix(".json").read_text())
            with h5py.File(path) as f:
                actions = np.array(f["traj_0/actions"])
            kept, rejected, total = extract(path, None, None, 250, log=lambda *_: None)
            self.assertEqual((len(kept), total), (1, len(meta["episodes"])))
            np.testing.assert_allclose(kept[0]["act"], np.clip(actions, -1, 1))
            self.assertEqual(len(kept[0]["obs"]["tcp_pose"]), len(actions))


if __name__ == "__main__":
    unittest.main()
