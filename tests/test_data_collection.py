"""Selection must precede actions; incomplete episodes never enter the dataset."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium.spaces import Box

from data_collection.collect_episodes import collect_episodes, ZeroActions


class FakeEnv:
    action_space = Box(-1, 1, (1, 2), np.float32)
    device = "cpu"

    @property
    def unwrapped(self):
        return self

    def reset(self, seed):
        self.seed = seed
        self.step_count = 0
        return {"raw": torch.tensor([seed], dtype=torch.int16)}, {}

    def step(self, action):
        self.step_count += 1
        return (
            {"raw": torch.tensor([self.seed + self.step_count], dtype=torch.int16)},
            torch.zeros(1),
            torch.tensor([self.step_count == 2]),
            torch.tensor([False]),
            {},
        )


def gate(env, observation):
    return {
        "requested_pass": env.seed % 2 == 1,
        "strict_100_pass": False,
        "corners_uv": [float("nan")],
    }, {}


class CollectionTests(unittest.TestCase):
    @patch("data_collection.collect_episodes.initial_visibility", side_effect=gate)
    def test_rejected_reset_gets_no_action_and_accepted_t0_is_preserved(self, _):
        env = FakeEnv()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            result = collect_episodes(
                env, root, num_episodes=1, max_attempts=2, steps=5, policy=ZeroActions()
            )
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["attempted"], 2)
            self.assertEqual(result["episodes"][0]["seed"], 1)
            self.assertEqual(result["episodes"][0]["observations"], 3)
            self.assertEqual(env.step_count, 2)
            with np.load(
                root / "episode_000000/frame_000000.npz", allow_pickle=False
            ) as saved:
                np.testing.assert_array_equal(saved["observation/raw"], [1])
                self.assertEqual(saved["observation/raw"].dtype, np.int16)
            with np.load(
                root / "episode_000000/frame_000002.npz", allow_pickle=False
            ) as saved:
                np.testing.assert_array_equal(saved["observation/raw"], [3])
                self.assertIn("action", saved)
            self.assertFalse(list(root.glob("*.partial")))
            attempts = [
                json.loads(line)
                for line in (root / "attempts.jsonl").read_text().splitlines()
            ]
            self.assertEqual([item["accepted"] for item in attempts], [False, True])
            self.assertIsNone(attempts[0]["visibility"]["corners_uv"][0])
            with self.assertRaises(FileExistsError):
                collect_episodes(env, root, num_episodes=1, max_attempts=1)

    @patch("data_collection.collect_episodes.initial_visibility", side_effect=gate)
    def test_strict_selection_and_budget_exhaustion(self, _):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            result = collect_episodes(
                FakeEnv(),
                root,
                num_episodes=1,
                max_attempts=2,
                visibility_mode="full_surface",
            )
            self.assertEqual(result["status"], "EXHAUSTED")
            self.assertEqual(result["episodes"], [])

    @patch("data_collection.collect_episodes.initial_visibility", side_effect=gate)
    def test_policy_error_retains_partial_but_never_approves_it(self, _):
        class BrokenPolicy(ZeroActions):
            def act(self, observation):
                raise RuntimeError("policy failed")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            with self.assertRaisesRegex(RuntimeError, "policy failed"):
                collect_episodes(
                    FakeEnv(),
                    root,
                    num_episodes=1,
                    max_attempts=1,
                    start_seed=1,
                    steps=3,
                    policy=BrokenPolicy(),
                )
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "ERROR")
            self.assertEqual(manifest["episodes"], [])
            self.assertTrue((root / "episode_000000.partial/frame_000000.npz").exists())

    @patch("data_collection.collect_episodes.initial_visibility", side_effect=gate)
    def test_success_filter_preserves_failures_outside_approved_manifest(self, _):
        class ExpertEnv(FakeEnv):
            def step(self, action):
                observation, reward, term, trunc, info = super().step(action)
                info["success"] = torch.tensor([self.seed == 3])
                return observation, reward, term, trunc, info

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            result = collect_episodes(
                ExpertEnv(), root, num_episodes=1, max_attempts=4,
                steps=5, policy=ZeroActions(), require_success=True,
            )
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["attempted"], 4)
            self.assertEqual(result["visibility_passed"], 2)
            self.assertEqual(result["expert_attempted"], 2)
            self.assertEqual(result["expert_failed"], 1)
            self.assertEqual(result["episodes"][0]["seed"], 3)
            self.assertTrue(result["episodes"][0]["success"])
            self.assertEqual(result["episodes"][0]["steps"], 1)
            rejected = json.loads((root / "rejected_seed_000001/episode.json").read_text())
            self.assertFalse(rejected["success"])
            self.assertEqual(rejected["observations"], 3)

    @patch("data_collection.collect_episodes.initial_visibility", side_effect=gate)
    def test_optional_step_ground_truth_aligns_with_post_action_observation(self, _):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            with patch(
                "data_collection.collect_episodes.get_gt_state",
                side_effect=lambda env: SimpleNamespace(t=torch.tensor([env.step_count])),
            ):
                collect_episodes(
                    FakeEnv(), root, num_episodes=1, max_attempts=1,
                    start_seed=1, steps=2, policy=ZeroActions(), save_step_gt=True,
                )
            with np.load(root / "episode_000000/frame_000002.npz") as saved:
                np.testing.assert_array_equal(saved["ground_truth/t"], [2])

    @patch("data_collection.collect_episodes.initial_visibility", side_effect=gate)
    def test_moving_box_refreshes_saved_frame_and_policy_observation(self, _):
        class MovingEnv(FakeEnv):
            def get_info(self):
                return {"success": torch.tensor([False])}

            def get_obs(self, info):
                return {"raw": torch.tensor([99], dtype=torch.int16)}

        class Motion:
            def __init__(self, env, settings, policy):
                self.offset = 0

            def advance(self, step, policy):
                self.offset = step
                return True

            def frame_data(self):
                return {"offset_m": [self.offset, 0, 0], "status": "moving"}

        class RecordingPolicy(ZeroActions):
            def reset(self, env, seed):
                super().reset(env, seed)
                self.inputs = []

            def act(self, observation):
                self.inputs.append(observation["raw"].item())
                return self.action

        settings = dict(mode="continuous", start_seconds=0.1, end_seconds=2,
                        duration_seconds=1, minimum_distance_m=0.08,
                        translation_m=[0.06, 0, 0])
        policy = RecordingPolicy()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            with patch("data_collection.collect_episodes.EpisodeHoleMotion", Motion):
                collect_episodes(MovingEnv(), root, num_episodes=1, max_attempts=1,
                                 start_seed=1, steps=2, policy=policy, hole_motion=settings)
            self.assertEqual(policy.inputs, [1, 99])
            with np.load(root / "episode_000000/frame_000001.npz") as saved:
                self.assertEqual(saved["observation/raw"].item(), 99)
                np.testing.assert_array_equal(saved["hole_motion/offset_m"], [1, 0, 0])
            with np.load(root / "episode_000000/frame_000002.npz") as saved:
                np.testing.assert_array_equal(saved["hole_motion/offset_m"], [1, 0, 0])


if __name__ == "__main__":
    unittest.main()
