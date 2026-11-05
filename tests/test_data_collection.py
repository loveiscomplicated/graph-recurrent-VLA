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


if __name__ == "__main__":
    unittest.main()
