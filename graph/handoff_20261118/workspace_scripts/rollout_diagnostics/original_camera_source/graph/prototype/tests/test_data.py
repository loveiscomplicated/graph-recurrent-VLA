import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from graph.prototype.data import episode_split, state_at, validate_trajectory, load_dataset, split_indices, phase_label
from graph.prototype.geometry import GeometryState
from graph.prototype.graph_builder import GraphBuilder
from graph.prototype.normalization import FeatureNormalizer
from .fixtures import geometry


class DataTests(unittest.TestCase):
    def test_episode_splits_and_reset_groups_no_leakage(self):
        ids = list(range(30))
        groups = [i//2 for i in ids]
        splits = episode_split(ids, seed=0, groups=groups)
        self.assertEqual(splits, episode_split(ids, seed=0, groups=groups))
        self.assertEqual(set(sum(splits.values(), [])), set(ids))
        assigned = [{groups[i] for i in values} for values in splits.values()]
        for i in range(3):
            for j in range(i):
                self.assertFalse(assigned[i] & assigned[j])
        with self.assertRaises(ValueError):
            episode_split([0]*10)

    def test_timestep_and_corruption_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"test.h5"
            with h5py.File(path, "w") as f:
                g = f.create_group("traj_0")
                actions = np.arange(21, dtype=np.float32).reshape(3, 7)/21
                g.create_dataset("actions", data=actions)
                g.create_dataset("obs", data=np.zeros((4, 43), dtype=np.float32))
                state = np.arange(4, dtype=np.float32)[:, None]*np.ones((1, 13), np.float32)
                g.create_dataset("env_states/actors/peg", data=state)
                a, _ = validate_trajectory(g)
                for t in range(3):
                    self.assertEqual(float(state_at(g["env_states"], t)["actors"]["peg"][0]), t)
                    np.testing.assert_array_equal(a[t], actions[t])
                del g["env_states/actors/peg"]
                g.create_dataset("env_states/actors/peg", data=state[:3])
                with self.assertRaises(ValueError):
                    validate_trajectory(g)

    def test_normalizer_shared_information_and_no_test_statistics(self):
        g = geometry(8)
        train = g.select(slice(0, 6))
        norm = FeatureNormalizer.fit(train)
        modified = g.as_dict()
        modified["peg_points"] = modified["peg_points"].clone()
        modified["peg_points"][6:] += 1000
        again = FeatureNormalizer.fit(GeometryState(**modified).select(slice(0, 6)))
        for k, v in norm.as_dict().items():
            torch.testing.assert_close(v, again.as_dict()[k], atol=0, rtol=0)
        graph = norm.apply("surface", g, GraphBuilder())
        flat = norm.apply("flat", g, None)
        torch.testing.assert_close(flat, torch.cat((graph.x.flatten(1), graph.context), -1), atol=0, rtol=0)

    def test_phase_does_not_require_simulator_contacts(self):
        g = geometry(1)
        label = phase_label(None, g)
        self.assertIn(label, range(5))

    def test_actual_dataset_alignment_split_and_finiteness(self):
        path = Path("outputs/graph/dataset.pt")
        if not path.exists():
            self.skipTest("official extracted dataset unavailable")
        data, g = load_dataset(path)
        self.assertEqual(data["actions"].shape, (len(data["episode_id"]), 7))
        for split in ("train", "validation", "test"):
            self.assertGreater(len(split_indices(data, split)), 0)
        splits = data["manifest"]["splits"]
        self.assertFalse(set(splits["train"]) & set(splits["test"]))
        self.assertFalse(set(splits["validation"]) & set(splits["test"]))
        self.assertFalse({"phase", "success", "reward", "actions"} & set(g.as_dict()))
        for value in g.as_dict().values():
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(value.shape[0], len(data["episode_id"]))
        with h5py.File(data["manifest"]["source"], "r") as f:
            for check in data["manifest"]["snapshot_checks"]:
                eid = check["episode_id"]
                rows = data["episode_id"] == eid
                steps = data["timestep"][rows]
                torch.testing.assert_close(steps, torch.arange(check["timesteps"]))
                np.testing.assert_array_equal(data["actions"][rows].numpy(), f[f"traj_{eid}/actions"][:])
                self.assertLess(check["max_observation_error"], 2e-5)
        for check in data["manifest"]["replay_checks"]:
            self.assertTrue(check["success"])
            self.assertLess(check["max_obs_error"], .001)


if __name__ == "__main__":
    unittest.main()
