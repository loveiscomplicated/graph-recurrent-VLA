"""Check fresh GNN rollout provenance and resumable optimizer snapshots."""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from training.detected_corner_gnn_fresh import checkpoint_state, load_frozen_rollout, save_atomic


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FreshGNNTests(unittest.TestCase):
    def test_source_rollout_requires_matching_temporal_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporal_path = root / "temporal.pt"
            temporal_path.write_bytes(b"frozen temporal")
            identity = dict(dataset_sha256="data", threshold_mm=30)
            cache = root / "rollout_inputs.pt"
            torch.save(dict(identity=identity, arrays={}), cache)
            old_config = dict(trainer="old", seed=1, training=dict(epochs=20, batch_size=96),
                              temporal=dict(checkpoint=str(temporal_path), sha256=digest(temporal_path)), output={})
            source = dict(config=old_config, epoch=20, identity=identity, temporal_sha256=digest(temporal_path),
                          rollout_cache_sha256=digest(cache))
            checkpoint = root / "last.pt"
            torch.save(source, checkpoint)
            config = dict(trainer="fresh", seed=1, training=dict(epochs=40, batch_size=96),
                          temporal=old_config["temporal"], output={},
                          rollout_source=dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint)))
            _, cache_path, checked, _ = load_frozen_rollout(config)
            self.assertEqual(cache_path, cache.resolve())
            self.assertEqual(checked["identity"], identity)
            config["training"]["batch_size"] = 64
            with self.assertRaisesRegex(ValueError, "settings do not match"):
                load_frozen_rollout(config)

    def test_atomic_snapshot_contains_real_optimizer_and_rng_states(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        source = dict(identity={}, gnn_initial_sha256="initial", rollout_cache_sha256="cache",
                      temporal_sha256="temporal", run_id="source")
        state = checkpoint_state(model, optimizer, {}, source, "run", 1, np.random.default_rng(2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_atomic(path, state)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            self.assertTrue(loaded["optimizer"]["state"])
            self.assertIn("torch_rng", loaded)
            self.assertIn("cuda_rng", loaded)
            self.assertIn("numpy_rng", loaded)
            self.assertIn("python_rng", loaded)
            self.assertEqual(loaded["epoch"], 1)
            self.assertFalse(path.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()
