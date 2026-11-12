"""Contract checks for 6D corner training and detector-only inference."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from graph.temporal.corner_rotation_6d import (
    Rotation6DCornerPredictor, encode_inputs, load_predictor, load_tracker,
)
from training.fixed_id_corner_rotation_6d import save_checkpoint, train, training_statistics


class Rotation6DTests(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.zeros(2, 2, 13)
        self.inputs[:, 0, 7:11] = torch.tensor([[0.0035, -0.707, 0.707, -0.0035],
                                                [-0.0035, 0.707, -0.707, 0.0035]])
        self.valid = torch.tensor([[True, False], [True, False]])

    def test_sign_matrix_and_padding(self):
        encoded = encode_inputs(self.inputs, self.valid)
        torch.testing.assert_close(encoded[0, 0], encoded[1, 0])
        q = self.inputs[0, 0, 7:11].numpy()
        matrix = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        np.testing.assert_allclose(encoded[0, 0, 7:13].numpy(), matrix[:, :2].T.reshape(-1), atol=1e-6)
        self.assertTrue(torch.equal(encoded[:, 1], torch.zeros(2, 15)))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            encode_inputs(torch.zeros(1, 1, 13), torch.ones(1, 1, dtype=torch.bool))

    def test_checkpoint_tracker_and_training_features(self):
        config = {"model": {"architecture": "transformer", "width": 16, "layers": 1, "heads": 2}}
        model = Rotation6DCornerPredictor(**config["model"])
        torch.nn.init.normal_(model.delta_head.weight)
        mean, std = training_statistics({"inputs": self.inputs, "context": self.valid})
        model.input_mean.copy_(mean)
        model.input_std.copy_(std)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(path, model, optimizer, config, "dataset", "run", [{"epoch": 1}],
                            np.random.default_rng(0))
            loaded, checkpoint = load_predictor(path)
            self.assertEqual(checkpoint["input_schema"], "corner_history_tcp_rotation_6d_v1")
            torch.testing.assert_close(loaded.input_mean, mean)
            with torch.no_grad():
                result = loaded(self.inputs, self.valid)
            torch.testing.assert_close(result[0, 0], result[1, 0])
            tracker, _ = load_tracker(path, .1)
            robot = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
            tracker.step(np.array([[0.1, 0, 0]], dtype=np.float32), np.array([0]), robot)
            result = tracker.step(np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int64), robot)
            self.assertEqual(result["positions"].shape, (1, 3))
            torch.save({"format": "fixed_id_corner_predictor_v1"}, path)
            with self.assertRaisesRegex(ValueError, "rotation-6D"):
                load_predictor(path)


    def test_training_workflow_on_synthetic_data(self):
        inputs = self.inputs.clone()
        context = self.valid.clone()
        data = dict(inputs=inputs, context=context, targets=torch.ones(2, 2, 3) * 0.01,
                    supervised=context, observed=context)
        config = dict(seed=0, model=dict(architecture="transformer", width=16, layers=1, heads=2),
                      data=dict(root="unused", expected_dataset_sha256="test"),
                      training=dict(device="cpu", precision="float32", epochs=1, batch_size=2,
                                    log_every_batches=1, cpu_threads=1, learning_rate=0.001,
                                    weight_decay=0.01, gradient_clip=1.0), output={})
        with tempfile.TemporaryDirectory() as directory:
            config["output"]["root"] = str(Path(directory) / "outputs")
            with patch("training.fixed_id_corner_rotation_6d.load_splits",
                       return_value=({"train": data, "validation": data}, {"dataset_sha256": "test"})), \
                 patch.dict("os.environ", {"WANDB_MODE": "disabled", "EXPERIMENTS_DIR": str(Path(directory) / "experiments")}):
                output = Path(train(config))
            model, checkpoint = load_predictor(output / "best.pt")
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertTrue(torch.isfinite(model.input_mean).all())
            self.assertEqual(model.input_mean.shape, (15,))

if __name__ == "__main__":
    unittest.main()
