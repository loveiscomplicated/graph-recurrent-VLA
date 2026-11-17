"""Real two-backbone runs and failure checks for independent aggregation."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

from main import load_config
from evaluation.gnn.temporal.corner_tracking_aggregate import aggregate, check_paired_runs, inspect_run
from training.gnn.temporal.corner_tracking import compare


class IndependentAggregationTest(unittest.TestCase):
    def test_completed_runs_aggregate_and_unfinished_or_unpaired_runs_fail(self):
        config = load_config("graph/configs/temporal/corner_tracking/smoke.yaml")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "WANDB_MODE": "disabled", "EXPERIMENTS_DIR": str(Path(directory) / "experiments")}):
            config["output"] = {"root": str(Path(directory) / "runs")}
            config["data"]["frames"] = 8
            source_report = json.loads(Path(compare(config)).read_text())
            paths = [row["output"] for row in source_report["runs"]]
            results = [inspect_run(path) for path in paths]
            rows, configs = list(map(list, zip(*results)))
            check_paired_runs(rows, configs, [0])
            result_path = aggregate(dict(runs=paths, seeds=[0], output={"root": str(Path(directory) / "aggregate")}))
            result = json.loads(Path(result_path).read_text())
            self.assertEqual(len(result["runs"]), 2)
            self.assertIn("mamba_minus_transformer", result["aggregate"]["scenarios"]["masked"])

            altered = deepcopy(rows)
            altered[1]["training_mask_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "training_mask_sha256"):
                check_paired_runs(altered, configs, [0])
            altered = deepcopy(rows)
            altered[1]["validation_mask_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "validation_mask_sha256"):
                check_paired_runs(altered, configs, [0])
            altered = deepcopy(rows)
            altered[1]["shared_initialization_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "shared_initialization_sha256"):
                check_paired_runs(altered, configs, [0])
            altered_configs = deepcopy(configs)
            altered_configs[1]["dataset_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "dataset_sha256"):
                check_paired_runs(rows, altered_configs, [0])

            sidecar = json.loads((Path(paths[0]) / "best.pt.tracking.json").read_text())
            metadata_path = Path(sidecar["experiment_directory"]) / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["status"] = "failed"
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "unfinished, failed"):
                inspect_run(paths[0])


if __name__ == "__main__":
    unittest.main()
