"""Contract tests for Graph's seed-specific joint training configs."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from main import load_config, run
from training.gnn.temporal.config import resolve_joint_job_config as resolve_config


class JointConfigTests(unittest.TestCase):
    def test_all_seed_configs_resolve_to_joint_trainer(self):
        directory = Path("graph/configs/temporal/insertion")
        for architecture in ("mamba", "transformer"):
            for seed in range(3):
                with self.subTest(architecture=architecture, seed=seed):
                    job = load_config(directory / f"train_joint_{architecture}_{seed}.yaml")
                    effective = resolve_config(job)
                    self.assertEqual(effective["trainer"], "training.gnn.temporal.joint_finetuning:train")
                    self.assertEqual(effective["model"]["architecture"], architecture)
                    self.assertEqual(effective["seed"], seed)
                    self.assertEqual(effective["checkpoint"], job["checkpoint"])
                    self.assertEqual(effective["temporal_checkpoint"], job["temporal_checkpoint"])
                    self.assertEqual(effective["output"], job["output"])

    def test_root_entrypoint_passes_full_config_to_existing_trainer(self):
        path = Path("graph/configs/temporal/insertion/train_joint_mamba_0.yaml")
        expected = resolve_config(load_config(path))
        with patch("training.gnn.temporal.joint_finetuning.train") as trainer:
            run(path)
        trainer.assert_called_once_with(expected)

    def test_legacy_joint_job_entrypoint_passes_the_same_resolved_config(self):
        from training.gnn.temporal import joint_config, joint_finetuning

        self.assertIs(joint_config.resolve_config, resolve_config)
        self.assertIs(joint_config.train, joint_finetuning.train_from_base)
        job = load_config("graph/configs/temporal/insertion/train_joint_mamba_0.yaml")
        job["trainer"] = "training.gnn.temporal.joint_config:train"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.yaml"
            path.write_text(yaml.safe_dump(job))
            with patch("training.gnn.temporal.joint_finetuning.train") as trainer:
                run(path)
            trainer.assert_called_once_with(resolve_config(job))

    def test_rejects_unexpected_or_conflicting_settings(self):
        job = load_config("graph/configs/temporal/insertion/train_joint_mamba_0.yaml")
        with self.assertRaisesRegex(ValueError, "exactly"):
            resolve_config({**job, "training": {"epochs": 1}})
        with tempfile.TemporaryDirectory() as directory:
            base_path = Path(directory) / "base.yaml"
            base_path.write_text(yaml.safe_dump({
                "trainer": "training.gnn.temporal.joint_finetuning:train",
                "seed": 99,
            }))
            with self.assertRaisesRegex(ValueError, "seed-specific"):
                resolve_config({**job, "base_config": str(base_path)})

    def test_legacy_mamba_entrypoint_resolves_to_same_functions(self):
        from training.gnn.temporal import selected_estimator, selected_mamba

        self.assertIs(selected_mamba.train_and_evaluate, selected_estimator.train_and_evaluate)
        self.assertIs(selected_mamba.aggregate, selected_estimator.aggregate)


class EarlyNodeJointTests(unittest.TestCase):
    def test_seed_recipes_keep_the_matching_early_stopped_source(self):
        directory = Path('graph/configs/temporal/insertion')
        outputs = set()
        for seed in range(3):
            source = load_config(directory / f'train_node_token_early_stopping_{seed}.yaml')
            joint = load_config(directory / f'train_joint_early_node_token_mamba_{seed}.yaml')
            for key in ('seed', 'checkpoint', 'split_manifest', 'selected_m0', 'data', 'mode',
                        'model', 'loss', 'evaluation_selection'):
                self.assertEqual(joint[key], source[key], (seed, key))
            self.assertEqual(joint['training']['early_stopping'], source['training']['early_stopping'])
            self.assertEqual(joint['training']['epochs'], 500)
            self.assertIn(f'/node_token_early_stopping/training/seed_{seed}/', joint['temporal_checkpoint'])
            self.assertNotIn('continuation_checkpoint', joint)
            self.assertEqual(joint['trainer'], 'training.gnn.temporal.joint_finetuning:train_and_evaluate')
            outputs.add(joint['output']['root'])
        self.assertEqual(len(outputs), 3)

    def test_training_wrapper_evaluates_best_and_preserves_input_config(self):
        from training.gnn.temporal.joint_finetuning import train_and_evaluate
        config = {'output': {'root': 'training'}, 'comparison_output': 'evaluation'}
        with patch('training.gnn.temporal.joint_finetuning.train', return_value=Path('training/run')) as train:
            with patch('training.gnn.temporal.joint_finetuning.evaluate', return_value='finished') as evaluate:
                self.assertEqual(train_and_evaluate(config), 'finished')
        train.assert_called_once_with(config)
        evaluated = evaluate.call_args.args[0]
        self.assertEqual(evaluated['joint_checkpoint'], 'training/run/best.pt')
        self.assertEqual(evaluated['output']['root'], 'evaluation')
        self.assertEqual(config['output']['root'], 'training')

    def test_continuation_wrapper_still_requires_a_resume_checkpoint(self):
        from training.gnn.temporal.joint_finetuning import continue_and_evaluate
        with self.assertRaisesRegex(ValueError, 'continuation_checkpoint'):
            continue_and_evaluate({'comparison_output': 'evaluation'})


if __name__ == "__main__":
    unittest.main()
