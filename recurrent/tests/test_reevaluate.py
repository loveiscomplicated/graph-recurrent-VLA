"""Restored evaluation planning and provenance; no training or simulator runs."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from recurrent.common import sha256
from recurrent.execution import atomic_json
from recurrent.reevaluate import main, prepare, verify_task
from recurrent.run_experiments import bind_experiment
from recurrent.tests.test_main_execution import config, demos, evaluated, fixture, trained


class ReevaluateTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.cfg = config(self.source)
        demos(self.source / "data.npz")
        bind_experiment(self.cfg)
        self.run, _ = trained(self.cfg)
        self.cfg.update(data="data.npz", out_dir="out")
        self.out = self.root / "reevaluation"

    def test_new_environment_keeps_source_and_exact_commands(self):
        path = self.source / "out/identity.json"
        original = json.loads(path.read_text())
        original["versions"]["torch"] = "original-torch-version"
        atomic_json(path, original)
        before = {p: sha256(p) for p in self.source.rglob("*") if p.is_file()}
        identity, tasks, fixture = prepare(self.cfg, self.source, self.out, 30)
        self.assertEqual(identity["source_versions"]["torch"], "original-torch-version")
        self.assertNotEqual(identity["evaluation_versions"]["torch"], "original-torch-version")
        self.assertEqual(identity["checkpoint_sha256"]["mlp_none_s0"], sha256(self.run / "best.pt"))
        self.assertEqual([t["backend"] for t in tasks], ["physx_cuda", "physx_cpu"])
        self.assertEqual([t["expected_rows"] for t in tasks], [4, 4])
        self.assertTrue(all(t["command"][1:3] == ["-m", "recurrent.evaluate"] for t in tasks))
        self.assertEqual(fixture, self.out / "initial_states.pt")
        self.assertEqual(before, {p: sha256(p) for p in self.source.rglob("*") if p.is_file()})

    def test_refuses_training_data_or_config_mismatch_and_source_output(self):
        changed = copy.deepcopy(self.cfg)
        changed["train"]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            prepare(changed, self.source, self.out, 30)
        with self.assertRaisesRegex(ValueError, "separate"):
            prepare(self.cfg, self.source, self.source / "new_eval", 30)
        (self.run / "last.pt").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete source training"):
            prepare(self.cfg, self.source, self.out, 30)

    def test_main_plan_does_not_start_children_and_batch_can_change(self):
        argv = ["reevaluate", "--source-root", str(self.source), "--out-dir", str(self.out)]
        with patch.object(sys, "argv", argv), patch("recurrent.reevaluate.load_config", return_value=self.cfg), \
                patch("recurrent.reevaluate.Children") as children:
            main()
            children.assert_not_called()
        self.assertFalse((self.out / "initial_states.pt").exists())
        first = json.loads((self.out / "reevaluation.identity.json").read_text())
        with patch.object(sys, "argv", argv + ["--num-envs", "16"]), \
                patch("recurrent.reevaluate.load_config", return_value=self.cfg):
            main()
        self.assertEqual(first, json.loads((self.out / "reevaluation.identity.json").read_text()))

    def test_verified_completion_requires_matching_labels(self):
        _, tasks, _ = prepare(self.cfg, self.source, self.out, 30)
        cfg = copy.deepcopy(self.cfg)
        cfg["out_dir"] = str(self.out)
        initial = self.root / "initial.pt"
        fixture(initial, seeds=range(2000, 2002))
        path, _ = evaluated(cfg, self.run / "best.pt", initial=initial)
        task = {**tasks[0], "backend": "physx_cpu", "output": str(path)}
        self.assertTrue(verify_task(cfg, task, initial))
        with self.assertRaisesRegex(ValueError, "labels/count mismatch"):
            verify_task(cfg, {**task, "model": "gru"}, initial)

    def test_execute_uses_separate_evaluation_commands_and_resumes(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["out_dir"] = str(self.out)
        initial = self.out / "initial_states.pt"

        def child(cmd, **kwargs):
            if cmd[2] == "recurrent.initial_states":
                if not initial.exists():
                    fixture(initial, seeds=range(2000, 2002))
            else:
                self.assertEqual(cmd[2], "recurrent.evaluate")
                ckpt = cmd[cmd.index("--ckpt") + 1]
                evaluated(cfg, ckpt, backend=cmd[cmd.index("--sim-backend") + 1],
                          initial=initial, name=cmd[cmd.index("--name") + 1])

        argv = ["reevaluate", "--source-root", str(self.source),
                "--out-dir", str(self.out), "--execute"]
        with patch.object(sys, "argv", argv), patch("recurrent.reevaluate.load_config", return_value=self.cfg), \
                patch("recurrent.reevaluate.Children") as children:
            runner = children.return_value.__enter__.return_value.run
            runner.side_effect = child
            main()
            self.assertEqual(runner.call_count, 3)
            self.assertEqual([c.args[0][2] for c in runner.call_args_list],
                             ["recurrent.initial_states", "recurrent.evaluate", "recurrent.evaluate"])
            before = {p: p.read_bytes() for p in (self.out / "eval").glob("*")}
            runner.reset_mock()
            main()
            self.assertEqual(runner.call_count, 1)  # fixture validation only
            self.assertEqual(before, {p: p.read_bytes() for p in (self.out / "eval").glob("*")})
        self.assertTrue((self.out / "reevaluation.complete.json").exists())


if __name__ == "__main__":
    unittest.main()
