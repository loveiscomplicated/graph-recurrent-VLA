"""Archive integrity and statistics used in exported reports; no simulator runs."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from recurrent.common import sha256
from recurrent.evaluate import summarize
from recurrent.execution import atomic_json
from recurrent.integrity import finish_evaluation, identity_hash
from recurrent.reevaluate import prepare
from recurrent.report import group_rate, paired_diff, spearman
from recurrent.report_sources import load_reevaluation
from recurrent.run_experiments import bind_experiment
from recurrent.tests.test_main_execution import config, demos, evaluated, fixture, trained


class StatisticsTest(unittest.TestCase):
    def test_sample_standard_deviation_over_training_seeds(self):
        rows = [dict(train_seed=s, success=value)
                for s, values in enumerate([[False, False], [False, True], [True, True]])
                for value in values]
        result = group_rate(rows)
        self.assertEqual(result["mean"], .5)
        self.assertEqual(result["std"], .5)
        self.assertEqual((result["k"], result["n"], result["seeds"]), (3, 6, 3))

    def test_spearman_uses_average_ranks_for_ties(self):
        self.assertAlmostEqual(spearman([1, 1, 2, 3], [0, 0, 1, 1]), 2*np.sqrt(2)/3)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))

    def test_paired_interval_resamples_both_crossed_seed_dimensions(self):
        a = [dict(train_seed=s, episode_seed=e, success=s == 0) for s in range(3) for e in range(30)]
        b = [{**r, "success": False} for r in a]
        result = paired_diff(a, b)
        self.assertAlmostEqual(result["diff"], 1/3)
        self.assertEqual((result["lo"], result["hi"]), (0, 1))
        self.assertEqual(result, paired_diff(a, b))
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_diff(a, b[:-1])
        with self.assertRaisesRegex(ValueError, "unique"):
            paired_diff(a + a[:1], b)


class RestoredReportTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        absolute = config(self.source)
        demos(self.source / "data.npz")
        bind_experiment(absolute)
        self.run, _ = trained(absolute)
        self.cfg = copy.deepcopy(absolute)
        self.cfg.update(data="data.npz", out_dir="out")
        self.out = self.root / "reevaluation"
        self.out.mkdir()
        self.identity, self.tasks, self.initial = prepare(self.cfg, self.source, self.out, 2)
        atomic_json(self.out / "reevaluation.identity.json", self.identity)
        atomic_json(self.out / "plan.json", dict(identity=self.identity, num_envs=2))
        fixture(self.initial, seeds=range(2000, 2002))
        evaluation_cfg = copy.deepcopy(self.cfg)
        evaluation_cfg["out_dir"] = str(self.out)
        for task in self.tasks:
            evaluated(evaluation_cfg, task["checkpoint"], backend=task["backend"],
                      initial=self.initial, name=task["name"])
        self.complete()

    def complete(self):
        atomic_json(self.out / "reevaluation.complete.json", dict(
            source_identity_sha256=self.identity["source_identity_sha256"],
            fixture_sha256=sha256(self.initial), policy_episodes=4, expert_episodes=4,
            completions={t["name"]: sha256(Path(t["output"]).with_suffix(".complete.json")) for t in self.tasks},
        ))

    def test_all_inputs_are_verified_without_mutation_or_unplanned_rows(self):
        (self.out / "eval/unplanned.jsonl").write_text("corrupt unplanned file")
        before = {p: sha256(p) for p in self.root.rglob("*") if p.is_file()}
        rows, source, provenance = load_reevaluation(self.cfg, self.source, self.out)
        self.assertEqual(len(rows), 8)
        self.assertEqual(source, self.source / "out")
        self.assertEqual(provenance["training_runs"], 1)
        self.assertEqual(before, {p: sha256(p) for p in self.root.rglob("*") if p.is_file()})

    def test_root_completion_corruption_is_rejected(self):
        path = self.out / "reevaluation.complete.json"
        value = json.loads(path.read_text())
        value["policy_episodes"] += 1
        atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, "completion mismatch"):
            load_reevaluation(self.cfg, self.source, self.out)

    def test_episode_or_summary_corruption_is_rejected(self):
        target = Path(self.tasks[0]["output"]).with_suffix(".summary.json")
        value = json.loads(target.read_text())
        value["summary"]["normal/none"]["successes"] = 99
        atomic_json(target, value)
        with self.assertRaisesRegex(ValueError, "Summary values mismatch"):
            load_reevaluation(self.cfg, self.source, self.out)

    def test_reporting_uses_archived_evaluator_versions(self):
        self.identity["evaluation_versions"]["torch"] = "archived-evaluator-version"
        atomic_json(self.out / "reevaluation.identity.json", self.identity)
        atomic_json(self.out / "plan.json", dict(identity=self.identity, num_envs=2))
        value = torch.load(self.initial, weights_only=False)
        value["versions"]["torch"] = "archived-evaluator-version"
        torch.save(value, self.initial)
        for task in self.tasks:
            path = Path(task["output"])
            identity_path = path.with_suffix(".identity.json")
            identity = json.loads(identity_path.read_text())
            identity.update(versions=self.identity["evaluation_versions"], initial_states=sha256(self.initial))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            for row in rows:
                row.update(evaluation_identity=identity_hash(identity), initial_state_source=sha256(self.initial))
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            atomic_json(identity_path, identity)
            finish_evaluation(path, identity, {"summary": summarize(rows)})
        self.complete()
        _, _, provenance = load_reevaluation(self.cfg, self.source, self.out)
        self.assertEqual(provenance["evaluation_versions"]["torch"], "archived-evaluator-version")


if __name__ == "__main__":
    unittest.main()
