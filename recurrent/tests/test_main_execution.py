"""Main-run integrity and orchestration with synthetic files and child processes.

No real training, simulator, CUDA, W&B transport, or publishing is performed.
"""
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from recurrent.common import RAW_KEYS, environment_record, load_config, versions
from recurrent.execution import Children, atomic_json, lock_environment, writer_lock
from recurrent.integrity import (bind_identity, evaluation_complete, finish_evaluation,
    identity_hash, read_eval_rows, training_complete, validate_demos)
from recurrent.run_experiments import (bind_experiment, eval_cmd, eval_identity, eval_jobs,
    expected_rows, experiment_jobs, job_identity, validate_results)


def demos(path, count=100):
    meta = {"source": "expert", "args": dict(num_demos=count, start_seed=0, noise=0.02,
            perturb_frac=0.5, horizon=250), "environment": environment_record(),
            "versions": versions(), "expert_success": [count, count], "shift_plans": {str(s): None for s in range(count)}}
    np.savez(path, lengths=np.ones(count, dtype=np.int64), seeds=np.arange(count),
             act=np.zeros((count, 7), np.float32), perturbed=np.zeros(count, bool),
             meta=np.array(json.dumps(meta)),
             **{f"raw_{k}": np.zeros((count, d), np.float32) for k, d in RAW_KEYS.items()})


def fixture(path, seeds=range(2000, 2030)):
    dims = {"qpos": 9, "qvel": 9, "ee_pose": 7, "left_finger_pose": 7,
            "right_finger_pose": 7, "left_fingertip_pos": 3, "right_fingertip_pos": 3,
            "peg_pose": 7, "box_pose": 7, "hole_pose": 7, "gripper_qpos": 2,
            "peg_half_sizes": 3, "hole_radius": None, "success": None}
    gt = {k: torch.zeros((1, d) if d else (1,)) for k, d in dims.items()}
    states = {s: {"state": {"actor": torch.zeros(1, 13)}, "gt": gt,
                  "hole_offset": torch.zeros(1, 7)} for s in seeds}
    torch.save({"source": "legacy physx_cpu reset(seed)", "seeds": list(seeds),
                "environment": environment_record(), "versions": versions(), "episodes": states}, path)


def config(root):
    cfg = load_config("experiment.yaml")
    cfg.update(data=str(root / "data.npz"), out_dir=str(root / "out"),
               models=["mlp"], levels=["none"], seeds=[0])
    cfg["train"].update(steps=2, log_every=1)
    cfg["eval"].update(episodes=2, cross_eval_clean=False, stress={})
    return cfg


def trained(cfg, name="mlp_none_s0"):
    identity = job_identity(cfg, "mlp", "none", 0)
    run = Path(cfg["out_dir"]) / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    atomic_json(run / "identity.json", identity)
    history = [{"step": i, "val_optimization_loss": float(i), "val_action_mse_z": float(i),
                "val_command_mse_6d": float(i), "elapsed_s": float(i)} for i in (1, 2)]
    c = dict(model="mlp", train_level="none", seed=0, device="cuda:0")
    payload = dict(identity=identity, kind="mlp", cfg=identity["model_config"], train_level="none",
                   seed=0, config=c, state_dict={"weight": torch.zeros(1)}, step=1,
                   val={"optimization_loss": 1.0, "action_mse_z": 1.0, "command_mse_6d": 1.0})
    torch.save(payload, run / "best.pt")
    torch.save({**payload, "step": 2, "val": history[-1]}, run / "last.pt")
    atomic_json(run / "history.json", dict(identity=identity, config=c, history=history,
        summary=dict(best_step=1, best_val_loss=1.0, final_val_action_mse_z=2.0,
                     final_val_command_mse_6d=2.0, params=1, train_time_s=2.0)))
    return run, identity


def row(identity, seed=2000, name="mlp_none_s0", model="mlp", cond="normal"):
    return dict(ckpt=name, model=model, train_level="none", train_seed=None if model == "expert" else 0,
                condition=cond, eval_level="none", episode_seed=seed, success=False, steps=250,
                shift_applied=False, mean_action_delta=0.0,
                mean_infer_ms=None if identity["backend"] == "physx_cuda" else 0.0,
                acquisition=False, transport=False, alignment=False, failure_stage="acquisition",
                evaluation_identity=identity_hash(identity), sim_backend=identity["backend"],
                initial_state_source=identity["initial_states"])


def evaluated(cfg, ckpt, level="none", backend="physx_cpu", initial=None, name="mlp_none_s0"):
    from recurrent.evaluate import summarize
    identity = eval_identity(cfg, ckpt, level, backend, initial)
    path = Path(cfg["out_dir"]) / "eval" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path.with_suffix(".identity.json"), identity)
    rows = [row(identity, s, name, "expert" if ckpt == "expert" else "mlp", cond)
            for cond, _, _ in identity["jobs"] for s in range(2000, 2000 + cfg["eval"]["episodes"])]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    finish_evaluation(path, identity, {"summary": summarize(rows)})
    return path, identity


class IntegrityTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = config(self.root)
        demos(self.root / "data.npz")

    def test_eight_shards_and_exact_main_totals(self):
        cfg = load_config("experiment.yaml")
        all_jobs = experiment_jobs(cfg)
        shards = [experiment_jobs(cfg, i, 8) for i in range(8)]
        self.assertEqual([len(s) for s in shards], [11] * 4 + [10] * 4)
        self.assertEqual(len(set().union(*map(set, shards))), 84)
        self.assertEqual(sum(map(len, shards)), 84)
        self.assertEqual(sum(expected_rows(cfg, l) for _, _, l, _ in all_jobs), 11340)
        self.assertEqual(expected_rows(cfg, "none", expert=True), 60)
        for _, _, level, _ in all_jobs:
            self.assertEqual(len(eval_jobs(cfg, level)) * 30, expected_rows(cfg, level))

    def test_data_shape_finite_metadata_and_seed_validation(self):
        path = self.root / "data.npz"
        validate_demos(path, num_demos=100)
        from recurrent.train import load_episodes
        episodes, metadata = load_episodes(path)
        self.assertEqual(len(episodes), 100)
        self.assertEqual(metadata["source"], "expert")
        with np.load(path) as source:
            original = dict(source)
        changes = [dict(act=np.zeros((99, 7))), dict(raw_tcp_pose=np.full((100, 7), np.nan)),
                   dict(seeds=np.zeros(100, dtype=int)), dict(meta=np.array('{}')),
                   dict(lengths=np.ones(99, dtype=int))]
        for change in changes:
            np.savez(path, **{**original, **change})
            with self.assertRaises((ValueError, KeyError)):
                validate_demos(path, num_demos=100)

    def test_completed_training_reuse_incomplete_restart_and_identity_refusal(self):
        run, identity = trained(self.cfg)
        self.assertTrue(training_complete(run, identity))
        from recurrent.train import train, get_parser
        from recurrent.run_experiments import train_cmd
        args = get_parser().parse_args(train_cmd(self.cfg, "mlp", "none", 0, run)[3:])
        with patch("recurrent.train._train") as implementation:
            train(args)
            implementation.assert_not_called()
        best = (run / "best.pt").read_bytes()
        (run / "last.pt").unlink()
        self.assertFalse(training_complete(run, identity))
        with patch("recurrent.train._train", return_value=run) as implementation:
            train(args)
            implementation.assert_called_once()
        self.assertEqual(next((run / "incomplete").glob("*/best.pt")).read_bytes(), best)
        changed = copy.deepcopy(identity)
        changed["optimizer"]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "mismatch"):
            training_complete(run, changed)
        (run / "identity.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            training_complete(run, identity)

    def test_root_identity_rejects_changed_config_and_data(self):
        bind_experiment(self.cfg)
        changed = copy.deepcopy(self.cfg)
        changed["train"]["batch"] += 1
        with self.assertRaisesRegex(ValueError, "mismatch"):
            bind_experiment(changed)
        path = self.root / "data.npz"
        with path.open("ab") as stream:
            stream.write(b"different data hash")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            bind_experiment(self.cfg)

    def test_concurrent_shards_serialize_the_shared_identity_snapshot(self):
        from concurrent.futures import ThreadPoolExecutor
        from recurrent.execution import atomic_json as real_write
        def delayed_write(path, value):
            if Path(path).name == "applied_config.json":
                time.sleep(0.02)
            real_write(path, value)
        with patch("recurrent.execution.atomic_json", side_effect=delayed_write):
            with ThreadPoolExecutor(max_workers=8) as pool:
                identities = list(pool.map(lambda _: bind_experiment(self.cfg), range(8)))
        self.assertTrue(all(value == identities[0] for value in identities))
        path = Path(self.cfg["out_dir"]) / "applied_config.json"
        self.assertEqual(json.loads(path.read_text())["identity"], identities[0])

    def test_fixture_reuse_and_version_seed_environment_refusal(self):
        from recurrent.initial_states import collect, validate
        path = self.root / "initial.pt"
        fixture(path)
        before = path.read_bytes()
        with patch("recurrent.initial_states.make_env") as factory:
            collect(path, range(2000, 2030))
            factory.assert_not_called()
        self.assertEqual(before, path.read_bytes())
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate(path, range(2001, 2031))
        value = torch.load(path, weights_only=False)
        value["versions"]["mani_skill"] = "other"
        torch.save(value, path)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            collect(path, range(2000, 2030))

    def test_jsonl_tail_repair_newline_duplicates_and_corruption(self):
        run, _ = trained(self.cfg)
        identity = eval_identity(self.cfg, run / "best.pt", "none")
        path = self.root / "partial.jsonl"
        valid = json.dumps(row(identity))
        path.write_text(valid)
        self.assertEqual(len(read_eval_rows(path, identity, repair=True)), 1)
        self.assertTrue(path.read_bytes().endswith(b"\n"))
        path.write_text(valid + '\n{"episode_seed":20')
        original = path.read_bytes()
        self.assertEqual(len(read_eval_rows(path, identity, repair=True)), 1)
        self.assertEqual(next(self.root.glob("*.bak")).read_bytes(), original)
        self.assertEqual(path.read_text(), valid + "\n")
        for fragment in ('{"value": 0.', '{"value": 1e-', '{"value": tru', r'{"value": "\u12'):
            path.write_text(valid + "\n" + fragment)
            self.assertEqual(len(read_eval_rows(path, identity, repair=True)), 1)
            self.assertEqual(path.read_text(), valid + "\n")
        for text in (valid + "\n" + valid + "\n", '{broken}\n' + valid + "\n",
                     valid + "\ngarbage", json.dumps({**row(identity), "episode_seed": 100}) + "\n",
                     json.dumps({**row(identity), "success": "false"}) + "\n"):
            path.write_text(text)
            with self.assertRaises(ValueError):
                read_eval_rows(path, identity, repair=True)
            self.assertEqual(path.read_text(), text)

    def test_complete_results_still_validate_checkpoint_identity_and_exact_set(self):
        run, _ = trained(self.cfg)
        path, identity = evaluated(self.cfg, run / "best.pt")
        self.assertTrue(evaluation_complete(path, identity))
        changed = copy.deepcopy(identity)
        changed["backend"] = "physx_cuda"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            evaluation_complete(path, changed)
        path.write_text(path.read_text() + json.dumps(row(identity)) + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluation_complete(path, identity)

    def test_partial_evaluation_rejects_foreign_summary_before_repair(self):
        run, _ = trained(self.cfg)
        path, identity = evaluated(self.cfg, run / "best.pt")
        path.write_text(json.dumps(row(identity)) + '\n{"torn":')
        original = path.read_bytes()
        summary_path = path.with_suffix(".summary.json")
        summary = json.loads(summary_path.read_text())
        summary["evaluation_identity"] = "foreign"
        atomic_json(summary_path, summary)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            evaluation_complete(path, identity, repair=True)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(list(path.parent.glob("*.bak")))

    def test_best_checkpoint_report_ignores_unplanned_files_and_counts_exactly(self):
        from recurrent.report import load_histories, load_rows, fig_offline_online
        bind_experiment(self.cfg)
        run, _ = trained(self.cfg)
        evaluated(self.cfg, run / "best.pt")
        evaluated(self.cfg, "expert", name="expert")
        counts = validate_results(self.cfg)
        self.assertEqual(counts, dict(training_jobs=1, policy_episodes=4, expert_episodes=4))
        exp = Path(self.cfg["out_dir"])
        (exp / "eval" / "unplanned.jsonl").write_text("bad data")
        (exp / "runs" / "unplanned").mkdir()
        (exp / "runs" / "unplanned/history.json").write_text("bad data")
        histories = load_histories(exp, self.cfg)
        self.assertEqual(len(histories), 1)
        self.assertEqual(histories[0]["evaluated_val_action_mse_z"], 1)
        self.assertEqual(histories[0]["final_val_action_mse_z"], 2)
        self.assertEqual(len(load_rows(exp, self.cfg)), 8)
        (run / "best.pt").unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete training"):
            validate_results(self.cfg)

    def test_eval_only_no_training_and_correct_policy_expert_arguments(self):
        from recurrent.run_experiments import main
        path = self.root / "cfg.json"
        path.write_text(json.dumps(self.cfg))
        argv = ["run", "--config", str(path), "--eval-only", "--no-report"]
        with patch.object(sys, "argv", argv), patch("recurrent.run_experiments.execute_queue") as queue:
            with self.assertRaisesRegex(ValueError, "completed training"):
                main()
            queue.assert_not_called()
        run, _ = trained(self.cfg)
        initial = self.root / "initial.pt"
        fixture(initial, range(2000, 2002))
        calls = []
        def capture(tasks, workers, children, exp, kind, verify):
            calls.extend((kind, task) for task in tasks)
        args = argv + ["--sim-backend", "physx_cuda", "--num-envs", "7", "--initial-states", str(initial)]
        with patch.object(sys, "argv", args), patch.dict(os.environ, CUDA_VISIBLE_DEVICES="5"), \
             patch("recurrent.run_experiments.execute_queue", side_effect=capture):
            main()
        self.assertEqual([kind for kind, _ in calls], ["eval", "eval"])
        for kind, (name, cmd, env) in calls:
            def option(flag): return cmd[cmd.index(flag) + 1]
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "5")
            self.assertEqual(option("--initial-states"), str(initial))
            self.assertEqual(option("--episodes"), "2")
            self.assertEqual(option("--sim-backend"), "physx_cpu" if name == "expert" else "physx_cuda")
            self.assertEqual(option("--num-envs"), "1" if name == "expert" else "7")

    def test_mocked_gpu_batch_resume_changes_size_without_new_episodes(self):
        from types import SimpleNamespace
        from recurrent.evaluate import evaluate_gpu, verify_resume
        from recurrent.common import NullLogger
        self.cfg["eval"]["episodes"] = 5
        run, _ = trained(self.cfg)
        initial = self.root / "initial.pt"
        fixture(initial, range(2000, 2005))
        path = self.root / "gpu.jsonl"
        jobs = [("normal", "none", {})]
        identity = verify_resume(path, run / "best.pt", jobs, 5, 2000, 250,
                                 "corruption.yaml", "physx_cuda", initial)
        path.write_text(json.dumps(row(identity)) + "\n")
        env = SimpleNamespace(device="cuda:0", backend=SimpleNamespace(
            sim_backend="physx_cuda", render_backend="none"), close=lambda: None)
        env.unwrapped = env
        meta = torch.load(run / "best.pt", weights_only=False)
        learned = SimpleNamespace(ckpt=meta, name="mlp")
        def batch(env, policy, seeds, *args):
            return [row(identity, seed) for seed in seeds], dict(reset_s=0, rollout_s=0)
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.reset_peak_memory_stats"), \
             patch("torch.cuda.max_memory_allocated", return_value=0), \
             patch("envs.maniskill.make_peg_insertion_env", return_value=env) as factory, \
             patch("mani_skill.utils.registration.TimeLimitWrapper", side_effect=lambda e, h: e), \
             patch("recurrent.evaluate.LearnedPolicy", return_value=learned), \
             patch("recurrent.gpu_eval.TensorPolicy", return_value=object()), \
             patch("recurrent.gpu_eval.run_batch", side_effect=batch) as rollout, \
             patch("recurrent.evaluate.experiment_logger", return_value=NullLogger()):
            evaluate_gpu(run / "best.pt", jobs, 5, 2000, 250, path, None,
                         "corruption.yaml", lambda _: None, 3, initial)
            self.assertEqual([c.args[2] for c in rollout.call_args_list], [[2001, 2002, 2003], [2004]])
            self.assertEqual([c.kwargs["num_envs"] for c in factory.call_args_list], [3, 1])
            before = path.read_bytes()
            evaluate_gpu(run / "best.pt", jobs, 5, 2000, 250, path, None,
                         "corruption.yaml", lambda _: None, 2, initial)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(rollout.call_count, 2)
        self.assertTrue(evaluation_complete(path, identity))

    def test_report_renders_only_verified_synthetic_results(self):
        from recurrent.report import main
        bind_experiment(self.cfg)
        run, _ = trained(self.cfg)
        evaluated(self.cfg, run / "best.pt")
        evaluated(self.cfg, "expert", name="expert")
        path = self.root / "cfg.json"
        path.write_text(json.dumps(self.cfg))
        with patch.object(sys, "argv", ["report", "--config", str(path), "--latency", str(self.root / "absent.json")]):
            main()
        report = Path(self.cfg["out_dir"]) / "report"
        text = (report / "report.md").read_text()
        self.assertIn("best.pt", text)
        self.assertIn("physx_cpu", text)
        self.assertIn("cuda:0", text)
        import csv
        with (report / "episodes.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 8)

    def test_mutually_exclusive_modes_and_single_gpu_evaluator_constraint(self):
        from recurrent.run_experiments import main
        for args in (["--train-only", "--eval-only"],
                     ["--sim-backend", "physx_cuda", "--eval-workers", "2", "--initial-states", "fake.pt"]):
            with patch.object(sys, "argv", ["run", *args]), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)

    def test_main_rejects_invalid_environment_before_children_start(self):
        from recurrent.main_multi_gpu import main
        for settings in (dict(EVAL_WORKERS="2"), dict(EVAL_WORKERS="1", EVAL_NUM_ENVS="0")):
            with patch.dict(os.environ, settings), patch.object(sys, "argv", ["main"]), \
                 patch("recurrent.main_multi_gpu.run_pipeline") as run, self.assertRaises(SystemExit):
                main()
            run.assert_not_called()


FAKE = r'''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
module = args[1]
def option(flag): return args[args.index(flag) + 1]
root = Path(os.environ["FAKE_ROOT"])
def event(action):
    with (root / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(action=action, module=module, args=args,
                     gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))) + "\n")
event("start")
if module == "recurrent.run_experiments":
    index = option("--shard-index")
    if "--train-only" in args:
        if index == os.environ.get("FAKE_FAIL"):
            raise SystemExit(3)
        time.sleep(0.03 * (8 - int(index)))
    else:
        assert "--eval-only" in args
        assert option("--sim-backend") == "physx_cuda"
        assert option("--num-envs") == "7"
        assert option("--initial-states").endswith("initial_states.pt")
    (root / (("train" if "--train-only" in args else "eval") + index)).touch()
event("finish")
'''


class OrchestrationTest(unittest.TestCase):
    def pipeline(self, fail=None):
        from recurrent.main_multi_gpu import run_pipeline
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        cfg = load_config("experiment.yaml")
        cfg.update(out_dir=str(root / "out"), data=str(root / "data.npz"))
        path = root / "main.json"
        path.write_text(json.dumps(cfg))
        runner = root / "fake-python"
        runner.write_text(f"#!{sys.executable}\n" + FAKE)
        runner.chmod(0o755)
        phases = []
        def validate(cfg, *args, **kwargs):
            self.assertEqual(len(list(root.glob("train[0-9]"))), 8)
            if not kwargs.get("training_only"):
                self.assertEqual(len(list(root.glob("eval[0-9]"))), 8)
            phases.append(kwargs.get("training_only", False))
            return dict(training_jobs=84, policy_episodes=11340, expert_episodes=60)
        env = dict(FAKE_ROOT=str(root), FAKE_FAIL=fail or "")
        with patch.dict(os.environ, env), patch("recurrent.main_multi_gpu.validate_demos"), \
             patch("recurrent.main_multi_gpu.bind_experiment", return_value={}), \
             patch("recurrent.main_multi_gpu.validate_results", side_effect=validate):
            if fail:
                with self.assertRaisesRegex(RuntimeError, "FAIL"):
                    run_pipeline(path, [str(i) for i in range(8)], 7, str(runner))
            else:
                run_pipeline(path, [str(i) for i in range(8)], 7, str(runner))
        events = [json.loads(l) for l in (root / "events.jsonl").read_text().splitlines()]
        return root, cfg, events, phases

    def test_all_training_precedes_evaluation_eight_gpu_shards(self):
        root, cfg, events, phases = self.pipeline()
        training = [e for e in events if "--train-only" in e["args"]]
        evaluation = [e for e in events if "--eval-only" in e["args"]]
        self.assertEqual((len(training), len(evaluation)), (16, 16))
        first_eval = next(i for i, e in enumerate(events) if "--eval-only" in e["args"])
        self.assertEqual(sum(e["action"] == "finish" for e in events[:first_eval] if "--train-only" in e["args"]), 8)
        self.assertEqual(phases, [True, False])
        assigned = []
        for e in training:
            if e["action"] == "start":
                index = int(e["args"][e["args"].index("--shard-index") + 1])
                self.assertEqual(e["gpu"], str(index))
                assigned += experiment_jobs(cfg, index, 8)
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(len(assigned), 84)
        self.assertTrue(all(e["gpu"] == "0" for e in events if e["module"] in ("recurrent.collect", "recurrent.initial_states")))
        self.assertTrue((root / "out/verified_results.json").exists())

    def test_failed_child_prevents_eval_report_and_verification(self):
        root, cfg, events, phases = self.pipeline("0")
        self.assertFalse(any("--eval-only" in e["args"] or e["module"] == "recurrent.report" for e in events))
        self.assertEqual(phases, [])
        self.assertFalse((root / "out/verified_results.json").exists())

    def test_lock_survives_parent_close_until_child_exits(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "lock"
            with writer_lock(path) as fd:
                env = lock_environment(fd)
                proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                        env=env, pass_fds=(fd,))
            try:
                with self.assertRaisesRegex(RuntimeError, "Another writer"):
                    with writer_lock(path): pass
            finally:
                proc.terminate()
                proc.wait(timeout=5)
            with writer_lock(path): pass

    def test_failure_kills_descendants_in_owned_process_group(self):
        with tempfile.TemporaryDirectory() as d:
            pidfile = Path(d) / "pid"
            code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); open(sys.argv[1],'w').write(str(p.pid)); time.sleep(30)"
            with self.assertRaises(RuntimeError):
                with Children() as children:
                    parent = children.start([sys.executable, "-c", code, str(pidfile)])
                    deadline = time.monotonic() + 5
                    while not pidfile.exists() and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(pidfile.exists())
                    raise RuntimeError("synthetic failure")
            pid = int(pidfile.read_text())
            stat = Path(f"/proc/{pid}/stat")
            self.assertTrue(not stat.exists() or stat.read_text().split()[2] == "Z")
            self.assertIsNotNone(parent.returncode)

    def test_sigint_and_sigterm_reap_shard_descendants_and_release_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sig in (signal.SIGINT, signal.SIGTERM):
                lock = root / f"lock-{sig}"
                ready = root / f"ready-{sig}"
                # The simulated shard reaps its writer when the launcher signals the group.
                shard = """import sys,time
from recurrent.execution import Children
with Children(groups=False) as children:
    writer=children.start([sys.executable,'-c','import time; time.sleep(60)'])
    open(sys.argv[1],'w').write(str(writer.pid))
    time.sleep(60)
"""
                launcher = """import sys,time
from recurrent.execution import Children,writer_lock,lock_environment
with writer_lock(sys.argv[1]) as fd:
    with Children() as children:
        shard=children.start([sys.executable,'-c',sys.argv[3],sys.argv[2]],env=lock_environment(fd))
        time.sleep(60)
"""
                proc = subprocess.Popen([sys.executable, "-c", launcher, str(lock), str(ready), shard],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    deadline = time.monotonic() + 5
                    while (not ready.exists() or not ready.read_text()) and time.monotonic() < deadline:
                        time.sleep(0.02)
                    self.assertTrue(ready.exists())
                    with self.assertRaisesRegex(RuntimeError, "Another writer"):
                        with writer_lock(lock): pass
                    pid = int(ready.read_text())
                    proc.send_signal(sig)
                    stdout, stderr = proc.communicate(timeout=8)
                    self.assertEqual(proc.returncode, 128 + sig, stdout + stderr)
                    self.assertFalse(Path(f"/proc/{pid}").exists())
                    with writer_lock(lock): pass
                finally:
                    if proc.poll() is None:
                        proc.kill()
                        proc.communicate(timeout=5)

    def test_shell_forwards_main_config_and_setup_environment(self):
        import shutil
        repository = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "recurrent").mkdir()
            (root / ".venv/bin").mkdir(parents=True)
            (root / "outputs/recurrent").mkdir(parents=True)
            shutil.copyfile(repository / "recurrent/run_main_multi_gpu.sh", root / "recurrent/run_main_multi_gpu.sh")
            (root / "outputs/recurrent/env.sh").write_text("export SETUP_MARKER=loaded\n")
            runner = root / ".venv/bin/python"
            runner.write_text(f"#!{sys.executable}\nimport json,os,sys; print(json.dumps([sys.argv[1:],os.environ['SETUP_MARKER']]))\n")
            runner.chmod(0o755)
            result = subprocess.run(["bash", "recurrent/run_main_multi_gpu.sh"], cwd=root,
                                    env=dict(os.environ, MAIN_CONFIG="custom config.yaml"), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), [["-m", "recurrent.main_multi_gpu", "--config", "custom config.yaml"], "loaded"])


if __name__ == "__main__":
    unittest.main()
