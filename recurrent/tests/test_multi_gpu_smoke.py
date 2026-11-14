"""Exercise shell orchestration and failure checks without a simulator or GPU."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = r'''
import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if args[0] == "-":
    os.execv(sys.executable, [sys.executable, *args])
if args[0] == "-c":
    print(os.environ.get("FAKE_GPU_COUNT", "2"))
    raise SystemExit(0)

module = args[1]
def option(name):
    return args[args.index(name) + 1]

device = os.environ.get("CUDA_VISIBLE_DEVICES")
with open("calls.jsonl", "a") as stream:
    stream.write(json.dumps({"module": module, "args": args, "device": device,
                             "wandb_mode": os.environ.get("WANDB_MODE")}) + "\n")

if module in ("recurrent.collect", "recurrent.initial_states"):
    Path(option("--out")).write_text("fixture")
elif module == "recurrent.run_experiments":
    import yaml
    assert "--train-only" in args and "--no-report" in args
    cfg = yaml.safe_load(Path(option("--config")).read_text())
    root = Path(cfg["out_dir"])
    index, count = int(option("--shard-index")), int(option("--shard-count"))
    jobs = [(m, l, s) for l in cfg["levels"] for m in cfg["models"] for s in cfg["seeds"]]
    (root / "logs").mkdir(exist_ok=True)
    for m, l, s in jobs[index::count]:
        name = f"{m}_{l}_s{s}"
        run = root / "runs" / name
        run.mkdir(parents=True)
        for artifact in ("best.pt", "last.pt", "history.json"):
            (run / artifact).write_text("checkpoint")
        (root / "logs" / f"train_{name}.log").write_text("device cuda:0\n")
elif module == "recurrent.evaluate":
    if device == os.environ.get("FAKE_EVAL_FAIL_GPU"):
        print("simulated GPU evaluation failure")
        raise SystemExit(3)
    assert option("--sim-backend") == "physx_cuda"
    assert Path(option("--ckpt")).is_file()
    assert Path(option("--initial-states")).is_file()
    n = int(option("--episodes"))
    assert int(option("--num-envs")) == n
    start = int(option("--start-seed"))
    out = Path(option("--out"))
    seeds = list(range(start, start + n))
    row_seeds = seeds.copy()
    if device == os.environ.get("FAKE_DUPLICATE_GPU"):
        row_seeds[-1] = row_seeds[0]
    rows = [{"condition": "normal", "eval_level": "none", "episode_seed": s} for s in row_seeds]
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out.with_suffix(".identity.json").write_text(json.dumps({"backend": "physx_cuda"}))
    out.with_suffix(".summary.json").write_text(json.dumps({
        "performance": {"batches": [{"seeds": seeds}], "episodes_per_s": 1.0}}))
    directory = out.parent / f"tracking-{out.stem}"
    directory.mkdir()
    backend = "physx_cpu" if device == os.environ.get("FAKE_CPU_GPU") else "physx_cuda"
    (directory / "metadata.json").write_text(json.dumps({"status": "finished", "environment": {
        "sim_backend": backend, "render_backend": "none", "device": "cuda:0", "num_envs": n}}))
    out.with_name(out.name + ".tracking.json").write_text(json.dumps({
        "experiment_directory": str(directory.resolve())}))
else:
    raise AssertionError(module)
'''


class MultiGPUSmokeTest(unittest.TestCase):
    def run_smoke(self, devices="5,2", **overrides):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        (root / "recurrent/configs").mkdir(parents=True)
        (root / "outputs/recurrent").mkdir(parents=True)
        (root / ".venv/bin").mkdir(parents=True)
        for path in ("recurrent/smoke_multi_gpu.sh", "recurrent/configs/smoke.yaml"):
            shutil.copyfile(ROOT / path, root / path)
        (root / "outputs/recurrent/env.sh").write_text("")
        python = root / ".venv/bin/python"
        python.write_text(f"#!{sys.executable}\n" + RUNNER)
        python.chmod(0o755)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices, WANDB_MODE="disabled",
                   FAKE_GPU_COUNT=str(len(devices.split(","))))
        for key in ("EVAL_NUM_ENVS", "FAKE_EVAL_FAIL_GPU", "FAKE_DUPLICATE_GPU", "FAKE_CPU_GPU"):
            env.pop(key, None)
        env.update(overrides)
        result = subprocess.run(["bash", "recurrent/smoke_multi_gpu.sh"], cwd=root,
                                env=env, capture_output=True, text=True, timeout=30)
        calls = root / "calls.jsonl"
        calls = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return root, result, calls

    def test_default_runs_gpu_evaluation_on_each_selected_device(self):
        root, result, calls = self.run_smoke()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(next(root.glob("outputs/recurrent/exp/*/smoke_summary.json")).read_text())
        self.assertEqual((summary["training_jobs"], summary["evaluation_episodes"]), (14, 60))
        self.assertEqual([e["gpu"] for e in summary["evaluations"]], ["5", "2"])
        self.assertIn("Multi-GPU smoke PASS", result.stdout)
        cpu = [c for c in calls if c["module"] in ("recurrent.collect", "recurrent.initial_states")]
        self.assertEqual(len(cpu), 2)
        self.assertTrue(all(c["device"] == "5" for c in cpu))
        evaluations = [c for c in calls if c["module"] == "recurrent.evaluate"]
        self.assertEqual({c["device"] for c in evaluations}, {"5", "2"})
        outputs = [c["args"][c["args"].index("--out") + 1] for c in evaluations]
        self.assertEqual(len(set(outputs)), 2)

    def test_eight_devices_and_smaller_batch(self):
        root, result, calls = self.run_smoke("7,5,3,1,6,4,2,0", EVAL_NUM_ENVS="3")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(next(root.glob("outputs/recurrent/exp/*/smoke_summary.json")).read_text())
        self.assertEqual((summary["gpu_count"], summary["envs_per_gpu"],
                          summary["evaluation_episodes"]), (8, 3, 24))
        self.assertEqual(sum(c["module"] == "recurrent.run_experiments" for c in calls), 8)
        self.assertEqual(sum(c["module"] == "recurrent.evaluate" for c in calls), 8)

    def test_failed_worker_prevents_success(self):
        root, result, calls = self.run_smoke(FAKE_EVAL_FAIL_GPU="5")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU evaluation 0 FAIL", result.stderr)
        self.assertNotIn("Multi-GPU smoke PASS", result.stdout)
        self.assertFalse(list(root.glob("outputs/recurrent/exp/*/smoke_summary.json")))

    def test_actual_cpu_backend_prevents_success(self):
        _, result, _ = self.run_smoke(FAKE_CPU_GPU="2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not finish with GPU physics", result.stderr)
        self.assertNotIn("Multi-GPU smoke PASS", result.stdout)

    def test_duplicate_episode_prevents_success(self):
        _, result, _ = self.run_smoke(FAKE_DUPLICATE_GPU="2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete or duplicate", result.stderr)
        self.assertNotIn("Multi-GPU smoke PASS", result.stdout)

    def test_invalid_batch_rejected_before_work(self):
        _, result, calls = self.run_smoke(EVAL_NUM_ENVS="0")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
