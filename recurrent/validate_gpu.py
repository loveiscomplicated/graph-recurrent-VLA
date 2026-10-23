"""Authorized comparison; CPU correctness workers, isolated performance runs."""

import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pynvml

ROOT = Path("outputs/recurrent/gpu_validation")
ENV = {**os.environ, "WANDB_MODE": "disabled", "CUDA_VISIBLE_DEVICES": "0"}


def command(model, backend, batch, out, extra):
    return [
        sys.executable,
        "-m",
        "recurrent.evaluate",
        "--ckpt",
        str(ROOT / f"{model}_none_s0/best.pt"),
        "--sim-backend",
        backend,
        "--num-envs",
        str(batch),
        "--initial-states",
        str(ROOT / "initial.pt"),
        "--out",
        str(out),
        *extra,
    ]


def run(cmd):
    out = Path(cmd[cmd.index("--out") + 1])
    print("RUN", out, flush=True)
    before = out.read_bytes() if out.exists() else b""
    record_path = out.with_suffix(".process.json")
    previous = json.loads(record_path.read_text()) if record_path.exists() else None
    stop = threading.Event()
    peak = [0]

    def sample():
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        while not stop.is_set():
            peak[0] = max(peak[0], pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            stop.wait(0.1)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        with out.with_suffix(".log").open("a") as log:
            result = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=ENV, check=False
            )
    finally:
        stop.set()
        sampler.join()
    record = {
        "output": str(out),
        "command": cmd,
        "returncode": result.returncode,
        "process_s": time.perf_counter() - start,
        "peak_device_vram_bytes": peak[0],
    }
    record["completed_episodes_before"] = len(before.splitlines())
    record["new_episodes"] = (
        len(out.read_bytes().splitlines()) - len(before.splitlines())
        if out.exists()
        else 0
    )
    if (
        result.returncode == 0
        and previous
        and before == out.read_bytes()
        and previous["command"] == cmd
    ):
        previous["resume_verification_s"] = record["process_s"]
        record = previous
    record_path.write_text(json.dumps(record, indent=2))
    if result.returncode:
        raise RuntimeError(f"Failed: {out}")
    return record


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for level in ["none", "low", "mid", "high"]:
        for condition in ["normal", "hole_shift"]:
            jobs.append(
                command(
                    "gru",
                    "physx_cpu",
                    1,
                    ROOT / f"cpu_gru_{condition}_{level}.jsonl",
                    ["--conditions", condition, "--levels", level],
                )
            )
    for blackout in [0, 10, 20, 40]:
        jobs.append(
            command(
                "gru",
                "physx_cpu",
                1,
                ROOT / f"cpu_gru_blackout_{blackout}.jsonl",
                ["--conditions", "--blackout", str(blackout)],
            )
        )
    jobs.append(
        command(
            "mlp",
            "physx_cpu",
            1,
            ROOT / "mlp_physx_cpu_1.jsonl",
            ["--conditions", "normal"],
        )
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(run, jobs))
    # Only correctness CPU jobs run concurrently. All timing repeats are isolated.
    for backend, batch in [("physx_cpu", 1), ("physx_cuda", 16), ("physx_cuda", 30)]:
        if backend == "physx_cuda":
            for model in ["mlp", "gru"]:
                extra = ["--conditions", "normal"]
                if model == "gru":
                    extra += [
                        "hole_shift",
                        "--levels",
                        "none",
                        "low",
                        "mid",
                        "high",
                        "--blackout",
                        "0",
                        "10",
                        "20",
                        "40",
                    ]
                records.append(
                    run(
                        command(
                            model,
                            backend,
                            batch,
                            ROOT / f"{model}_{backend}_{batch}.jsonl",
                            extra,
                        )
                    )
                )
        for repeat in range(3):
            records.append(
                run(
                    command(
                        "gru",
                        backend,
                        batch,
                        ROOT / f"perf_{backend}_{batch}_{repeat}.jsonl",
                        ["--conditions", "normal"],
                    )
                )
            )
        (ROOT / "validation_runs.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
