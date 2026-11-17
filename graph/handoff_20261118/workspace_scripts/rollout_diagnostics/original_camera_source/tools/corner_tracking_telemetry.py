"""Record non-invasive process/GPU telemetry for independently running jobs."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


JOBS = tuple(f"{architecture}_{seed}" for architecture in ("mamba", "transformer") for seed in range(3))


def process_status(root):
    """Find each launcher by its exact config path, without attaching a debugger."""
    output = subprocess.check_output(["ps", "-eo", "pid=,pcpu=,rss=,etime=,args="], text=True)
    processes = {}
    for line in output.splitlines():
        parts = line.split(maxsplit=4)
        if len(parts) != 5 or not any(
            command in parts[4] for command in ("main.py --config", "-m tools.corner_tracking_progress --config")
        ):
            continue
        for name in JOBS:
            config_path = str(root / "train_configs" / f"{name}.yaml")
            if config_path in parts[4]:
                processes[name] = dict(pid=int(parts[0]), cpu_percent=float(parts[1]),
                                       rss_mib=round(int(parts[2]) / 1024, 1), elapsed=parts[3])
    return processes


def last_completed_epoch(root, name):
    """Return a saved epoch, never infer progress from GPU or CPU utilization."""
    architecture, seed = name.rsplit("_", 1)
    histories = list((root / "training" / architecture / f"seed_{seed}").glob("*/history.json"))
    epochs = []
    for path in histories:
        history = json.loads(path.read_text())
        if history:
            epochs.append(history[-1]["epoch"])
    return max(epochs, default=0)


def snapshot(root):
    """Read one point-in-time GPU sample and the six process/checkpoint states."""
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).splitlines()[0]
    memory_used, memory_total, utilization = (float(value.strip()) for value in gpu.split(","))
    running = process_status(root)
    jobs = {}
    for name in JOBS:
        exit_file = root / "logs" / f"train_{name}.exit"
        exit_code = int(exit_file.read_text().strip()) if exit_file.exists() else None
        log_path = root / "logs" / f"train_{name}.log"
        progress = None
        if log_path.exists():
            with log_path.open(encoding="utf-8", errors="replace") as file:
                for line in file:
                    if line.startswith("[progress]"):
                        progress = line.strip()
        jobs[name] = dict(process=running.get(name), exit_code=exit_code,
                          completed_epoch=last_completed_epoch(root, name), episode_progress=progress)
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                gpu=dict(memory_used_mib=memory_used, memory_total_mib=memory_total,
                         utilization_percent=utilization), jobs=jobs)


def main():
    parser = argparse.ArgumentParser(description="Log live corner-training resource telemetry")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30)
    options = parser.parse_args()
    if options.interval <= 0:
        parser.error("--interval must be positive")
    root = options.root.resolve()
    log = root / "logs" / "telemetry.jsonl"
    while True:
        current = snapshot(root)
        with log.open("a", encoding="utf-8") as file:
            file.write(json.dumps(current, allow_nan=False) + "\n")
        gpu = current["gpu"]
        print(f"[{current['timestamp_utc']}] GPU {gpu['utilization_percent']:.0f}% "
              f"{gpu['memory_used_mib']:.0f}/{gpu['memory_total_mib']:.0f} MiB", flush=True)
        for name, job in current["jobs"].items():
            process = job["process"]
            if process is None:
                state = f"not running; exit={job['exit_code']}"
            else:
                state = "running"
            print(f"  {name}: epoch_completed={job['completed_epoch']} {state}", flush=True)
            if job["episode_progress"]:
                print(f"    {job['episode_progress']}", flush=True)
        time.sleep(options.interval)


if __name__ == "__main__":
    main()
