"""Run six calibration jobs, select one threshold, and replay validation."""

import argparse
from pathlib import Path
import subprocess
import sys


def run_six(configs, logs, stage):
    jobs = []
    for architecture in ("mamba", "transformer"):
        for seed in (0, 1, 2):
            name = f"{stage}_{architecture}_seed_{seed}"
            config = configs / f"{name}.yaml"
            if not config.is_file():
                raise FileNotFoundError(f"Missing config: {config}")
    for architecture in ("mamba", "transformer"):
        for seed in (0, 1, 2):
            name = f"{stage}_{architecture}_seed_{seed}"
            log = (logs / f"{name}.log").open("w")
            job = subprocess.Popen([sys.executable, "-u", "main.py", "--config", str(configs / f"{name}.yaml")],
                                   stdout=log, stderr=subprocess.STDOUT)
            jobs.append((name, job, log))
            print(f"Started {name}: pid={job.pid}", flush=True)
    failures = []
    for name, job, log in jobs:
        status = job.wait()
        log.close()
        print(f"Finished {name}: exit={status}", flush=True)
        if status:
            failures.append(name)
    if failures:
        raise SystemExit(f"{stage} failed: {', '.join(failures)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", required=True, type=Path)
    parser.add_argument("--logs", required=True, type=Path)
    args = parser.parse_args()
    args.logs.mkdir(parents=True, exist_ok=True)
    run_six(args.configs, args.logs, "calibrate")
    subprocess.run([sys.executable, "-u", "main.py", "--config", str(args.configs / "select.yaml")], check=True)
    run_six(args.configs, args.logs, "evaluate")
    subprocess.run([sys.executable, "-u", "main.py", "--config", str(args.configs / "compare.yaml")], check=True)


if __name__ == "__main__":
    main()
