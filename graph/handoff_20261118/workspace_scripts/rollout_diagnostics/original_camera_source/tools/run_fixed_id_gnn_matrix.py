"""Launch six independent matched GNN jobs, then compare after all complete."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", required=True, type=Path)
    parser.add_argument("--logs", required=True, type=Path)
    args = parser.parse_args()
    args.logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for architecture in ("mamba", "transformer"):
        for seed in (0, 1, 2):
            name = f"{architecture}_seed_{seed}"
            config = args.configs / f"{name}.yaml"
            if not config.is_file():
                raise FileNotFoundError(f"Missing run configuration: {config}")
    for architecture in ("mamba", "transformer"):
        for seed in (0, 1, 2):
            name = f"{architecture}_seed_{seed}"
            log = (args.logs / f"{name}.log").open("w")
            process = subprocess.Popen([sys.executable, "-u", "main.py", "--config",
                                        str(args.configs / f"{name}.yaml")],
                                       stdout=log, stderr=subprocess.STDOUT)
            processes.append((name, process, log))
            print(f"Started {name}: pid={process.pid}", flush=True)
    failures = []
    for name, process, log in processes:
        result = process.wait()
        log.close()
        print(f"Finished {name}: exit={result}", flush=True)
        if result:
            failures.append(name)
    if failures:
        raise SystemExit(f"Training failed: {', '.join(failures)}; see individual logs")
    subprocess.run([sys.executable, "-u", "main.py", "--config",
                    str(args.configs / "compare.yaml")], check=True)


if __name__ == "__main__":
    main()
