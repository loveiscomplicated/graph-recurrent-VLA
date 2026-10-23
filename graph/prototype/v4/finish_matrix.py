"""Wait for full-budget state/both runs, then finish fixed evaluations."""
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    root=Path('outputs/graph/v4')
    for mode in ['state','both']:
        path=root/'training'/f'surface_{mode}'/'training.json'
        while not path.exists():
            failed=root/f'tmux_mps_{mode}.failed'
            if failed.exists():raise RuntimeError(f'MPS {mode} training failed; see trainer log')
            time.sleep(10)
        completed=json.loads(path.read_text())
        if completed['optimizer_steps']!=10590 or len(completed['history'])!=30 or completed['smoke_max_batches'] is not None:
            raise AssertionError('full-budget training required before evaluation')
        command=[sys.executable,'-m','graph.prototype.v4.run_matrix','--phase','evaluate','--models','surface','--histories',mode]
        subprocess.run(command,check=True)
    subprocess.run([sys.executable,'-m','graph.prototype.v4.run_matrix','--phase','ablate','--models','surface'],check=True)
    # No training or other evaluation runs should be active at this point.
    for module in ['latency','distribution','compare','figures','report']:
        subprocess.run([sys.executable,'-m',f'graph.prototype.v4.{module}'],check=True)
    print('Completed all matrix evaluations, ablations and comparisons.',flush=True)


if __name__=='__main__':main()
