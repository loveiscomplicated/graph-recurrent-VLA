"""Back up everything needed to restore this pod's work to a private HF repo.

Included: team code (untracked in the working copy), all baseline outputs and
checkpoints, ExperimentLogger runs, the earlier /workspace/vision prototype
(code, docs, results, logs), Claude memory, and a ``restore/`` folder with git
commits, third-party repo pins, pip freezes and RESTORE.md.
Excluded (re-downloadable or regenerable): datasets, prepared episodes, public
model weights, third-party clones, virtualenvs, caches, credentials.

Example:
    python visual/baseline/backup_to_hf.py kdh1270/visual-baseline-backup
"""

import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

WORKSPACE = Path("/workspace")
REPO_DIR = WORKSPACE / "graph-recurrent-VLA"
VISION_DIR = WORKSPACE / "vision"
STAGING = WORKSPACE / "backup_staging" / "restore"

RESTORE_GUIDE = """# Restore guide

1. Clone the team repo at the recorded commit (see `graph_recurrent_vla_git.txt`):
   `git clone https://github.com/loveiscomplicated/graph-recurrent-VLA.git && git checkout <commit>`
   then copy `graph-recurrent-VLA/` from this backup over it (team code, outputs, experiments).
2. Environments: create venvs from `pip_freeze_<env>.txt` (python version in the file header).
   The baseline env also used the PyTorch cu128 index:
   `uv pip install -r pip_freeze_baseline.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match`
3. Dataset: HF `Graph-Recurrent-VLA/graph-recurrent-rollouts`, revision
   94cb85e8009f20b5b47063e8295f6c69040549c1, pattern `integration/c555bed7462d4fc09e55ae3bb04421a1/*`
   into `data/hf`; then `python -m visual.baseline.prepare_data --archives data/hf/integration/c555bed7462d4fc09e55ae3bb04421a1 --output data/visual_baseline/episodes`.
4. Vision prototype: copy `vision/` to /workspace/vision; re-clone third-party repos at the
   commits in `third_party.txt` and re-download the model weights listed in `vision_checkpoints.txt`.
5. Claude memory: copy `claude_memory/` to ~/.claude/projects/-root/memory/.
6. Results: baseline eval JSONs are in `graph-recurrent-VLA/outputs/visual_baseline/*/eval_*.json`.
"""


def run(command: str) -> str:
    """Run a shell command and return stdout plus stderr (never raises)."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr


def write_restore_metadata() -> None:
    """Record what is not uploaded so it can be reconstructed."""
    STAGING.mkdir(parents=True, exist_ok=True)
    (STAGING / "RESTORE.md").write_text(RESTORE_GUIDE)
    (STAGING / "graph_recurrent_vla_git.txt").write_text(
        run(f"cd {REPO_DIR} && git rev-parse HEAD && git branch --show-current && git status --short"))
    (STAGING / "graph_recurrent_vla_tracked.diff").write_text(run(f"cd {REPO_DIR} && git diff HEAD"))

    pins = []
    for clone in sorted((VISION_DIR / "third_party").iterdir()):
        url = run(f"git -C {clone} remote get-url origin").strip()
        commit = run(f"git -C {clone} rev-parse HEAD").strip()
        pins.append(f"{clone.name} {url} {commit}")
    (STAGING / "third_party.txt").write_text("\n".join(pins) + "\n")
    (STAGING / "vision_checkpoints.txt").write_text(run(f"ls -la {VISION_DIR / 'checkpoints'}"))

    for env in sorted((WORKSPACE / "envs").iterdir()):
        python = env / "bin" / "python"
        header = run(f"{python} --version")
        freeze = run(f"VIRTUAL_ENV={env} uv pip freeze")
        (STAGING / f"pip_freeze_{env.name}.txt").write_text(f"# {header}{freeze}")
    (STAGING / "system.txt").write_text(run("nvidia-smi; uname -a; env | grep ^RUNPOD_ | grep -v KEY"))


def main() -> None:
    repo = sys.argv[1]
    api = HfApi()
    assert api.repo_info(repo).private, f"{repo} must be private"
    write_restore_metadata()

    uploads = [
        # (local folder, path in repo, ignore patterns)
        (STAGING, "restore", []),
        (REPO_DIR / "visual" / "baseline", "graph-recurrent-VLA/visual/baseline", ["__pycache__/*"]),
        (REPO_DIR / "visual" / "configs", "graph-recurrent-VLA/visual/configs", []),
        (REPO_DIR / "experiments", "graph-recurrent-VLA/experiments", []),
        (REPO_DIR / "outputs" / "visual_baseline", "graph-recurrent-VLA/outputs/visual_baseline", []),
        (VISION_DIR, "vision", ["data/*", "checkpoints/*", "third_party/*", "**/__pycache__/*", "*.pyc"]),
        (WORKSPACE / "home/.claude/projects/-root/memory", "claude_memory", []),
    ]
    for folder, target, ignore in uploads:
        if not folder.exists():
            print("skip missing", folder)
            continue
        api.upload_folder(repo_id=repo, folder_path=str(folder), path_in_repo=target,
                          ignore_patterns=ignore, commit_message=f"backup {target}")
        print("uploaded", target, flush=True)
    print("backup done:", len(api.list_repo_files(repo)), "files")


if __name__ == "__main__":
    main()
