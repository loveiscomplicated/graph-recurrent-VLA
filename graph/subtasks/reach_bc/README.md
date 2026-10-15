# Reach Flat MLP BC v1

Measured on Vast.ai **55512455**, Linux Python 3.11.16, ManiSkill 3.0.1,
SAPIEN 3.0.3, Torch 2.8.0+cpu, `physx_cpu`, one environment. Training and
inference ran on CPU. RTX 3090 supplied Vulkan rendering; CUDA simulation and
CUDA training were not run. See [technical report](../../../docs/reach_bc_baseline.md).

The unchanged v2 `CommonPolicy('flat', cfg, 'goal', 'regression')` has 299,655
parameters. Its 1,915-dimensional geometry input and 31-dimensional state bypass
use existing train-only normalizers. The goal feature is v2's current-geometry
canonical **grasp** target, distinct from the subtask's 5 cm approach target.
No planner action, phase, future state, or future waypoint is an input.

Measured main result: held-out **0/5**, fresh **3/50** (6%; Wilson 95% CI
2.06–16.22%). Cartesian reference: **5/5 and 50/50**. Fixed-open diagnostic:
**0/5 and 5/50**. These diagnostics do not replace the learned-policy result.

## Runtime preparation

The new instance was empty. Repository and `outputs/subtasks/generated_50`
were copied to the SAME absolute repository path, so original index hashes and
absolute trajectory references stayed unchanged:

```bash
cd /Users/jeong-yunseong/Documents/programming/graph-recurrent-VLA
uv venv --python 3.11 /workspace/reach-bc-55512455/.venv
uv pip install --python /workspace/reach-bc-55512455/.venv/bin/python \
  torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /workspace/reach-bc-55512455/.venv/bin/python \
  -r graph/subtasks/requirements-generator.txt
uv pip check --python /workspace/reach-bc-55512455/.venv/bin/python
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
REACH_PY=/workspace/reach-bc-55512455/.venv/bin/python
"$REACH_PY" examples/check_env.py --control-mode pd_ee_delta_pose
```

The exact resolved package list is in the experiment's
`reports/linux_requirements_freeze.txt`. The old Mac venv and datasets are preserved.
New-source replay is tested on Linux; Mac is used only for old-data regression
and analysis of already measured artifacts.

Before fixing fresh seeds, metadata from the local existing experiments was
scanned and copied as `reports/prior_seed_inventory.json` (43 sources, 175 unique
seed values, no intersection with 12000–12049). On the local repository:

```bash
.venv/bin/python - <<'PY'
from graph.subtasks.reach_bc.evaluate import seed_inventory
from graph.subtasks.reach_bc.policy import ROOT
from graph.prototype.common import write_json
write_json(ROOT/'reports/prior_seed_inventory.json', seed_inventory())
PY
```

Copy this report to the matching Linux experiment directory. The manifest CLI
also scans metadata present on Linux and merges this prior inventory. The
executed run first built the manifest and then merged the full local inventory
before training; its fixed 55 starts were preserved.

## Dataset → smoke → training → actual rollout → reporting

These are the Python commands actually executed in order on Linux. Existing
prepared data/checkpoints/manifests/evaluation outputs are refused; choose a new
experiment directory for a new run. No CLI overwrites the source export or split.

```bash
"$REACH_PY" -m graph.subtasks.reach_bc.prepare
"$REACH_PY" -m graph.subtasks.reach_bc.evaluate --manifest
"$REACH_PY" -m graph.subtasks.reach_bc.train --smoke
"$REACH_PY" -m graph.subtasks.reach_bc.train
"$REACH_PY" -m graph.subtasks.reach_bc.evaluate --mode flat
"$REACH_PY" -m graph.subtasks.reach_bc.evaluate --mode reference
"$REACH_PY" -m graph.subtasks.reach_bc.evaluate --mode flat --repeat
"$REACH_PY" -m graph.subtasks.reach_bc.evaluate --mode fixed_open
"$REACH_PY" -m graph.subtasks.reach_bc.diagnostics
"$REACH_PY" -m graph.subtasks.reach_bc.evaluate --mode flat --video-seed 12005
"$REACH_PY" -m graph.subtasks.reach_bc.evaluate --mode flat --video-seed 12000
"$REACH_PY" -m graph.subtasks.reach_bc.evaluate --mode flat --video-seed 11033
"$REACH_PY" -m graph.subtasks.reach_bc.evaluate --mode flat --video-seed 12017
"$REACH_PY" -m graph.subtasks.reach_bc.evaluate --mode reference --video-seed 12000
"$REACH_PY" -m graph.subtasks.reach_bc.report
```

`prepare` replays all original 2,084 actions, audits the diagnostic tail, checks
the complete offline/live geometry and raw features, retains 1,603 actions up to
first live success, stores T+1 states per episode, and checks normalized agreement.
Training always starts from a fresh seed-0 model, never from the overfit checkpoint.
It uses 30 epochs, batch 32, AdamW 1e-3 / weight decay 1e-5, existing cosine scheduler,
and selects only validation MSE. Test is read once after selection.

The CLI defaults to `outputs/subtasks/reach_bc_v1`. For a fresh reproduction, pass
`--output NEW_ROOT` to every command and `--dataset NEW_ROOT/data/reach_first_success.pt`
after preparation. Copy the prior seed inventory to `NEW_ROOT/reports/` or retain
the original metadata tree for the manifest's scan. `--source-root`
can point at a copied generated_50 tree; its internal absolute paths must resolve.
Training never re-splits episodes. Reproduced dataset container bytes may differ,
so regenerated dataset/checkpoint/manifest hashes are recorded together.

Reference and repeat evaluations ran as `reach_bc_diagnostics` under supervisor
using `run_diagnostics.sh`; fixed-open ran as a separate foreground supervisor job
`reach_bc_fixed_open`. Both jobs have `autorestart=false`, finished normally, and
do not restart on an existing result. This avoids relying on an SSH connection
for completion. Supervisor logs are in `reports/`. The first flat command's SSH
connection dropped after partial console output; the process completed and saved
all 55 results. The independent repeat subsequently matched every state/action
trace exactly.

## Tests actually executed

Run source-render and shared host-readback tests in separate processes. SAPIEN
3.0.3 cannot switch its process-global render device from the recorded source
environment's `sapien_cuda` to shared `sapien_cpu`. Physics is CPU in both.

```bash
REACH_BC_INTEGRATION=1 "$REACH_PY" -m unittest \
  graph.subtasks.reach_bc.tests.test_reach_bc -v
SUBTASK_GENERATION_INDEX=outputs/subtasks/generated_50/segments/index.json \
  "$REACH_PY" -m unittest graph.subtasks.tests.test_subtasks \
  graph.subtasks.tests.test_environment graph.prototype.tests.test_geometry \
  graph.prototype.tests.test_data graph.prototype.tests.test_models \
  graph.prototype.v2.tests.test_v2.StateTests \
  graph.prototype.v2.tests.test_v2.ModelTests -v
"$REACH_PY" -m unittest tests.test_maniskill_env tests.test_graph_gt_nodes -v
# Mac legacy-data integration, not new-data benchmark:
SUBTASK_INTEGRATION=1 .venv/bin/python -m unittest \
  graph.subtasks.tests.test_subtasks graph.subtasks.tests.test_environment -v
```

Reach suite: 9/9 pass, including actual CPU feature/action checks and all-55 repeat.
Linux source/model regression: 52 tests, 44 pass / 8 expected skips (legacy artifacts
not copied and opt-in old-data integration). Shared Linux CPU/RGB: 8/8 pass.
Mac legacy suite: 32 tests, 31 pass / 1 expected skip (generated Linux index).
The local model/data regression also passed 50 tests with 3 opt-in skips.
The initial combined Linux suite's RGB CUDA-readback error and subsequent
render-device conflict are preserved in logs. The shared factory now consistently
selects host readback for CPU-only Torch; the final isolated suites pass.

Coverage: first-success and T/T+1 alignment; original samples/splits; train-only
statistics via held-out poisoning; loading; gradients; measured overfit; checkpoint
roundtrip; complete offline/live agreement; installed-controller action/gripper
decoding; reset/step; real success/timeout; all seed fingerprints and overlap;
exact repeated closed-loop policy evaluation; old subtask/model regressions.

## Artifacts

`data/reach_first_success.pt`, `reports/trimming_audit.json`,
`checkpoints/best.pt`, `training/config.json`, `training/normalization.pt`,
`training/history.json`, `training/training.json`, `evaluation/paired_starts.json`,
`evaluation/{flat,flat_repeat,reference,fixed_open}.json`,
`reports/{failure_summary,distribution_audit}.json`, `reports/results.md`,
`figures/*.png`, and five annotated rollout videos with state/action JSON sidecars.
All were copied back to the local experiment directory. Videos are confirmed to
match the main evaluation traces. See the report for limitations and next-step judgment.

## Frozen-checkpoint bottleneck review

Follow-up diagnosis preserves the selected checkpoint and main SR. See the
[added technical analysis](../../../docs/reach_bc_baseline.md#추가-병목-진단-frozen-checkpoint의-제어-피드백)
and `outputs/subtasks/reach_bc_v1/bottleneck_v1/summary.json`.

```bash
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --audit
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --mode finger_mean
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --mode qvel_zero
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --mode ee_floor_clip
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --mode low_variance_clip
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --feedback
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --reference-switch
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck --offline
"$REACH_PY" -m graph.subtasks.reach_bc.bottleneck_report
"$REACH_PY" -m unittest graph.subtasks.reach_bc.tests.test_bottleneck -v
```

The audit replays all original 5,418 learner steps exactly. Four inference-only
interventions run 220 paired episodes, with fixed train-derived statistics and
original success/budget. Additional physical one-step perturbation and reference
switch probes locate velocity feedback and finger-velocity coupling. New unit
tests passed 2/2 on Linux and Mac. No model is trained, and diagnostic SRs do not
replace the main result. `bottleneck_v1.tar.gz` preserves full raw traces locally.
