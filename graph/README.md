# Graph team

## Corner prediction and tracking

The separate 6D TCP rotation corner-predictor recipe uses existing fixed-ID
sequence data. After setting the dataset path and checksum, run:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/fixed_id_corner_rotation_6d.yaml
```

The default Mamba recipe requires CUDA and `mamba_ssm`; Transformer can also run
on CPU with `model.architecture: transformer`, `training.device: cpu`, and
`training.precision: float32`. For the model contract and detector-only
inference example, see [6D TCP rotation input](../docs/corner_prediction.md#6d-tcp-rotation-input).

To render a continuously moving box and Hole with the leaderboard Transformer
and GNN, configure the checkpoint paths in
`graph/configs/temporal/visualize_moving_hole.yaml` and run:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/visualize_moving_hole.yaml
```

Use `graph/configs/temporal/visualize_jumping_hole.yaml` for a single abrupt
translation. Both recipes use a time window and a GT-based proximity guard.
By default they step the simulator with zero actions; configure an existing
`module:factory` policy to exercise robot motion.

For the existing GT-guided scripted expert, run:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/visualize_moving_hole_expert.yaml
```

This recipe uses `pd_ee_delta_pose`, requires the repository's ManiSkill and
Pinocchio dependencies, and stops motion when the expert enters insertion.
For a single abrupt translation with the same expert and guards, use
`graph/configs/temporal/visualize_jumping_hole_expert.yaml`.

This requires ManiSkill, the RGB-D corner detector dependencies, trusted local
checkpoints and FFmpeg. See [moving Hole visualization](../docs/detected_corner_gnn.md#moving-hole-visualization)
for the motion and video limitations.

Run recipes from the repository root with Python 3.11, Torch, NumPy, SciPy and
PyYAML. Set dataset, checkpoint and checksum paths before running. Official
`mamba_ssm` requires CUDA; camera videos also require OpenCV and FFmpeg with
libx264. The lifecycle smoke recipe below runs on CPU without the CUDA runtime.

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/corner_tracking/smoke.yaml
```

Select the workflow by the identity information and training target:

| Workflow | Recipe | Guide |
| --- | --- | --- |
| Prepare fixed-ID corner histories | `data_collection/configs/fixed_id_corner_sequences.yaml` | [Dataset and certificates](../docs/data_collection.md#fixed-id-per-corner-sequences) |
| Shared fixed-ID Mamba or Transformer | `graph/configs/temporal/fixed_id_corners.yaml` | [Training, continuation and early stopping](../docs/corner_prediction.md#shared-fixed-id-corner-prediction) |
| Fixed-ID prediction videos | `graph/configs/temporal/visualize_fixed_id_corners.yaml`, `visualize_all_fixed_id_corners.yaml` | [Single-corner and full-episode video](../docs/corner_prediction.md#fixed-id-corner-video-visualization) |
| Fixed-ID prediction followed by GNN | `graph/configs/temporal/prepare_fixed_id_gnn.yaml`, `train_fixed_id_gnn.yaml`, `compare_fixed_id_gnn.yaml` | [Cache, training and comparison](../docs/corner_prediction.md#fixed-id-corner-predictor-followed-by-a-gnn) |
| Detector-only association | `graph/configs/temporal/calibrate_detected_corner_tracking.yaml`, `select_detected_corner_threshold.yaml`, `evaluate_detected_corner_tracking.yaml` | [Association and rollout evaluation](../docs/detected_corner_gnn.md) |
| Detector-only rollouts followed by GNN | `graph/configs/temporal/train_detected_corner_gnn.yaml` | [20-epoch, fresh 40-epoch and resumed training](../docs/detected_corner_gnn.md#detector-only-rollout-gnn-training) |
| Lifecycle-managed corner completion | `graph/configs/temporal/corner_tracking/prepare.yaml`, `compare.yaml`, `compare_compiled.yaml` | [Tracking, CUDA execution and diagnostics](../docs/corner_tracking.md) |

For example, after configuring the frozen temporal checkpoint, matched dataset,
label cache and calibrated threshold:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/train_detected_corner_gnn.yaml
```

The six detector-only GNN leaderboard results have an [exact evaluation
preservation requirement](../docs/detected_corner_gnn.md#evaluation-preservation).
The 40-epoch and continuation YAML files describe additions to a complete source
effective config; follow their instructions before execution.

## Node-token and joint temporal training

With the original graph cache, observed robot frames, matching source checkpoints
and CUDA Mamba runtime, set the paths in the selected recipe and run:

```bash
python main.py --config graph/configs/temporal/insertion/train_joint_node_token_mamba_0.yaml
```

Use the corresponding seed 1/2 recipes for paired independent runs.

| Workflow | Recipe under `graph/configs/temporal/insertion/` | Guide |
| --- | --- | --- |
| Joint training from pretrained node-token models | `train_joint_node_token_mamba_0.yaml` | [Joint protocol](../docs/temporal_estimation.md#joint-fine-tuning-from-the-120-epoch-node-token-checkpoints) |
| Joint continuation with early stopping | `continue_joint_node_token_mamba_0.yaml` | [Checkpoint and stopping contract](../docs/temporal_estimation.md#early-stopped-joint-continuation) |
| Joint training from independently early-stopped sources | `train_joint_early_node_token_mamba_0.yaml` | [Source pairs and evaluation](../docs/temporal_estimation.md#joint-training-from-independently-early-stopped-sources) |

The five node-token shell launchers accept checkout, Python, runtime and log
paths while retaining their original server defaults. For example:

```bash
bash tools/run_joint_node_token_seed.sh 0 \
  --experiment-root /path/to/run \
  --source-root /path/to/checkout \
  --python /path/to/venv/bin/python \
  --runtime-root /path/to/runtime
```

Use `--runtime-root ""` when no extra runtime module directory is needed, and
`--logs PATH` to select a log directory. Dataset/checkpoint paths inside each YAML
must also be configured. See [launcher behavior and failure markers](../docs/temporal_estimation.md#node-token-launch-scripts).

## Frozen selected-frame M0 with temporal Transformer

With PyTorch, NumPy, PyYAML and Matplotlib installed, provide the original
contiguous graph cache, matching raw robot observations, selected-M0 manifest,
parent split and frozen checkpoint. Set their paths in the config, then run:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/insertion/train_selected_m0_transformer.yaml
```

Training selects a Transformer checkpoint on held-out development parents and
automatically evaluates the 40 original validation episodes and 960 sampled
frames. See [temporal estimation](../docs/temporal_estimation.md#causal-transformer-with-frozen-m0)
for feature, cache and comparison details.

## Frozen selected-frame M0 with temporal Mamba

With the original full-episode graph cache, matching observed robot frames,
parent-disjoint split manifest, selected M0 dataset manifest and a 60-epoch M0
checkpoint available, configure their paths and run:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/insertion/train_selected_m0_mamba.yaml
```

This trains a Mamba behind a frozen M0, evaluates complete episodes and reports
the original 960 selected validation frames. See [temporal estimation](../docs/temporal_estimation.md)
for the split and measurement details.

## M0 Rim-only input

With the existing family development dataset and Torch, NumPy, SciPy, OpenCV and
PyYAML installed, set the dataset paths and run from the repository root:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/train_rim_only.yaml
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_rim_only.yaml
```

Set the trained matrix path in the evaluation config. The default detector-based
filter keeps PEG and candidate Rim points and excludes other BOX points. An
`oracle_rim` filter uses GT-certified labels for analysis only. See
[vertex-family semantics](../docs/vertex_family_semantics.md#m0-rim-only-입력-실험-설정)
for UNKNOWN handling, comparison requirements and limitations.

## Vertex-family semantic development study

Prerequisites: Torch, NumPy, SciPy, OpenCV, PyYAML and Matplotlib; original TRAIN
RGB-D plus its frame hashes/parent manifest, or the prepared development dataset.
Set the dataset and completed-matrix paths in these configs, then run from the
repository root:

```bash
WANDB_MODE=disabled python main.py --config data_collection/configs/vertex_family.yaml
WANDB_MODE=disabled python main.py --config graph/configs/train_vertex_family.yaml
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_vertex_family.yaml
```

This experimental path keeps the existing AB GNN and Object21D heads, compares
no family tokens, detector-derived tokens, partial GT-certified Oracle tokens,
and redundant object-type capacity control on identical graphs. It uses only a
derived original-TRAIN cohort. See [the measured study](../docs/vertex_family_semantics.md)
for label limitations, three-seed results, information budgets and reproduction.

## Vertex GNN bottleneck diagnostics

With Python 3.11, Torch, NumPy, PyYAML and Matplotlib, provide the original paired
predictions, extracted graph cache, parent manifest and the newly trained Baseline
checkpoint. Set their paths and a new output directory in the recipes:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/analyze_vertex_diagnostics.yaml
WANDB_MODE=disabled python main.py --config graph/configs/train_readout_probe.yaml
WANDB_MODE=disabled python main.py --config graph/configs/analyze_vertex_geometry.yaml
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_readout_probe.yaml
```

The probe captures node features immediately before the original pools, freezes
all source parameters and trains only freshly initialized existing pools/heads.
All candidates reuse the same checksummed feature cache, anchors, confidence,
Object21 loss and paired frames. `prepare_only: true` captures features and raw
geometry without training. The separate probe budget is batch 256 × 30 epochs;
seed/capacity controls and execution details are recorded in effective configs.
See [bottleneck diagnostics](../docs/vertex_gnn_bottlenecks.md) for the measured
results, oracle definitions and limitations. No original GNN architecture or
controller is changed.

## Vertex geometry/readout ablation

With Python 3.11, Torch, NumPy and PyYAML installed, supply the original gated
MPNN checkpoint and its extracted published graph cache. Set `base_config`,
`reference_checkpoint` and `data.cache_dir` in the training recipe; set the
resulting `ablation_manifest.json` path in the evaluation recipe. From the root:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/train_vertex_ablation.yaml
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_vertex_ablation.yaml
```

This trains Baseline/A/B/AB from scratch with the existing trainer and loss,
then scores all four on the same saved validation frames. Models remain under
`graph/`, orchestration under `training/gnn/` and `evaluation/gnn/`, and the data
reader remains the existing cache-only factory. See [vertex ablation](../docs/gnn_training.md#vertex-geometry-and-readout-ablation)
for the checkpoint audit, AG-Pose/GMT-inspired scope, parameter counts, metric
definitions, synthetic verification and the real-data comparison status.
`parallel_models` can run 2–4 independent processes; select `data.num_workers`
for the available host memory. Both are recorded in the effective recipe.

## Temporal estimation workflows

Install Torch, NumPy, PyYAML and Matplotlib, then run from the repository root.
Set the selected recipe's checkpoint, cache and parent-manifest paths first.
Pose GRU can add checksum-matched raw robot frames; insertion Mamba/GRU requires
those frames by default. Hybrid workflows require the original frozen checkpoints
and the identified validation prediction snapshot.

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/hole_gru/train.yaml
WANDB_MODE=disabled python main.py --config graph/configs/temporal/hole_gru/evaluate.yaml
```

Choose the workflow by its input/output task:

| Workflow | Config under `graph/configs/temporal/` | Guide |
| --- | --- | --- |
| Hole position GRU | `hole_gru/train.yaml`, `hole_gru/evaluate.yaml` | [Hole GRU](../docs/temporal_estimation.md#gru-temporal-object-state-estimator) |
| Peg/Hole pose GRU | `pose_gru/train_hole.yaml`, `train_peg.yaml`, `train_joint.yaml`, `evaluate.yaml` | [Pose GRU](../docs/temporal_estimation.md#joint-peghole-temporal-pose-estimator) |
| Pose/relative/lateral loss ablations | `precision/train_*.yaml`, `precision/evaluate.yaml` | [Insertion objectives](../docs/temporal_estimation.md#gru-insertion-precision-objectives) |
| Insertion Mamba and matched GRU | `insertion/train_mamba.yaml`, `train_gru_matched.yaml`, `evaluate.yaml`, `compare_gru_mamba.yaml` | [Insertion estimation](../docs/temporal_estimation.md#mamba-insertion-precision-estimator) |
| Frozen Mamba + Gate hybrid | `hybrid/evaluate.yaml` | [Hybrid evaluation](../docs/hybrid_estimation.md#frozen-mamba--learned-gate-hybrid) |
| Hybrid Peg-head fine-tuning | `hybrid/finetune_peg.yaml`, `hybrid/analyze_finetuning.yaml` | [Fine-tuning](../docs/hybrid_estimation.md#h1-peg-head-relative-geometry-fine-tuning) |
| Saved hybrid error analysis | `hybrid/analyze_errors.yaml` | [Error attribution](../docs/hybrid_estimation.md#hybrid-h1-relative-geometry-error-attribution) |

`insertion.py` supports both Mamba and GRU; `hole_gru.py` and `pose_gru.py` name
their separate original estimators. CPU Mamba uses the explicit `torch_reference`
backend; official `mamba_ssm` kernels require a compatible CUDA environment.
The guides retain the measured results, split roles and device limitations.

| Location | Responsibility |
| --- | --- |
| `graph/temporal/` | Models, streaming state and observation features |
| `graph/pose_geometry.py` | Pose and insertion geometry |
| `data_collection/temporal/` | Aligned episodes, windows, splits and frozen feature caching |
| `training/gnn/temporal/` | Config validation, frozen-source preparation, training and losses |
| `evaluation/gnn/temporal/` | Paired evaluation, diagnostics and plots |

Run the related CPU tests, including tiny synthetic training workflows:

```bash
WANDB_MODE=disabled .venv/bin/python -m unittest \
  tests.test_temporal_hole_gru tests.test_temporal_pose_gru \
  tests.test_temporal_insertion tests.test_temporal_precision \
  tests.test_temporal_hybrid tests.test_temporal_hybrid_finetuning \
  tests.test_temporal_hybrid_error_analysis tests.test_temporal_gate -v
```

## Learned temporal update gate

With Torch, NumPy, PyYAML and Matplotlib installed, the original gated MPNN
`best.pt`, its extracted published cache, and an explicit parent-disjoint split
manifest, run from the repository root (full runs require separate approval):

```bash
WANDB_MODE=disabled python main.py --config graph/configs/train_temporal_gate.yaml
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_temporal_gate.yaml
```

Set checkpoint/cache/manifest paths in both configs and the learned checkpoint
path in the evaluation config. The gate trains on GNN-train parents only; the
original GNN validation split is reserved for the final three-method comparison.
See [GNN training](../docs/gnn_training.md#learned-temporal-update-gate) for manifest
preparation, initialization, features, assembly ablation and smoke tests.

## Frozen GNN temporal-memory evaluation

With Torch, NumPy and PyYAML installed, an extracted graph cache and its trained
`best.pt`, run from the repository root:

```bash
python main.py --config graph/configs/evaluate_temporal_memory.yaml
```

Set the checkpoint/cache paths in the YAML. Set `evaluation.visibility_labels`
to `null` if the optional earlier depth-visibility diagnostics are unavailable.
This evaluates the saved validation split without training or generating caches.
See [temporal memory](../docs/gnn_training.md#frozen-gnn-temporal-memory-evaluation)
for observation gates, causal state, metrics and limits.

## Simulator GT points

`graph.utils.gt_nodes.get_gt_node_positions(env)` returns TCP-relative gripper,
fingertip, peg-corner and hole-corner coordinates from simulator GT. See
[`utils/gt_nodes.py`](utils/gt_nodes.py) and
[`examples/visualize_gt_points.py`](../examples/visualize_gt_points.py).

## Observed corners to GNN input

Install `requirements.txt` and `visual/corner_detection/requirements.txt` in a
Python 3.11 environment. From the repository root:

```bash
python examples/build_corner_graph.py --input /path/to/frame.npz \
  --tcp-xyz 0.0 0.0 0.2 --model gated --device cpu
```

Supply the actual observed TCP world position (metres). This is an untrained
forward smoke check. `graph/build_graph/` converts Visual `TensorResult` into a
shared `GraphBatch` consumed by both GNNs, preserving PEG/BOX/HOLE distinctions,
unknown semantics/quality and padding masks. See [graph input](../docs/graph_input.md)
for the contract, confidence policy, edge settings and runnable Python example.

## GNN training

For a downloaded graph cache, extract `graph_cache.tar` into `/workspace/gnn_cache`
and run from the repository root with Torch, NumPy and PyYAML installed:

```bash
python main.py --config graph/configs/train_gated_mpnn_cached.yaml
```

This cache-only factory reads the published split, graph tensors and GT targets.
It checks the inventory and file checksums without raw RGB-D, detection, or cache
regeneration. Change `data.cache_dir` for another extraction location. See
[cache-only training](../docs/gnn_training.md#train-from-a-downloaded-cache).

Install `requirements.txt` and `visual/corner_detection/requirements.txt`, extract
approved episode shards and set `data.root` in the gated config. On the training host:

```bash
python main.py --config graph/configs/train_gated_mpnn.yaml
```

The gated recipe enables `training.compile.enabled: true` and dynamic-shape
compilation on CUDA. CPU/MPS use eager execution. Set the flag to `false` to
disable compilation. The first train/validation calls can spend time compiling.

The common `training.gnn.trainer` handles optimization, validation, tracking and
resume; task implementations own output-specific loss/metrics. Frame detection
and graph caches are generated on first access when `data.use_cache` is true.
See [GNN training](../docs/gnn_training.md) for dataset layout, splits, cache policy,
checkpoint semantics and other GNN extensions. No training or frame caching was
executed locally during this implementation.

`train_gnn.yaml` remains a BasicGNN initialization example (`mode: initialize`);
its node outputs need a suitable task before actual training.

## Existing prototypes

See [the independent surface-geometry prototype](prototype/README.md) and
[the subtask pipeline](subtasks/README.md) for their existing workflows.

## Corner prediction model

The vertex graph model is defined in [gated_mpnn.py](gated_mpnn.py).
Its training objectives live in
[training/gnn/loss.py](../training/gnn/loss.py):
`object21_loss` compares geometry with GT and regularizes endpoint corrections;
`paired_delta_loss` supervises position changes between paired samples.

## BOX-to-Hole observability diagnostics

With the original prediction/geometry exports and four checkpoints available,
run from the repository root:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/analyze_box_hole.yaml
```

Adjust artifact paths in the config. Torch, NumPy, SciPy, PyYAML and Matplotlib
are required; validation-seed replay also requires ManiSkill/SAPIEN and Vulkan.
The evaluation preserves checkpoints and reserved test frames. See
[BOX-to-Hole observability](../docs/box_hole_observability.md) for GT information
budgets, missing physical correspondence, measured results and CUDA verification.

## Expanded vertex-family training

Prepare the 590-training-parent cohort with
`python main.py --config data_collection/configs/vertex_family_expanded.yaml`.
It preserves the earlier 40 validation parents; set the source paths first.
Set `family_dataset` to the emitted dataset directory in
`graph/configs/train_vertex_family_expanded.yaml` and run
`python main.py --config graph/configs/train_vertex_family_expanded.yaml`.
The maximum is 60 epochs; after at least 15 epochs, training stops when validation
loss fails to improve by 0.001 for eight consecutive epochs. The best validation
checkpoint and actual epoch counts are retained. Equal stopping criteria can
produce unequal optimization budgets across variants.

For the completed full-frame cohort (590 training episodes / 60,600 frames and
40 development-validation episodes / 4,102 frames), use
`python main.py --config graph/configs/train_vertex_family_all_frames.yaml`.
This recipe starts fresh for four family modes and three seeds, enables CUDA
compilation and retains the same 60-epoch maximum and early-stopping settings.

To train a fresh temporal estimator with early stopping on a selected frozen GNN,
set the data and checkpoint paths in
[`configs/temporal/insertion/train_selected_gnn_mamba_early_stopping.yaml`](configs/temporal/insertion/train_selected_gnn_mamba_early_stopping.yaml)
or the corresponding Transformer config, then run
`python main.py --config graph/configs/temporal/insertion/train_selected_gnn_mamba_early_stopping.yaml`.
The graph cache, raw proprioception, selected GNN checkpoint and split manifest
must already be available. See [temporal estimation](../docs/temporal_estimation.md)
for validation selection and resume behavior.

### Joint GNN and temporal fine-tuning

The team-owned configurations `graph/configs/temporal/insertion/train_joint_mamba_0.yaml`
and `train_joint_transformer_0.yaml` (and their seed 1/2 counterparts) specify a
20-epoch joint run. They reference the completed 120-epoch continuation's
best temporal checkpoints and their original matching GNN checkpoints on the
4175 server. Adjust paths if the artifacts are moved. The running early-stopping
experiments are separate and are not used as inputs.

Each seed file contains its checkpoint and output paths. Shared settings live in
`configs/temporal/insertion/joint_mamba.yaml` or `joint_transformer.yaml`.
Edit the seed file for run-specific paths and the corresponding base for shared
settings. Paths are relative to the repository root. The launcher combines them
before invoking the existing trainer, which records the full effective config.
Seed recipes dispatch to `training.gnn.temporal.joint_finetuning:train_from_base`;
the resolver lives in `training.gnn.temporal.config`. The old `joint_config`
module retains aliases for existing external imports and recipes.

From the repository root, after preparing the configured graph cache, raw robot
observations, pretrained checkpoints, and the Mamba CUDA environment, run for
example:

```bash
python main.py --config graph/configs/temporal/insertion/train_joint_mamba_0.yaml
```

The new optimizer updates both networks, using separate configured learning
rates. It reuses the temporal checkpoint's train-only normalization, observes the
same parent-disjoint development split, and resets state at the same window
boundaries. It saves a baseline evaluation at epoch 0, each epoch's train and
validation losses, validation physical-error distributions, both network
weights, optimizer state, effective configuration, and experiment logger records.
The best checkpoint is selected by validation loss, including the baseline.
The validation statistics use reset windows and are not a full-episode test;
no test set or original 960-frame evaluation is run by the training entrypoint.
After training, copy the training config, set `trainer` to
`training.gnn.temporal.joint_finetuning:evaluate`, set `joint_checkpoint` to
the produced `best.pt`, set `evaluation_selection` to the original validation
frame selection, and change `output.root` to a separate evaluation directory.
Run this new config with the same root entrypoint. The evaluator loads both
trained models and writes per-frame errors and physical-condition summaries.

The sample configuration caps each process's CUDA allocator at 10% and is not a
GPU scheduler. Check other GPU jobs and schedule these six runs conservatively.
The learning rates are initial experimental choices, not validated optimal values.
The joint checkpoint is a new format and cannot be loaded by the existing frozen
GNN evaluation entrypoint; use the paired evaluator above.

Evaluate all completed full-frame seed matrices with
`python main.py --config graph/configs/evaluate_vertex_family_all_frames.yaml`.
The evaluator verifies checkpoint/data identities and matching training settings,
accepts recorded early stopping, and reports all Object21 components plus endpoint
and relative-vector diagnostics on the same development-validation frames.
Output includes seed-specific/averaged errors, Parent-clustered intervals and
label-removal/swapping interventions. No optimizer step or reserved-test read is
performed. Update the dataset/matrix paths if running on a different host.

For saved full-frame prediction comparisons by exact PEG/target/total and joint
PEG×target counts, run
`WANDB_MODE=disabled python main.py --config graph/configs/analyze_vertex_family_counts.yaml`.
The inputs are per-frame natural predictions and an aligned `vertex_counts.npz`
exported from the original validation graphs. It reports all Object21 components,
seed means, Parent bootstrap intervals and figures without new model inference.

## Frame-node-token Mamba

With the selected M0 checkpoints, graph cache, robot frames and CUDA Mamba runtime,
run `python main.py --config graph/configs/temporal/insertion/train_node_token_mamba_0.yaml`
from the repository root after updating paths. Repeat for seeds 1 and 2, then run
`python main.py --config graph/configs/temporal/insertion/aggregate_node_token_mamba.yaml`.
The implementation and comparison limits are described in
[temporal estimation](../docs/temporal_estimation.md#frame-node-token-mamba-experiment).

### Node-token Mamba with early-stopped GNN

After preparing the original graph cache, robot frames, three early-stopped GNN
checkpoints and CUDA Mamba runtime, update the paths in
`configs/temporal/insertion/train_node_token_early_stopping_0.yaml` (and seed 1/2)
and run `python main.py --config graph/configs/temporal/insertion/train_node_token_early_stopping_0.yaml`.
After all seeds complete, run the `compare_node_token_early_stopping.yaml` config.
The 4175 server also has a parallel launcher in `tools/run_node_token_early_stopping.sh`.
See [temporal estimation](../docs/temporal_estimation.md#node-token-mamba-with-an-early-stopped-gnn)
for stopping and comparison details.

For a detector-only GNN continuation, copy the completed run's effective YAML,
add the resume fields from `configs/temporal/train_detected_corner_gnn_resume.yaml`,
and run `python main.py --config /path/to/effective_resume.yaml`. This requires
its 40-epoch checkpoints and frozen rollout cache; see
[temporal estimation](../docs/temporal_estimation.md) for early stopping details.

## Random-action corner diagnostics

With trusted local leaderboard seed-0 checkpoints, ManiSkill, Pinocchio,
OpenCV, Matplotlib and FFmpeg installed, run:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/random_action_diagnostics.yaml
```

The config selects environment/action seeds, action amplitude, checkpoint paths
and output directory. It records expert controls, random rollouts, per-frame
results, diagnostic GT substitutions and videos using `ExperimentLogger`.
See [random-action diagnostics](../docs/detected_corner_gnn.md#random-action-diagnostics)
for the GT intervention limits and measured results.
