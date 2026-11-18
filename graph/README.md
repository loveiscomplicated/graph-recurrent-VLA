# Graph team

To compare the frozen epoch-443 surface tracker with the epoch-34 spatial-temporal
online tracker on common held-out physical corners, install the spatial-temporal
evaluation dependencies below and prepare the hash-pinned test datasets, surface
prediction bundle and published online result at the paths in this config:

```bash
WANDB_MODE=disabled .venv/bin/python main.py --config graph/configs/temporal/compare_corner_trackers.yaml
```

This reuses saved surface predictions and the verified online test rollout,
without training or calibration. See [common tracker comparison](../docs/spatial_temporal_corner_tracking.md#common-physical-corner-comparison)
for masks, initialization coverage and independent result verification.

For full chronological spatial-temporal tracking followed by frozen-tracker
Object21 training, install Torch/CUDA, NumPy, SciPy, PyYAML and OpenCV, prepare
the full detector/surface episode dataset, and run from the repository root:

```bash
WANDB_MODE=disabled .venv/bin/python main.py --config graph/configs/temporal/run_spatial_temporal_workflow.yaml
```

See [spatial-temporal corner tracking](../docs/spatial_temporal_corner_tracking.md)
for preparation, checksum verification, validation calibration and prerequisites.

For an explicitly authorized held-out test, prepare the separate test dataset
with `data_collection/configs/spatial_temporal_test_data.yaml`, then evaluate the
validation-selected checkpoints without retraining or association calibration:

```bash
WANDB_MODE=disabled .venv/bin/python main.py --config graph/configs/temporal/evaluate_spatial_temporal_test.yaml
```

The recipe pins the two checkpoint hashes and association settings. The guide
above describes test-data transfer and the separate test rollout cache.

## Recovered Five-mode graph training

To verify the unified current-frame router locally, install the repository and
corner-detector dependencies and place the two pinned Peg-surface `best.pt`
files at the paths in the smoke config. No dataset or simulator is needed:

```bash
WANDB_MODE=disabled .venv/bin/python main.py --config graph/configs/smoke_five_mode_hard_routing.yaml
.venv/bin/python -m pytest tests/test_five_mode_hard_routing.py tests/test_detected_corner_visibility.py tests/test_five_mode_supervised_data.py -q
```

The smoke command loads the actual C1 epoch-57 and C2 epoch-35 weights on CPU,
runs synthetic preprocessed C1/C2 observations, and checks C3 abstention. It
records results through `ExperimentLogger`; it does not execute RGB-D detection
or measure dataset accuracy. See the [inference contract and verification](../docs/gnn_training.md#five-mode-current-frame-hard-routing).

To diagnose the frozen C2 epoch-35 Peg-surface baseline on the pinned 6,744
validation frames, use the same Python dependencies, the checkpoint, and the
supervised release shards at the paths in this config:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 .venv/bin/python main.py --config graph/configs/diagnose_five_mode_condition_2_validation.yaml
```

The CPU recipe reports C2 A–E geometry errors, policy/motion distributions and
per-frame predictions through `ExperimentLogger`. It accepts validation only
and performs no model selection. See the [subtype results](../docs/gnn_training.md#five-mode-c2-validation-subtypes).

To audit whether the current release contains typed Box surface points and
prepare the controlled C2 continuation experiments, use the same GNN Python
environment, the pinned release shards, and the epoch-35 checkpoint:

```bash
WANDB_MODE=disabled .venv/bin/python main.py --config data_collection/configs/audit_five_mode_surfaces.yaml
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 .venv/bin/python main.py --config graph/configs/prepare_five_mode_condition_2_experiments.yaml
```

Preparation emits three training YAMLs and runs one disposable training minibatch
for each ready Peg-only arm. The current release has no typed Box points, so
the Box arm fails preflight until new surface data is ready. Full training is
an explicit run of an emitted YAML; preparation does not replace the baseline.
See [C2 improvement experiments](../docs/gnn_training.md#c2-expert-improvement-experiments)
for sampling, validation-only selection and the Box-data requirements.

The completed production Box cache has a dedicated CUDA preparation config:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/prepare_five_mode_box_surface_experiment.yaml
```

It pins the typed candidate while preserving the original reference. Run its
emitted `peg_box_focus.yaml` explicitly for formal training. See
[production Box-surface inputs](../docs/gnn_training.md#production-box-surface-candidate).

To read the complete Five-mode train split as chronological graph episodes,
use `training.gnn.five_mode_sequence_data:build_train_loader`. This loader does
not accept a visibility condition or condition index. It keeps all frames in
each episode in ascending `frame_indices` order and can shuffle only the episode
order. For the current full release, pin the expected inventory in the calling
config:

```python
import torch

from training.gnn.five_mode_sequence_data import build_train_loader

config = {
    "data": {
        "manifest": "outputs/five_mode_recovery/release/manifest.json",
        "include_surface": False,
        "expected_train_episodes": 3600,
        "expected_train_frames": 681842,
        "shuffle_episodes": True,
        "num_workers": 0,
        "max_edge_m": 0.08,
    },
    "training": {"episode_batch_size": 4},
}
loader, identity = build_train_loader(config, torch.Generator().manual_seed(0))
for batch in loader:
    frame_graphs = batch.graph       # B*T B2-2 Delaunay graphs in time order
    valid_frames = batch.frame_mask  # [B,T], excluding sequence padding
    frame_numbers = batch.frame_indices
```

The loader verifies each episode's `corners`, optional Peg `surface`, and
`targets` members against their release checksums when read. It rejects missing
or noncontiguous frame indices. Variable episode lengths are right-padded in a
batch; padded graph frames have no valid nodes and frame index `-1`. Use
`batch.reshape_frames(frame_encoder(batch.graph))` to restore a frame encoder's
`[B*T,...]` result to `[B,T,...]`, and call `loader.dataset.close()` after a
single-process read when the loader will no longer be reused.

Frame edges follow B2-2 rather than the older Five-mode complete-graph baseline.
The loader triangulates Peg and Box/Hole nodes separately, emits both directions
for every retained Delaunay link, rejects links longer than `data.max_edge_m`
(8 cm by default), and creates no Peg-to-Box/Hole cross edge. Detector corners
are vertex nodes and optional TAPNext++ points are non-corner tracker nodes.

For fresh training on the full release (900 seeds), place the release
`manifest.json`, `condition_index.json`, and its 18 TAR shards under
`dataset/` in the training checkout on a PyTorch CUDA host. The detector
index selects the same 502,598 training and 60,011 validation frames for
both variants. Each run initializes new weights and uses validation loss
for early stopping.

```bash
python main.py --config graph/configs/train_five_mode_condition_1_detector_corners.yaml
python main.py --config graph/configs/train_five_mode_condition_1_peg_surface.yaml
```

Tune `training.batch_size` for the target GPU. The complete directed graph
requires more memory when the Peg surface tracks are included.

Run the paired evaluation on the 62,309 held-out `condition_1` frames
using the same release files. Set the two checkpoint paths in the config to
the completed runs' `best.pt` files.

```bash
python main.py --config graph/configs/evaluate_five_mode_condition_1.yaml
```

To fine-tune both validation-best Condition 1 models on the detector-selected
Condition 2 split, place each source `best.pt` at the `initial_weights` path in
its config. Both recipes use 43,383 training frames, 6,744 validation frames,
and validation-loss early stopping.

```bash
python main.py --config graph/configs/finetune_five_mode_condition_2_detector_corners.yaml
python main.py --config graph/configs/finetune_five_mode_condition_2_peg_surface.yaml
```

After both runs finish, set their validation-best checkpoint paths in the paired
evaluation config and evaluate the 5,181 detector-selected Condition 2 test
frames.

```bash
python main.py --config graph/configs/evaluate_five_mode_condition_2_finetuning_comparison.yaml
```

For staged Peg-surface fine-tuning, use the same release and place the
pinned Condition 1 validation-best checkpoint at the config's
`initial_weights` path.

```bash
python main.py --config graph/configs/finetune_five_mode_peg_surface_curriculum.yaml
```

The three stages sample 80,000 training frames per epoch at Condition 1
to Condition 2 ratios of 80:20, 60:40, and 50:50. The selection criterion
is Condition 2 validation endpoint error, with Condition 1 validation
endpoint error capped at 110% of the initial checkpoint's value. Each
stage records its handoff checkpoint in `selection.json`. A stage can
hand off its original input if no eligible checkpoint improves it.

For the earlier visibility-based routing comparison, use the same release and the three
SHA-256-pinned checkpoints in the config. The RTX 3090 recipe fixes the
57-epoch model on Condition 1 and selects the Condition 2 checkpoint by
mean endpoint error on the validation split.

```bash
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_five_mode_routed.yaml
```

The evaluator saves per-frame test endpoints, mean and P95 errors, and
GT-adjusted changes across adjacent frames in the same episode. Its mixed-model
candidate selection is a historical diagnostic, separate from the fixed C1
epoch-57/C2 epoch-35 inference mapping below. See the
[Five-mode routing result](../docs/GNN_leaderboard_Nov_wee3.md#five-mode-가시성-조건별-모델-선택).

For current-frame inference, `graph.five_mode_hard_routing` combines detector
execution, condition classification, complete directed graph construction, and
selection of the Condition 1 or Condition 2 Peg-surface expert. The detector is
called once; its merged corners are used unchanged for both classification and
graph corner nodes. Pass only the current frame's visible, depth-valid,
non-corner TAPNext++ Peg points as `peg_surface_xyz`.

```python
from graph.five_mode_hard_routing import build_hard_routing_pipeline

pipeline = build_hard_routing_pipeline({
    "device": "cuda:0",
    "detector_device": "cuda:0",
    "coplanarity_tolerance_m": 0.001,
    "checkpoints": {
        "condition_1": {
            "path": "outputs/five_mode_condition_1_peg_surface/adddc72d35f94deb995385af2e3f89e7/best.pt",
            "sha256": "9c2b05a27353acd001a5a342b137f7df58f6e824c9dc4a0cc89034a7a6ea703a",
        },
        "condition_2": {
            "path": "outputs/five_mode_condition_2_peg_surface/bd92c40236124b8f84a945f90f3544f3/best.pt",
            "sha256": "4720dcaadd73a72c98fa0197c6fc26f53a0830995d19a4368e05473054097311",
        },
    },
})
result = pipeline.predict_rgbd(
    rgb, depth_raw_mm, intrinsic_cv, extrinsic_cv,
    peg_surface_xyz, tcp_xyz,
)
if result.prediction_status == "unavailable":
    # condition_3 has no configured expert yet.
    handle_condition_3(result)
else:
    consume_object21(result.object21, result.condition)
```

The two checkpoint hashes and their recorded training condition are verified at
load time, together with the Peg-surface model and packed-data contracts. The
result includes corner/surface counts and `condition_evidence` with Peg plane
RMS, noncoplanarity and tolerance. A Condition 3 frame returns
`prediction_status="unavailable"`, `object21=None` and `selected_model=None`
without running either expert. `predict_detection` is the equivalent entrypoint
when the caller has already run the current-frame detector. The output tensor
uses the model device; detector tensors may use another device. Pass empty
surface inputs as shape `[0,3]`. Object21 centers are TCP-relative with world
axes, and `object21[7:10]` is the world-aligned Peg endpoint relative to Hole.

With the same release and the selected 50:50 checkpoint, run the
Condition 2 endpoint diagnostics on the CUDA host:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_five_mode_endpoint_diagnostics.yaml
```

The evaluator reports disjoint visibility groups and the effect of
substituting each geometric term with its GT value after inference. See
the [endpoint diagnostics](../docs/GNN_leaderboard_Nov_wee3.md#five-mode-condition_2-끝점-오차-진단).

With the full supervised release, its `original_rgbd` archives, `zstd`,
and the detector-only Condition 1 best checkpoint, run the recent-corner
completion diagnostic on validation from the repository root:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_five_mode_corner_oracle.yaml
```

The frozen GNN compares current detector corners with one or two recent GT
corners added. The evaluator verifies the detector conditions against
the release and saves paired errors and candidate coverage. See the
[evaluation protocol](../docs/corner_prediction.md#recent-gt-corner-completion-for-object21).

To train a detector-history recovery and selection model, use the supervised
release shards, detector condition index and the same detector-only checkpoint.
The local recipe uses PyTorch, NumPy, SciPy and PyYAML. It selects a checkpoint
by Condition 2 validation Object21 loss and compares it with recency rules on
the full validation cohort.

```bash
WANDB_MODE=disabled python main.py --config graph/configs/train_recent_corner_completion.yaml
```

For the held-out comparison, use the saved epoch-6 checkpoint and the same
release. The evaluator sends Condition 1 directly to the detector-only GNN
and applies learned and recency selection to Conditions 2 and 3.

```bash
WANDB_MODE=disabled python main.py --config graph/configs/evaluate_recent_corner_completion.yaml
```

With the held-out `test_errors.npz`, replay the saved model to audit which
Hole rim candidates reached the GNN on Condition 3:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/diagnose_recent_corner_hole.yaml
```

See the [recent-corner training protocol](../docs/corner_prediction.md#learned-recent-corner-completion-for-object21).

With the same Five-mode release, frozen Condition 1 GNN checkpoint, and
saved recent-model validation trace, run matched persistent trackers with
and without a shape loss from the repository root. The local recipe uses
PyTorch, NumPy, SciPy and PyYAML.

```bash
WANDB_MODE=disabled python main.py --config graph/configs/train_persistent_corner_tracking.yaml
```

The pilot trains on an equal sample of Conditions 2 and 3 and selects
each checkpoint by their mean validation Object21 loss. The result includes
the full validation split and groups Condition 3 by the last Hole rim detection.
See [persistent corner tracking](../docs/corner_prediction.md#persistent-corner-tracking-with-shape-loss).

### Recovered subset

On a PyTorch CUDA host, place the supervised subset manifest and its pinned
`batches/` learning archives under `dataset/` in the training checkout.
Run from that checkout with the experiment YAML:

```bash
python main.py --config condition_1_detector_corners.yaml
python main.py --config condition_1_peg_surface.yaml
```

The recipes use `training.gnn.packed_five_mode_data:build_data` to decode and
check the train and validation frames once and keep the observations and
targets on the device. Both recipes select `data.condition: condition_1`
from the same frozen splits. Every corner comes from the current-frame detector.
`data.include_surface` adds non-corner Peg tracks. See
[Five-mode graph input](../docs/data_collection.md#five-mode-graph-input). The trainer
records the effective configuration, validation metrics and checkpoints via
ExperimentLogger.

With both runs complete and their checkpoint directories in the checkout, run
the paired holdout evaluation:

```bash
python main.py --config graph/configs/evaluate_recovered_five_mode.yaml
```

The config selects both validation-best snapshots, verifies matching held-out
frame IDs and targets, and records mean 3D errors through ExperimentLogger.

For two-condition fine-tuning, `data.condition` selects the primary train
condition and the validation condition. Set `data.balanced_train_condition`
to add a second training condition. Batches default to 50:50. Use
`data.train_mix.weights` to set integer weights by condition and
`data.train_mix.frames_per_epoch` to fix the total sampled training frames.
Each epoch draws without replacement from both train pools. The batch size
must be divisible by the sum of the weights.

The guarded Peg-surface continuation uses
`training.gnn.finetune_five_mode_curriculum:train` to evaluate both validation
conditions and retain a checkpoint only when its Condition 1 endpoint error
stays within the configured degradation limit. After that decision is fixed,
`graph/configs/evaluate_five_mode_peg_surface_balanced_conditions.yaml` compares
the selected checkpoint and the unselected final epoch on both pinned test
cohorts. This diagnostic comparison records full Object21 metrics and does not
select or replace a checkpoint using test results.

## B2-2 corner tracking

With the prepared train/validation episode manifest from the per-frame corner
and non-corner Peg surface caches, run:

```bash
.venv/bin/python main.py --config graph/configs/state_estimation/spatial_corner_tracking.yaml
```

B2-2 applies spatial message passing before per-vertex causal attention. It
predicts XYZ displacements from the last actual observation and passes through
currently observed coordinates. The default 20-epoch CPU recipe applies causal
attention over each complete chronological episode with a batch size of two,
before any visibility-condition filter. The training logger reports
missing-corner RMSE and the last-observation baseline. The recorded full-history
experiment remains reproducible with
`spatial_corner_tracking_full_history.yaml`, which writes to its original output
root. See [the fixed-ID supervision contract](../docs/corner_prediction.md#b2-2-spatial-first-corner-recovery).

## Spatial-temporal corner tracking

The B2-3 workflow adds TAPNext++ surface context to a detector-only tracker:
an object-internal Delaunay GNN encodes each frame, a causal Transformer
forecasts and associates corner tracks, and a separately trained complete-graph
GNN estimates Object21 from frozen tracker rollouts. Training, evaluation,
input contracts and the four root-entrypoint configs are documented in
[spatial-temporal corner tracking](../docs/spatial_temporal_corner_tracking.md).

## Corner visibility GNN

For a new Five-mode visibility index, use the
[per-frame detector corner workflow](../docs/data_collection.md#꼭짓점-가시성-프레임-인덱스).
For current Five-mode training and evaluation, use the
[Five-mode graph inputs](../docs/data_collection.md#five-mode-graph-input).
Both corner nodes and visibility labels come from the current-frame detector.
Optional tracked non-corner surface points contribute only to the graph.

The recipes below reproduce earlier experiments with their recorded inputs
and frame selections. Shared-cache access follows the
[repository rules](../AGENTS.md#forbidden-cache-access).

With the Pilot and Recovery raw datasets, their shared TAPNext++ caches, and the
[historical corner visibility index](../docs/data_collection.md#기존-추적-기반-프레임-선택)
in `outputs/`, run from the repository root:

```bash
OMP_NUM_THREADS=4 .venv/bin/python main.py --config graph/configs/train_corner_visibility_gnn.yaml
```

The config trains `VertexEdgeGatedMPNN` from scratch for 20 epochs on
`condition_1/train` and scores `condition_1/validation` each epoch. Each graph
uses the visible TAPNext++ corner queries with valid depth and complete directed
edges. The visibility index uses GT to select frames. Graph nodes use observed
coordinates and detector object types, and the Object21 targets use saved GT.
The common GNN trainer records progress and epoch metrics through ExperimentLogger
and writes the effective config, dataset identity, history, and checkpoints
under `outputs/corner_visibility_gnn/<run_id>/`.

To continue the 20-epoch run, set `resume` in the
[continuation config](configs/resume_corner_visibility_gnn.yaml) to its
`last.pt` and run:

```bash
OMP_NUM_THREADS=4 .venv/bin/python main.py --config graph/configs/resume_corner_visibility_gnn.yaml
```

This config caps the run at epoch 40, with validation-loss patience of eight
epochs and a minimum improvement of 0.001. The stopping counter starts with the
resumed validation scores, and checkpoint selection carries forward the best
validation loss from the original run.

The 40-epoch vertex input comparison uses these configs with the same root
entrypoint and the same `condition_1` inputs and prerequisites:

| Config | Graph inputs |
| --- | --- |
| [train_matched_vertex_gnn.yaml](configs/train_matched_vertex_gnn.yaml) | Observed coordinates retained by one-to-one GT vertex matching, with TCP |
| [train_corner_gnn_without_tcp.yaml](configs/train_corner_gnn_without_tcp.yaml) | All depth-valid observed corners, with the world origin as the coordinate reference for the graph and target |
| [train_oracle_vertex_gnn.yaml](configs/train_oracle_vertex_gnn.yaml) | Same vertices as the matched run, with exact GT coordinates and TCP |

Each run starts from seed 0 and evaluates validation loss every epoch. The
matched and oracle inputs depend on GT and are intended for offline comparison.

To train for 40 epochs on per-frame detector coordinates, use the raw datasets,
the same visibility index, and the Pilot/Recovery detector NPZ archives and
manifests listed in [train_direct_corner_gnn.yaml](configs/train_direct_corner_gnn.yaml).

```bash
OMP_NUM_THREADS=4 .venv/bin/python main.py --config graph/configs/train_direct_corner_gnn.yaml
```

This run keeps the same frame selection and TCP reference. Each graph uses the
detector's reported 3D corners and object types for that frame. Frames with no
detected corners still contribute their Object21 target. The original
visibility index determines membership, so its TAPNext++ and GT criteria
still apply when selecting frames.

The [expanded direct detector recipe](configs/train_expanded_direct_corner_gnn.yaml)
adds Five-mode `condition_1/train` frames selected by the
[historical index](../docs/data_collection.md#기존-추적-기반-프레임-선택)
to the original training set. It
requires the compact, checksum-verified detector cache at the path configured
under `data.imported_direct_train`. With that cache in place, run:

```bash
OMP_NUM_THREADS=4 .venv/bin/python main.py --config graph/configs/train_expanded_direct_corner_gnn.yaml
```

The 40-epoch run starts from fresh weights and uses the original validation
frame list. The original test split is reserved for evaluation.

The [expanded corner and surface recipe](configs/train_expanded_surface_corner_gnn.yaml)
uses the same split and fresh 40-epoch schedule. It requires the original and
imported compact caches configured under `data`. Each frame graph combines the
direct detector's 3D corners with up to 32 depth-valid TAPNext++ Peg surface
points, excluding points within two pixels of detector corners. It uses the
same complete directed graph topology and TCP reference as the direct-corner run.

```bash
OMP_NUM_THREADS=4 .venv/bin/python main.py --config graph/configs/train_expanded_surface_corner_gnn.yaml
```

The [Condition 2 fine-tuning recipe](configs/finetune_condition_2_surface_corner_gnn.yaml)
loads the expanded surface model's 44-epoch `best.pt` weights and requires the
compact Condition 2 train and validation cache referenced in the config.
It uses validation loss for early stopping.

```bash
OMP_NUM_THREADS=4 .venv/bin/python main.py --config graph/configs/finetune_condition_2_surface_corner_gnn.yaml
```

For the expanded Condition 2 run, use the same recipe with the added
Five-mode train-only detector-corner and surface cache in `data.imported_surface`.
Set `data.expected_frames.train` to the sum of the original and imported
selection counts. The generated run config records both inputs and keeps
the original Condition 2 validation split.

The [mixed Condition 1 and 2 recipe](configs/finetune_mixed_condition_surface_corner_gnn.yaml)
requires both conditions' original and imported compact graph caches. It
shuffles the combined train frames and selects checkpoints using the original
Condition 2 validation split.

```bash
OMP_NUM_THREADS=4 .venv/bin/python main.py --config graph/configs/finetune_mixed_condition_surface_corner_gnn.yaml
```

For detector-coordinate augmentation, copy either expanded recipe and set
`data.corner_jitter_std_mm: 1.0`. The value is the Gaussian standard deviation
per world-coordinate axis in millimetres. The surface recipe also needs
`data.corner_reference_config: graph/configs/train_expanded_direct_corner_gnn.yaml`
to verify the detector-node prefix of each training graph. Run the copied
config with the same `main.py --config` command. The train collator samples
fresh corner noise on each batch and rebuilds edge geometry. Validation reads
clean coordinates. A missing or zero standard deviation uses the clean recipe.

The following recipes use the clean models' `best.pt` files as initial weights
and train for 10 epochs with 1 mm jitter and a fresh AdamW optimizer. The input
cache prerequisites are the same as for the matching expanded recipes.

```bash
OMP_NUM_THREADS=4 .venv/bin/python main.py --config graph/configs/finetune_expanded_direct_corner_gnn_jitter.yaml
OMP_NUM_THREADS=4 .venv/bin/python main.py --config graph/configs/finetune_expanded_surface_corner_gnn_jitter.yaml
```

For the stronger perturbation experiment, use the same commands with
`finetune_expanded_direct_corner_gnn_strong_jitter.yaml` and
`finetune_expanded_surface_corner_gnn_strong_jitter.yaml`. These recipes start
from the 1 mm fine-tuning best weights and train for 10 epochs with a 5 mm
per-axis standard deviation. The paired test compares clean inputs and per-axis
standard deviations of 4, 5, and 6 mm.

The continuation recipes `resume_expanded_direct_corner_gnn_strong_jitter.yaml`
and `resume_expanded_surface_corner_gnn_strong_jitter.yaml` restore the 10-epoch
`last.pt` and cap training at epoch 20. Run either with `main.py --config`.
They keep the original validation frame IDs and fix the validation perturbation
per frame using `data.validation_corner_jitter_std_mm` and
`data.validation_corner_jitter_seed`. The restored checkpoint is evaluated
under the noisy validation condition before early stopping begins.

## Fixed-Hole Expert curriculum

With CUDA PyTorch, SciPy, PyYAML, the paired corner and State36 caches, and the
frozen reset-seed split and Robot9 archive listed in the config, run:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/fixed_hole_expert_curriculum.yaml
```

The split groups all four policy variants by reset seed. This first stage uses
fixed-Hole Expert episodes, with 80 train and 10 validation seeds. The historical
seed-0 Transformer/GNN pair supplies a validation baseline. The new Transformer
and GNN each start with fresh weights. Detector-to-GT matching uses the same
object kind and a 3 mm one-to-one gate for offline supervision. Detector-only
rollouts use measured robot observations and the existing 30 mm association.
The reserved test seed group stays outside training and model selection.

After the 20-epoch run, continue its GNN with the same frozen Transformer,
optimizer state, and reset-seed split:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/fixed_hole_expert_gnn_continuation.yaml
```

The continuation config pins the source checkpoints and rollout receipt. It
verifies the regenerated rollouts and replays epoch 20 on the original
validation frames before updating the GNN. The new run's `comparison.json`
records validation loss, Peg and Hole rotation errors, and Peg endpoint–Hole
position error for epochs 20 through 30.

With the original tracker checkpoint and sequence cache, run the Transformer
continuation independently through epoch 40:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/fixed_hole_expert_transformer_continuation.yaml
```

The fixed-ID trainer restores the optimizer and shuffle state and records
validation loss and corner RMSE for all, visible, and missing targets in
`history.json` alongside separate `best.pt` and `last.pt` checkpoints.

With the epoch 40 checkpoints and the same CUDA environment and sequence cache,
resume the Transformer with validation-loss early stopping:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/fixed_hole_expert_transformer_early_stopping.yaml
```

This run uses the original train split for updates and the validation split to
select the best checkpoint and stop after 15 epochs without improvement. It
saves `epoch_045.pt`, `epoch_050.pt`, and subsequent five-epoch checkpoints
alongside `best.pt` and `last.pt`. The total epoch budget is 300.

With the same caches and the selected epoch 208 Transformer, run a fresh GNN:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/fixed_hole_expert_gnn_early_stopping.yaml
```

This recipe freezes the Transformer, regenerates `rollout_inputs.pt`, and
checks the original ordered validation frame IDs. GNN updates use the original
train split. Validation loss selects `best.pt` and stops the run after 15
epochs without improvement, with a maximum of 200 epochs. `history.json`
records validation loss, Peg and Hole rotation, and Peg endpoint–Hole errors.

## Object21D state estimation

From the repository root with PyTorch, NumPy, SciPy and PyYAML installed:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/compare.yaml
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/tapnext_smoke.yaml
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/shared_tapnext_smoke.yaml
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/pilot_recovery_cuda_smoke.yaml
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/fixed_hole_curriculum.yaml
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/fixed_hole_holdout.yaml
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/fixed_hole_spatial_first_resume.yaml
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/fixed_hole_spatial_first_holdout.yaml
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/fixed_hole_spatial_first_early_stopping.yaml
WANDB_MODE=disabled python main.py --config graph/configs/state_estimation/fixed_hole_spatial_first_loss_comparison.yaml
```

The comparison config defaults to model A on synthetic episodes with seeds
0, 1 and 2. Change `model.name` or use the individual YAML files in
`graph/configs/state_estimation/` to select B1, B2, C1 or C2. The TAPNext++
smoke config requires the saved surface replay manifest and its raw RGB-D
source. The shared-cache config runs all five models for one epoch on the same
reset-seed-disjoint subset. It requires `tapnext_state36.npz` and the matching
raw mixed-trajectory pilot manifest and frame files. The CUDA comparison adds
`tapnext_recovery_state36.npz` and its raw recovery dataset, checks the
existing split receipt, and samples the return-to-Expert interval. Set
`phase: evaluate` and `evaluation.checkpoint` to the output
`last.pt` to evaluate the validation-selected weights. See the
[Object21D guide and verification](../docs/state21_implementation_audit.md).

The fixed-Hole Object21D curriculum uses both caches and the frozen split. Set
`phase: prepare` in a copy of its YAML and run it once to prepare the shared
graphs. With CUDA PyTorch and the prepared graphs and split receipt available,
run one `phase: train` config per `model.name`. Each run starts with fresh
weights and automatically trains the four fixed-Hole stages in order. See the
[curriculum setup and schedule](../docs/state21_implementation_audit.md#fixed-hole-curriculum).
The holdout command needs the source caches and raw test frames, the curriculum
graph manifest, and the completed status files and final checkpoints for all
five models. It evaluates the same 40 test episodes and reports endpoint error
and settling time after the first renewed Expert action.
The B2 resume config needs its completed curriculum status, final checkpoint,
the saved graphs, and the split receipt. It extends the final stage to the
configured total epoch count using the saved optimizer and shuffle state.
The B2 holdout config scores the original and resumed checkpoints on the same
test frames and records the 3 mm endpoint hit rate and Object21D loss.
The B2 early-stopping config continues from the 10-epoch checkpoint on the
same graphs and stops according to validation Object21D loss.
The B2 loss comparison needs the frozen split receipt and either the source
caches and raw frames or the saved graph bundle and a verified GT pose archive.
It trains four variants with matched initial seeds and epoch budgets.
`comparison.json` records coverage and validation metrics. `comparison.md`
reports final-epoch corner RMSE and endpoint hit rate within 3 mm, with
changes relative to the baseline.

## Surface graph GNN smoke test

With the saved mixed-trajectory pilot, the TAPNext++ 256 checkpoint, both
Visual tracking submodules, and an Apple Silicon Python environment with
PyTorch, torchvision, MLX, OpenCV, SciPy and einops, run from the repository root:

```bash
PYTHONPATH=visual/third_party/tapnet:visual/third_party/co_tracker \
  WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python main.py --config graph/configs/surface_graph_gnn.yaml
```

To train from scratch on those graphs, copy the YAML, set `phase: train` and
point `graph_manifest` to the generated `manifest.json`. Run the same root
entrypoint with the copied config. `training.validation_every_updates` controls
how often the full validation set is scored and its Object21 errors are logged.
See the [graph, timing and validation protocol](../docs/gnn_training.md#surface-graph-gnn-smoke-test).

## Hole rim depth verification

With the saved fresh trajectory frames and the shared PyTorch environment,
compare the existing recovery with the fallback depth check:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/hole_depth_fallback.yaml
```

With ManiSkill and the saved protocol and exclusion manifests, collect
new trajectories for the fixed Hole rim comparison:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/hole_depth_fallback_fresh.yaml
```

With the same collection prerequisites, evaluate the fixed longer-span
replacement on a separate set of scene seeds:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/hole_rim_replacement_fresh.yaml
```

## Corner detector benchmark

Use the frozen benchmark dataset for current Peg, Hole rim and Box exterior
3 mm comparisons. With its saved frames and the shared PyTorch environment,
run the checksum-verified CPU replay:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/replay_corner_detection_benchmark.yaml
```

The [corner detector benchmark](../docs/corner_detection.md#corner-detector-benchmark)
records the dataset selection, baseline and collection command.

## Peg position correction

With ManiSkill and the saved mixed-trajectory camera protocol, collect a fresh
cohort and evaluate the fixed Peg position correction with and without centered
bounds:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/peg_plane_center_fresh.yaml
```

With the shared PyTorch environment and the saved Box exterior holdout
selection and trajectory frames, replay the fixed Peg correction on CPU:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/peg_position_holdout.yaml
```

Use `graph/configs/temporal/peg_position_correction.yaml` with the frozen
development benchmark to reproduce the development scores. Compare centered
Peg prism bounds with the position-corrected baseline on the same holdout:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/peg_plane_center_holdout.yaml
```

See [Peg trajectory position correction](../docs/corner_detection.md#peg-trajectory-position-correction)
for the settings and paired results.

## Box exterior corner recovery

With the frozen benchmark and the shared PyTorch environment, run the paired
CPU replay of the exterior pixel-center correction:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/box_exterior_recovery.yaml
```

With ManiSkill and the saved Hole recovery holdout manifests and trajectory
frames, replay the fixed correction on seeds disjoint from the tuning
benchmark. The workflow renders the initial frames using the saved protocol.

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/box_exterior_holdout.yaml
```

See [Box exterior corner recovery](../docs/corner_detection.md#box-exterior-corner-recovery)
for the selection rule and paired results.

## Trajectory Hole rim recovery

With the saved corner benchmark frames and the shared PyTorch environment,
replay the CPU detector and causal rim recovery in trajectory order:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/trajectory_rim_recovery.yaml
```

See [trajectory Hole rim recovery](../docs/corner_detection.md#trajectory-hole-rim-recovery)
for the stationary-Box assumption and comparison.

## Initial-state corner detector evaluation

With the restored Peg and Hole audit JSONs listed in the config and the shared
ManiSkill/PyTorch environment, re-render the 1,200 initial states and evaluate
the CPU RGB-D detector:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/initial_corner_detection.yaml
```

## Pilot corner detector evaluation

With the completed local mixed-trajectory pilot and shared ManiSkill/PyTorch
environment, run the CPU RGB-D detector and GT-supported scoring with:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/pilot_corner_detection.yaml
```

The fixed sampling, approximate visibility criterion and measured results are
described in [corner detection](../docs/corner_detection.md#pilot-trajectory-corner-detection-evaluation).

## Hole opening recovery evaluation

With the restored reset audits, the completed pilot dataset, their baseline
evaluation JSONs, and the shared ManiSkill/PyTorch environment, run the paired
CPU evaluations from the repository root. Each command compares the base
detector, the original opening recovery, and the refined opening recovery.

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/hole_rim_reset.yaml
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/hole_rim_pilot.yaml
```

For the fixed-parameter holdout, also provide the frozen pilot protocol and
the two tuning-result JSONs named in the holdout config. The evaluator renders
fresh resets and rollouts and keeps the sampled trajectory observations.

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/hole_rim_holdout.yaml
```

See [Hole opening recovery](../docs/corner_detection.md#hole-opening-recovery)
for the detector option, stage definitions, and results.

## Small geometry recovery evaluation

With the sampled frames and selection from the Hole holdout run, use the
shared ManiSkill/PyTorch environment to replay the development comparison:

```bash
WANDB_MODE=disabled OPENBLAS_NUM_THREADS=1 python main.py --config graph/configs/temporal/small_geometry.yaml
```

See [small geometry recovery](../docs/corner_detection.md#small-geometry-recovery).

## Independent generalization evaluation

With the frozen test dataset and original Transformer/GNN checkpoints available
at the paths in the config, evaluate the independent test on CPU with:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/independent_test_evaluation.yaml
```

The dataset, checkpoint, and detector prerequisites, visibility-matched
follow-up, and limits of the original leaderboard replay are documented in
[independent generalization test](../docs/detected_corner_gnn.md#independent-generalization-test-and-leaderboard-replay).

With the original held-out RGB-D, archived CPU predictions, and verified
checkpoint pairs available, run the corner rigidity audit with:

```bash
PYTHONPATH=outputs/corner_rigidity/detector_source/visual WANDB_MODE=disabled \
  python main.py --config graph/configs/temporal/corner_rigidity.yaml
```

See [held-out corner rigidity](../docs/detected_corner_gnn.md#held-out-corner-rigidity)
for checkpoint and detector setup, metrics, and the paired intervention.

With the same detector and checkpoint setup, compare the Transformer's
pre-association forecasts on expert and random actions:

```bash
PYTHONPATH=outputs/corner_rigidity/detector_source/visual WANDB_MODE=disabled \
  python main.py --config graph/configs/temporal/corner_action_generalization.yaml
```

Use `graph/configs/temporal/corner_action_generalization_matched_visibility.yaml`
for the separate visibility-matched recordings. See
[held-out action generalization](../docs/detected_corner_gnn.md#held-out-action-generalization).

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

The random-action diagnostic also has a selected-camera recipe:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/random_action_selected_camera.yaml
```

It requires the same simulator, detector and checkpoint prerequisites as the
original recipe. The recorded paired results and camera comparison are in
[the detector-only GNN guide](../docs/detected_corner_gnn.md#selected-camera-random-action-comparison).

To replay the seed-2 camera failure and inspect its exact detector and GNN
inputs, run `WANDB_MODE=disabled python main.py --config graph/configs/temporal/hole_input_diagnostics.yaml`
with the recorded results and checkpoints present. See the
[Hole input inspection](../docs/detected_corner_gnn.md#seed-2-selected-camera-hole-input-inspection).

To scan camera poses against saved random-action states, configure and run
`graph/configs/temporal/camera_visibility_search.yaml`. The evaluation and
visibility limits, held-out comparison recipes and recorded results are in the
[multi-state camera search](../docs/detected_corner_gnn.md#multi-state-camera-visibility-search-and-held-out-action-comparison).

For the broader randomized camera study, configure
`graph/configs/temporal/camera_distribution_search.yaml` and its separate
held-out counterpart. The sample construction and results are described in
[the camera study](../docs/detected_corner_gnn.md#broader-random-position-camera-study).
