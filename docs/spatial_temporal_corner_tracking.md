# Spatial-temporal corner tracking

This guide defines the B2-3 prototype as two separately trained stages. Stage A
optimizes detector-only corner tracking. Stage B freezes Stage A, creates
chronological rollouts, and trains a complete-graph `VertexEdgeGatedMPNN` to
estimate Object21. The six preserved detector-only leaderboard implementations
and checkpoints are not modified.

With a verified prepared dataset and the Five-mode release available on a CUDA
host, run the validation-only sequence from the repository root:

```bash
WANDB_MODE=disabled .venv/bin/python main.py \
  --config graph/configs/temporal/run_spatial_temporal_workflow.yaml
```

Prerequisites are PyTorch with CUDA, NumPy, SciPy, PyYAML and OpenCV for imported
geometry utilities. Raw preparation additionally needs the `zstd` executable.
The workflow records each effective stage recipe, pins the selected checkpoints
by SHA-256, calibrates association on validation, evaluates the full validation
split, freezes Stage A, and then trains/evaluates Stage B. It stops if full
validation exceeds the 128-track guard. It never starts held-out test evaluation.

## Model contract

Each current frame contains two observed node roles:

- `corner`: Peg, Box exterior, or Hole-rim coordinates produced by the current
  RGB-D corner detector;
- `surface`: TAPNext++ non-corner Peg or Box/Hole surface coordinates.

The spatial graph uses bidirectional, distance-limited Delaunay edges inside
each object. There are no Peg-to-Box edges in Stage A. The spatial GNN returns
node embeddings and one pooled context per object. A corner forecast at frame
`t` receives its associated node embedding from `t-1`, its observation age,
and the pooled current-object context at `t`. It never receives its own
current detector coordinate before association. A missing corner therefore
continues to receive current TAPNext++ surface context.

The Stage-A recipe explicitly sets `model.coordinate_frame: world`. The model
encodes current world XYZ, scaled in metres, without changing the shared graph
builder or spatial backbone. Encoding only TCP-relative XYZ made a joint Peg/TCP
translation invisible to the pooled context; a 139.75 mm counterexample produced
zero feature/context change. This would hide grasped-object motion during
occlusion. The world-coordinate mode retains that motion. Checkpoints predating
the explicit field load with their original TCP-relative interpretation.

The causal Transformer predicts displacement from the last actual detection,
one uncertainty value, and an association embedding. Online assignment allows
only same-object pairs inside the configured distance gate. It first maximizes
the number of matches and then minimizes position distance plus embedding cost.
Unmatched detections initialize tracks; unmatched tracks remain active. GT
corner identity and reference geometry are excluded from this forward path.

Canonical IDs in the prepared Stage-A dataset are supervision metadata. They
define target histories, association positives, reappearance events and shape
losses, but are not node features or Transformer inputs.

## Stage A: train and evaluate tracking

Prepare the full chronological train and development-validation inventory from
the current Five-mode release and its pinned raw RGB-D archives:

```bash
python main.py \
  --config data_collection/configs/spatial_temporal_corner_tracking.yaml
```

This produces 3,600 train and 450 validation episodes. It reads current
detector coordinates, TAPNext++ non-corner surface coordinates, and raw
simulator poses used only to create offline certified IDs and separate
reference XYZ. It does not accept a visibility condition. Each completed raw
batch has a checksummed receipt, so an interrupted build resumes without
silently trusting partial files. Do not point this workflow at the forbidden
legacy shared State36 caches.

Raw TAR entries need not be chronological, and `episode_index.json` need not
precede the frames. A separate streaming metadata pass resolves episode paths;
the pose pass places each frame at its explicit index and rejects missing or
duplicate frames. RGB-D images are never extracted to disk.

Audit every prepared episode against its checksum and the current detector and
surface caches with:

```bash
WANDB_MODE=disabled .venv/bin/python main.py \
  --config data_collection/configs/verify_spatial_temporal_corner_tracking.yaml
```

The audit checks all chronological offsets, unchanged observed coordinates,
offline certificates, unique episode/seed membership, split totals and dataset
identity. It relies on the preparation's verified raw pose pass rather than
re-reading raw GT. `data.source_manifest` may relocate the source release only
when its bytes match the original manifest SHA-256.

Historical Five-mode releases omit `surface_object_id` because their TAPNext++
surface stream is Peg-only; the preparation records this as
`historical_peg_only` and assigns those rows to Peg. New two-object surface
caches preserve their explicit per-row Peg or Box/Hole IDs.

Set the manifest path, device and output root, then run:

```bash
WANDB_MODE=disabled python main.py \
  --config graph/configs/temporal/train_spatial_temporal_corner_tracker.yaml
```

Training uses every train frame before visibility-condition filtering. The
objective combines visible/missing position forecast, association contrast,
within-object rigidity and uncertainty calibration. Checkpoint selection uses
development-validation missing-corner RMSE. Episode order may shuffle, but
frame order inside every episode remains chronological.

`data.graph_cache_root` optionally stores CPU episode graphs. Source episode
checksums are still checked on cache hits. Cache keys include graph/loader source
hashes, edge limit and NumPy/SciPy/Torch versions. Entries publish atomically for
multiworker loading. This changes preprocessing cost without changing episode
order or graph tensors. The batched association objective retains the same
within-object eligible negatives; tests compare its loss and gradients against
the original objectwise reference. Training records all loss components,
missing-corner hold baseline, throughput and peak CUDA tensor memory.

Set the selected checkpoint, its SHA-256 and the same manifest before running:

```bash
WANDB_MODE=disabled python main.py \
  --config graph/configs/temporal/evaluate_spatial_temporal_corner_tracker.yaml
```

Evaluation reports two distinct views:

- teacher-bound forecast RMSE and association accuracy on offline ID histories;
- detector-only online reappearance retention, ID switches, known-track
  conflicts, track growth and missing-track RMSE.

The evaluator removes canonical IDs from each graph before calling the online
tracker. Certificates are consulted only after each step to score the result.
The configured association gate and embedding weight are part of the evaluation
protocol and should be selected on a calibration split before a held-out run.

For a fixed validation-best checkpoint, configure its path and SHA-256 in
`graph/configs/temporal/calibrate_spatial_temporal_association.yaml` and run it
through `main.py`. The default sweep uses four deterministic validation seed
groups, retaining every mode for those seeds, with distance gates
10/20/30/50 mm and embedding weights 0/5/10 mm. All candidates use identical
episodes and frozen weights. Candidates exceeding 128 tracks fail explicitly.
Among feasible candidates, selection minimizes the sum of certified identity
error rate and reappearance failure rate, then missing RMSE, then mean final
track count. All candidates and the selected association are saved. Full
validation is evaluated afterward; this is development calibration, not an
independent estimate of generalization.

Online metrics distinguish pre-assignment forecasts from unchanged measured
coordinates. Reappearance RMSE scores the prior lifecycle track's forecast;
association accuracy excludes physical corners without prior certified history.
Incremental online attention retains every normalized state at every layer.
Only the current query is recomputed; no history window is truncated. Tests
compare this path with full-sequence causal attention, including late tracks.

## Transfer to a host with the same Five-mode release

If uploading observed geometry again is expensive, export only GT reference
targets and offline certified IDs:

```bash
.venv/bin/python main.py \
  --config data_collection/configs/export_spatial_temporal_labels.yaml
```

Copy the resulting supervision packet to the GPU host. Set its packet path,
the checksum-identical source release manifest and the prepared output root in
`data_collection/configs/restore_spatial_temporal_labels.yaml`, then run that
config through `main.py`. Restoration obtains all observed coordinates from the
remote detector/surface release. It requires **every reconstructed NPZ file's
SHA-256 to equal the original local file's checksum**, then copies the unchanged
prepared manifest. It never substitutes approximate GT geometry or recomputes
certificates at a different precision. Test episode members remain unopened.

## Stage B: frozen rollouts and Object21

Stage B reads the unfiltered recovered Five-mode manifest. It replays every
train and validation episode in chronological order using the frozen Stage-A
checkpoint. Current detector corners and both objects' TAPNext++ non-corner
surface points enter Stage A. Only the resulting corner tracks are passed to
the downstream estimator.

The downstream graph is complete and directed, excluding self-edges. Each
track node contains XYZ, Peg/Box type, and three reliability features:
current detection flag, log-scaled observation age and predicted uncertainty.
Track and canonical IDs are not GNN features. This complete graph supplies the
Peg-to-Box interaction intentionally omitted from the Stage-A Delaunay graph.

Set the frozen tracker, full Five-mode manifest and expected train inventory;
the supplied recipe pins 3,600 train episodes:

```bash
WANDB_MODE=disabled python main.py \
  --config graph/configs/temporal/train_spatial_temporal_object21.yaml
```

The workflow saves `frozen_tracker_rollouts.pt` with a checksum before fitting
only the downstream GNN. Rollouts use ragged frame offsets rather than padding
the entire corpus to the episode with the most duplicate tracks. A configurable
track-count guard fails before the complete graph becomes unexpectedly
quadratic; it never truncates tracks. The cache records the tracker checkpoint identity, input
manifest, association settings, per-episode match/track counts and effective
config. Development-validation Object21 loss selects `best.pt`.

Evaluate the selected checkpoint and its exact sibling rollout cache with:

```bash
WANDB_MODE=disabled python main.py \
  --config graph/configs/temporal/evaluate_spatial_temporal_object21.yaml
```

The evaluator saves aligned frame IDs, Object21 predictions, targets, physical
error arrays and aggregate statistics. Test episodes remain unopened by the
supplied training and development-validation recipes.

## Current verification boundary

On 2026-11-18 KST, the full preparation and current-frame lineage audit passed
on macOS CPU from a dirty source snapshot based on `0a5e33d`:

| Split | Episodes | Frames |
| --- | ---: | ---: |
| train | 3,600 | 681,842 |
| validation | 450 | 84,081 |

All 4,050 episodes use `historical_peg_only` surfaces. The audit checked
11,701,590 detector rows, 19,605,700 surface rows and 10,898,699 certified
correspondences. No visibility filtering or test episode access occurred.
Dataset SHA-256 is
`5e12276ee0f1888bd62a1f20cb663efaa41d0ad955a22d4cd49c1c3488124325`;
manifest SHA-256 is
`55707055bee93332a6ba75383e6e478322ae3e8011b18af87cff65d5fb168aa9`.
See the local [audit](../outputs/spatial_temporal_corner_tracking/verification.json)
and [prepared manifest](../outputs/spatial_temporal_corner_tracking/data/manifest.json).

Actual CUDA forward/backward and a four-episode train/validation smoke ran on an
RTX 3090 24 GB with Torch `2.14.1+cu132`, Python 3.12.15, NumPy 2.5.5 and
SciPy 1.18.1 in a separate server `.venv`. CPU-only macOS uses Python 3.11,
Torch 2.14.1, NumPy 2.4.6 and SciPy 1.17.1. No simulator rendering was needed.
Small CUDA smoke results do not establish full-data model quality. Full training,
association calibration, frozen rollouts and Stage B results are recorded by
the workflow; held-out test evaluation requires a separate user request.

An initial TCP-relative training run, `699184c8d8a040778fb8ee159a5efd21`, was
interrupted after the observability counterexample above. Its checkpoints,
history and source snapshot are retained as a diagnostic baseline; its partial
results are not the requested full world-coordinate experiment. The corrected
recipe starts fresh with the same model dimensions, optimizer and batch size.

## Completed full-data validation

The full workflow completed on 2026-11-18 at 11:01 KST on the RTX 3090.
The workflow and all training/evaluation experiment records finished normally.
No held-out test episode was evaluated and no artifact was published remotely.

| Component | Run ID | Selection |
| --- | --- | --- |
| Workflow | `c075de63a7dd466797e7fcb9bcef6e68` | all five stages completed |
| Stage A | `52b03758bf794e3a859821ae67d54427` | 40 epochs; best epoch 34 |
| Association calibration | `dc3cb46cc24d4209af7fa21a4bdf0ba1` | 12 candidates, 20 validation episodes |
| Stage-A validation | `f0a2d290f5d94c559bf70e61c7a24f01` | all 450 episodes / 84,081 frames |
| Stage B | `e960d262ca2d491f97313f13fa1b692e` | early stopping at epoch 26; best epoch 16 |
| Stage-B validation | `d4c7ced08fd7491ab45ca8864cfeb00a` | all 84,081 validation frames |

Stage A selects by teacher-bound missing RMSE, 13.5940 mm, against a
55.0371 mm hold baseline. Its teacher-bound visible/reappearance RMSEs are
2.0804/10.1312 mm. The selected detector-only online association uses a 30 mm
distance gate and 10 mm embedding weight. Full validation online missing and
reappearance RMSEs are 19.9839 and 11.9018 mm; association accuracy is 99.8795%
and reappearance same-ID retention is 94.7189%. There are 1,426 identity switches
and 5,483 known-track conflicts. Mean final tracks per episode are 24.0556,
with a validation maximum of 54.

The frozen rollout cache contains all 765,923 train/validation frames with
15,592,332 ragged track rows. The largest frame has 67 tracks, below the
128-track guard. Completion verification checked all three artifact hashes,
unique frame IDs, chronological ragged offsets and the exact alignment of
validation prediction IDs, targets and observed/total track counts with the
frozen cache. Stage A remained frozen; semantic IDs are disabled in Stage B.

Stage-B normalized validation Object21 loss is 0.6143091. Physical validation
errors below are frame-weighted Euclidean distances, not corner RMSE:

| Metric | Mean | Median | P95 |
| --- | ---: | ---: | ---: |
| Peg position | 1.5385 mm | 0.8243 mm | 4.0214 mm |
| Hole position | 5.5579 mm | 1.3672 mm | 38.4766 mm |
| Peg endpoint position | 2.5822 mm | 1.0950 mm | 5.2481 mm |
| Peg endpoint relative to Hole | 7.0245 mm | 1.8002 mm | 39.4466 mm |
| Peg rotation | 1.6326 deg | 0.4716 deg | 3.8346 deg |
| Hole rotation | 0.6371 deg | 0.4651 deg | 1.7444 deg |

The Hole and relative-endpoint tails remain substantial despite the lower
medians. These are development-validation results after checkpoint selection
and calibration on validation. They do not establish held-out generalization
or closed-loop insertion success, and no matched downstream baseline comparison
was run here.

Exact artifact SHA-256 values:

- Stage-A `best.pt`:
  `ce43b0d36d3a8aee58cbcaa401cd28072d70b426b31666f663bc49d9e9702a72`
- Stage-B `best.pt`:
  `c5a08925a6040a48a54c54d5775f27c3d30dd988ecb306ab1c73f87899e94e83`
- `frozen_tracker_rollouts.pt`:
  `088c5d0b045e9219b74ce67961dbd8302625e9e68076df0ebbb33802d07b5e21`

The server root is `/workspace/spatial_temporal_corner_tracking`; checkpoints
and the rollout cache remain in their run directories under `outputs/`.
Local copies of configs, histories, calibration candidates and results are in
[`completed_workflow/`](../outputs/spatial_temporal_corner_tracking/server_results/completed_workflow/).
See the [completion verification](../outputs/spatial_temporal_corner_tracking/server_results/completed_workflow/completion_verification.json)
and [final physical-error statistics](../outputs/spatial_temporal_corner_tracking/server_results/completed_workflow/outputs/spatial_temporal_object21_evaluation/d4c7ced08fd7491ab45ca8864cfeb00a/results.json).
The large binary cache, checkpoints and per-frame prediction NPZ remain on the
server; this local record bundle contains metadata rather than those binaries.

## Held-out test with fixed validation choices

After explicit authorization on 2026-11-18, the test workflow fixes the Stage-A
epoch-34 and Stage-B epoch-16 checkpoints above, the 30 mm association gate and
10 mm embedding weight. It performs no optimizer step, checkpoint selection,
calibration sweep or visibility filtering on test.

Prepare the 450 test episodes / 86,179 frames separately:

```bash
WANDB_MODE=disabled .venv/bin/python main.py \
  --config data_collection/configs/spatial_temporal_test_data.yaml
```

The ordinary preparation entrypoint still refuses test. The explicit
`prepare_test` entrypoint checks seed separation and adds Object21 labels to
the existing offline corner-label arrays. Object21 and canonical reference
labels remain outside the model forward path. To move these labels to a host
with the pinned source release, use `export_spatial_temporal_test_labels.yaml`
and `restore_spatial_temporal_test_labels.yaml` under `data_collection/configs/`.
Restoration requires byte-identical episode checksums, including Object21 labels.

On the CUDA host, run:

```bash
WANDB_MODE=disabled .venv/bin/python main.py \
  --config graph/configs/temporal/evaluate_spatial_temporal_test.yaml
```

`evaluation.spatial_temporal_held_out` checks both checkpoint hashes and requires
association settings to equal the frozen Stage-B training identity. It validates
the full test inventory, source manifest lineage, episode order and seed
separation. Teacher-bound forecasts use offline labels. Detector-only tracking
removes canonical IDs before inference and collects frozen outputs during that
same online scoring pass. The downstream GNN receives only those tracks and
their reliability features. The existing 128-track guard stays fixed.

The new cache is `frozen_test_rollouts.pt`, format
`spatial_temporal_test_rollouts_v1`, explicitly labelled `test`. It never
replaces or relabels the training/validation rollout cache. The validation
evaluator retains its original cache-hash guard and shares the prediction and
physical-metric implementation with test evaluation. Outputs include aligned
`seed:policy_mode:frame` IDs, predictions, targets, physical errors, observed
and total track counts, and checkpoint/cache/prediction SHA-256 values.

Before test execution, GPU regression passed 60 tests. Replaying the previous
84,081-frame validation evaluation preserved frame IDs, targets and track
counts exactly; the maximum prediction component difference was
`2.980232238769531e-7` (not a bitwise inference match). No protected leaderboard
implementation or model weights were changed. Test results are reported
separately from validation results.

### Held-out results

Run `07c77c1f1f4740aebde8c747b8d110c9` completed normally at 13:09 KST on
2026-11-18. All 450 test episodes / 86,179 frames were evaluated with the same
checkpoint SHA-256 values and association settings as validation. Before/after
file hashes verified unchanged weights. No optimizer or selection step ran.

| Tracking metric | Validation | Held-out test |
| --- | ---: | ---: |
| Teacher-bound missing RMSE | 13.5940 mm | 12.4478 mm |
| Online missing RMSE | 19.9839 mm | 19.7708 mm |
| Online reappearance RMSE | 11.9018 mm | 10.4751 mm |
| Online association accuracy | 99.8795% | 99.8679% |
| Reappearance same-ID retention | 94.7189% | 95.7030% |
| Identity switches | 1,426 | 1,606 |
| Known-track conflicts | 5,483 | 6,764 |
| Mean final tracks per episode | 24.0556 | 24.6400 |
| Maximum tracks per episode | 54 | 56 |

The held-out missing-corner hold baseline is 58.7400 mm. Online missing RMSE
is approximately 66.3% lower. ID/error counts use the same definitions as
validation; raw counts span different frame/certificate inventories and are not
paired differences on identical samples.

| Object21 physical error | Validation mean | Test mean | Test median | Test P95 |
| --- | ---: | ---: | ---: | ---: |
| Peg position | 1.5385 mm | 1.5592 mm | 0.9355 mm | 4.2318 mm |
| Hole position | 5.5579 mm | 6.4307 mm | 1.4368 mm | 45.9096 mm |
| Peg endpoint position | 2.5822 mm | 1.9991 mm | 1.2323 mm | 5.2710 mm |
| Peg endpoint relative to Hole | 7.0245 mm | 7.2218 mm | 2.0306 mm | 49.1819 mm |
| Peg rotation | 1.6326 deg | 1.0940 deg | 0.5405 deg | 3.2735 deg |
| Hole rotation | 0.6371 deg | 0.8063 deg | 0.5850 deg | 2.2224 deg |

Test Object21 loss is 0.6321594, versus validation 0.6143091. Online tracking
point estimates remain close to validation. Hole/relative-vector large-error
tails remain the main limitation; this evaluation does not measure closed-loop
insertion success.

The separate test cache contains 1,778,249 ragged track rows and a maximum of
56 tracks per frame. Completion verification checked every prepared episode
checksum, source-ordered unique sample IDs, prediction/target alignment,
observed and total track counts, finite outputs and both unchanged checkpoint
hashes. Its artifact hashes are:

- `frozen_test_rollouts.pt`:
  `d76f72c6bceeeb9d2000192e8a0f3bf350c5bce3ce12a0c77012996ea326e4f3`
- `predictions.npz`:
  `50f3557367fff1a595cd292d19502136b09dab0162da8bf5b39ef520a71fe127`
- Prepared test manifest:
  `86971fa25ae752ac4e89a905b86df1c9992eaa246bdd12eff31abd4c9bd928f4`

The server artifacts are under
`/workspace/spatial_temporal_corner_tracking/outputs/spatial_temporal_held_out/07c77c1f1f4740aebde8c747b8d110c9/`.
Local [results](../outputs/spatial_temporal_corner_tracking/server_results/held_out/results.json),
[verification](../outputs/spatial_temporal_corner_tracking/server_results/held_out/verification.json),
[locked config](../outputs/spatial_temporal_corner_tracking/server_results/held_out/held_out_lock.json)
and source/validation-replay metadata are preserved in `server_results/held_out/`.
Binary rollout/prediction files remain on the server; no HF publishing occurred.
