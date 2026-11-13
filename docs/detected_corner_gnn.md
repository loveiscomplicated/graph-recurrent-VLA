# Detector-only corner association and GNN

This guide covers detector-only association, frozen rollout inputs, downstream
GNN training and saved-result visualization. [Fixed-ID prediction](corner_prediction.md)
supplies the frozen temporal model; [lifecycle tracking](corner_tracking.md) is
a separate experiment.

## Evaluation preservation

The following six [leaderboard results](experiment_leaderboard.md#전체-4102프레임-종합-순위)
are preservation targets for cleanup work. These displayed values are rounded;
the original per-frame arrays and full-precision JSON metrics are the references.

| Training protocol | Architecture | Normalized aggregate error | Hole-position mean |
| --- | --- | ---: | ---: |
| Resumed with early stopping | Transformer | 0.1162 | 1.831 mm |
| Resumed with early stopping | Mamba | 0.1196 | 1.749 mm |
| Fresh 40 epochs | Transformer | 0.1297 | 1.923 mm |
| Fresh 40 epochs | Mamba | 0.1433 | 1.970 mm |
| Initial 20 epochs | Transformer | 0.1694 | 2.304 mm |
| Initial 20 epochs | Mamba | 0.1847 | 2.549 mm |

Before changing a numerical dependency, preserve its source bytes, the selected
checkpoint hashes, frozen temporal checkpoint identities, rollout cache hashes,
effective configs, and reference predictions. The evaluation uses seeds 0/1/2,
the same ordered 4,102 validation frame IDs and Object21 GT, and the original
batch size, dtype, device, package versions and aggregation order. Compare every
per-frame prediction and error array and every full-precision summary exactly.
Rounded table equality or a tolerance-based match does not establish exact
reproduction. Training refactors also need initialization and optimizer/RNG
restoration checks, including the original stopping and checkpoint selection.

When the complete evaluation inputs or original execution environment are
unavailable, retain the affected implementation and its dependencies. Saved
array checks validate existing results; report them separately from new model
inference. Existing checkpoint/source checks remain in force.

For the 2026-11-09 cleanup, the local baseline is in
[baseline.json](../outputs/refactoring_baseline/baseline.json). The local host
has no CUDA and no per-seed frozen rollout caches. The 20-epoch and resumed
results have saved per-frame predictions and GNN checkpoints; the 40-epoch local
copy has summary JSON only. Consequently the six-result numerical dependency
tree is preserved, and full inference reproduction is deferred until the
original inputs and CUDA environment are available. The baseline and other
`outputs/` artifacts are local verification evidence and are excluded from Git.

The [saved-result audit](../outputs/refactoring_baseline/protected_result_verification.json)
confirmed unchanged bytes for 52 source/config files and 581 artifacts. For the
20-epoch and resumed results, all 16 stored error metrics reproduced the original
full-precision aggregate statistics exactly across the 12 prediction archives,
with identical frame IDs and GT. The 40-epoch summaries retained their original
bytes. This audit did not execute model inference.

## Published checkpoint backup

The 2026-11-09 KST backup preserves every checkpoint found in the automatic
association GNN workflows and their corner-predictor lineage. The private
[complete checkpoint index](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/blob/53587c96ad1a620a152a0bc6a23774a06b2a2dfd/graph/a6a3d5bc7df34172957302eb9524ba25/artifacts/metrics/25baea12a4f5d553282f75ce256ed6b736e2735a5eb317438498793603ba868c/checkpoint_index.json)
contains original run IDs, source paths, sizes, SHA-256 and immutable download
links for each file.

| Workflow | Checkpoints |
| --- | ---: |
| Detector-only GNN, initial 20 epochs | 12 |
| Detector-only GNN, fresh 40 epochs | 12 |
| Detector-only GNN, resumed early stopping | 12 |
| Shared corner predictor, initial training | 12 |
| Shared corner predictor, continuation | 12 |
| Shared corner predictor, early stopping | 12 |
| Corner-predictor preflight | 2 |
| Total | 74 |

Counts include both best/last files for Mamba/Transformer seeds 0/1/2, plus
the two preflight files. The 18 registered frozen rollout-cache files are also
preserved. Existing `tools.publish_experiment` published checkpoints, configs,
metrics and original run metadata to the Model repository; the three comparison
runs' prediction arrays and metrics use the private Dataset repository.
Original experiment IDs and training provenance were retained. Source code was
committed locally as `96cb93d`; this is the backup source revision, rather than
a replacement for the original runs' recorded Git provenance.

Independent remote checks verified 412 published files and matched all 74
checkpoints and 18 caches against server-side LFS SHA-256 and sizes. The index
and its publication files were checked separately. Saved arrays reproduced all
six full-precision result groups, including the fresh 40-epoch results; all 18
archives have identical frame IDs and GT. Model inference was not rerun.

Vast.ai instance `59441593` was stopped only after publication and verification:
the API reported `actual_status=exited`, `cur_state=stopped`, and the instance
remained present with its disk. No destroy operation was requested; other
instances were left alone. Local copies and receipts are under
`outputs/automatic_corner_backup/`, including the [remote verification](../outputs/automatic_corner_backup/remote_verification.json),
[saved-result audit](../outputs/automatic_corner_backup/six_result_verification.json)
and [stop confirmation](../outputs/automatic_corner_backup/instance_stop.json).
Historical notes below about artifacts not being uploaded refer to the state
at the original experiment's completion.

## Detector-only corner association

### Moving Hole visualization

`evaluation.moving_hole:visualize` renders a new scene with the configured
leaderboard Transformer and GNN checkpoints. Run the command in the
[Graph README](../graph/README.md#corner-prediction-and-tracking); set the two
checkpoint paths, device, frame count, translation and output directory in its
YAML. The temporal checkpoint hash is checked against the GNN's recorded source.
The existing detector, tracker, graph construction and experiment logger are
reused. The video shows actual detections and predicted tracks with internal IDs,
plus ground-truth and estimated Hole centers. Results JSON records each center,
error, track count and new-track count and the two checkpoint hashes.

The simulator's Hole is part of the kinematic box mesh, so this experiment moves
the entire box in a straight line; it does not move the hole independently within
the box. It advances the simulation once per frame. Elapsed time is the frame
index divided by ManiSkill's control frequency. The default recipe uses zero
actions; set `policy` to `module:factory` for a policy implementing the existing
`reset(env, seed)` and `act(observation)` interface. Playback FPS is independently
configured.

`motion.start_seconds` and `motion.end_seconds` set the permitted time window.
`continuous` interpolates the configured world translation over
`duration_seconds`; `jump` applies it once at the start. Before each move, the
guard uses simulator GT to measure the Euclidean distance from the peg head to
the Hole center. At or below `minimum_distance_m` it permanently stops motion,
even if the peg subsequently retreats. It also rejects a proposed translation
if any point along its straight path enters the same radius, including for a
jump. This conservative GT-based proximity proxy is checked at control-step
boundaries; it is not an insertion-phase detector or a deployable perception
signal. A policy can bring the peg closer between checks. Frame records include
time, pre-move distance, actual offset and guard status. GT is otherwise used for
display and error, never model input.

The optional `stop_expert_insert` setting additionally latches the guard when
the policy's `phase` is `insert`. It requires a policy with that field; the
existing `data_collection.scripted_policy:make_policy` exposes it and requires
`control_mode: pd_ee_delta_pose`. This expert uses simulator GT for control,
including the current moving goal; it does not use the GNN predictions. Phase
labels and simulator success are recorded per frame, including the terminal
frame. Configure `motion.mode: jump` for the same expert with an abrupt move.

New tracks can arise from newly visible corners as well as failed associations;
track counts alone do not establish identity switches.

On 2026-11-11, the Mac CPU simulator, detector and model ran seed 0 for 60
frames with 60 mm total world-X translation. The Hole-position error averaged
1.27 mm (maximum 2.75 mm); 14 initial tracks grew to 18. This is one illustrated
scene, not an independent accuracy or robustness benchmark. CUDA was not run.
The [video](../outputs/moving_hole/2e4a7b6287b44d81963946a04cd27a31/moving_hole.mp4)
and [per-frame results](../outputs/moving_hole/2e4a7b6287b44d81963946a04cd27a31/results.json)
are ignored local artifacts and may not exist in another checkout.

With the time-window implementation, 2026-11-11 Mac CPU zero-action examples
used 20 Hz, a 0.5–2.5 s window, an 80 mm guard and a 60 mm displacement.
[Continuous motion](../outputs/moving_hole/547ebac312db4cd488594430c74a61e1/moving_hole.mp4)
ran from 0.5 to 2.0 s; mean Hole error was 1.04 mm across 70 frames.
[The jump](../outputs/moving_hole/bb42c59fdce54ee39c9dc6d5df372289/moving_hole.mp4)
occurred at 0.5 s and added six tracks at that frame; mean Hole error after the
jump was 29.66 mm. The peg remained at least 282 mm away in these examples, so
they do not demonstrate proximity stopping during an actual insertion. Guard
behavior has separate tests; these examples are not aggregate benchmarks.

The [expert example](../outputs/moving_hole/94eebbdfa83748e6940242a6342983c8/moving_hole.mp4)
on the same Mac CPU used the expert recipe, seed 0 and a 3.5 s start. Motion
stopped at 14 mm when the expert entered insertion at 4.25 s; the peg head was
still 127 mm from the Hole center, showing why the distance guard alone does
not establish the phase boundary. The offset remained fixed throughout the
insert phase and ManiSkill success became true at 5.1 s. The 80 mm proximity
guard was also reached at 4.65 s. This is one GT-guided episode, not a policy
success-rate measurement. Its [frame records](../outputs/moving_hole/94eebbdfa83748e6940242a6342983c8/results.json)
include the guard, phase and Object21 errors.

The matching [abrupt-motion expert video](../outputs/moving_hole/1bc9da6aab6a484c9a3cd15e36fd9100/moving_hole.mp4)
uses seed 0 and jumps 60 mm at 3.5 s, during the expert's align phase. The
guard froze the offset when insertion began at 4.3 s, and ManiSkill success
was true at 5.25 s. Mean Hole prediction error from the jump through the
terminal frame was 55.58 mm; this GT-guided expert does not consume those
predictions. [Per-frame results](../outputs/moving_hole/1bc9da6aab6a484c9a3cd15e36fd9100/results.json)
record the motion, tracking and phase data. This is a single example.

With the same expert and guards, 120 mm examples are available for
[continuous motion](../outputs/moving_hole/414c77586d65416cb96e8f5d3accb5c5/moving_hole.mp4)
and an [abrupt jump](../outputs/moving_hole/bb5ffae219d148b7ae18fec7ce906cbd/moving_hole.mp4).
Both reached the full 120 mm displacement before the expert entered insertion,
then held the box fixed and reached simulator success on seed 0. Mean Hole
prediction error from 3.5 s to the terminal frame was 59.84 mm and 86.75 mm,
respectively. The controller uses GT, not these predictions. These are single
illustrations; configs retain their exact settings under
`graph/configs/temporal/visualize_moving_hole_expert_120mm.yaml` and
`graph/configs/temporal/visualize_jumping_hole_expert_120mm.yaml`.

#### Camera position for the moving-Hole example

On 2026-11-11, a camera position was selected for the continuous 120 mm,
seed 0 expert episode above to reduce robot occlusion of the Hole before
insertion. The camera is fixed in the world frame; positions are in metres:

| Setting | Value |
| --- | --- |
| `base_camera` position (`eye`) | `[0.65, -0.30, 0.65]` |
| Look-at point (`target`) | `[0.065, 0.297, 0.11]` |
| Resolution | 512 × 512 |

For another caller using the shared environment factory, the pose can be set
explicitly at creation:

```python
from mani_skill.utils import sapien_utils
from envs.maniskill import make_peg_insertion_env

camera_pose = sapien_utils.look_at(
    eye=[0.65, -0.30, 0.65],
    target=[0.065, 0.297, 0.11],
)
env = make_peg_insertion_env(
    seed=0,
    obs_mode="rgb+depth",
    sensor_configs={"base_camera": {"width": 512, "height": 512, "pose": camera_pose}},
)
```

This snippet configures the camera only; it does not reproduce the moving-Hole
episode or its control settings. The existing `evaluation.moving_hole:visualize`
workflow currently fixes its sensor configuration internally and does not read
`camera_view` from YAML. The recorded run used a local wrapper to override the
base-camera pose while retaining the existing evaluation implementation.

The comparison tested the task's original camera plus 12 candidates, all on
the same 152-frame episode. For each frame, 49 points on the front opening
were projected into the image and checked against rendered depth with a 3 mm
tolerance; off-screen, back-facing, and invalid-depth samples counted as not
visible. Segmentation identified robot- and peg-occluded samples. This is a
sampled visibility measure, not a pixel-exact certificate. Insertion was
excluded from selection because the motion guard stops box movement then.

| Metric | Original camera | Selected camera |
| --- | ---: | ---: |
| Mean opening visibility before insertion | 10.1% | 88.1% |
| Mean opening visibility during actual movement (3.55–6.50 s) | 0.0% | 75.0% |
| Robot-occluded sample fraction during actual movement | 74.8% | 0.0% |
| Peg-occluded sample fraction during actual movement | 25.2% | 25.0% |

The [camera comparison video](../outputs/moving_hole/414c77586d65416cb96e8f5d3accb5c5/camera_visibility/comparison.mp4)
shows the original view on the left and selected view on the right; green and
red marks show visible and blocked opening samples. The
[per-frame visibility results](../outputs/moving_hole/414c77586d65416cb96e8f5d3accb5c5/camera_visibility/visibility_results.json)
and [scan script](../outputs/moving_hole/414c77586d65416cb96e8f5d3accb5c5/camera_visibility/scan_hole_cameras.py)
record the candidates and method. The 13 sampled positions and one episode do
not establish a globally optimal camera location or visibility across episodes.

The [selected-camera model video](../outputs/moving_hole/camera_model_runs/2a87ac49cc854f9880cca35d5e7536b9/moving_hole.mp4)
shows GT Hole center in green and the predicted center in magenta. It runs the
seed 0 early-stopped resumed Transformer + GNN, the overall 4,102-frame
leaderboard's first-ranked model family, using the same checkpoint hashes as
the original video. Yellow points are detected tracks and blue points are
unmatched track forecasts. The [per-frame model results](../outputs/moving_hole/camera_model_runs/2a87ac49cc854f9880cca35d5e7536b9/results.json),
[effective settings](../outputs/moving_hole/camera_model_runs/config.yaml) and
[local camera wrapper](../outputs/moving_hole/camera_model_runs/run_camera_visualization.py)
are available alongside the video. On this episode, mean Hole-position error
before insertion was 28.31 mm with the original camera and 12.52 mm with the
selected camera; during actual movement it was 57.40 mm and 18.40 mm,
respectively. Maximum movement-period error with the selected camera was
51.56 mm. The same recorded peg-tip and Hole-center positions and expert phases
matched at every frame; this does not establish generalization to other episodes.

The [Hole error and unmatched-track plot](../outputs/moving_hole/camera_model_runs/hole_error_and_unmatched_tracks.png)
and [Box-track position plot](../outputs/moving_hole/camera_model_runs/hole_and_unmatched_box_positions.png)
share the episode time axis. The latter also shows predicted and GT Hole center
coordinates. Unmatched means an existing track had no detection assigned in that
frame; the Box-track average is over forecast positions, with a possibly changing
set of tracks, and is not expected to equal the Hole center. Its source
[per-frame CSV](../outputs/moving_hole/camera_model_runs/hole_and_unmatched_box_positions.csv)
and [unmatched Box-track records](../outputs/moving_hole/camera_model_runs/unmatched_box_replay.json)
are available for inspection. The tracking counts and positions were captured
in separate reruns whose saved frame records matched the selected-camera model
run exactly.

All linked `outputs/` files are ignored local artifacts, not files guaranteed
to be present in another checkout. These runs used the Mac CPU simulator and
CPU inference; CUDA was not executed. The GNN and tracker implementation and
the preserved leaderboard results were not modified.

For a view without a moving arm crossing the Hole, zero-action 120 mm examples
show [continuous movement](../outputs/moving_hole/4d1a71156d3648e79a6008801509d229/moving_hole.mp4)
and an [abrupt jump](../outputs/moving_hole/ba42aebd6ec3497985938c7fab78d924/moving_hole.mp4).
The Hole is visible in the inspected camera frames, but no per-pixel occlusion
certificate was computed. The actual displacement was 120 mm in both; mean
Hole error from 0.5 s through the end was 6.47 mm and 59.45 mm respectively.
The jump introduced eight new tracks in the jump frame, not necessarily eight
identity errors. Recipes are `visualize_unobstructed_moving_hole.yaml` and
`visualize_unobstructed_jumping_hole.yaml` under `graph/configs/temporal/`.

`graph.temporal.detected_corner_tracking.DetectedCornerTracker` builds one
internal history per encountered detector point. Before each frame, the frozen
shared temporal model predicts all existing tracks from actual past detection
history and the current robot state. Same-detector-type pairs within the distance
limit are matched one-to-one, maximizing match count before minimizing distance.
Matched outputs are the actual detector coordinates, unmatched old tracks use
the predicted coordinates, and unmatched detections initialize new tracks.
Predictions are never inserted into the history as measurements. Tracks are not
retired, so ID fragmentation and growth are measured rather than hidden.

Tracking receives only detector XYZ, detector peg/box labels and observed robot
state; no GT identity or GT geometry enters matching. The detector does not
provide a GT hole-family label. Offline scoring uses certified IDs and, only
when several detections have the same certificate, the reference vertex selects
one representative for metrics. Unknown-ID detections still participate in
matching but cannot be scored for their physical identity. A reappearance is a
certified detection following one or more frames without that certificate; it may
therefore include physical occlusions, detector misses and certificate failures.

The calibration recipe runs candidate distance gates on calibration episodes.
One common gate for all six model/seed pairs maximizes their mean reappearance
same-ID rate among candidates with at most 1% known-track identity conflicts in
every individual calibration run; ties choose the smaller gate. This rule is an
initial prototype criterion, not a universal safety bound. An identity conflict
means a previously certified track is subsequently certified as another corner.
When no candidate passes, selection fails instead of using validation to tune it.
The evaluation recipe replays separate validation episodes and reports
reappearance retention, new tracks on reappearance, identity switches, known
track conflicts, and growth. The old GT-ID GNN training remains a separate
experiment; this association evaluation does not train or evaluate a GNN.

Set paths and checkpoint hashes in the team YAML files. The matrix launcher
`tools/run_detected_corner_matrix.py --configs DIR --logs DIR` runs six matched
calibration jobs, selects the shared gate, runs six validation jobs, and creates
a comparison only after every preceding stage succeeds. This process waits for
its own child jobs and does not create a background status watcher.

### Detector-only tracking video

`graph/configs/temporal/visualize_detected_corner_tracking.yaml` renders
complete development-validation episodes using the source RGB camera frames,
seed-0 checkpoints, the calibration-selected 30 mm distance gate and the same
online tracker as validation. Configure the dataset, checkpoint and validation
result paths and hashes; CUDA, official `mamba_ssm`, OpenCV and FFmpeg/libx264
are required. The run verifies dataset lineage, input checksums, checkpoint
identity and aggregate replay counts against the recorded validation results.
It writes H.264 videos, a manifest and per-frame track coordinates locally.

Each dot color identifies an internally assigned track within its own panel;
the two panels assign IDs independently. A black rim denotes an actual detector
coordinate and a white rim denotes a prediction for a track not matched in that
frame. All valid detections are included, even those whose physical ID cannot
be certified. The tracker uses no GT identity. Predicted points can project on
top of an occluder, and overlapping dots may hide one another. Selected episodes
illustrate behavior and should not be treated as aggregate accuracy.

### Detector-only rollout RMSE

`graph/configs/temporal/evaluate_detected_corner_rollout_rmse.yaml` replays the
recorded detector-only validation with the selected threshold. It verifies the
source checkpoint, dataset, validation episode identities and per-episode match,
new-track and final-track counts. Configure these paths and hashes first.

Scoring is permutation-invariant: at each frame, all track output coordinates
are optimally paired by minimum total squared 3D distance with GT corner points
whose physical IDs have been certified at or before that frame. Assignment is
separate for detector peg and box types. All available pairs are matched without
a maximum-distance cutoff, and unmatched extra tracks and uncovered GT points
are counted separately. GT identities and coordinates are used only to form
and score this target set, never as input to the tracker. Unknown-ID detections
do participate in tracking and matching, but an otherwise unseen GT corner
without a certificate cannot yet be included in the target set.

The metric is `sqrt(mean(||output_xyz - gt_xyz||²))`, in millimetres, over
matched track-frame pairs. It is reported separately for currently detected
points and unmatched tracks using predictions. The identity-free geometric
pairing is an oracle evaluation, not the tracker's own matching algorithm;
it can associate a point with a physically different corner if this minimizes
distance. Extra unmatched nodes do not contribute to RMSE, so RMSE alone must
not be interpreted as full-set accuracy. Counts and per-frame records are saved.

On instance 59441593, all six rollouts reproduced the corresponding earlier
association counts across 40 validation episodes / 4,102 frames. The target set
contained 52,029 eligible GT corner-frames, all of which had a geometric pair.
The mean of the three individual seed RMSE values was:

| Model | All matched outputs | Currently detected | Predicted / unmatched by detector |
| --- | ---: | ---: | ---: |
| Mamba | 4.052 mm | 3.465 mm | 4.681 mm |
| Transformer | 3.723 mm | 3.472 mm | 4.007 mm |

The mean unmatched-output fractions were 19.50% for Mamba and 19.15% for
Transformer. An unmatched output is not automatically a false positive: it
may correspond to a physical corner whose GT identity has not been certified.
These track-frame outputs are excluded from the displayed RMSE, so the values
are conditional on geometric pairing, not a full graph error or a 21D metric.
See `outputs/detected_corner_rollout_rmse/comparison/300b51596ab4443bb5ff8083b75f4350/comparison.json`
for per-seed results and the corresponding per-episode frame counts.

### Detector-only rollout GNN training

`training.detected_corner_gnn:train` replays each complete source episode using
the calibrated detector-only tracker and the frozen, shared per-corner temporal
checkpoint. It creates graph nodes for every track, including tracks whose
physical ID cannot be certified. Current detections retain their exact observed
coordinates; unmatched tracks use predicted coordinates. Node kinds are only
the detector's peg/box labels, and the extra feature indicates detection versus
prediction. GNN semantic IDs are always unknown; GT corner identity, reference
geometry and GT-derived hole type never enter graph construction. Track IDs are
used internally to maintain history but are not a GNN feature.

The synchronized Object21 target cache and source episodes are checked against
the fixed-ID dataset lineage and sample/TCP order. Existing recorded validation
rollouts must reproduce the same match, new-track and final-track counts before
training proceeds. The full frozen rollout cache is saved with a checksum;
only GNN parameters enter AdamW. The team recipe uses 20 epochs, batch size 96,
learning rate 0.0003 and identical objective, initialization, seed and data
settings for paired Mamba/Transformer runs. Validation loss selects `best.pt`;
`last.pt`, history, progress and source metadata are also retained. No early
stopping is used for this fixed 20-epoch comparison.
Future runs print train/validation loss once per completed epoch; intermediate
batch progress is retained in `progress.json` without printing batch lines.

`evaluation.detected_corner_gnn_comparison:compare` verifies paired initial GNN
weights and settings and evaluates the saved rollout inputs using each selected
GNN checkpoint. It requires exact validation sample IDs, order and Object21 GT
against the existing leaderboard reference, and exports per-frame predictions,
21D error statistics and validation-loss curves. Results use development
validation, which also selects the checkpoint; GT IDs are not used for tracking.
Differences in input information and training settings should still be stated
when adding results beside older leaderboard entries.

On instance 59441593 all six 20-epoch jobs and the comparison completed. The
best GNN epochs were 20 / 19 / 20 for Mamba and 20 / 16 / 20 for Transformer.
Hole-position mean error was 2.549 mm and 2.304 mm respectively, averaged over
the three paired seeds on 4,102 development-validation frames. The nine 21D
metrics, per-frame predictions, seed-specific checkpoint identities and
independent aggregation audit are in the [experiment leaderboard](experiment_leaderboard.md#자동-코너-대응--gnn--20-epoch--검증-전체-4102프레임).
GNN best/last checkpoints, logs, configs, source snapshot and predictions are
backed up locally under `outputs/detected_corner_gnn_training/`; the much larger
per-seed `rollout_inputs.pt` caches remain on the server and can be reconstructed
from the recorded data and frozen temporal checkpoints. No artifacts were uploaded.

#### Fresh 40-epoch run with optimizer checkpoints

`training.detected_corner_gnn_fresh:train` validates and reuses each earlier
run's checksummed frozen rollout inputs, but constructs a **fresh** GNN and a
fresh AdamW. It does not load the earlier GNN weights or optimizer. The initial
GNN hash must match the original matched seed; the temporal checkpoint and all
data/model settings must match the source cache. The cache is hard-linked into
each new run, so the input artifact remains available without a second copy.

The run trains for 40 epochs with the same optimizer hyperparameters and data
split. `last.pt` and validation-selected `best.pt` are written atomically and
include GNN weights, AdamW state, epoch, Python/NumPy/Torch/CUDA RNG state,
effective configuration, source identities and cache hashes. The recipe saves
the state needed for a future resume; it does not itself implement resumption.
Epoch-level train/validation losses are printed, and intermediate batch progress
remains in `progress.json`. The original 20-epoch artifacts remain unchanged.

The detector-only GNN can continue from a completed 40-epoch run with
`training.detected_corner_gnn_resume:train` and a copy of its effective config.
Set `training.epochs` to the maximum total epoch, supply the source `last.pt`
and checksums for both `last.pt` and `best.pt` in `resume`, and set `patience`.
The continuation restores the GNN, optimizer, and random states, retains the
prior best checkpoint and history, and resets the patience counter at resume.
Only a strictly lower validation loss resets patience. Frozen detector-only
rollout inputs are reused, and checkpoints are saved at each epoch. The run
ends at the patience limit or the maximum epoch, whichever comes first.
See `graph/configs/temporal/train_detected_corner_gnn_resume.yaml` for the
resume fields; a full effective configuration is required.

### Object21 validation video

`python main.py --config graph/configs/temporal/visualize_object21.yaml` renders
original camera frames using previously saved Object21 comparison predictions;
fill in the comparison, family dataset, rollout cache and their checksums in a
copy of the config. The two panels use the same validation frame and seed. Peg
wireframes and hole rings depict world poses and dimensions; crosses show the
independently predicted endpoint. Hole axes depict the full rotation. Green is
GT and magenta is prediction. Coordinates are translated by the synchronized
TCP before projection. The visualization uses saved evaluation outputs, not a
new inference, and the selected examples are not aggregate test performance.

## Random-action diagnostics

`graph/configs/temporal/random_action_diagnostics.yaml` runs the frozen
leaderboard seed-0 Transformer and GNN in the live simulator. For each
configured environment seed it replays an expert until the simulator reports a
successful grasp, the peg has lifted by at least 30 mm, and the expert reaches
alignment. It then compares expert continuation against separately seeded random
6-D arm commands, keeping the gripper closed. A run that never reaches that
switch is recorded rather than silently treated as a successful rollout.

GT is used for the switch check and for **diagnostics only**. Detector error is
the distance to the nearest same-kind GT corner, allowing duplicates and without
asserting visibility. At track birth, the nearest same-kind reference within
10 mm is assigned for diagnostic comparison; unassigned tracks remain in the
actual input and are counted separately. Track-ID switches, ambiguous corners,
unknown detections, and never-observed corners make this an approximate oracle,
not a true detector recall or identity certificate. Matched-node RMSE excludes
unassigned tracks and therefore must be read alongside its coverage.

The GNN counterfactuals replace observed and/or predicted coordinates of
assigned tracks with those GT coordinates while preserving their original
provenance flags and leaving unassigned tracks unchanged. Another drops
unassigned tracks; another changes the assigned tracks' provenance flags to
observed. A final deliberately out-of-distribution condition supplies all 24 GT
corners as observed, including occluded corners. These interventions can interact
and change graph support, and do not establish an additive causal attribution or
a deployable model score. The original GNN input never uses GT.

A 2026-11-13 Mac CPU run used environment seeds 0–4, action seeds 73 and 74,
and 35 post-switch steps per rollout. All 15 runs reached the switch, all 525
post-switch frames retained the simulator grasp, and each expert/random prefix
matched before switching. Across the ten random versus five expert rollouts,
episode-weighted mean Peg position errors were 51.01 versus 18.80 mm, and Hole
position errors were 11.63 versus 7.57 mm. Random rollouts had zero detector
outputs in 107/350 frames; expert controls had zero in 8/175 frames. These are
five paired initial states and two reused action sequences, not ten independent
initial states. The model seed was 0 and no initial visibility filter was applied.

Across random rollout detections, nearest-reference 3D RMSE was 3.21 mm for
1,645 detections; 9 were over 10 mm. For birth-assigned track nodes, observed and
predicted 3D RMSE were 2.53 mm (1,632 nodes) and 18.72 mm (2,483 nodes),
respectively; 326/4,441 total track nodes were unassigned and omitted from those
track errors. Averaged over random episodes, Peg error was 51.01 mm originally,
46.70 mm after replacing assigned predicted positions, 46.73 mm after replacing
all assigned positions, and 5.09 mm using the complete 24-corner GT graph. The
last condition adds hidden corners and changes provenance; the gap is not a pure
GNN error measurement. In one random rollout all 35 frames after switching had
zero detections. Its assigned peg tracks came only from one physical end; fixing
their coordinates still left 128.5 mm mean Peg error, while the complete GT
condition gave 3.9 mm. This points to lost or insufficient graph support and
tracking errors as major failure modes, without proving the training distribution
is their sole cause. Model inference on CUDA and the other two leaderboard model
seeds were not executed.

The [run results](../outputs/random_action_diagnostics/5a5605be3d2c4d71b97b945082601647/results.json),
[main plot](../outputs/random_action_diagnostics/5a5605be3d2c4d71b97b945082601647/diagnostics.png),
[stage plot](../outputs/random_action_diagnostics/5a5605be3d2c4d71b97b945082601647/stage_diagnostics.png)
and per-episode videos are local, ignored verification artifacts.

### Selected-camera random-action comparison

The selected moving-Hole camera can be used in the same diagnostic workflow:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/random_action_selected_camera.yaml
```

`camera_view` specifies a fixed world `eye` and `target`; omitting it retains
ManiSkill's original camera. The recipe uses the same five environment seeds,
two action seeds and model checkpoint as the earlier default-camera run. All
15 expert/random episodes reached the same switch and recorded 35 post-switch
steps with grasp maintained. The comparison checked every paired frame's action,
Peg GT and Hole GT: the maximum absolute difference was zero. The selected
camera's live `cam2world_gl` translation was independently checked against
`[0.65, -0.30, 0.65]` metres.

| Random post-switch metric | Default camera | Selected camera |
| --- | ---: | ---: |
| Peg position mean | 51.01 mm | 19.52 mm |
| Hole position mean | 11.63 mm | 30.62 mm |
| No detections | 107 / 350 frames | 0 / 350 frames |
| Detected point to nearest same-kind reference RMSE | 3.21 mm | 4.97 mm |
| Birth-assigned predicted-track RMSE | 18.72 mm | 39.85 mm |

The selected-camera Hole mean is dominated by both random trajectories from
environment seed 2 (115.2 mm each); the error was already 115.2 mm at the
switch, before random actions. For the other four environment seeds the Hole
means were 10.41 mm default and 9.46 mm selected. In that seed-2 random
trajectory none of the 35 post-switch frames had a detected point whose nearest
GT reference was an aperture-mouth corner. This nearest-reference observation
is not a visibility certificate, and it does not by itself establish the cause
of the Hole error. Correction of assigned tracked coordinates did not remove
it. The selected camera was chosen using one moving-Hole episode, not optimized
for all initial states or this action distribution.

The [paired comparison](../outputs/camera_diagnostics_comparison/87b24df5ad27461ba5a13800a38d27dc/comparison.json),
[comparison figure](../outputs/camera_diagnostics_comparison/87b24df5ad27461ba5a13800a38d27dc/camera_comparison.png),
[side-by-side seed-2 video](../outputs/camera_diagnostics_comparison/87b24df5ad27461ba5a13800a38d27dc/env_2_camera_comparison.mp4),
and [selected-camera results](../outputs/random_action_diagnostics/0919f71ee5f84fd09d4d32df6384cdf5/results.json)
are local ignored artifacts. Re-run the paired audit using
`graph/configs/temporal/compare_camera_diagnostics.yaml` after updating its two
result paths. CPU inference was executed on this host; CUDA and model seeds 1/2
were not executed. The node RMSEs use different node subsets, and remain subject
to the approximate birth-assignment and unassigned-node limitations above.

### Seed-2 selected-camera Hole input inspection

Re-run the frame-level inspection with:

```bash
WANDB_MODE=disabled python main.py --config graph/configs/temporal/hole_input_diagnostics.yaml
```

It replays the original saved actions for the default and selected cameras,
reconstructs tracking from frame zero, and verifies Peg GT, Hole GT and the
original Hole prediction against the recorded results at every frame. It saves
frames 0, 68 (random-action switch), and 78, including actual detector and graph
nodes, per-target pooling weights, GT geometry, counterfactuals and figures.

For seed 2, the near opening on frame 0 projects to a 17.5 × 19.7-pixel
axis-aligned box from the default camera, versus 3.8 × 12.5 pixels from the
selected camera. Its view angle relative to the opening-plane normal is 25.8°
versus 76.6°. These are GT geometric projections, not proof of visibility.
At the center pixel of the near opening in the selected image, measured depth
was 1.017 m versus 0.994 m to the GT opening plane; this single depth sample
alone does not establish the visibility of the entire contour.

On selected-camera frame 68, all five target-side GNN nodes are currently
detected box vertices, each 2.3–2.9 mm from its nearest GT exterior vertex;
none is nearest an aperture vertex. The graph contains 8 Peg and 5 Box nodes
and 156 directed edges. Target pooling weights on Box nodes 0 and 5 sum to
0.919. This is a readout statistic, not an isolated causal contribution.
The target anchor is 73.5 mm from the Hole GT and the learned position offset
is approximately `[46.6, 0.0, 11.2]` mm in world axes, producing 115.2 mm
error. The same frame in the default-camera run retains four aperture-near
tracks and yields 18.7 mm error, although none is currently detected then.

Diagnostic input interventions for this frame leave the frozen GNN unchanged:
removing Peg nodes yields 107.3 mm, snapping the five target nodes within 10 mm
to their nearest GT references yields 115.9 mm, and appending all eight GT
aperture corners as observed Box nodes yields 22.3 mm. The last probe uses
hidden GT and changes the node distribution, so it is evidence of sensitivity
to opening-corner support, not a deployment result or proof of a unique cause.
The selected camera's error is already 116.0 mm at frame 0, before grasping or
random actions. The evidence points to an oblique low-resolution aperture,
missing aperture-near detections, and the GNN extrapolation from exterior nodes;
it does not establish which detector component rejected the opening or whether
this generalizes across initial states.

Local evidence: [aperture comparison](../outputs/hole_input_diagnostics/6acece0f46454f7ab970c56c9fef769b/aperture_comparison.png),
[selected frame graph](../outputs/hole_input_diagnostics/6acece0f46454f7ab970c56c9fef769b/selected_frame_68.png),
[error timeline](../outputs/hole_input_diagnostics/6acece0f46454f7ab970c56c9fef769b/hole_error_timeline.png),
and [numeric snapshots](../outputs/hole_input_diagnostics/6acece0f46454f7ab970c56c9fef769b/snapshots.json).
CPU inference and CPU simulator were run on this host; CUDA was not tested.

### Multi-state camera visibility search and held-out action comparison

The candidate search reuses saved expert-prefix/random-action trajectories and
restores their physical states before rendering each camera. It verifies Peg and
Hole GT positions on replay and restoration. Run with
`graph/configs/temporal/camera_visibility_search.yaml` and the source results
configured there. Ten fixed viewpoints were examined on environment seeds
0–2 and action seeds 73/74. Six states per trajectory were sampled: the initial
frame and steps 0, 9, 18, 27 and 35 after lifting. The identical initial state
is consequently counted once per action seed.

At each state the scan projects a 7×7 grid on the front Hole opening and
compares each point with rendered depth, using a 3 mm tolerance and simulator
segmentation to identify robot/peg blockage. Back-facing, off-screen and invalid
samples count as not visible. This is a sparse GT-based diagnostic, not a
pixel-exact visibility certificate; nearby grid points may map to the same
pixel, especially at oblique views. Camera ranking maximizes the minimum of the
three per-environment mean visibility scores; ties use mean visibility, lower
robot blockage and then minimum projected width. The fixed target is
`[0.065, 0.297, 0.11]` metres. The selected eye was `[0.35, -0.30, 0.35]`.
A second candidate `[-0.35, -0.30, 0.35]` was added after inspecting the
selection results because its worst-seed visibility was above 95% and its
minimum projected opening width was larger. Its choice is exploratory.

Separate evaluation used environment seeds 3–6, with two saved random-action
seeds and an expert continuation for each environment. Seeds 3–6 were not used
for camera selection; this does not establish that they are outside the model's
training data. Initial-state visibility filtering was not applied. The original
seed-0 Transformer/GNN pair was fixed throughout. Actions, Peg GT positions and
Hole GT positions matched exactly at every frame of all 12 paired episodes;
the comparison does not separately verify every robot joint or object rotation.
Each random episode contributes 35 post-lift frames and maintained grasp.

| Camera | Eye (m) | Sampled Hole visibility | Robot-blocked samples | Random Hole mean | Random Peg mean |
| --- | --- | ---: | ---: | ---: | ---: |
| Default | simulator default | 63.6% | 36.4% | 11.02 mm | 64.58 mm |
| Previous | `[0.65, -0.30, 0.65]` | 99.0% | 0.0% | 44.78 mm | 18.00 mm |
| Visibility-selected | `[0.35, -0.30, 0.35]` | 99.7% | 0.3% | 14.11 mm | 71.96 mm |
| Wider-opening candidate | `[-0.35, -0.30, 0.35]` | 100.0% | 0.0% | 24.04 mm | 70.89 mm |

Visibility figures are means over 48 sampled states in the held-out scan, with
49 projected points per state; they should not be interpreted as a guarantee
across continuous time. Error means use all 280 random-action frames for each
camera. The previous-camera Hole mean is affected by both seed-6 random runs
at about 151.5 mm, while the visibility-selected Peg mean includes both seed-3
runs around 146 mm. Better projected Hole visibility alone therefore did not
identify a camera with stable Hole **and** Peg predictions. The small set of
initial states and reused action sequences are not an independent deployment
benchmark. No full inference evaluation was run on CUDA or model seeds 1/2.

[Selection results](../outputs/camera_visibility_search/0dc652a1bcd84916b1519bd0814dc8d9/search_results.json),
[held-out visibility](../outputs/camera_visibility_holdout/cad2bf22360a44baa261a75f84cfbf4e/search_results.json),
[paired numerical results](../outputs/camera_holdout_comparison/d78175519f5d409d99c451740cd1bc40/holdout_results.json),
[paired error figure](../outputs/camera_holdout_comparison/d78175519f5d409d99c451740cd1bc40/heldout_camera_comparison.png),
and [four-camera video](../outputs/camera_holdout_comparison/d78175519f5d409d99c451740cd1bc40/env_6_four_cameras.mp4)
are local ignored artifacts. Individual evaluation and comparison configs are
under `graph/configs/temporal/`; paths refer to the saved runs and must be updated
when the experiments are repeated.

### Broader random-position camera study

The 2026-11-13 follow-up samples 40 independently seeded simulator initial
states (100–139), with 100–119 used to choose a camera and 120–139 held out
for visibility verification. The source action collector is configured in
`data_collection/configs/camera_visibility_trajectories.yaml`: it records the
expert approach and grasp, then 35 random arm actions with the gripper closed.
The NumPy action RNG uses `73 + environment_seed`. Failed grasps are retained
with their expert-prefix samples; in this run all 40 reached the lift switch.
No initial visibility filter was applied. Across these 40 resets, the initial
Hole center ranges in world metres were approximately X `[-0.056, 0.064]`, Y
`[0.201, 0.393]`, Z `[0.061, 0.164]`; Box and Hole poses were randomized by the
installed simulator. Held-out means only held out from **camera selection**,
not proven disjoint from the predictor's original training dataset.

The [selection config](../graph/configs/temporal/camera_distribution_search.yaml)
compares 13 fixed camera positions on eight samples per trajectory: reset, three
approach/grasp points, lift switch, and random steps 10, 20 and 35. It maximizes
mean sampled opening visibility across the 20 selection seeds, breaking ties by
the 10th percentile of per-seed visibility, robot blockage and minimum projected
width. The choice was eye `[0.45, -0.30, 0.35]` m and target
`[0.065, 0.297, 0.11]` m. The scan uses the 7×7 depth/segmentation procedure
above, with the same caveats: points may coincide in image pixels, aperture
samples are GT-based, and eight frames cannot cover every occlusion in time.
The run replays actions and verifies saved Peg and Hole positions and restored
states for each sampled frame.

On the separate 20 held-out seeds, each seed's sampled mean opening visibility
improved over the default camera: mean 41.44% → 99.68%, 10th percentile of
per-seed means 12.50% → 99.85%, and mean robot-blocked samples 55.06% → 0.32%.
These are sample-based comparisons, not a guarantee of an unoccluded physical
camera. The selected camera's minimum projected opening bounding-box width on
these states was 8.3 pixels (default 14.8), so the view is smaller even when
less occluded. The held-out comparison is configured in
`graph/configs/temporal/camera_distribution_comparison.yaml`.

The 10 even-numbered held-out seeds were also replayed end-to-end through the
fixed seed-0 Transformer/GNN for both views, using the exact saved actions. All
10 paired episodes matched in recorded actions, Peg positions and Hole positions
at every frame. Over 350 post-lift frames per view, no-detection frames changed
37 → 0 and mean Peg position error 38.50 → 32.51 mm, but mean Hole error rose
7.56 → 25.07 mm. Nine of ten episodes had higher mean Hole error under the new
camera. This model result should not be conflated with camera visibility: the
cause of the error increase and the benefit of retraining have not been tested.
CPU simulation/rendering and CPU inference ran locally; CUDA and the other
model seeds were not run.

[Camera selection](../outputs/camera_distribution_search/ecec199d855f4879bdec9cda5cb672b7/search_results.json),
[held-out visibility scan](../outputs/camera_distribution_holdout/af05ceee263248688d872620da2c02b9/search_results.json),
[paired visibility statistics](../outputs/camera_distribution_comparison/311588939ada4c6b815ca5f4154dafbe/distribution_results.json),
[visibility figure](../outputs/camera_distribution_comparison/311588939ada4c6b815ca5f4154dafbe/distribution_comparison.png),
[example views](../outputs/camera_distribution_comparison/311588939ada4c6b815ca5f4154dafbe/distribution_scenes.png)
and [model comparison](../outputs/camera_distribution_model_comparison/f97458193f0f49a9a3ec2575eea5557a/comparison.json)
are local, ignored artifacts. The saved-source paths in configs must be updated
when reproducing the study with new runs.

#### Additional 60-state validation

A follow-up retained the 40 states above and collected 60 new simulator seeds
140–199 using the same collector and action settings. The camera remained fixed
at `[0.45, -0.30, 0.35]`; the new states were not used to select it. Three new
states (143, 167, 186) did not reach the lift switch and remain in the visibility
scan with six expert-prefix samples each; the other 57 have eight samples each.
The 100-state total therefore contains 794 sampled states. The original 20
selection seeds are included in that total and are not independent validation.

Over all 100 initial states, the equally weighted per-seed mean opening
visibility was 39.25% for the default camera and 99.06% for the selected camera.
Considering only the 60 new seeds, the corresponding means were 38.23% and
98.59%; 59 of 60 per-seed means improved. Seed 193 fell from 67.09% to 46.43%,
so the chosen view is not uniformly better. These figures use the sparse 7×7
GT/depth diagnostic and its limitations described above, not continuous-time
visibility guarantees. On the new sample, minimum projected opening width was
16.16 pixels for the default view and 6.77 pixels for the selected view.

The fixed seed-0 Transformer/GNN was also run on all 57 newly collected
trajectories with a completed lift and 35 random steps; the three failed-lift
cases cannot contribute post-lift model metrics. Paired actions and recorded Peg
and Hole GT positions matched exactly for every frame. Across 1,995 post-lift
frames per view, zero-detection frames fell from 408 to 0; mean Peg position
error changed from 59.84 to 52.72 mm and mean Hole error from 11.94 to 17.89 mm.
Hole error increased in 41 of 57 episode means. These are model-seed-0 results,
not an independent model test: the relation of simulator seeds to the original
training data is unknown. Failed grasps are not represented in the post-lift
metrics. CPU simulation and CPU model inference were used; rendering used the
host's SAPIEN macOS backend with MoltenVK on Apple M4 Pro. CUDA and other model
seeds were not evaluated.

[Run configs, summary and analysis script](../outputs/camera_expanded_study/),
[visibility and model comparison figure](../outputs/camera_expanded_study/comparison.png),
[visibility results](../outputs/camera_expanded_study/visibility/59df67e3ab8c4aa490956d76160ca4b7/search_results.json),
and [paired model results](../outputs/camera_expanded_study/model_comparison/36dd73af6f604719afb797a5d7cef4de/comparison.json)
are local ignored artifacts. The runs used the dirty working tree; the Git commit
alone does not capture their code. No checkpoints or artifacts were published.

## Independent generalization test and leaderboard replay

The fixed unfiltered test protocol is
[protocol.json](../outputs/independent_generalization_test/protocol.json),
SHA-256 `9f0052660248e4934656eca7a0a23b750346e98786455e23b920ff250a70e49d`.
Before test observations or model results, 20 reset seeds were drawn from
1,000,000–1,999,999 with a fixed RNG. They do not overlap the verified original
900 collection seeds, its 630 Graph development seeds, the separate 900-episode
BC collection inventory, or explored seeds 0–139. The Graph exclusion inventory's
dataset hash was checked against all three GNN checkpoints. Disjointness from
other unknown external datasets is not established.

The protocol fixes the original default 512×512 base camera, the three
leaderboard Transformer/GNN checkpoint pairs, the 30 mm tracking gate, a
GT-guided scripted expert, and two trajectories per reset: expert continuation
or 35 random 6-D arm actions after confirmed grasp/lift. Random seeds are
`73 + reset_seed`. No visibility or success filter and no replacement are used.
All 40 planned cases reached the lift switch; one expert trajectory terminated
at 31 post-lift actions. Its final terminal observation was not archived, so
its post-lift metric contains 30 rather than 35 frames. All other cases reached
the planned 35-frame post-lift window. Full native base-camera RGB-D,
calibration, robot observations, GT, and actions are saved with per-frame hashes
in the [raw manifest](../outputs/independent_generalization_test/raw/manifest.json).
The same 1,342 prefix frames per mode were verified equal in images, robot
observations, GT and pre-switch actions. Expert GT guides the controller; it is
not supplied to the detector, tracker or GNN.

The single detector is run once per raw frame, then all three model pairs receive
identical detections. The main statistic follows the leaderboard convention:
average the three model-seed errors per frame, then average over frames. The
[unfiltered evaluation](../outputs/independent_generalization_test/evaluation/3758115c26414a729b7a0d6c3e83de9f/results.json)
contains every frame, per-seed prediction and missing-output counts:

| Unfiltered test | Frames | Peg mean / p90 | Hole mean / p90 |
| --- | ---: | ---: | ---: |
| Expert, all observed frames | 2,037 | 60.23 / 137.49 mm | 11.79 / 25.04 mm |
| Expert, after lift | 695 | 17.33 / 45.87 mm | 5.35 / 14.40 mm |
| Random after lift | 700 | 58.24 / 127.69 mm | 10.54 / 20.93 mm |

A [post-hoc initial-visibility audit](../outputs/independent_generalization_test/visibility_audit/245bbff060224c4baa557e528187c42b/initial_visibility.json)
found only 4/20 resets pass the historical ends-and-grasp initial gate. These
four expert trajectories averaged Peg 1.13 mm and Hole 2.77 mm over all
frames; the other sixteen averaged 72.87 mm and 13.71 mm. At reset, their
average Peg detection counts were 6.5 versus 2.6. This grouping was not a
filter, and it does not prove the gate itself caused the error.

A separate [matched-visibility follow-up protocol](../outputs/independent_matched_visibility_test/protocol.json),
SHA-256 `01bc87ff10cc64c2066f4e22e24581ae0ac090206704d9ac78d321585037639d`,
was preregistered after seeing the first test. It applies the original initial
gate to a fixed candidate order and retains the first 20 passing new seeds; no
model error was used for selection. Twenty passed among 59 attempts. The test
seeds are disjoint from the first test and both verified collection inventories;
initial RGB-D/pose fingerprints were checked again during collection. All 40
planned expert/random cases and all 35 post-lift frames per case were evaluated.
The follow-up [results](../outputs/independent_matched_visibility_test/evaluation/640bd18751a54fdd819f66301d8a7aa0/results.json)
show:

| Matched-visibility follow-up | Frames | Peg mean / p90 | Hole mean / p90 |
| --- | ---: | ---: | ---: |
| Expert, all observed frames | 1,810 | 2.02 / 4.81 mm | 6.80 / 5.65 mm |
| Expert, after lift | 700 | 2.38 / 4.81 mm | 4.66 / 9.11 mm |
| Random after lift | 700 | 12.45 / 24.88 mm | 12.50 / 15.43 mm |

The Hole mean exceeds p90 due to a small number of large errors; one follow-up
episode averaged 47.1 mm expert and 133.4 mm random Hole error after lift.
The follow-up only matches the initial visibility condition, not all training
trajectory or success-filter conditions. Its design was prompted by the first
test, so it is a separate hypothesis-follow-up result, not part of the original
preregistered test. Do not tune models on either test and continue to call them
untouched held-out tests. [Comparison figure](../outputs/independent_generalization_test/cohort_comparison/bd941dfa11a44ee3b5bbc48fa4de405d/cohort_comparison.png).

For the original leaderboard, all three frozen Transformer rollout caches were
downloaded from the pinned backup and matched their checkpoint SHA-256. The
protected evaluator was run unchanged on all 4,102 original validation frames
on CPU. Across seeds, GNN-only mean Peg/Hole errors were 1.4248108/1.8311576 mm,
versus stored 1.4248109/1.8311574 mm. CPU predictions were **not bit-identical**
to the original CUDA arrays; maximum absolute Object21 element differences were
`2.98e-7`, `3.58e-7`, and `3.58e-7` for seeds 0/1/2. This is a CPU rerun of
frozen tracker graph inputs, not exact original-device reproduction or a rerun
of detection/tracking. [GNN replay](../outputs/leaderboard_reproduction/evaluation/e089802df1504215993e8f1c04d3d72c/reproduction.json).

For 960 hash-verified original validation RGB-D frames present locally, the
current CPU detector's same-type point sets matched original cached detection
counts on 959 frames, with one missing recorded point across the remaining
frame. Of 7,654 matched points, mean distance was 0.0041 mm, p90 0.00047 mm,
and maximum 4.94 mm; these are not bit-exact outputs. Original CUDA and current
CPU detector runtime differ. [Detector audit](../outputs/leaderboard_reproduction/detector_audit/16f3fcdbe12c40caa3c3cd1af4907ebe/detector_audit.json).

The pinned raw shard `shard_000000.tar` was downloaded and verified by SHA-256.
Its four full validation episodes (412 contiguous frames) were rerun causally
from RGB-D through detector, Transformer tracker and GNN using model seed 0.
Recomputed Object21 GT matched exactly. Reference versus current CPU mean Peg
error was 1.176 versus 1.193 mm; Hole was 1.347 versus 1.325 mm. Node matching
found provenance/type or count differences in two frames. This is a close but
not exact CPU replay on **4/40 episodes**, not a full original-pipeline replay
of all 4,102 raw frames. [Full-pipeline subset replay](../outputs/leaderboard_reproduction/raw_replay/4650bb50aaaa4073950edb7c4345066c/raw_replay.json).
The remaining 36 raw episodes and the original CUDA numerical environment are
still required for complete end-to-end/exact reproduction. No protected
leaderboard implementation or dependency was changed.
