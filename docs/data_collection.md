# Episode collection

## Collection batch defaults

The dataset workflows target eight episodes per batch and use two collection
workers by default on the local machine and on servers. Override with
`batch_episodes` and `workers` in workflow YAML. The sharded RGB-D workflow
also accepts its existing `episodes_per_shard` setting. Each sharded worker
collects one shard at a time. Workflows that share one simulator across all
policies for a seed keep the whole seed together. A three-policy pilot therefore
runs two seed jobs (six episodes) per group, while four-policy Moving-Hole RAM
staging archives two seeds (eight episodes) per group. The caller-owned
`collect_episodes` function and CLI run one simulator and serve as the workers
for the sharded workflow.

The RGB-D dataset workflows stage uncompressed batches and compress completed
batches in a background writer while collection proceeds. This applies to
sharded expert and Object21 BC datasets, mixed Pilot/Main and Recovery,
independent/matched test collection, and the corner-detector holdout. Their
existing NPZ paths, manifest schemas and raw frame arrays remain readable.
The Moving-Hole archive workflow uses the same staging scheduler and writes
Zstandard archives directly. The camera-visibility trajectory study produces
only JSON metadata.

Set `ram_root` to a writable RAM-backed mount with space for the active batches.
The default is `/dev/shm` when present and the system temporary directory
otherwise. On macOS, point `ram_root` to an existing RAM disk to enable
memory-backed staging. For the sharded workflows, each worker produces one
eight-episode shard, so a collection wave can have two active shards.

## Fixed-ID per-corner sequences

This dataset-only workflow isolates per-corner temporal prediction before adding
any downstream graph. Set `source.corner_dataset` to a completed, full-frame
`observed_corner_tracks_v1` directory in
`data_collection/configs/fixed_id_corner_sequences.yaml`, then run from the root
with Python 3.11, NumPy and PyYAML:

```bash
WANDB_MODE=disabled python main.py --config data_collection/configs/fixed_id_corner_sequences.yaml
```

IDs are defined in object-local coordinates, independent of detector output
order or world pose. Lexicographic sign combinations `(-1,-1,-1)` through
`(+1,+1,+1)` identify eight corners per family:

| IDs | Frame | Local coordinates |
| --- | --- | --- |
| 0–7 | peg | signs × peg half sizes |
| 8–15 | box | signs × `[peg_half_sizes.x] * 3` |
| 16–23 | hole | signs × `[peg_half_sizes.x, hole_radius, hole_radius]` |

The source certificates attach actual detections to these IDs offline. UNKNOWN
certificates stay unassigned; the adapter does not force them to the nearest
corner. These inherited GT-assisted certificates have imperfect coverage and
are not evidence of perfect visual identity recognition. When certificates
duplicate an ID in one frame, offline annotation retains the actual detection
closest to that ID's GT coordinate, without averaging or snapping its position.
The manifest counts unknown detections and duplicate certificates.

Each `(episode, vertex_id)` sample begins at its first certified detection.
Never-observed IDs and frames before that first sighting are omitted. The first
sighting initializes history; all subsequent frames are prediction targets,
including frames with detections. An ID first seen in the final frame has no
future prediction target and is counted separately. There is **no artificial
corner deletion, random masking or synthetic occlusion** in this workflow.

- Visible observation and target: unchanged actual detector XYZ in metres.
- Missing after first sighting: zero observation with a false observation flag;
  GT world XYZ is a supervision label only.
- `observed` means a certified detection is available. Missing intervals can
  reflect occlusion, detector misses, or an unresolved ID certificate.

`iter_sequences(root, split)` yields independent corner sequences. Their model
inputs have shape `[T,13]`: previous-frame observed XYZ (3), previous-frame
observation flag (1), and current observed robot state (9: TCP world XYZ, wxyz
quaternion, two finger positions). The current detection is shifted to the next
input so that predicting a visible target cannot become copying that target.
`prediction_valid` excludes only initialization. `targets` contains world XYZ;
`target_delta_xyz` subtracts `last_observed_xyz_world`, the most recent detection
strictly before each valid prediction, for displacement training. Predictions
are not fed back as measured observations. A future consumer can return the
unchanged measurement when visible and the predicted position when missing.

ID is sample metadata, not an embedding or numeric input. All corner samples
can reuse one temporal model's weights. Mamba and Transformer can consume the
same samples, labels, masks and parent-separated splits. This command builds
data only; it imports no tracker, GNN or temporal model and starts no training.
It retains train/calibration/validation membership, opens no reserved-test files,
checks input/output hashes and writes a completion manifest only after all
episodes succeed. Experiment identity and the effective config use the shared
`ExperimentLogger`; nothing is uploaded. Run
`.venv/bin/python -m unittest tests.test_fixed_id_corner_sequences -v` to check
first-sighting rules, real-coordinate preservation, ordering invariance,
causality, parent separation and interrupted/corrupt-source behavior.

### Prepared dataset verification

On 2026-11-08 the server converted all 630 episodes from source run
`c9c8da5bc0ac46f8a5cfad4219b082e6`, preserving the previous development splits:

| Split | Episodes | Corner sequences | Visible prediction targets | Missing prediction targets |
| --- | ---: | ---: | ---: | ---: |
| train | 531 | 6,786 | 353,892 | 318,322 |
| calibration | 59 | 753 | 38,430 | 36,769 |
| validation | 40 | 519 | 27,719 | 23,791 |

The dataset is at
`/workspace/fixed_id_corner_sequences/data/278debee127a4c8cad63271e9ff13e22`;
its dataset SHA-256 is
`989ad1a9096d3841a5e0d903eb7afea35a598c9cd57820205ddc9e9e14354d3e`.
The exact server config is `/workspace/fixed_id_corner_sequences/prepare.yaml`.
The tmux session `fixed_id_corner_data` retains `prepare` and `audit` windows;
logs are `/workspace/fixed_id_corner_sequences/prepare.log` and `audit.log`.

The full source/output audit passed for every episode and sequence: 428,099
retained observations match original detector coordinates exactly, all eligible
certified observations remain present, visible and missing targets follow the
declared rule, causal inputs contain no current measurement, and parent splits
are unchanged. Of 428,105 certified source detections, five duplicate certificates
were collapsed to one original measurement each, and one terminal-only first
sighting had no future prediction target. There were zero artificially removed
certified observations. The 60,300 UNKNOWN detections remain unassigned and
7,061 never-seen episode/vertex combinations are excluded. UNKNOWN intervals
must not be interpreted as confirmed physical occlusion.

Canonical ID geometry and observed robot state additionally matched 18 sampled
original raw-GT frames, including world rotations. All seven local contract
tests passed. This preparation/audit ran NumPy on CPU; no simulator, CUDA model
or new training was executed. The implementation was a dirty working snapshot
based on `320b7cc`; exact source-file hashes are recorded in the effective config
and manifest. The original sources and previous model checkpoints are preserved.
Local copies of the [verification result](../outputs/fixed_id_corner_sequences/278debee127a4c8cad63271e9ff13e22/verification.json),
[summary](../outputs/fixed_id_corner_sequences/278debee127a4c8cad63271e9ff13e22/summary.json)
and [effective config](../outputs/fixed_id_corner_sequences/278debee127a4c8cad63271e9ff13e22/prepare.yaml)
are retained as ignored verification artifacts.

## Full episodes for tracked corner completion

`python main.py --config graph/configs/temporal/corner_tracking/prepare.yaml`
packs an existing **full-frame** vertex-family dataset into checksummed temporal
episodes. This preparation adapter also requires OpenCV for the existing label
utilities. Set `source.family_dataset` first. It reuses observed graph coordinates,
BOX physical certificates and the original raw observed TCP/finger state; it
does not run detection or edit RGB-D. PEG supervision reuses the existing unique
coordinate/projection association. Those GT-assisted IDs are labels only and
can remain UNKNOWN. The source certification policies have different strength;
they are not a claim of perfect corner identity.

The adapter rejects selected-frame/noncontiguous cohorts. It preserves the
source development-validation parents and reserves 10% of source training
parents, seed 0, exclusively for calibration. These parents do not receive
optimizer updates. The source dataset is original-TRAIN-only; its development
validation is not a new independent test. Prepared manifests support reserved
test metadata, but training/evaluation here never open those episode files.
Raw files, graph files and certificate archives are checked against source hashes.
Prepared episode bytes and the manifest are also hashed. Every episode carries
an explicit parent and split; overlapping parents are rejected.

For faster preparation, set the source path and `workers` in
`data_collection.temporal.corner_tracks_parallel:prepare` via
`graph/configs/temporal/corner_tracking/prepare_parallel.yaml` and run the same
root entrypoint. Workers process independent episodes, check the same source
hashes and GT/robot alignment, and report progress. The final manifest is written
atomically only after every episode succeeds; failed runs retain partial files
without a completion manifest. `tests.test_corner_tracks_parallel` compares the
prepared episode arrays and splits with the serial adapter and checks that source
bytes remain unchanged and corrupted input cannot publish a manifest.

See [tracked corner completion](corner_tracking.md#tracked-corner-completion-before-gnn)
for synthetic-removal manifests, chronological eligibility, supervision boundaries,
and the shared Mamba/Transformer workflow.

For the separate Graph-team development cohort that preserves selected original
TRAIN frames, detector metadata, physical-role certificates and immutable graph
inputs, see [the vertex-family study](vertex_family_semantics.md). Its
`data_collection.vertex_family:prepare` workflow uses the root config entrypoint
and does not change the collection gate, raw image contract or original splits.

Run commands from the repository root after installing `requirements.txt`.

For the explicitly requested 900-episode expert dataset on a Linux CUDA/Vulkan
host, also install `requirements-tracking.txt`, configure HF write access, and run:

```bash
WANDB_MODE=disabled python -m data_collection.collect_dataset \
  --config data_collection/configs/visible_peg.yaml
```

That config requests native Main/Wrist 512×512 RGB-D, CPU physics, CUDA rendering,
900 successful scripted-expert trajectories, 250-step horizons, raw GT at every
frame, two independent workers and eight-episode archives. Initial Main visibility
is recomputed from every saved initial frame. Every saved frame's raw image schema
and action/observation alignment are checked before publication. Success filtering
changes the reset distribution; no train/validation/test partition is imposed.
`publish: true` explicitly invokes the shared `tools.publish_experiment` trajectory
workflow after collection; `publish: false` keeps the result local. Existing output
directories are refused. A process failure is reported rather than automatically
resuming or overwriting its output. Failed expert rollouts remain local and are
excluded from the approved manifest and published archives. HF failures retain
local archives and the existing experiment run identity for a publishing retry.

`data_collection/scripted_policy.py` extracts the existing GT-guided servo expert
without importing team experiment packages. It uses `pd_ee_delta_pose` and the
installed controller's negative rotational scaling, and returns normalized native
actions matching the actual environment action-space shape. The original team
implementation remains unchanged. This is a privileged expert, not a learned policy.

```bash
python -m data_collection.collect_episodes --help
python -m data_collection.collect_episodes --output outputs/initial_frames \
  --num-episodes 10 --max-attempts 1000 --resolution 512
```

The collector tries consecutive reset seeds starting at `--start-seed` (default
0). It evaluates only the first `base_camera` observation and retains Main and
Wrist raw observations for approved episodes. No action is applied to rejected
states. The attempt budget is finite; insufficient approved episodes produce
`EXHAUSTED` and a nonzero CLI exit. Use a new output directory for each run;
existing output is never overwritten and automatic resume is not implemented.

### Paired Object21 demonstrations for BC

Use `data_collection.collect_object21_bc:collect` and
`data_collection.object21_bc:export` with the root launcher. The first
workflow reuses the raw shard collector and independent verification. Keep the
raw collection settings from `data_collection/configs/visible_peg.yaml`,
change the output directory and seed range, and set `publish: false`.
Provide a JSON exclusion inventory with the original 900 seeds in
`original_collection_seeds`, the 630 Graph development seeds in
`graph_development_seeds`, and their dataset and collection identities in
`graph_dataset_sha256` and `original_collection_run_id`.

```bash
WANDB_MODE=disabled python main.py --config /path/to/collect.yaml
WANDB_MODE=disabled python main.py --config /path/to/export.yaml
```

The collection YAML supplies `trainer`, `collection` (path to the raw settings)
and `excluded_seeds` (path to the inventory). The export YAML supplies
`trainer`, `raw_root`, `output`, `device`, `workers`, `split_seed`,
`split_counts`, and `models`. Set `split_counts` to the intended numbers
of train, validation and test episodes. For each of `models.transformer`
and `models.mamba`, supply `gnn`, its `sha256` and the matching `corner`
checkpoint. Use the resumed seed-0 pairs from the
[checkpoint index](detected_corner_gnn.md#published-checkpoint-backup).
Export needs CUDA, the official Mamba selective scan extension and the
[corner detector dependencies](corner_detection.md#install-and-run).

The exporter may run alongside collection and processes only verified shards.
It fixes split positions by sorted episode seed before inference and stores both
models' predictions for each expert action. Each split contains aligned
`robot_state`, `object21_transformer`, `object21_mamba`, and
`expert_action` arrays. Use `episode_offsets` to recover trajectories. The
Object21 export supplies the 9D robot input used by the corner tracker.

For the Recurrent BC release, run `data_collection.recurrent_bc_dataset:export`
with `raw_root`, `object21_root` and a new `output` in its YAML. It joins
Robot15 by reset seed and step, matching
`recurrent.features.FEATURE_LAYOUT`. The primary inputs are
`state36_transformer` and `state36_mamba`, each `[Robot15, Object21]` with
the same expert action and episode splits. The README includes the legacy
clean-observation State38 adapter, appending `[obs_fresh=1, obs_age=0]`.
The compact archive includes a loader example, split checksums and model
provenance. Publish it after export through
[the shared publisher](huggingface_artifacts.md).

The first [paired Recurrent BC release](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/eca8ba8b122e7d2e7007e7dc20200dad5d3e3725/integration/293fbe2c68764fb8901a40f857a1ade6)
contains 900 fresh successful episodes with 92,566 aligned action rows. The
train/validation/test splits have 720/90/90 episodes and
74,053/9,201/9,312 action rows. Reset seeds range from 200005 to 229115 and
have zero overlap with the previous 900-episode collection, including its 630
Graph development episodes. The archive's SHA-256 is
`aec78bdef35d8f873d0cc8215d1e986050940c9041e7e0948e21081d5531c1f0`.

## Visibility

### Initial Peg placement under the current main camera

The installed ManiSkill 3.0.1 `PegInsertionSide-v1` initialization draws the
Peg center uniformly from X `[-0.1, 0.1]` m and Y `[-0.3, 0]` m. Its Z is
the randomly configured half-width, `[0.015, 0.025]` m. The half-length is
`[0.085, 0.125]` m and yaw is in `[30°, 150°]`; roll and pitch are zero.
These are pre-dynamics initialization bounds, not a post-action trajectory.

For the shared 512×512 camera at `[0.65, -0.30, 0.65]` m looking at
`[0.065, 0.297, 0.11]` m, an analytical enclosing box for the entire Peg is
X `[-0.25, 0.25]`, Y `[-0.45, 0.15]`, Z `[0, 0.05]` m. Its positive-depth
vertices project to U `[5.65, 265.01]`, V `[253.21, 488.50]` pixels. Since
the box is entirely in front of the camera, its projected convex hull encloses
every allowed Peg geometry and lies inside the image. This proves the initial
Peg is geometrically in frame under these settings; it does not prove visible
surface area or prevent occlusion. Robot joints are randomized and the task
also contains the box, so visibility cannot be settled from the spawn bounds
alone. The box used for projection is deliberately larger than the realizable
Peg volume and its margins are conservative, not observed extremal values.

The analysis used the camera pose and FOV in the shared factory and checked
the resulting calibration against a previously saved frame, without sampling
Peg initial states. [Geometry figure](../outputs/peg_initial_geometry/peg_initial_geometry.png),
[analysis and bounds](../outputs/peg_initial_geometry/analysis.json), and
[calculation script](../outputs/peg_initial_geometry/analyze.py) are local
ignored artifacts. CPU-side geometry was evaluated; CUDA was not run.

### Initial Hole opening occlusion under the current main camera

On 2026-11-14, a separate diagnostic rendered one initial state for each of
the 1,200 simulator seeds 200–1399 with the current 512×512 main camera. It
used the four physical corners of the negative-local-X Hole opening. All four
projected corners were in frame and the opening was front-facing in every
sample. Only reset frames were tested; these counts do not describe grasping,
motion, other cameras or the complete seed space.

The existing 7×7 opening-interior depth/segmentation diagnostic found at least
one robot-blocked sample in 47/1,200 frames (3.92%). In 32 of those 47 frames
(68.1%), the nearest projected pixel of at least two of the four corners was
robot-labelled and at least 1.5 mm closer than the corner. Across all frames,
the per-corner robot-blocked counts were 0: 1,144, 1: 22, 2: 19, 3: 9 and
4: 6; hence at least two were robot-blocked in 34/1,200 frames (2.83%). The
interior grid and corner pixels measure different points, so their counts do
not have to be nested. The 7×7 grid covers only 70% of the radius in each
opening direction and neighboring samples can land on the same pixel.

As a separate approximation, a corner is supported if a box-labelled surface
within its 7×7-pixel neighborhood lies within 6 mm of the 3-D GT corner.
Thirty frames had two or more unsupported corners; 25 of the 34 frames with
two or more robot-blocked corner pixels also had two or more unsupported
corners. The two measurements need not agree at boundaries: the neighborhood
can contain box support even when the nearest corner pixel belongs to the
robot. The measurements are raster/depth diagnostics, not exact continuous
visibility or detector performance. A 3×3 robot-neighborhood sensitivity check
marked at least two corners in 43 frames, versus 34 at the nearest pixel.

[Example images](../outputs/hole_corner_occlusion/examples.png),
[summary](../outputs/hole_corner_occlusion/summary.json),
[per-frame data](../outputs/hole_corner_occlusion/results.json), and the
[calculation](../outputs/hole_corner_occlusion/analyze.py) are local ignored
artifacts. The run used CPU physics and SAPIEN/MoltenVK rendering on the Mac;
CUDA was not run. GT and simulator segmentation are diagnostic inputs only.

`data_collection/visibility.py` extracts the historical GT/depth gate, including its calibration
checks, signed depth handling, pixel centers and expert grasp-band calculation.
It imports no Graph, perception, tracker or model packages from this repository.

The default `--visibility-mode ends_and_grasp` means:

- All eight GT peg box corners are in the Main image, in front of the 1 cm near
  plane; projection boundary epsilon is 0.0001 pixels.
- Both end strips (17.5 mm each) and the historical expert grasp band (17.5 mm
  total width) have projected camera-facing surface pixels.
- Every predicted pixel in those regions has valid observed depth within
  1.5 mm of the GT box front surface. Nonpositive signed int16 depth is invalid.

`--visibility-mode full_surface` additionally requires depth confirmation for
all projected camera-facing peg surface pixels. Neither mode requires the
hidden back surface or visibility at later timesteps. These are GT-assisted
dataset selection rules, not a visual detector. Physical collision validity,
expert success and historical diversity/split selection are separate historical
filters and are not included in this extracted visibility collector.

## Trajectories

The default `--steps 0` stores only the approved initial observation. To collect
actual trajectories, provide `--steps N --policy your_module:make_policy`.
The factory takes no arguments and returns an object implementing:

```python
class Policy:
    def reset(self, env, seed):
        # Initialize policy state; do not reset, step, or reposition the env.
        ...

    def act(self, observation):
        # Native batched action matching env.action_space.shape.
        ...
```

The policy receives the same initial observation that passed selection and
controls its own stochastic seed initialization. Collection stops on terminated,
truncated or the requested step count. The policy's native action convention
must match `--control-mode` (omitting it uses the installed task default).

For a simulator/storage smoke check, use explicit zero actions:

```bash
python -m data_collection.collect_episodes --output outputs/collection_smoke \
  --num-episodes 1 --max-attempts 40 --steps 2 --zero-actions
```

This is not an expert demonstration. No old expert, training or model workflow
is imported by this extraction.

To use the standalone scripted expert through the existing collector:

```bash
python -m data_collection.collect_episodes --output outputs/expert_episodes \
  --num-episodes 3 --max-attempts 100 --resolution 512 \
  --steps 250 --max-episode-steps 250 --require-success --save-step-gt \
  --policy data_collection.scripted_policy:make_policy \
  --control-mode pd_ee_delta_pose --render-backend cuda --observation-device cuda
```

`--max-episode-steps` explicitly overrides ManiSkill's default 100-step time limit.
`--require-success` stops a trajectory on the first official success and keeps
unsuccessful attempts in `rejected_seed_*/`, outside the approved manifest.
`--save-step-gt` adds raw `ground_truth/` fields after each action, in addition to
the existing initial GT. These options are opt-in; initial-only defaults remain.

### Moving Hole collection

The existing sharded collection workflow can opt into the same motion schedule
as the Graph moving-Hole visualization. For a local example, run:

```bash
WANDB_MODE=disabled python -m data_collection.collect_dataset \
  --config data_collection/configs/moving_hole_collection.yaml
```

This example collects one successful expert episode with CPU physics and keeps
the output local. It uses the visualization's fixed 60 mm world +X translation,
3.5–9 s motion window, 3 s duration, 80 mm peg-tip-to-hole guard and expert
insertion-phase stop. These are example settings, not an agreed experiment
distribution; the 120 mm visualization uses `[0.12, 0, 0]`. `hole_motion`
accepts `continuous` or `jump`; distance and direction remain fixed by
`translation_m` for every episode. Per-episode randomization is not implemented.
Omit `hole_motion` for the original stationary collection. Motion collection
currently requires one CPU physics environment per worker, a positive start time,
and, when `stop_expert_insert` is true, a policy exposing `phase` after reset.

The Hole belongs to the box, so the whole kinematic box moves. The schedule is
reset for each accepted episode. After each nonterminal simulator step, the
collector applies the scheduled translation and refreshes the observation, GT
and evaluation info before saving and supplying the next policy input. The saved
action, reward and termination flags come from the preceding simulator step,
before that translation; they are not recomputed after moving the box. There is
no further movement after a terminal or truncated step or a success that ends
collection. Each frame additionally records `hole_motion/offset_m` and
`hole_motion/status`, and the effective settings are saved in `config.json`.
The initial visibility gate and both visibility modes are unchanged.

The lower-level `collect_episodes` CLI accepts the same mapping as a JSON string
via `--hole-motion`; use `--save-step-gt` to save GT after every action. The
sharded workflow passes the YAML mapping to each worker and retains its normal
success filtering, verification and explicit publishing behavior.

## GPU observations

Add `--render-backend cuda --observation-device cuda` on a supported Linux GPU
host. See [GPU observations](gpu_observations.md) for native rendering versus
output transfer, the runnable device check and backend verification limits.

## Saved data

```text
output/
├── config.json                 # parameters, actual backends, versions, source revision
├── attempts.jsonl              # every seed's first-frame visibility decision
├── manifest.json               # run status and committed episode list
└── episode_000000/
    ├── episode.json            # seed, frame/step counts, stop flags
    ├── visibility.json         # detailed first-frame selection evidence
    ├── frame_000000.npz        # raw initial observation, info and selection GT
    ├── frame_000001.npz        # next observation and action that led to it
    └── ...
```

NPZ keys flatten nested dictionaries with `/`, e.g.
`observation/sensor_data/base_camera/rgb`. Shapes, batch dimensions and dtypes
are preserved. Depth stays signed int16 millimetres; overflow/invalid values
are not repaired. Load with `np.load(path, allow_pickle=False)`. Each post-action
frame includes `action`, `reward`, `terminated`, `truncated`, and `info`.

An episode has T+1 observations and T actions. Every frame is read back before
being exposed. Episode folders are renamed from `.partial` only after their
receipt is written; interrupted/failed folders remain excluded from the manifest.
Exact simulator-state replay, automatic split assignment and video encoding are
not provided. The raw collector itself does not publish; the dataset workflow
above explicitly validates and publishes selected archives through the shared CLI.
Successful-task filtering requires the explicit `--require-success` option.

## Checks

```bash
python -m unittest tests.test_visibility tests.test_data_collection \
  tests.test_observation_device tests.test_maniskill_env -v
python -m unittest tests.test_observation_device_native_gpu -v
```

Recorded results and execution limits: [collection verification](data_collection_verification.md).

## Temporal-gate episode selection

`data_collection/temporal_gate_data.py` selects complete episodes from the existing
published graph cache. Its explicit parent manifest keeps Gate development within
the GNN train split and reserves original GNN validation for final comparison.
It neither collects new simulator data nor modifies cached frames. See
[learned temporal gate](gnn_training.md#learned-temporal-update-gate) for source
parent mapping requirements and the manifest-generation command.

`data_collection/temporal/frozen_episodes.py` shares frozen-frame inference with the
learned gate and GRU estimator. It sorts predictions, targets, observed TCP and
quality by one sample-ID permutation. `FrozenEpisodeDataset` requires complete
zero-based, contiguous episodes and validates row shapes and frame/episode IDs.
Graph/GT pairing is checked against the existing cache identity and byte checksum
before sequence construction. Inputs contain only current GNN outputs and actual
observation metadata; labels are returned separately as Hole world XYZ. GRU uses
the same explicit parent manifest and never reads original GNN test frames.
See [GRU estimator](temporal_estimation.md#gru-temporal-object-state-estimator).

`TemporalPoseDataset` in `data_collection/temporal/pose_sequences.py` extends this path
for Peg/Hole poses. It reuses sample ordering, checksum and parent-split checks,
adds valid PEG/BOX/HOLE quality and causal TCP translation changes, and returns
world-frame Object21 labels separately. Padding is structural and masked; input
normalization is fitted on train observations only. Optional `data.raw_root`
reads only `observation/extra/tcp_pose` and `observation/agent/qpos` from matching
raw frames, verifies every frame against the cache's source SHA256 and checks TCP
agreement. It adds wxyz TCP orientation, causal angular motion and the two Panda
finger positions. There is no fallback to `ground_truth/` fields. Cache-only mode
omits these nine unavailable values. See
[joint pose estimator](temporal_estimation.md#joint-peghole-temporal-pose-estimator).

Insertion-precision GRU retains this same complete-episode input/normalization
contract. `prepare_pose_sequences` can also reuse the independent-cache selector
in `data_collection/temporal/insertion_sequences.py`: a `test_role: independent` manifest must provide
explicit parent identities for both original and external cache episodes. All
external parents must be absent from every original GNN split. Training only
checks external split metadata for leakage; external frame tensors are opened
only for explicit test evaluation. `data.independent_raw_root` is required when
the independent test uses the observed-robot feature schema. The previous GNN
validation comparison remains available and is labeled reused reference data.

`data_collection/temporal/insertion_sequences.py` extends the same frozen-episode path
for insertion Mamba/GRU experiments. It adds observed-support ages and causal
windows, and requires an explicit `gnn_validation_reference` or `independent`
test role. Independent-cache parents must be absent from every original GNN
split. Training validates external metadata without reading independent test
frames. See [Mamba insertion estimator](temporal_estimation.md#mamba-insertion-precision-estimator)
for manifest creation, raw-proprioception joins and the independent-test workflow.

`data_collection/temporal/finetuning_split.py` partitions only the original Mamba
training parents into fine-tuning train/development (seed 0, 80/20 by parent).
The existing 126-episode validation is reserved as a reused development benchmark;
it enters neither gradients, auxiliary-weight calibration nor checkpoint selection.
Pretrained Mamba, Gate and GNN have already seen the new development parents, so
this is held out only from the fine-tuning updates. The original reference test is
also reserved and is not promoted to an independent test. Existing cache checksums,
raw observed robot joins, frame identities and visibility conventions are reused.

## Physical vertex coverage filtering

`data_collection.physical_vertex_selection:prepare` applies a GT-assisted,
frame-level filter to all original TRAIN frames from the 590 parents remaining
after excluding the 40 development-validation parents. Configure paths in
`data_collection/configs/physical_vertex_selection.yaml` and run:

```bash
WANDB_MODE=disabled python main.py --config data_collection/configs/physical_vertex_selection.yaml
```

The detector processes the original RGB-D, with no GT fed to it. For each PEG
candidate, match to a unique physical peg corner within 3 mm in each GT-local
coordinate and 2 pixels in projection; unmatched or ambiguous detections do not
count. The condition requires at least four distinct matched corners, excluding
any set whose corners all share one physical box face. For the Hole condition,
the existing GT/RGB-D topology certificate in `vertex_family_labels.py`
requires three distinct RIM physical IDs; exterior and UNKNOWN candidates do not
count. The coordinate, projection, face-depth and image-support tests are
finite-tolerance operational certificates, not proofs of physical identity.
GT is used in dataset selection and can introduce selection bias; it is not a
reliable deployable selection method without GT. Passing the gate is evidence
under these rules, not a claim about every physically visible corner.

The run verifies source frame hashes, retains the original source IDs, and writes
all decisions to `frames.jsonl` plus aggregate counts and source/config hashes.
It does not alter the input dataset or exclude frames from other workflows.
`max_frames` is only for smoke checks; only an unrestricted 590-parent run can
produce complete population counts. The JSONL can be used as a frame-ID filter
for a later training workflow; no training is initiated here.

The raw and cached entrypoints share frame selection, input validation, ordered
spawn-worker execution, aggregation and output recording in
`data_collection/physical_vertex_selection.py`. The cached entrypoint remains
available at `data_collection.physical_vertex_cache_selection:prepare`; existing
config files and commands still work. Raw workers retain detector UV, while
cached workers project checksum-verified XYZ and exclude padded graph nodes.
Episode auditing reuses the cached workers and preserves the original 960-frame
decisions separately. Cached provenance records both the shared implementation
and the compatibility entrypoint's code hashes.

Run `.venv/bin/python -m unittest tests.test_physical_vertex_selection tests.test_physical_vertex_filter -v`
for parent/ID validation, frame order, population and smoke counts, checksum
rejection, distinct pixel-source decisions, preservation of the original
validation decisions, and actual CPU spawn-worker execution on synthetic native
RGB-D frames. These checks do not rerun the full server dataset or CUDA detector.

For an expanded vertex-family development cohort, `selection.validation_selection`
can point to a prior `selection.json` to preserve its validation episodes while
selecting training parents from the remaining original TRAIN pool. See
[vertex-family semantics](vertex_family_semantics.md#확장-학습-설정).

`selection.frames_per_parent: null` retains all recorded frames for the selected
parents. `raw_storage: hardlink` avoids duplicating raw files when the source and
output are on the same filesystem; the linked raw files share source inodes and
must be treated as immutable. The default `copy` creates independent files.

### Verified cached-graph selection result

On 2026-11-08 KST, the published graph cache was used for the full 590-parent
selection with `data_collection/configs/physical_vertex_cached_selection.yaml`.
Every raw frame and graph archive was hash-checked, and detected UV was recovered
by projecting cached world XYZ through the saved camera calibration. This is not
a fresh detection for all frames: on the first 8,952 frames, independent GPU
re-detection and cached-graph selection produced identical per-frame decisions.
That check does not prove equivalence on every remaining frame. The detector
source files match the cache's recorded source hashes; a 150-frame inspection
showed reconstructed UV differences at most 0.000069 pixels.

| Rule | Frames |
| --- | ---: |
| Peg condition | 21,176 |
| Hole condition | 9,481 |
| Both | 8,093 |

These totals are under the finite-tolerance GT-assisted rules above and apply
to the 60,600 recorded original TRAIN frames in 590 parents only. Local
[summary](../outputs/physical_vertex_selection/7cd80170f5304b69b8384cbf99b457c3/summary.json)
and [frame decisions](../outputs/physical_vertex_selection/7cd80170f5304b69b8384cbf99b457c3/frames.jsonl)
contain all IDs; `both_pass` selects the intersection. No model training was run.

For family preparation, `reuse_family` lists dataset roots with completed or
recovered manifests. Source dataset, label-generating code, settings, archive
hashes and graph/label node ordering are checked before graph/label files are
hardlinked into the new dataset. `reuse_graph_cache` points to the original
cache with metadata/inventory: after obtaining missing detector provenance,
its graph is reused only if XYZ, object order, confidence, TCP and Object21
match exactly. Detector mismatches rebuild that frame's graph rather than
attaching fresh labels to a different node set. Reuse totals are logged and
recorded in the final manifest. An interrupted run's completed frame archives
can be recovered after validating their identities and content; incomplete
archives must be recomputed.

### Physical conditions across complete validation episodes

`data_collection/configs/physical_vertex_episode_selection.yaml` applies the same
GT-assisted physical vertex criteria to every frame represented in a temporal
prediction file. Run `python main.py --config data_collection/configs/physical_vertex_episode_selection.yaml`
after setting the prediction, original 960-frame selection, parent manifest, raw
frame and cache paths. Raw and cached files are checksum verified. New frames use
pixels reprojected from cached detector XYZ; the previously selected 960 retain
the original detector-pixel decisions. The workflow records any disagreements.
For the seed-0 validation episodes, 4,102 frames across 40 parents were checked;
all 960 original decisions agreed with the reprojection audit. The result is a
conservative GT/RGB-D certificate, not a model input or a claim that uncertified
corners are invisible.

### Restore the frozen-M0 temporal experiment on another host

`data_collection/temporal/restore_hf_temporal.py` restores the already published
original graph cache and contiguous raw training-pool episodes, plus explicitly
published temporal checkpoints and the matching runtime/split bundle. It pins
repository revisions, verifies every archive and selected raw frame against its
SHA-256, restores only the original 630-episode training pool, and deletes each
verified raw archive after extraction to limit disk use. The transfer uses a
private Hugging Face dataset and model repository and requires an existing
read-capable token supplied to the process; credentials are not written to the
config, logs or artifact bundle. The six temporal continuations retain the
original frozen 60-epoch M0 checkpoints; the independently continued GNN is not
substituted into them. The restored temporal configurations resume epoch 90 with
Adam and random state, then run 30 additional epochs. Local focused tests cover
selected archive members and checksum rejection.

## Camera visibility trajectory source

`data_collection/configs/camera_visibility_trajectories.yaml` records seeded,
camera-independent expert-prefix and random-action episodes for camera studies.
It saves raw action sequences, Peg/Hole/Box GT positions, Hole orientation,
sampling frames and whether grasp/lift succeeded; failures remain in the
manifest. Each environment seed receives a distinct deterministic random-action
RNG stream. After configuring seeds and output, run:

```bash
WANDB_MODE=disabled python main.py --config data_collection/configs/camera_visibility_trajectories.yaml
```

This collection provides no model training or RGB-D dataset.
See [the broader camera study](detected_corner_gnn.md#broader-random-position-camera-study)
for replay, sample selection and limitations.

## Frozen independent simulator tests

`data_collection.independent_test_protocol:freeze` checks the named original
collection and Graph-development seed inventories before drawing test resets,
records checkpoint hashes and fixes policy/camera settings. Its configuration is
`data_collection/configs/independent_test_protocol.yaml`. The separate
`data_collection.independent_test_collection:collect` workflow then records
all planned expert and random-action cases without silently filtering failures.
Native base-camera RGB-D and calibration, observed robot state, raw GT and the
applied action are stored with per-frame hashes. The source protocol and output
paths cannot be overwritten by these workflows.

The separate matched-visibility follow-up is generated with
`data_collection/configs/matched_visibility_test_selection.yaml`. It fixes the
candidate seed order before using the existing GT-assisted initial visibility
gate and records every attempted seed, including failures. It was designed after
the first unfiltered test and must remain identified as a follow-up. Full
criteria, limitations and saved results are in the
[independent test and leaderboard replay guide](detected_corner_gnn.md#independent-generalization-test-and-leaderboard-replay).

## Unfiltered mixed trajectory collection

The mixed trajectory workflow freezes pilot and main reset seeds and split
membership before collection, excludes the configured historical seed inventories,
and records Expert, Random, and Expert→Random from each same initial state. It
uses the explicitly configured main camera and preserves unsuccessful episodes,
unreached transition phases and terminal observations. RGB, depth, segmentation,
calibration, robot state, action, GT and status are stored as native arrays in
per-frame NPZ files. Frames `0..T-1` contain the action applied from that
observation; frame `T` is the terminal observation with `action_valid=false`.
The source expert uses live GT and is not a learned policy.

Run the 100-state pilot with Python 3.11 and the shared simulator dependencies:

```bash
WANDB_MODE=disabled python main.py --config data_collection/configs/mixed_trajectory_pilot.yaml
WANDB_MODE=disabled python main.py --config data_collection/configs/mixed_trajectory_pilot_audit.yaml
```

Pilot and all main split assignments, phase choices, delays and random streams
are preregistered in the saved protocol. Each run also records the effective
config and hashes through the shared `ExperimentLogger`. Run the main config
only after the pilot audit; choose a sufficiently large, writable output path in
`data_collection/configs/mixed_trajectory_main.yaml`. The main workflow checks
that its protocol matches the audited pilot and that at least 120% of the
pilot-based storage estimate plus the configured reserve is available before it
creates an output directory. `outputs/` is local and ignored; nothing is uploaded.
The audit checks all frame hashes, camera position, action alignment and split
membership, and reports initial Peg and sampled Hole visibility. These are
GT/depth diagnostics, not exact visibility throughout the trajectories.

Output directories cannot be overwritten. An interrupted collection keeps its
partial episodes and manifest for inspection; it does not currently support
in-place resumption. Keep these files when planning a retry. On macOS use CPU
physics and the host SAPIEN/MoltenVK renderer; CUDA has not been tested for this
workflow on this host.

### 파일럿 State36 쌍 캐시

`recurrent/configs/tapnextpp_state36_cache_cuda.yaml`은 파일럿 원본과 완성된
[TAPNext++ 꼭짓점 캐시](tapnextpp.md#꼭짓점-검출-추가)를 조인한다. 원본 프레임과
부모 캐시 해시를 확인하고, 기존 배열을 복사한 궤적별 NPZ를 새 실행 폴더에 저장한다.

`state36_gt`와 `state36_transformer`는 같은 `frame_indices`에 대응하는
`float32 [T, 36]` 배열이다. 두 배열의 첫 15열은 동일한 로봇 상태다.
나머지 21열은 각각 저장된 물체 GT와 해당 프레임까지의 검출을 기존 Transformer
추적기와 GNN(seed 0)으로 처리한 추정값이다. Graph 입력은 저장된 꼭짓점과 관측된
로봇 상태다. 추적기는 궤적마다 초기화한다. `counts`는 관측된 Peg 및 Box 노드
수이며, `graph_has_nodes`는 GNN에 해당 프레임의 노드가 있었는지 나타낸다.

`action[t]`는 관측 `t`에서 적용한 실제 제어 입력이다. 마지막 관측까지 보존하며,
행동 학습에서는 `action_valid`가 참인 행을 고른다. 두 State36 배열과 `action`은
같은 마스크로 인덱싱한다. 결과에는 세 정책의 궤적이 모두 포함된다.

### 공유 TAPNext++·State36 캐시

**FORBIDDEN** · 이 섹션은 기존 캐시의 용도를 기록한다. 에이전트가 캐시 파일이나
복제본에 직접 또는 로더를 통해 접근하려면 현재 작업에 대한 사용자의 명시적
허락이 필요하다. 접근 범위는 [저장소 에이전트 지침](../AGENTS.md#forbidden-cache-access)을
따른다.

공유 NPZ는 파일럿의 원본 episode와 frame 순서를 유지한다. Recurrent 형식
`float32[N,36]` GT는 `state36_gt`에 프레임당 한 행씩 넣는다.
`episode_offsets[e:e+2]`가 에피소드의 반열린 행 범위이고,
`frame_indices`는 에피소드 시작에서 0으로 돌아간다. 종료 관측도 포함한다.
`metadata_json`에 원본 manifest, detector cache, TAPNext++ 체크포인트의
SHA-256과 State36 필드 순서를 기록한다.

`surface_xy`와 `surface_visible`은 현재 프레임의 Peg 표면점 추적 결과다.
`corner_frame_offsets[t:t+2]`는 같은 프레임의 `corner_xy`와
`corner_visible` 행 범위다. `corner_query_offsets[e:e+2]`는 그 에피소드
초기 꼭짓점의 `corner_query_xy` 및 `corner_query_object_id` 범위다.
좌표는 원본 카메라의 픽셀이며, 꼭짓점 ID는 에피소드 내부 초기 쿼리 번호다.
검출된 꼭짓점과 깊이로 검증된 Peg 표면점은 같은 causal TAPNext++ stream에서
추적한다. 초기화 전은 NaN 좌표와 false 가시성으로 표시한다.

이 공유 캐시는 Pilot/Recovery의 State36 GT와 추적 결과를 프레임별로 정렬하는 데
쓰인다. `corner_xy`는 에피소드 초기에 검출한 꼭짓점 쿼리를 TAPNext++로 추적한
좌표다. 이 좌표는 [기존 GNN 실험의 프레임 선택](#기존-추적-기반-프레임-선택)에
사용됐다. 새 조건 집계와 [Five-mode 그래프 입력](#five-mode-graph-input)의
프레임별 꼭짓점 좌표는 별도의 `corners` 캐시에서 가져온다.

```python
import numpy as np

with np.load("tapnext_state36.npz", allow_pickle=False) as cache:
    start, stop = cache["episode_offsets"][:2]
    state36 = cache["state36_gt"][start:stop]
    surface = cache["surface_xy"][start:stop]
    corner_start, corner_stop = cache["corner_frame_offsets"][start : start + 2]
    first_frame_corners = cache["corner_xy"][corner_start:corner_stop]
```

`data_collection.shared_tapnext_cache`는 원본·검출·GT의 에피소드 순서와
프레임 수를 검사하고, CUDA 추적 결과를 같은 순서의 NPZ로 합친다.
원본이나 캐시를 공유할 때는 `(episode_ids, frame_indices)`로 정렬한다.

2026-11-16 KST에 같은 파일럿 원본의 300개 episode와 종료 관측을 포함한
62,395개 frame을 CUDA에서 추적·병합했다. 압축 파일은 24,319,089 bytes다.
전체 raw frame의 해시와 GT 행, 관측 배열의 episode/frame 범위를 검증했다.
초기 표면점과 Peg·Hole 조립체 쪽 꼭짓점 쿼리는 300개 episode에서 확보했다.
원본 policy mode와 reset seed를 보존했다.
[공유 NPZ](../outputs/shared_tracking_cache/tapnext_state36.npz)

### 꼭짓점 가시성 프레임 인덱스

Five-mode 선택 매니페스트와 그 매니페스트가 참조하는 `corners` TAR가 있는
환경에서 저장소 루트 기준으로 실행한다.

```bash
.venv/bin/python main.py --config data_collection/configs/corner_visibility_index.yaml
```

조건 분류에는 현재 프레임에서 검출기가 병합한 꼭짓점만 사용한다.
검출이 없는 프레임도 분류하며, 검출 누락을 이전 프레임이나 추적으로 채우지 않는다.
`corner_object_id == 0`인 점을 Peg로 센다. `corner_object_id == 1`이면서
`corner_contour_hole == true`인 점을 Hole 림으로 센다. Box 외곽, TAPNext++ 추적점,
GT 매칭은 분류에서 제외한다.

| 키 | 판정 |
| --- | --- |
| `condition_1` | Peg 4개 이상이며 비공면, Hole 림 3개 이상 |
| `condition_2` | `condition_1`에 해당하지 않으며 Peg 2개 이상, Hole 림 1개 이상 |
| `condition_3` | Peg 0–1개 또는 Hole 림 0개 |

모든 프레임은 세 조건 중 하나에 속한다. 예를 들어 Peg 4개가 공면이고
Hole 림 4개가 검출되면 `condition_2`, Peg 7개와 Hole 림 0개면
`condition_3`이다.

비공면은 Peg의 검출 3D 좌표를 최적 적합 평면에 투영했을 때 수직 거리의
RMS가 `coplanarity_tolerance_m`을 초과하는 경우다. 기본값은 1 mm다.
검출기 좌표의 작은 오차로 공면 점이 비공면으로 판정되는 것을 줄이기 위해
허용 오차를 사용한다. 캐시의 림 표시는 삽입구의 로컬 −X 방향이나 물리
꼭짓점 ID를 지정하지 않는다. 분류는 검출기 병합 후 림 표시를 그대로 센다.

결과 JSON은 매니페스트의 원래 split을 보존한다. `counts[condition][split]`에
프레임 수를 기록하고, `episodes`에는 seed, 정책, split, 조건별 프레임 번호를
저장한다. 다시 집계할 때는 설정의 `output`에 새 경로를 지정한다.

2026-11-17 KST에 선택 매니페스트의 800개 seed, 4,000개 에피소드, 756,004개 프레임을
집계했다. 각 detector NPZ의 SHA-256을 확인하고 프레임별 분류를 독립 계산으로 검증했다.
전체 중 112,174프레임에서 검출기가 Hole 림 점을 4개 넘게 표시했다.
[선택 매니페스트](../outputs/five_mode_caches/selected_800_seeds/manifest.json) ·
[분류 결과](../outputs/five_mode_caches/selected_800_seeds/detected_corner_visibility_all_frames.json)

### 기존 추적 기반 프레임 선택

Git 커밋 `0cfca96`에 기록된 기존 Pilot/Recovery 프레임 인덱스는
TAPNext++가 추적한 꼭짓점 쿼리를 현재 깊이로 3D 복원하고, 원본 GT와
물체별 1:1 대응을 구했다. 기본 대응 거리는 5 mm 이내이고, 차순위
후보와의 거리 차이는 2 mm 이상이다. Hole은 로컬 −X 삽입구의 네 모서리만
세고, Peg는 매칭된 물리 꼭짓점의 affine rank를 계산했다. 이 방식은
초기 Five-mode 확장 실험의 가져온 프레임 선택에도 사용됐다.
이 인덱스의 용도는 해당 실험의 재현에 한정한다. 현재 Five-mode 프레임
선택에는 [꼭짓점 가시성 프레임 인덱스](#꼭짓점-가시성-프레임-인덱스)를 사용한다.

기존 `outputs/corner_visibility_index/frame_locations.json`은 그 실험들의
재현용 프레임 목록이다. 조건마다 `train`, `validation`, `test` 원본 NPZ
경로가 들어 있다. 당시 `condition_2`는 Peg 3개와 Hole 림 2개,
`condition_3`은 Peg 0–1개와 Hole 림 0개였다.
Pilot과 Recovery의 reset seed를 합쳐 분할 난수 seed 0,
validation과 test 각 10개 seed로 나눴다. 동일 reset seed의 에피소드는
같은 split에 속한다.

### 프레임별 검출 꼭짓점과 TAPNext++ 표면점 연결

아래 명령은 기존 Pilot/Recovery 연결본의 재현 절차다. 입력은 원본
Pilot/Recovery와 대응하는 기존 공유 TAPNext++ NPZ다. 해당 NPZ에 접근할 때는
[캐시 접근 규칙](../AGENTS.md#forbidden-cache-access)을 따른다. 현재 Five-mode의
캐시 생성 절차는 [Five-mode mixed trajectory dataset](#five-mode-mixed-trajectory-dataset)에
있다. 기존 설정은 로컬 MPS의 메인 RGB-D 검출기를 사용한다.

```bash
.venv/bin/python main.py --config data_collection/configs/link_detector_corners_pilot.yaml
.venv/bin/python main.py --config data_collection/configs/link_detector_corners_recovery.yaml
```

검출 캐시의 원본 매니페스트 해시, 에피소드 순서, 프레임 번호와 각 NPZ의
해시를 확인한 뒤 두 연결본을 저장한다. 두 연결본에서 프레임 `i`의
검출점은 `corner_frame_offsets[i:i+2]`가 가리키는 `corner_uv`,
`corner_world_xyz`, `corner_object_id` 범위다. 점 순서는 매 프레임
검출기 출력 순서이며 프레임 간 꼭짓점 ID는 없다.

- `tapnext_with_detector_corners.npz`에는 기존 State36과 밀집 표면 추적 배열,
  새 검출점이 들어 있다.
- `visible_surface_with_detector_corners.npz`에는 새 검출점과 기존 추적에서
  보인 표면점이 들어 있다. 같은 프레임의 새 검출점과 반올림한 픽셀이
  겹치는 표면점은 제외하며, `surface_frame_offsets`와 `surface_track_id`로
  표면점을 조회한다.

두 연결본에는 각각 새 포맷 이름이 기록된다. 기존
`shared_tapnext_state36` 로더는 그대로 이전 포맷을 읽는다. 이미 완료한
검출 캐시를 다시 연결하려면 설정에 `detection_manifest`를 지정한다.

`shared_tapnext_state36`의 `corner_xy`는 초기 검출 꼭짓점을 TAPNext++로
추적한 좌표다. 프레임별 검출 꼭짓점과 비꼭짓점 Peg 표면점을 함께 쓰려면
`data_collection.detector_surface_cache.track_episode`로 표면점 전용 추적을
실행하고 같은 프레임의 검출기 `corner_world_xyz`와 추적 결과의
`surface_world_xyz`를 합친다. 이 표면점 전용 경로는 쿼리 초기화와 각
프레임 출력에서 검출 꼭짓점 반경 안의 점을 제외한다.

### Mixed trajectory pilot results

On 2026-11-14, the pilot collected 100 reset seeds and all 300 planned
trajectories with the explicit main camera. A sequential prefix was interrupted
for partitioned collection; completed overlapping episodes and the interrupted
partial directory were preserved separately. The merged manifest records both
collector source hashes and links the worker manifests. The post-merge audit
verified 62,395 frame hashes, action-valid alignment, camera position, initial
state fingerprints and exact case/split coverage. All 100 initial states passed
the existing GT/depth ends-and-grasp visibility diagnostic; this is not a
continuous-time visibility guarantee.

| Mode | Episodes | Successes | Mean maximum Peg displacement | P90 maximum Peg displacement |
| --- | ---: | ---: | ---: | ---: |
| Expert | 100 | 98 | 364.17 mm | 512.47 mm |
| Random from reset | 100 | 0 | 1.12 mm | 0.0012 mm |
| Expert→Random | 100 | 4 | 103.05 mm | 268.09 mm |

Displacement is the maximum relative to the initial Peg position across the
whole episode, so the Expert→Random figure includes its expert prefix. Full
Random rarely moved the Peg substantially, even though its robot actions were
random; the mean is influenced by a small number of larger movements. Of 100
Expert→Random episodes, 96 switched; all four non-switches had a target `close`
phase that ended before the selected delay. No failures were discarded.

An exploratory recovery probe replayed one case per expert phase through 35
random steps, verified Peg/Hole pose and robot joint state at the replay point,
then restarted the GT-guided expert for up to 100 steps. Two of six succeeded.
The six selected cases and reset-based expert restart are not a recovery success
rate estimate.

### Expert → Random → Expert recovery collection

Use the completed 100-seed mixed-trajectory pilot on the same ManiSkill and
rendering environment. Run from the repository root:

```bash
WANDB_MODE=disabled python main.py --config data_collection/configs/mixed_trajectory_recovery.yaml
```

The recovery collection preserves the pilot reset seeds and initial state
fingerprints. It replays the original applied controls through 35 Random
steps, verifies the live branch state, then restarts the scripted expert
in place for up to 100 steps. If the pilot missed its scheduled transition,
Random begins at the last action in the scheduled phase. If the pilot began
with Random, the corresponding Expert trajectory supplies one starting
action. Adjusted cases use the original per-seed random stream and gripper
setting and record the reason in `switch_adjustment`.

Each seed produces one episode regardless of success. The raw observation
and action schema matches the pilot, including the terminal frame. Use
`switch_step` and `recovery_step` in `episode.json` as the first Random and
returning Expert action indices. `successful` records the simulator outcome,
and `expert_recovery_actions` counts actions after the return. The collection
preserves the pilot's seed grouping for future splits.

On 2026-11-16 KST, CPU physics and SAPIEN/MoltenVK rendering on the Mac
produced 100 episodes and 18,250 frames, including 40 successful insertions
and 60 episodes that reached the recovery limit. A raw-frame audit confirmed
all three action segments, exact seed coverage, the parent hashes of the
replayed prefixes, and action alignment through the terminal observation.
[Recovery manifest](../outputs/mixed_trajectory_recovery/manifest.json) ·
[Raw-frame audit](../outputs/mixed_trajectory_recovery/audit.json)

The [private Hugging Face recovery publication](https://huggingface.co/datasets/Graph-Recurrent-VLA/mixed-trajectory-pilot/tree/62282f2f298acb27b4c0f7504d636b5c306653f3/graph/ccae6db8266a4347bb7341bb6ed620ab)
uses 301 MB of archives and references 8,454 byte-identical frames in the
original pilot. Download the four `recovery_arrays_*.tar.zst` archives and
extract `recovery_metadata.tar` beside them. With the original pilot already
restored and `zstd` installed, run from that directory:

```bash
python restore_recovery.py --shard-root . --pilot-root /path/to/mixed_trajectory_pilot \
  --output /path/to/mixed_trajectory_recovery
```

The restore checks every output NPZ against the original recovery manifest.
Full restoration was verified with Python 3.11.18 and zlib 1.2.12.

The pinned publication was downloaded to Vast.ai instance `60669483` and
restored at `/workspace/data/mixed_trajectory_recovery`. All 18,250 raw frame
checksums were verified. CUDA produced the TAPNext++ tracks and Recurrent
State36 labels for all 100 episodes. The
[shared recovery cache](../outputs/shared_tracking_cache/tapnext_recovery_state36.npz)
is stored on Vast.ai at
`/workspace/data/shared_tracking_cache/tapnext_recovery_state36.npz`.
Its SHA-256 is
`80d3166c12bf8cd2a90e2534915d64c59948519882b9af5dc74051a32f92ff77`.

The recovery cache inherits the pilot's reset-seed assignment. The
train/validation/test splits contain 80/10/10 recovery episodes and
14,574/1,923/1,753 frames. An exact-hash audit using both source manifests
confirmed that shared source frames and initial scenes stay within their
split. The Graph State21 loader read real train and validation samples
from the recovery cache.

### Five-mode mixed trajectory dataset

The five-mode RGB-D dataset groups these collection conditions under one
policy-mode label. The label identifies both the action policy and whether
the Box moves during the episode.

| Policy mode | Box motion | Action policy |
| --- | --- | --- |
| `Fixed_Hole_expert` | Fixed | Expert throughout |
| `Fixed_Hole_expert_random` | Fixed | Expert, then random |
| `Fixed_Hole_expert_random_expert` | Fixed | Expert, then random, then expert recovery |
| `Fixed_Hole_random` | Fixed | Random throughout |
| `Moving_Hole_expert` | Moving | Expert throughout |

For a new five-mode evaluation set, select reset seeds outside the training
and validation splits and collect all five modes for each selected seed.
This gives a paired expert comparison on the same initial scene and measures
Moving-Hole performance on held-out scenes. Keep every episode from a reset
seed in the same split.

Run the five-mode collection from the repository root:

```bash
WANDB_MODE=disabled python main.py --config data_collection/configs/five_mode_collection.yaml
```

On the configured collection instance, invoke the same collector with the
server's seed-inventory paths:

```bash
cd /workspace/moving_hole_collection
./run_collection.sh -u gpu_benchmark/code/main.py \
  --config gpu_benchmark/code/data_collection/configs/five_mode_collection_server.yaml
```

The example uses CPU physics with CUDA rendering, and requires `tar`, `zstd`,
a writable staging mount, and the historical seed inventories configured by
`base_config`, `exclusion_sources`, and `excluded_seed_protocols`. The seed
sampler excludes every reset seed in the original mixed-trajectory protocol
and all seeds in the named inventories and ranges. Add other experiment
protocols with `records[].seed` to `excluded_seed_protocols` when their seeds
also need to be excluded. The output protocol records the chosen seeds and
the inventory hashes.

With `batch_episodes: 8` and `workers: 2`, each base collection batch
contains two seed jobs of four episodes each. Each worker collects three
Fixed-Hole trajectories and one Moving-Hole expert trajectory per seed.
The same two workers then collect the Fixed-Hole recovery episodes by
replaying the matching Fixed-Hole expert-to-random trajectories. If the
requested phase is reached but the switch delay does not fit, recovery
branches at the last action in that phase. If the phase is never reached,
recovery branches after one expert action.

Each full archive contains ten episodes, with all five modes together for
each of its two seeds. Its `episode_index.json` records the policy labels,
episode paths, and measured Box offsets. The Moving-Hole expert uses the
configured translation with the proximity guard active.

For multiple collection hosts, generate one protocol with
`build_dataset_protocol(resolve_config(config))` before starting any workers.
Use the same `seed_count`, split counts, and exclusion inventories on each host.
Set `protocol_path` to the frozen JSON and give each worker its own
`record_range: [begin, end]` with the end excluded. Use a separate output
directory for each worker. When `resume: true`, rerunning the same config
checks archive hashes and collects ranges that have no valid receipt.

After collection, the cache builder runs the Main-camera RGB-D corner detector
independently on each saved frame and records its detections in `corners`.
It writes TAPNext++ non-corner tracks to `surface`. For this combined build,
use a CUDA host with the
[TAPNext++ prerequisites](tapnextpp.md#tapnext-point-tracking) and `tar`/`zstd`.

```bash
WANDB_MODE=disabled python main.py --config data_collection/configs/five_mode_cache_cuda.yaml
```

Set `source_root` to a completed shard and `output_root` to its cache directory.
The cache extracts one archive at a time and checks for 10 GiB of free space
on the output filesystem by default. Set `minimum_free_gib` when the
uncompressed input batch and cache outputs fit within a smaller reserve.
Its manifest joins the two per-episode NPZs by reset
seed, mode and split, records frame counts and hashes, and identifies episodes
where no surface query initialization was possible. Archive receipts make
completed batches reusable when the same cache configuration is rerun.

#### Five-mode graph input

**Five-mode 꼭짓점은 현재 프레임의 Main 카메라 RGB-D 꼭짓점 검출 결과로만
정의한다.** Peg, Box 외곽, Hole 림에 모두 적용한다. 해당 프레임에서 검출되지
않은 꼭짓점은 관측 꼭짓점 목록에서 빠진다. 이전 프레임의 검출이나 TAPNext++
추적으로 꼭짓점을 이어 가거나 채우는 방식은 이 입력에 사용하지 않는다.
`condition_1/2/3`과 현재 Five-mode 그래프의 꼭짓점 좌표는 같은 `corners`
캐시를 사용한다. TAPNext++는 선택적으로 추가하는 비꼭짓점 `surface` 점만
제공한다.

In the detector-corner + surface-point GNN experiment, each selected
`condition_1` frame contributes **one graph** with nodes from two caches:

| Cache | Nodes in the graph |
| --- | --- |
| `corners` | The Peg and Box/Hole corners detected directly in that frame, using `corner_world_xyz`. |
| `surface` | TAPNext++-tracked Peg surface points that are visible in that frame, have valid depth, and lie outside the configured exclusion radius around detected corners. Their 3D coordinates come from `surface_world_xyz`. |

TAPNext++ initializes its queries on the Peg surface outside the exclusion
radius around detected corners. To build each graph, pair the caches by episode
and frame index and include both sets of nodes. For new condition selection,
use the [per-frame detector index](#꼭짓점-가시성-프레임-인덱스).
Surface points contribute graph nodes only. The early expanded runs used the
[historical selection](#기존-추적-기반-프레임-선택). See the
[surface-cache implementation](../data_collection/detector_surface_cache.py).

### Mixed trajectory collection with a moving Box

The earlier four-mode Moving-Hole experiment ran all four moving policies on
the first 25 reset seeds in the frozen Fixed-Hole pilot protocol (100 episodes).
To reproduce it, keep the original pilot manifest and protocol locally.
The initial scenes, camera, controller, random action
streams, transition schedule, recovery procedure and raw frame schema come
from the Fixed-Hole workflow above. The recovery collector replays the newly
collected moving-Box prefix, including the same motion schedule. If the Box
movement prevents the expert from reaching the scheduled phase, recovery uses
the paired Fixed-Hole switch step (bounded by the available moving prefix).
Each adjustment is recorded in the recovery protocol.

```bash
WANDB_MODE=disabled python main.py --config data_collection/configs/mixed_trajectory_moving_hole.yaml
```

The Box translates 60 mm along world +X from 0.5 to 2.0 s and holds its new
position. The 80 mm proximity guard stays active throughout the move. The
schedule is independent of the expert phase. Outputs go to the configured
directory with separate `pilot/` (75 episodes) and `recovery/` (25 episodes)
manifests. `motion_audit.json` lists the final Box offset of each trajectory.
The run checks each reset against the Fixed-Hole reference. A fingerprint
mismatch is checked against the raw reset arrays with an absolute tolerance of
`1e-7` for platform rounding. Both Box and Hole must reach the target translation
in all episodes. Set `render_backend: cuda` on a Linux host with Vulkan/CUDA
interop to use GPU rendering with CPU physics.

### Moving-Hole collection with RAM staging

The example collects eight episodes (two seeds) with two collection workers and
writes compressed archives to `output_root`. Extend `record_range` to collect
additional batches. Set `render_backend: cuda` for GPU rendering on supported
Linux hosts. Linux defaults to `/dev/shm` when present. On macOS the default is
the system temporary directory, and `ram_root` can point to an existing RAM
disk for memory-backed staging. Allow space for two batches of uncompressed
frames when staging in RAM.

```bash
WANDB_MODE=disabled python main.py --config data_collection/configs/mixed_trajectory_moving_hole_streaming.yaml
```

The collector writes and verifies uncompressed NPZ frames in RAM. A background
worker compresses the completed batch into a `.tar.zst` file while the next
batch collects. Each archive is synchronized and verified before its RAM copy is
removed. The output manifest records the archive SHA-256 and seed range.
Extract each archive to its own directory with `tar --zstd -xf FILE` to read the
usual pilot, recovery, and motion-audit files with their original NPZ arrays and
frame checksums. The server needs both `tar` and `zstd`.

On 2026-11-16 KST, the RTX 3090 server collected the same eight episodes
(2,000 frames) with CUDA rendering and one seed per batch. Compressed NPZs
staged in RAM and copied to disk took 122.5 s and occupied 559.4 MB. Streaming
uncompressed NPZs through background Zstandard compression took 92.8 s and
occupied 194.1 MB on disk. The test included the final writes and archive
verification. All archived frames matched their collection checksums. The
server report is at
`/workspace/moving_hole_collection/ram_stream_benchmark/comparison.json`.
The same server later collected two eight-episode batches with two workers
in 94.8 s (3,416 frames). The first batch's compression, save, and
verification overlapped the second batch's collection by 5.7 s.

### Batched CUDA physics collection

On a CUDA host with Vulkan camera interop, run the same four-mode Moving-Hole
protocol with:

```bash
WANDB_MODE=disabled python main.py --config data_collection/configs/mixed_trajectory_moving_hole_cuda.yaml
```

Point `reference_pilot` at the verified Fixed-Hole pilot and choose an output
directory with enough space for the raw NPZs. `batch_size` sets the number of
parallel physics environments. `writers` sets the number of concurrent NPZ
writers. The internal batch dimension is split into the existing single-scene
raw frame format. Each scene keeps its own transition, proximity guard and
terminal observation. Fixed-Hole mixed collection can use
`physics_device: cuda` and `batch_size` without adding a motion schedule.

For Moving-Hole, seed-specific geometry is checked and the paired CPU reset
pose and robot joints are installed before collection. The cross-backend reset
audit allows absolute error of `1e-6` for float32 pose rounding. Recovery uses
the same GPU batch layout and recorded parent actions. It saves the live replay
frames because GPU PhysX can introduce small replay differences. The episode
receipt records the replay length and the differences at the branch. CPU and
CUDA dynamics can produce different grasp or insertion outcomes.

On 2026-11-16 KST, the RTX 3090 server ran the same 25 reset seeds and four
policies on both backends. Three CPU physics workers completed 100 episodes
and 20,853 frames in 444.4 s (46.9 frames/s). CUDA physics with a batch of
eight and three NPZ writers completed 100 episodes and 20,789 frames in
533.2 s (39.0 frames/s). Both used CUDA RGB-D rendering and the normal NPZ
compression, fsync, and readback checks on `/dev/shm`. GPU time sums the
completed pilot and the corrected recovery run (402.3 s and 130.9 s).
All 100 GPU episodes reached 60 mm and preserved the RGB-D and terminal
frame schema. The largest replay-boundary Peg deviations from the parent
were 6.81 mm and 7.70 degrees. Both backends completed 30 successful insertions.
The server receipts and logs are under
`/workspace/moving_hole_collection/gpu_benchmark/`.

### Pilot storage and additional diagnostics

The pilot occupied 17,505,350,273 bytes. Linear scaling estimates approximately
175 GB for 3,000 main episodes; the preflight requires approximately 215 GB
including a 20% margin and 5 GiB reserve. The current filesystem had only about
18 GB available, and the main collector correctly stopped before creating its
output. The weak reset-Random Peg movement should also be considered before
scaling the same protocol. Results used CPU physics and inference/diagnostics,
with SAPIEN/MoltenVK rendering on this Mac; CUDA was not executed.

[Audit and statistics](../outputs/mixed_trajectory_pilot_audit/audit.json),
[collection manifest](../outputs/mixed_trajectory_pilot/manifest.json),
[frozen protocol](../outputs/mixed_trajectory_pilot/protocol.json), and the
[recovery probe](../outputs/mixed_trajectory_recovery_probe/results.json) are
local ignored artifacts. The original worker and overlap directories remain on
disk; merged episode files share hard links with the worker copies.

A separate five-step-sampled analysis found that reset-Random brought the TCP
within 100 mm of the Peg center in 6/100 episodes; the median episode's minimum
sampled distance was 180.2 mm. The Peg moved more than 10 mm in 4/100 episodes.
This distance is to the Peg center, not a contact test, and sampling can miss
brief approaches. It supports the explanation that many reset-Random episodes
never approach the object closely, without proving why the Peg stayed still.
For the 96 switched Expert→Random cases, 71 moved the Peg over 10 mm after the
switch on sampled frames. The 18 reach-phase switches included only one such
case; all 16 align and all 12 insert switches did. See the local
[sampled proximity and movement results](../outputs/mixed_trajectory_pilot_diagnostics/results.json).
