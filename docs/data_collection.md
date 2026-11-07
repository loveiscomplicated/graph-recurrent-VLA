# Episode collection

Run commands from the repository root after installing `requirements.txt`.

For the explicitly requested 900-episode expert dataset on a Linux CUDA/Vulkan
host, also install `requirements-tracking.txt`, configure HF write access, and run:

```bash
WANDB_MODE=disabled python -m data_collection.collect_dataset \
  --config data_collection/configs/visible_peg.yaml
```

That config requests native Main/Wrist 512×512 RGB-D, CPU physics, CUDA rendering,
900 successful scripted-expert trajectories, 250-step horizons, raw GT at every
frame, four independent workers and 30-episode archives. Initial Main visibility
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

## Visibility

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
