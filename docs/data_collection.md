# Episode collection

Run commands from the repository root after installing `requirements.txt`.

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
No exact simulator-state replay, automatic split assignment, successful-task
guarantee, video encoding or remote publication is provided by this collector.

## Checks

```bash
python -m unittest tests.test_visibility tests.test_data_collection \
  tests.test_observation_device tests.test_maniskill_env -v
python -m unittest tests.test_observation_device_native_gpu -v
```

Recorded results and execution limits: [collection verification](data_collection_verification.md).
