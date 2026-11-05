# GPU observations

The environment separates physics, camera rendering and observation output:

| Option | Role |
| --- | --- |
| `device="cpu"` | CPU PhysX simulation |
| `render_backend="cuda"` | Native CUDA camera RGB-D on a supported host |
| `observation_device="cuda"` | Transfer remaining raw observation arrays to CUDA |

The observation wrapper alone can upload CPU images. To avoid camera image
GPU-to-CPU-to-GPU readback, select both the CUDA renderer and CUDA output.
Already-resident tensors are reused. The wrapper preserves shape and dtype and
leaves actions, rewards, flags, info and direct GT access unchanged.

## Environment creation

Run on a Linux host with working CUDA and Vulkan camera interop:

```python
from envs.maniskill import make_peg_insertion_env

env = make_peg_insertion_env(
    device="cpu",
    render_backend="cuda",
    observation_device="cuda",
    obs_mode="rgb+depth",
    sensor_configs={name: dict(width=512, height=512)
                    for name in ("base_camera", "hand_camera")},
)
try:
    observation, info = env.reset(seed=0)
finally:
    env.close()
```

RGB and depth retain their original batch dimensions. RGB is uint8; depth is
signed int16 millimetres, including invalid/overflow values. Calibration and
robot-state tensors may originate on CPU and are transferred at the output
boundary. Physics remains CPU in this example; CUDA physics is a separate option.

## Verify native camera output

```bash
python examples/check_gpu_observations.py
python -m unittest tests.test_observation_device_native_gpu -v
```

The example checks that both Main and Wrist RGB-D are already on CUDA before
the output wrapper, during reset and step. The native GPU test also checks the
CPU physics/backend and auxiliary observation transfer. Run these checks in a
separate process from CPU rendering because SAPIEN's render-device selection
is process-global.

For CPU output-wrapper contracts and simulator smoke checks:

```bash
python -m unittest tests.test_observation_device tests.test_maniskill_env -v
```

CUDA rendering, output transfer and physics were not executed on the current
Mac. CUDA tests skip when CUDA is unavailable. See the recorded
[verification results](data_collection_verification.md) for executed checks.

## Use during collection

```bash
python -m data_collection.collect_episodes --output outputs/gpu_initial_frames \
  --num-episodes 10 --render-backend cuda --observation-device cuda
```

The initial visibility check and disk storage explicitly materialize arrays on
CPU. This command does not imply an entirely GPU-resident collection pipeline.
See [data collection](data_collection.md) for visibility rules and storage format,
and the [simulator reference](../envs/maniskill/README.md) for raw observations.
