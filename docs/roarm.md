# RoArm M2-S in ManiSkill

From the repository root, use Python 3.11 with `requirements.txt` installed.
On macOS, install Homebrew `molten-vk` if SAPIEN cannot create a Vulkan device.

```bash
python -m examples.check_roarm --frames
python -m unittest tests.test_roarm tests.test_roarm_observation \
    tests.test_maniskill_env tests.test_observation_device -v
```

The smoke command writes a report and frames under `outputs/roarm_check/`.
It checks asset hashes, CPU physics, joint motion, FK, position reaching, reset,
and contact probes. The MPS regression runs on hosts where PyTorch reports MPS
available.

## Create the bench

```python
from envs.maniskill.roarm import make_roarm_env

env = make_roarm_env(obs_mode="rgb+depth", observation_device="mps")
try:
    obs, info = env.reset(seed=0)
    hold = obs["agent"]["qpos"].to("cpu")
    obs, reward, terminated, truncated, info = env.step(hold)
finally:
    env.close()
```

Importing the module registers agent `roarm_m2_s` and task
`RoArmM2Bench-v1`. The bench has a fixed mount, solid tabletop, an RGB-D
`base_camera`, optional static box obstacles, and a 2,000-step limit. The
factory defaults to one CPU environment, `state_dict` observations, and
absolute joint position control. Omit `observation_device` for native
observations. On macOS, PhysX runs on CPU and SAPIEN returns camera images on
CPU. The optional output device transfers only raw reset and step
observations. Pass CPU actions for CPU physics.

| Raw observation | Default shape and dtype |
| --- | --- |
| `sensor_data.base_camera.rgb` | `[B,128,128,3]` uint8 |
| `sensor_data.base_camera.depth` | `[B,128,128,1]` int16, raw millimetres |
| `agent.qpos`, `agent.qvel` | `[B,4]` float32 |
| `extra.tcp_pose` | `[B,7]` float32, world-frame XYZ followed by quaternion WXYZ |

Camera calibration is under `sensor_param.base_camera`. The four action
components follow the URDF joint order (base, shoulder, elbow, jaw).
`pd_joint_pos` uses absolute radians. `pd_joint_delta_pos` maps
`[-1,1]` to `±0.05` rad relative to the current joint position. The smoke
example uses SAPIEN Pinocchio FK and SciPy position-only IK in the robot base
frame. This bench covers joint actuation and reaching. Grasping and
real-hardware control are outside its scope.

## Assets and backend diagnostics

The simulator and smoke tests are based on the personal repository's
`5a9ae5b`. The original ROS files and five STL meshes are from the
[Waveshare RoArm workspace](https://github.com/waveshareteam/roarm_ws_em0/tree/f06b2ed6c1ddc5b7690975fdbba84aab626a2229).
The packaged source declaration and `source.json` record provenance and
asset hashes. Run `python tools/prepare_roarm_assets.py` to regenerate
the adapted URDF and SRDF offline.

For backend diagnostics, run the following in a fresh process. The checker
defaults to Linux CUDA with eight environments when backend flags are omitted.
On macOS, the CPU backend checker and MPS observation regression passed.
CUDA and physical hardware were outside this verification.

```bash
python -m examples.check_roarm_backends --device cpu --render-backend cpu \
    --num-envs 1 --output outputs/roarm_cpu
```

## Collision-aware reaching

Use the guarded controller and executor for a single bench with the fixed
mount, table, ground, and known static box obstacles. The example exercises
five reaching goals, a physical obstacle detour, unsafe goals, and execution
faults. To verify state observations on MPS while the bench runs on CPU,
run the second command.

```bash
python -m examples.check_roarm_motion --output outputs/roarm_motion_cpu
python -m examples.check_roarm_motion --observation-device mps \
    --output outputs/roarm_motion_mps
python -m unittest tests.test_roarm_motion tests.test_roarm_execution -v
```

```python
from envs.maniskill.roarm import make_roarm_env
from envs.maniskill.roarm.execution import RoArmGuardedExecutor
from roarm.collision import RoArmCollisionModel
from roarm.motion import RoArmMotionPlanner

# Obstacle center and target XYZ are in the robot base frame.
obstacles = [dict(center=[0.31, 0, 0.25], half_size=[0.025, 0.035, 0.025])]
env = make_roarm_env(control_mode="pd_joint_pos_guarded", obstacles=obstacles)
try:
    obs, _ = env.reset(seed=0)
    planner = RoArmMotionPlanner(RoArmCollisionModel(obstacles=obstacles))
    executor = RoArmGuardedExecutor(env, planner)
    start = obs["agent"]["qpos"][0].cpu().numpy()
    path = planner.plan_position(start, [0.30, -0.16, 0.23])
    stats = executor.execute(path)
finally:
    env.close()
```

The collision model checks URDF link convex hulls with the audited SRDF
exclusions. It models the tabletop as a plane at base-frame z=0 and checks
whole joint intervals against the scene geometry. The planner tries a direct
route, a route via home, and a bounded seeded RRT. The default bounds allow
5 mm geometric clearance and a ±0.01 rad tracking margin. A planning failure
means that the planner could not accept a route within these bounds and its
search budget.

The executor verifies scene geometry, tracking error, speed, clearance, and
sampled contact forces. Quintic time scaling limits commanded joint
speed to 0.25 rad/s and acceleration to 0.5 rad/s² with the default settings.
Measured limits are 0.4 rad/s speed, 0.01 rad tracking error, and 0.1 N sampled
contact force. Upon an execution fault it latches a `MotionRejected` and
sets a hold target from measured joints, with a last-valid fallback. Reset
the bench and create a new executor before resuming. These checks apply at
control steps on the known static simulated scene. Moving obstacles,
payloads, batched execution, and physical hardware are outside the guard's
scope.
