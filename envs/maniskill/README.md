# ManiSkill `PegInsertionSide-v1`

## Runtime checked on this host

Python 3.11, ManiSkill 3.0.1, SAPIEN 3.0.3, Gymnasium 1.3.0, PyTorch 2.14.1,
Pinocchio package `pin` 4.1.0. CUDA is unavailable. With `device="cpu"`, the
live `task.backend` reported `sim_backend="physx_cpu"`,
`render_backend="sapien_cpu"`, and `device=cpu` on macOS. This is ManiSkill +
SAPIEN + PhysX, not MuJoCo. The registered task has a 100-step time limit.

Linux Reach BC validation also executed this shared factory with Torch 2.8.0+cpu
on Vast.ai 55512455. CPU simulation now always selects `sapien_cpu` host image
readback, including when Torch can see CUDA; this avoids requesting a Vulkan
`cuda:0` renderer when only a CPU Vulkan device such as lavapipe is available.
Vulkan still needs a supported physical device. Shared CPU/RGB tests passed
on the CPU-only host (8 tests including GT nodes). The CUDA-visible Linux path
still needs verification on the target server.
The Graph subtask source wrapper separately preserves its recorded environment
configuration; SAPIEN 3.0.3 source-render and host-readback regression suites must
run in separate processes because render-device selection is process-global.

```python
from envs.maniskill import make_peg_insertion_env, get_gt_state
import torch

env = make_peg_insertion_env(seed=0, device="cpu", obs_mode="state")
try:
    obs, info = env.reset(seed=0)
    state = get_gt_state(env)
    action = torch.zeros((1, env.action_space.shape[-1]), device=env.unwrapped.device)
    obs, reward, terminated, truncated, info = env.step(action)
finally:
    env.close()
```

The factory accepts `seed`, `device`, `num_envs`, `obs_mode`, `control_mode`,
`render_mode`, and `sim_backend`. `seed` resets one episode during creation;
call `reset(seed=...)` again to start a chosen episode. `control_mode=None`
uses ManiSkill's default. `device="cpu"` maps to `physx_cpu`; `"cuda"` maps to
`physx_cuda`. Explicit `sim_backend` must agree with `device`. On this installed
version, CPU allows `num_envs=1` only. CUDA/vectorized execution has not been
run on this host.

## Observations and stepping

The factory defaults to `obs_mode="state"`. A reset returned a float32 Torch
tensor of shape `[1,43]`; `observation_space` was
`Box(-inf, inf, (1, 43), float32)`. `state_dict` returned:

```text
agent.qpos [1,9]      agent.qvel [1,9]
extra.tcp_pose [1,7]  extra.peg_pose [1,7]
extra.peg_half_size [1,3]
extra.box_hole_pose [1,7]  extra.box_hole_radius [1]
```

Every observation tensor has a leading environment dimension, even with one
CPU environment. A zero action step returned float32 reward `[1]`, bool
`terminated` `[1]`, bool `truncated` `[1]`; `info` contained `elapsed_steps`,
`success`, and `peg_head_pos_at_hole`. Reset `info` additionally contained
`reconfigure`. `success` and step flags were false in the smoke run. Other
observation modes are passed through to ManiSkill.

## Control modes and action convention

The task uses `panda_wristcam`. Its runtime controller exposed:
`pd_joint_delta_pos`, `pd_joint_pos`, `pd_ee_delta_pos`, `pd_ee_delta_pose`,
`pd_ee_pose`, `pd_joint_target_delta_pos`, `pd_ee_target_delta_pos`,
`pd_ee_target_delta_pose`, `pd_joint_vel`, `pd_joint_pos_vel`, and
`pd_joint_delta_pos_vel`.

| Mode checked | Action space | Convention from the installed controller config/code |
| --- | --- | --- |
| Default `pd_joint_delta_pos` | `Box(-1,1,(8,),float32)` | 7 normalized arm joint deltas followed by one normalized gripper target. Arm `±1` scales to `±0.1` rad. |
| `pd_ee_delta_pos` | `Box(-1,1,(4,),float32)` | Root-frame TCP translation `x,y,z` (normalized `±1` to `±0.1` m), then gripper. |
| `pd_ee_delta_pose` | `Box(-1,1,(7,),float32)` | Root-frame TCP translation `x,y,z`, then XYZ rotation components, then gripper. Translation bounds are `±0.1` m; rotation bounds are `±0.1` rad. The controller clips rotation vector norm to 1 before scaling and uses root-aligned body rotation. |
| `pd_ee_pose` | `Box(([-2]*3+[-2π]*3+[-1]),([2]*3+[2π]*3+[1]),(7,),float32)` | Absolute TCP pose in robot root frame plus normalized gripper. |

The gripper is a single normalized command for a mimic controller: `-1` maps
to a target finger joint position of `-0.01` m (closing force allowance), and
`+1` maps to `0.04` m (open). Both finger joints follow that one command.
The `pd_ee_delta_pose` implementation in this installed version multiplies the
normalized XYZ rotation components by its configured `rot_lower=-0.1`; check
this sign before interpreting learned action values as physical rotations.
The repository does not impose any policy action representation. For a live
check, run `python examples/check_env.py --control-mode pd_ee_delta_pose`.

## Simulator GT

`get_gt_state(env)` reads the live task after reset/step. All fields are
detached Torch tensors on the simulator device. `B` is number of environments.
Poses are world-frame `[x,y,z,qw,qx,qy,qz]`; metres for translation. Values
reported below were checked on a one-environment CPU reset.

| Field | Shape/dtype | Meaning and frame |
| --- | --- | --- |
| `qpos`, `qvel` | `[B,9]` float32 | Robot's seven arm joints, then two finger joints; joint positions (rad/m) and velocities (rad/s or m/s) in articulation coordinates. |
| `ee_pose` | `[B,7]` float32 | `panda_hand_tcp` world pose. |
| `left_finger_pose`, `right_finger_pose` | `[B,7]` float32 each | Finger link world poses. |
| `left_fingertip_pos`, `right_fingertip_pos` | `[B,3]` float32 each | World positions of rubber-tip collision box centers. These use fixed local offsets from installed `panda_v3.urdf`: `[0,+0.00758,0.04525]` and `[0,-0.00758,0.04525]` m. |
| `peg_pose`, `box_pose` | `[B,7]` float32 each | Peg and box actor world poses. |
| `hole_pose` | `[B,7]` float32 | Task's `box_hole_pose`: hole frame in world coordinates; includes the randomized hole offset in the box. |
| `gripper_qpos` | `[B,2]` float32 | Final two active joint positions from `qpos`, in metres. |
| `peg_half_sizes` | `[B,3]` float32 | Peg local half extents in metres. |
| `hole_radius` | `[B]` float32 | Task's hole clearance radius in metres. |
| `success` | `[B]` bool | Task `evaluate()["success"]`. |

The fingertip points are the only geometry derived by this accessor. The task
uses `panda_wristcam` with `panda_v3.urdf`; its finger pads are collision boxes,
not separate SAPIEN links. Contact events are not included in `GTState`; teams
can inspect `env.unwrapped.scene` if needed. No relative pose, graph, feature
vector, or sequence is constructed here.

## Cameras and images

Runtime `obs_mode="rgb"` exposed `base_camera` and robot-mounted
`hand_camera`. Each produced `sensor_data[name]["rgb"]` as
`[1,128,128,3]` `torch.uint8` in 0–255. In a reset, the observed pixel ranges
were 0–231 for base and 35–255 for hand. `sensor_param[name]` contained
`intrinsic_cv` `[1,3,3]`, `extrinsic_cv` `[1,3,4]`, and `cam2world_gl`
`[1,4,4]`, all float32. The CV extrinsic is a world-to-camera transform;
`cam2world_gl` is camera-to-world in the OpenGL frame. The live task also has
`render_camera`; with `render_mode="rgb_array"`, `env.render()` returned
`[1,512,512,3]` `torch.uint8`.

`obs_mode="rgb+depth+segmentation"` returned those three channels for both
sensors. RGB was `[1,128,128,3]` uint8; depth and segmentation were
`[1,128,128,1]` int16. ManiSkill documents depth units as millimetres, but a
base-camera pixel reached `-32768` in this run; inspect invalid/overflow values
before using raw depth numerically. Segmentation IDs were present (0–19 in the
base-camera reset). `python examples/render_episode.py` saves a short PNG frame
sequence plus one `render_camera.png` without a video encoder.

## Object geometry

The live task exposed `task.peg` and `task.box` as merged ManiSkill actors,
each containing one SAPIEN entity in this CPU run. Their actor poses are world
poses. The peg's entity had one box collision mesh (8 vertices, 12 faces) and
two box visual shapes; the box entity had four collision box meshes and four
visual box shapes. `hole_pose` is a task pose, not a separate mesh actor.

The installed `Actor.get_collision_meshes()` asserts on merged actors. Read
their SAPIEN entity components instead; these utility functions return meshes
in entity-local coordinates. Use the actor world pose when world vertices are
needed:

```python
import sapien.physx as physx
from mani_skill.utils.geometry.trimesh_utils import (
    get_actor_visual_meshes, get_component_meshes,
)

task = env.unwrapped
entity = task.peg._objs[0]  # one entity for the one-env CPU case
collision = [mesh for component in entity.components
             if isinstance(component, physx.PhysxRigidBaseComponent)
             for mesh in get_component_meshes(component)]
visual = get_actor_visual_meshes(entity)
peg_world_pose = task.peg.pose
```

For multiple environments, index the matching entity and actor pose. This
documents access; sampling a surface or defining graph nodes remains Graph
team work.

## Known limits

- CPU `num_envs>1` is rejected by this installed ManiSkill version. CUDA was
  unavailable and could not be checked here.
- On macOS, the simulator renderer still needs a Vulkan ICD for state-only
  tasks. The factory detects standard Homebrew MoltenVK locations. If your
  installation differs, set `VK_ICD_FILENAMES` explicitly.
- CPU end-effector controllers need Pinocchio (`pin` in requirements).
- This file records observed shapes/backend on this host and code-inspected
  controller conventions. Other package versions and hosts should be checked
  with the examples before reuse.

Reference: [ManiSkill observation modes](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html),
[controllers](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html),
and [camera sensors](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/sensors.html).
