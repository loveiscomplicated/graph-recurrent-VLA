"""Live GT adapter with per-episode randomized dimensions."""
import torch

from .geometry import GeometryState, rotation, transform
from .surface_sampling import peg_surface, hole_surface, task_box_bounds


class GTAdapter:
    def __init__(self, peg_points=64, hole_points=64, sampling_seed=17):
        self.peg_count, self.hole_count, self.seed = peg_points, hole_points, sampling_seed
        self._cache = None

    def __call__(self, env):
        from envs.maniskill import get_gt_state
        task = env.unwrapped
        s = get_gt_state(env)
        half = s.peg_half_sizes
        box_half = half[:, :1].expand(-1, 3).clone()
        offset = task.box_hole_offsets.p.detach().clone()
        radius = s.hole_radius[:, None]
        key = torch.cat((half, offset, radius), -1)
        if self._cache is None or not torch.equal(key, self._cache[0]):
            pl, pn = peg_surface(half, self.peg_count, self.seed)
            hl, hn = hole_surface(box_half, radius, offset, self.hole_count, self.seed)
            self._cache = (key.clone(), pl, pn, hl, hn)
        _, pl, pn, hl, hn = self._cache
        return GeometryState(
            s.ee_pose, s.left_fingertip_pos, s.right_fingertip_pos,
            s.gripper_qpos, s.qpos, task.agent.robot.pose.raw_pose.detach().clone(),
            s.peg_pose, half, pl.clone(), transform(pl, s.peg_pose),
            pn @ rotation(s.peg_pose[:, 3:]).transpose(-1, -2),
            s.box_pose, box_half, s.hole_pose, radius, offset, hl.clone(),
            transform(hl, s.box_pose), hn @ rotation(s.box_pose[:, 3:]).transpose(-1, -2))


def validate_live_collision(env, geometry):
    """Fail if installed task or finger collision primitives differ from adapter."""
    import numpy as np
    import sapien.physx as physx
    from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
    task = env.unwrapped
    expected = task_box_bounds(geometry.box_half_size, geometry.hole_radius, geometry.hole_offset)
    errors = []
    for i, entity in enumerate(task.box._objs):
        meshes = [m for c in entity.components if isinstance(c, physx.PhysxRigidBaseComponent)
                  for m in get_component_meshes(c)]
        actual = np.stack([m.bounds for m in meshes])
        np.testing.assert_allclose(actual, expected[i].cpu(), atol=2e-6, rtol=0)
        errors.append(float(np.max(np.abs(actual-expected[i].cpu().numpy()))))
    for i, entity in enumerate(task.peg._objs):
        meshes = [m for c in entity.components if isinstance(c, physx.PhysxRigidBaseComponent)
                  for m in get_component_meshes(c)]
        assert len(meshes) == 1
        np.testing.assert_allclose(meshes[0].bounds, torch.stack((-geometry.peg_half_size[i], geometry.peg_half_size[i])).cpu(), atol=2e-6, rtol=0)
    # Rubber-pad centre is not a contact-surface point or distal link origin.
    pad_errors = []
    for link, tip, offset in ((task.agent.finger1_link, geometry.left_fingertip, [0, .00758, .04525]),
                              (task.agent.finger2_link, geometry.right_fingertip, [0, -.00758, .04525])):
        for i, component in enumerate(link._objs):
            meshes = get_component_meshes(component)
            centres = np.stack([m.bounds.mean(0) for m in meshes])
            distance = np.linalg.norm(centres-np.asarray(offset), axis=1)
            assert distance.min() < 2e-6, (link.name, centres)
            local = torch.tensor(centres[distance.argmin()], dtype=torch.float32).reshape(1, 1, 3)
            world = transform(local, link.pose.raw_pose[i:i+1])[0, 0]
            pad_errors.append(float((world-tip[i]).abs().max()))
    return {"box_bounds_max_error_m": max(errors), "pad_center_max_error_m": max(pad_errors)}
