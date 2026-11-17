"""Metres/radians; world poses [x,y,z,qw,qx,qy,qz], hole +X insertion."""
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation


def rotation(pose):
    q=np.asarray(pose)[3:]
    if not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:
        raise ValueError('invalid wxyz quaternion')
    return Rotation.from_quat(q[[1,2,3,0]]).as_matrix()


def relative_pose(parent, child):
    r=rotation(parent)
    return r.T@(np.asarray(child)[:3]-np.asarray(parent)[:3]), r.T@rotation(child)


def rotation_error(r):
    return float(np.arccos(np.clip((np.trace(r)-1)/2,-1,1)))


def compose(parent, local):
    r=rotation(parent)
    q=Rotation.from_matrix(r@rotation(local)).as_quat()
    return np.r_[np.asarray(parent)[:3]+r@np.asarray(local)[:3],q[[3,0,1,2]]]


def compute_metrics(state, config):
    p,h,e=state['peg_pose'],state['hole_pose'],state['ee_pose']
    half=np.asarray(state['peg_half_sizes'])
    rel,rr=relative_pose(h,p)
    head=rel+rr@np.array([half[0],0,0])
    reach=compose(p,state['reach_local'])
    grasp=compose(p,state['grasp_local'])
    reach_pos,reach_rot=relative_pose(e,reach)
    ep,er=relative_pose(e,p)
    depth=float(head[0]+half[0]) # entry plane at hole-local x=-box half-length
    target=np.array([-2*half[0]+config['waiting_depth'],0,0])
    corners=np.asarray(list(product((-1,1),repeat=3)))*half
    hole_corners=rel+corners@rr.T
    surface_clearance=float(state['hole_radius']-np.abs(hole_corners[:,1:]).max())
    initial=state.get('initial_peg_pose',p)
    m=dict(position_error=float(np.linalg.norm(reach_pos)),rotation_error=rotation_error(reach_rot),
        gripper_open=bool(np.min(state['gripper_qpos'])>=config['open_finger']),
        gripper_width=float(np.sum(state['gripper_qpos'])),
        peg_motion=float(np.linalg.norm(np.asarray(p)[:3]-np.asarray(initial)[:3])),
        on_table=bool(abs(p[2]-half[2])<=config['on_table_tolerance']),
        grasp=bool(state['grasp']),contact_valid=bool(state['contact_valid']),
        transport_error=float(np.linalg.norm(rel-target)),relative_rotation_error=rotation_error(rr),
        head_lateral=float(np.linalg.norm(head[1:])),center_lateral=float(np.linalg.norm(rel[1:])),
        insertion_depth=depth,waiting_error=abs(depth-config['waiting_depth']),
        surface_clearance=surface_clearance,official_success=bool(state['success']),
        workspace_valid=bool(np.linalg.norm(np.asarray(p)[:2])<=config['workspace_radius'] and
            config['workspace_z_min']<=p[2]<=config['workspace_z_max']),
        grasp_relative_position=ep.tolist(),grasp_relative_rotation=er.tolist(),
        head_in_hole=head.tolist(),center_in_hole=rel.tolist(),
        reach_goal_pose=reach.tolist(),grasp_goal_pose=grasp.tolist(),
        preinsert_center_hole=target.tolist(),
        finger_force_norms=np.linalg.norm(state['finger_forces'],axis=-1).tolist(),
        peg_box_force_norm=float(np.linalg.norm(state['peg_box_force'])))
    return m
