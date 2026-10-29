"""Independent live kinematic Box motion with actual visual occluders.

GT is used only to construct/audit interventions. Estimated adapters receive RGB-D,
predicted masks and proprioception. No physics step follows a pose intervention.
"""
import argparse,json,inspect
from pathlib import Path
from copy import deepcopy
import numpy as np,torch
from scipy.spatial.transform import Rotation
from graph.persistent_geometry import branches
from graph.persistent_geometry.collect import collect_episode,inventory_sources,sensor_inputs,evaluation_sidecar
from graph.integration_baseline.geometry09b import memory,restore
from graph.integration_baseline import geometry09a as prior
from graph.integration_baseline.io import write_json,save_npz,sha256
from visual.runtime.grounding import GDino
from visual.runtime.tracking import SAM2Tracker
from utils.experiment_logger import ExperimentLogger

ORIGINAL_BRANCHES=branches.collect_branches
DELTAS=(('x-10',[-.01,0,0]),('x+10',[.01,0,0]),('y-10',[0,-.01,0]),('y+10',[0,.01,0]))


def occluded_branches(env,adapters,pre,baseline_sensor,baseline_masks,episode,timestep,folder,physics):
    rows,restored=ORIGINAL_BRANCHES(env,adapters,pre,baseline_sensor,baseline_masks,episode,timestep,folder,physics)
    from mani_skill.utils.structs.pose import Pose
    task=env.unwrapped
    baseline_cameras,robot76,pads,_=sensor_inputs(env,restored,episode,timestep)
    main=next(c for c in baseline_cameras if c.camera=='base_camera')
    old_pose=task.box.pose.raw_pose.clone()
    state=prior.physical_state(env);collision_before=prior.collision_audit(env,'box')
    post={n:memory(a) for n,a in adapters.items()}
    jpeg={n:(Path(a.directory.name)/f'{timestep:06d}.jpg').read_bytes() for n,a in adapters.items()}
    builder=task.scene.create_actor_builder()
    builder.add_box_visual(half_size=(.3,.3,.001),material=(.12,.12,.12,1.))
    blocker=builder.build_kinematic(f'task09e_visual_occluder_t{timestep}')
    rotation=Rotation.from_matrix(main.world_from_camera[:3,:3]).as_quat()
    quat=np.r_[rotation[3],rotation[:3]]
    base_gt=evaluation_sidecar(env,restored)
    box_pixels=np.argwhere(base_gt[main.camera+'/segmentation']==base_gt['actor_ids'][1])
    split_u=float(np.median(box_pixels[:,1])) if len(box_pixels) else float(main.intrinsic[0,2])
    token=physics.token()
    try:
        for occlusion in ('full','partial'):
            # A visual-only slab at camera z=100mm; no collision body/contact force.
            boundary_x=(split_u+.5-main.intrinsic[0,2])*.1/main.intrinsic[0,0]
            camera_xyz=np.array([0. if occlusion=='full' else boundary_x-.3,0.,.1])
            pos=main.world_from_camera[:3,:3]@camera_xyz+main.world_from_camera[:3,3]
            blocker.set_pose(Pose.create_from_pq(p=pos,q=quat))
            hidden_reference=None
            for label,delta in (('baseline',[0,0,0]),*DELTAS):
                raw=old_pose.clone();raw[0,:3]+=raw.new_tensor(delta);task.box.set_pose(Pose.create(raw))
                obs=task.get_obs();condition=f'box_{occlusion}_{label}'
                identity=episode+f':branch_{condition}_t{timestep}'
                cameras,state76,current_pads,_=sensor_inputs(env,obs,identity,timestep)
                np.testing.assert_array_equal(state76,robot76);np.testing.assert_array_equal(current_pads,pads)
                for c,bc in zip(cameras,baseline_cameras):
                    for key in ('intrinsic','extrinsic_cv','cam2world_gl','tcp_pose'):np.testing.assert_array_equal(getattr(c,key),getattr(bc,key))
                masks={};vision={}
                for c in cameras:
                    adapter=adapters[c.camera];restore(adapter,pre[c.camera]);adapter.identity=identity
                    masks[c.camera],vision[c.camera]=adapter.process(c.rgb,identity,timestep)
                sidecar=evaluation_sidecar(env,obs)
                current=next(c for c in cameras if c.camera==main.camera)
                if label=='baseline':hidden_reference=(current.rgb.copy(),current.depth_raw_mm.copy(),masks[main.camera].copy())
                same_rgb=np.array_equal(current.rgb,hidden_reference[0]);same_depth=np.array_equal(current.depth_raw_mm,hidden_reference[1])
                same_mask=np.array_equal(masks[main.camera],hidden_reference[2])
                visible=int((sidecar[main.camera+'/segmentation']==sidecar['actor_ids'][1]).sum())
                after=prior.physical_state(env);collision=prior.collision_audit(env,'box')
                allowed={'box_pose','hole_pose','goal_pose'}
                unchanged={key:np.array_equal(v,after[key]) for key,v in state.items() if key not in allowed}
                new_near=set(collision['near_contacts'])-set(collision_before['near_contacts'])
                valid=bool(all(unchanged.values()) and collision['valid'] and not new_near)
                output=folder/'branches'/f't{timestep:04d}'/condition;output.mkdir(parents=True,exist_ok=False)
                sensor={k:v.copy() for k,v in baseline_sensor.items()};sensor['episode']=np.array(identity)
                for c in cameras:
                    for k in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl'):sensor[c.camera+'/'+k]=getattr(c,k)
                save_npz(output/'input.npz',**sensor);save_npz(output/'masks.npz',**masks);save_npz(output/'gt.npz',**sidecar)
                row=dict(condition=condition,object='box',translation_world_m=delta,timestep=timestep,episode=identity,parent_episode=episode,
                    parent_history_through=timestep-1,physical_valid=valid,robot_state_exact=True,calibration_exact=True,no_physics_step=True,
                    visual_occluder=True,occlusion=occlusion,main_gt_visible_pixels=visible,
                    main_RGB_same_as_occluded_baseline=same_rgb,main_depth_same_as_occluded_baseline=same_depth,main_mask_same_as_occluded_baseline=same_mask,
                    source=str(output.relative_to(folder)),input_sha256=sha256(output/'input.npz'),masks_sha256=sha256(output/'masks.npz'),gt_sha256=sha256(output/'gt.npz'))
                write_json(output/'condition.json',dict(**row,vision=vision,collision=collision,unchanged=unchanged,
                    newly_near_contacts=sorted(new_near),note='live RGB-D; actual visual-only occluder, kinematic actor translation; no fresh contact-force claim'))
                physics.unchanged(token);rows.append(row)
            task.box.set_pose(Pose.create(old_pose))
        # Reappearance observations are the actual visible translated branches
        # captured earlier at exactly this robot state, not newly stepped physics.
    finally:
        task.box.set_pose(Pose.create(old_pose));blocker.set_pose(Pose.create_from_pq(p=[100,100,100]))
        obs=task.get_obs();cameras,robot,_,_=sensor_inputs(env,obs,episode,timestep)
        np.testing.assert_array_equal(robot,robot76)
        for c,b in zip(cameras,baseline_cameras):np.testing.assert_array_equal(c.rgb,b.rgb);np.testing.assert_array_equal(c.depth_raw_mm,b.depth_raw_mm)
        for name,a in adapters.items():
            restore(a,post[name]);a.identity=episode;(Path(a.directory.name)/f'{timestep:06d}.jpg').write_bytes(jpeg[name])
        physics.unchanged(token)
    return rows,obs


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(Path(__file__).with_name('protocol.json').read_text())
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    branches.SPECS=(('baseline_repeat','box',[0,0,0]),*((f'box_{n}','box',d) for n,d in DELTAS))
    branches.collect_branches=occluded_branches
    config=dict(task='09-E',stage='live_motion_controls',split=dict(train=[30002],validation=[],evaluation=[]),horizon=65,
        control_mode='pd_ee_delta_pose',branch_seeds=[30002],branch_timesteps=cfg['motion']['timesteps'],
        no_policy=True,no_physics_step_after_intervention=True,source='new diagnostic expert sequence, same development seed; no blind Final claim')
    with ExperimentLogger(team='integration',config=config,seed=42,model='task09e_live_motion_controls',job_type='evaluate',root=a.output) as log:
        root=log.directory;print('RUN_DIRECTORY='+str(root),flush=True)
        inventory_sources(root)
        detector=GDino(device='cuda');tracker=SAM2Tracker(device='cuda')
        ep=collect_episode(root,30002,'train',detector,tracker,{},config)
        write_json(root/'capture_manifest.json',dict(config=config,episodes=[ep]));log.link_output(root/'capture_manifest.json',kind='metrics')
if __name__=='__main__':main()
