"""Engineering-only live visual occlusion; never opens registered Final seeds.

The slab is a real rendered visual with no collision shape. Its placement uses
only camera calibration, not object pose/masks. We report actual visibility
after capture without selecting or repeating interventions by mask outcome.
"""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .gt_reference import camera_values,simultaneous_sidecar,array
from graph.task10i_stage4a.run_pipeline import CAMERAS,read,write,sha


def active_cameras(stratum):
    return {'main':CAMERAS[:1],'wrist':CAMERAS[1:],'both':CAMERAS}[stratum]


def occlusion_active(t):
    return 8<=t<20


def slab_pose(extrinsic_cv):
    if np.asarray(extrinsic_cv).shape!=(3,4):raise ValueError('3x4 OpenCV extrinsic required')
    world_from_camera=np.linalg.inv(np.vstack((extrinsic_cv,[0,0,0,1])))
    xyz=world_from_camera[:3,:3]@np.array([0.,0.,.1])+world_from_camera[:3,3]
    xyzw=Rotation.from_matrix(world_from_camera[:3,:3]).as_quat()
    return xyz,np.r_[xyzw[3],xyzw[:3]]


class VisualOccluders:
    def __init__(self,env,stratum):
        from mani_skill.utils.structs.pose import Pose
        self.Pose=Pose;self.env=env;self.actors={}
        for camera in active_cameras(stratum):
            builder=env.unwrapped.scene.create_actor_builder()
            builder.add_box_visual(half_size=(.3,.3,.001),material=(.12,.12,.12,1.))
            actor=builder.build_kinematic('task10i4a2_occluder_'+camera)
            # Match the previously executed visual-only control, and verify the
            # installed actor really contains no collision shapes.
            import sapien.physx as physx
            for entity in actor._objs:
                for component in entity.components:
                    if isinstance(component,physx.PhysxRigidBaseComponent):
                        assert len(component.get_collision_shapes())==0
            self.actors[camera]=actor
            actor.set_pose(Pose.create_from_pq(p=[100,100,100]))

    def place(self,observation,active):
        values=camera_values(observation)
        for camera,actor in self.actors.items():
            if active:
                # OpenCV camera z points forward. extrinsic_cv is world→camera.
                xyz,wxyz=slab_pose(values[camera]['extrinsic_cv'])
                actor.set_pose(self.Pose.create_from_pq(p=xyz,q=wxyz))
            else:actor.set_pose(self.Pose.create_from_pq(p=[100,100,100]))


def state(env):
    task=env.unwrapped
    return dict(clock=int(task.elapsed_steps.item()),peg=array(task.peg.pose.raw_pose),
        box=array(task.box.pose.raw_pose),qpos=array(task.agent.robot.get_qpos()),
        qvel=array(task.agent.robot.get_qvel()))


def smoke(root):
    from utils.experiment_logger import ExperimentLogger
    from graph.geometry_reach10d.runtime import environment
    root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(42)
    config=dict(engineering_only=True,final_seed_access=False,final_benchmark=False,
        seeds=[71300001,71300002,71300003],strata=['main','wrist','both'],frames=26,
        occlusion_steps=list(range(8,20)),action='zero pd_ee_delta_pose',
        collision_shapes=0,object_targeting=False,source_sha256=sha(Path(__file__)))
    write(root/'config.json',config)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_stress_engineering',
            seed=42,root=root/'experiments',job_type='evaluation',config=config) as logger:
        receipts=[]
        for stratum,seed in zip(config['strata'],config['seeds']):
            env=environment(Path('/workspace/task10g_f1_abc_20261031/full'))
            rows=[];arrays={}
            try:
                observation,_=env.reset(seed=seed)
                unoccluded=camera_values(observation);before=state(env)
                slabs=VisualOccluders(env,stratum)
                observation=env.unwrapped.get_obs()
                for c in CAMERAS:
                    for key,value in unoccluded[c].items():
                        np.testing.assert_array_equal(value,camera_values(observation)[c][key])
                for key,value in before.items():np.testing.assert_array_equal(value,state(env)[key])
                for t in range(config['frames']):
                    before=state(env);original=camera_values(observation)
                    slabs.place(observation,occlusion_active(t))
                    observation=env.unwrapped.get_obs()
                    for key,value in before.items():np.testing.assert_array_equal(value,state(env)[key])
                    current=camera_values(observation)
                    for c in CAMERAS:
                        for key in ('intrinsic','extrinsic_cv','cam2world_gl'):
                            np.testing.assert_array_equal(original[c][key],current[c][key])
                    sidecar,receipt=simultaneous_sidecar(env,observation)
                    assert receipt['reference_kind']=='EXACT_GT' and receipt['timestep']==t
                    rows.append(dict(timestep=t,occlusion_active=occlusion_active(t),
                        intended_cameras=list(active_cameras(stratum)),reference=receipt,
                        visible_pixels={c:{name:int(np.isin(sidecar['actor_segmentation'][i],ids).sum())
                            for name,ids in receipt['object_ids'].items()} for i,c in enumerate(CAMERAS)},
                        physical_state_unchanged_by_slab=True,calibration_unchanged=True))
                    for c,values in current.items():
                        for key,value in values.items():arrays[f't{t}_{c}_{key}']=value
                    arrays[f't{t}_segmentation']=sidecar['actor_segmentation']
                    if t<config['frames']-1:
                        action=torch.zeros((1,env.action_space.shape[-1]),device=env.unwrapped.device)
                        observation,*_=env.step(action)
                np.savez_compressed(root/(stratum+'.npz'),**arrays)
                path=root/(stratum+'_receipt.json')
                write(path,dict(stratum=stratum,seed=seed,frames=len(rows),rows=rows,
                    artifact_sha256=sha(root/(stratum+'.npz')),no_collision_shapes=True,
                    real_rendered_occlusion=True,zero_step_at_intervention=True,
                    reappearance_new_physical_timestep=True,mask_success_selection=False))
                receipts.append(dict(path=path.name,sha256=sha(path)))
            finally:env.close()
        write(root/'scope.json',dict(status='STRESS_CAPTURE_ENGINEERING_COMPLETE',run_id=logger.run_id,
            receipts=receipts,frames=78,final_data=False,whole_goal_complete=False))
        logger.link_output(root/'scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--wait-component',type=Path)
    a=p.parse_args()
    if a.wait_component:
        deadline=time.monotonic()+24*3600
        while True:
            file=a.wait_component/'pipeline_state.json'
            if file.exists() and read(file)['status'] in ('FAILED','COMPONENT_PREPROCESSING_AND_MATCHED_CROP_COMPLETE'):break
            live=False
            for command in Path('/proc').glob('[0-9]*/cmdline'):
                try:
                    value=command.read_bytes()
                    if b'graph.task10i_stage4a2.component_controls_runner' in value and str(a.wait_component).encode() in value:live=True
                except OSError:pass
            if not live:raise RuntimeError('component runner missing without terminal receipt')
            write(a.root.parent/'wait_state.json',dict(status='WAITING_FOR_LIVE_COMPONENT_RUNNER',pid=os.getpid()))
            if time.monotonic()>deadline:raise TimeoutError('stress serialization deadline')
            time.sleep(10)
    try:smoke(a.root)
    except BaseException as exc:
        write(a.root.parent/'failure.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),partial_preserved=True))
        raise
