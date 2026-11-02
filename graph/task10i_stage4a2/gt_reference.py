"""Evaluation-only synchronous Actor capture; never imported by inference.

Do not label a replayed frame exact just because depth agrees. This API captures
segmentation in the current render state and tests both-camera RGB/depth and
all recorded calibration fields around the readback without stepping physics.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from graph.task10i_stage4a.run_pipeline import CAMERAS,read,write,sha


def array(x):
    return x.detach().cpu().numpy().copy()


def camera_values(observation):
    result={}
    for camera in CAMERAS:
        sd=observation['sensor_data'][camera];sp=observation['sensor_param'][camera]
        result[camera]=dict(rgb=array(sd['rgb'])[0],depth_raw_mm=array(sd['depth'])[0,...,0],
            intrinsic=array(sp['intrinsic_cv'])[0],extrinsic_cv=array(sp['extrinsic_cv'])[0],
            cam2world_gl=array(sp['cam2world_gl'])[0])
    return result


def simultaneous_sidecar(env,observation):
    task=env.unwrapped;clock=int(task.elapsed_steps.item())
    before=camera_values(observation)
    state_before={name:array(getattr(task,name).pose.raw_pose) for name in ('peg','box')}
    qpos=array(task.agent.robot.get_qpos());qvel=array(task.agent.robot.get_qvel())
    segmentation={c:array(task._sensors[c].get_obs(rgb=False,depth=False,position=False,
                                               segmentation=True)['segmentation'])[0,...,0] for c in CAMERAS}
    after=camera_values(task.get_obs())
    checks={c:{k:bool(np.array_equal(v,after[c][k])) for k,v in before[c].items()} for c in CAMERAS}
    assert clock==int(task.elapsed_steps.item())
    for name,value in state_before.items():np.testing.assert_array_equal(value,array(getattr(task,name).pose.raw_pose))
    np.testing.assert_array_equal(qpos,array(task.agent.robot.get_qpos()))
    np.testing.assert_array_equal(qvel,array(task.agent.robot.get_qvel()))
    mapping={int(k):v.name for k,v in task.segmentation_id_map.items()}
    objects={name:[k for k,v in mapping.items() if v.startswith(name)] for name in ('peg','box')}
    robot_names={link.name for link in task.agent.robot.get_links()}
    robot=[k for k,v in mapping.items() if v in robot_names]
    assert all(objects.values()) and robot
    assert not set(objects['peg'])&set(objects['box'])
    assert not (set(objects['peg'])|set(objects['box']))&set(robot)
    for c in CAMERAS:
        assert segmentation[c].shape==before[c]['depth_raw_mm'].shape==before[c]['rgb'].shape[:2]
    exact=all(v for cam in checks.values() for v in cam.values())
    return dict(actor_segmentation=np.stack([segmentation[c] for c in CAMERAS]),
                exact_frame=np.bool_(exact)),dict(reference_kind='EXACT_GT' if exact else 'GT_UNSUPPORTED',
        timestep=clock,comparisons=checks,object_ids=objects,robot_ids=robot,actor_ids=mapping,
        clock_equal=True,actor_and_robot_state_equal=True,render_sensor_config_changed=False,
        capture_method='segmentation-only read from current sensors; complete RGB-D/calibration equality before/after')


def smoke(root):
    from utils.experiment_logger import ExperimentLogger
    from graph.geometry_reach10d.runtime import environment
    root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(42)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_synchronous_GT_engineering',seed=71300000,
         root=root/'experiments',job_type='evaluation',config=dict(engineering_only=True,final_benchmark=False)) as logger:
        env=environment(Path('/workspace/task10g_f1_abc_20261031/full'))
        rows=[];arrays={}
        try:
            obs,_=env.reset(seed=71300000)
            for t in range(6):
                sidecar,receipt=simultaneous_sidecar(env,obs)
                assert receipt['reference_kind']=='EXACT_GT',receipt
                assert receipt['timestep']==t
                rows.append(receipt)
                arrays[f't{t}_segmentation']=sidecar['actor_segmentation']
                for c,values in camera_values(obs).items():
                    for k,v in values.items():arrays[f't{t}_{c}_{k}']=v
                if t<5:
                    action=torch.zeros((1,env.action_space.shape[-1]),device=env.unwrapped.device)
                    obs,*_=env.step(action)
            np.savez_compressed(root/'synchronous_capture.npz',**arrays)
            write(root/'receipt.json',dict(status='SYNCHRONOUS_GT_ENGINEERING_PASS',run_id=logger.run_id,
                seed=71300000,frames=6,cameras=2,rows=rows,final_data=False,student_inference=False,
                source_sha256=sha(Path(__file__)),capture_sha256=sha(root/'synchronous_capture.npz')))
            logger.link_output(root/'receipt.json',kind='metrics')
        finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);smoke(p.parse_args().root)
