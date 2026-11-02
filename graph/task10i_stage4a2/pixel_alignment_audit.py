"""Evaluation-only native Position texture versus depth backprojection convention."""
import argparse
import importlib.metadata
import inspect
from pathlib import Path
import subprocess
import numpy as np
import torch
from .gt_reference import array,camera_values
from .stress_capture import state
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS


def stats(x):
    x=np.asarray(x)
    return dict(samples=len(x),mean=float(x.mean()),median=float(np.median(x)),
        p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95)),max=float(x.max())) if len(x) else dict(samples=0)


def run(root):
    from utils.experiment_logger import ExperimentLogger
    from graph.geometry_reach10d.runtime import environment
    root.mkdir(parents=True,exist_ok=False)
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    assert int(free.splitlines()[0])>=6144,'require6GiB free before creating diagnostic renderer'
    torch.set_num_threads(1);torch.manual_seed(42)
    config=dict(seed=71300004,frames=3,engineering_only=True,final_data=False,model_forwards=0,
        source_sha256=sha(Path(__file__)),packages={n:importlib.metadata.version(n) for n in ('mani_skill','sapien','torch','numpy')})
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_pixel_alignment_engineering',seed=71300004,
            root=root/'experiments',job_type='evaluation',config=config) as logger:
        env=environment(Path('/workspace/task10g_f1_abc_20261031/full'));rows=[];saved={};sources={}
        try:
            observation,_=env.reset(seed=71300004);task=env.unwrapped
            for t in range(3):
                before=camera_values(observation);physical=state(env)
                ids={int(k):v.name for k,v in task.segmentation_id_map.items()}
                object_ids={n:[k for k,v in ids.items() if v.startswith(n)] for n in ('peg','box')}
                robot_names={link.name for link in task.agent.robot.get_links()};object_ids['robot']=[k for k,v in ids.items() if v in robot_names]
                for camera in CAMERAS:
                    sensor=task._sensors[camera]
                    extra=sensor.get_obs(rgb=False,depth=True,position=True,segmentation=True)
                    position=array(extra['position'])[0,...,:3];seg=array(extra['segmentation'])[0,...,0]
                    depth=before[camera]['depth_raw_mm'];valid=depth>0
                    ratio=float(np.median(np.abs(position[...,2][valid])/depth[valid]))
                    scale=.001 if .9<ratio<1.1 else 1. if .0009<ratio<.0011 else None
                    assert scale is not None,('unknown Position units',str(position.dtype),ratio)
                    sign=float(np.median(np.sign(position[...,2][valid])))
                    convention='OpenGL' if sign<0 else 'OpenCV'
                    cv=position.astype(float)*scale*np.array([1,-1,-1] if sign<0 else [1,1,1])
                    np.testing.assert_allclose(cv[...,2][valid],depth[valid]*.001,atol=.002,rtol=0)
                    k=before[camera]['intrinsic'];yy,xx=np.indices(depth.shape);d=depth.astype(float)*.001
                    projected_u=k[0,0]*cv[...,0][valid]/cv[...,2][valid]+k[0,2]
                    projected_v=k[1,1]*cv[...,1][valid]/cv[...,2][valid]+k[1,2]
                    row=dict(camera=camera,timestep=t,position_dtype=str(position.dtype),position_shape=list(position.shape),
                        position_scale_to_m=scale,position_convention=convention,
                        projected_minus_integer_u=stats(projected_u-xx[valid]),
                        projected_minus_integer_v=stats(projected_v-yy[valid]),
                        z_quantization_error_m=stats(np.abs(cv[...,2][valid]-d[valid])))
                    for offset in (0.,.5):
                        cloud=np.stack([(xx+offset-k[0,2])*d/k[0,0],(yy+offset-k[1,2])*d/k[1,1],d],-1)
                        error=np.linalg.norm(cloud-cv,axis=-1)
                        row['backprojection_offset_'+str(offset)]={name:stats(error[valid&np.isin(seg,actor_ids)]) for name,actor_ids in object_ids.items()}
                    rows.append(row)
                    for name,value in before[camera].items():saved[f't{t}_{camera}_{name}']=value
                    saved[f't{t}_{camera}_native_position']=position;saved[f't{t}_{camera}_segmentation']=seg
                    for name,transform in sensor.config.shader_config.texture_transforms.items():
                        try:path=inspect.getsourcefile(transform);text=inspect.getsource(transform)
                        except (TypeError,OSError):path=None;text=None
                        sources[name]=dict(callable=getattr(transform,'__qualname__',type(transform).__qualname__),source_file=path,
                            source_sha256=sha(path) if path else None,source=text)
                after=camera_values(task.get_obs())
                for camera in CAMERAS:
                    for name,value in before[camera].items():np.testing.assert_array_equal(value,after[camera][name])
                for name,value in physical.items():np.testing.assert_array_equal(value,state(env)[name])
                if t<2:
                    observation,*_=env.step(torch.zeros((1,env.action_space.shape[-1]),device=task.device))
            np.savez_compressed(root/'native_position_alignment.npz',**saved)
            write(root/'installed_texture_transforms.json',sources)
            write(root/'alignment_metrics.json',rows)
            write(root/'scope.json',dict(status='NATIVE_PIXEL_ALIGNMENT_AUDIT_COMPLETE',run_id=logger.run_id,
                frames=3,cameras=2,RGB_depth_calibration_and_physical_state_unchanged_by_readback=True,
                model_forwards=0,final_data=False,source=config,artifact_sha256=sha(root/'native_position_alignment.npz'),
                interpretation='compare both offsets to native sensor texture; no geometry/production changes applied',whole_goal_complete=False))
            logger.link_output(root/'scope.json',kind='metrics')
        finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    try:run(a.root)
    except BaseException as exc:
        if a.root.exists():write(a.root/'failure.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),partial_preserved=True))
        raise
