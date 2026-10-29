"""Separate GT-mask/motion controls on already-rendered t0 interventions."""
import argparse,json,copy
from pathlib import Path
from dataclasses import replace
import numpy as np,torch
from graph.integration_baseline.io import write_json,save_npz
from graph.integration_baseline.geometry09b_probe import csv_write
from graph.persistent_geometry.evaluate import load_cameras
from graph.persistent_geometry.self_filter import LinkSurface
from graph.persistent_geometry.metrics import sample_meshes
from graph.plan_a_geometry.core import PlanA
from graph.plan_a_geometry.evaluate import save_records
from graph.persistent_geometry.contracts import transform
from visual.runtime.vision_packet import pose_matrix
from utils.experiment_logger import ExperimentLogger
from .oracle import OracleBoxMemory
from .motion import load,rebased,measured
from .core import BoxMemory


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(Path(__file__).with_name('protocol.json').read_text());acfg=json.loads(Path(__file__).parents[1].joinpath('plan_a_geometry/protocol.json').read_text());torch.set_num_threads(1)
    ep=json.loads((a.capture/'capture_manifest.json').read_text())['episodes'][0];source=a.capture/'collection/seed30002';geom=load(source/'robot_meshes.npz');meshes=sample_meshes(source/'evaluation_meshes.npz')
    system=PlanA('base_camera','hand_camera',acfg)
    def prepare(folder,oracle=False,baseline=False):
        inp=folder/('f0000_input.npz' if baseline else 'input.npz');mp=folder/('f0000_masks.npz' if baseline else 'masks.npz');gp=folder/('f0000_gt.npz' if baseline else 'gt.npz')
        cs,data=load_cameras(inp);masks=load(mp);gt=load(gp)
        if oracle:masks['base_camera'][1]=gt['base_camera/segmentation']==gt['actor_ids'][1]
        surfaces=[LinkSurface(n,geom[f'vertices_{i}'],data['robot_link_world'][i]) for i,n in enumerate(geom['links'])]
        prepared=system.prepare(cs,data,masks,surfaces);pools,_,_,_=system.represent(cs,prepared,'C3',('SF',))
        return next(c for c in cs if c.camera=='base_camera'),[r for r in pools['SF'] if r['object_id']==1],gt
    with ExperimentLogger(team='integration',config=dict(task='09-E',stage='separate_motion_oracle',physical_timestep=0,
        baseline_initialized_at_tick0=True,intervention_tick=1,estimated_rules_unchanged=True,gt_scope='oracle only',no_policy=True),
        seed=42,model='task09e_motion_oracle',job_type='evaluate',root=a.output) as log:
        root=log.directory;print('RUN_DIRECTORY='+str(root),flush=True);base={};rows=[]
        for n in ('O1','O2','O3'):
            camera,rr,gt=prepare(source,n in ('O1','O3'),True)
            m=BoxMemory('B1',cfg['memory']) if n=='O1' else OracleBoxMemory(cfg['memory'])
            if n=='O1':m.update_records(camera.episode,0,rr,camera)
            else:m.update_oracle(camera.episode,0,rr,camera,pose_matrix(gt['actor_pose'][1]))
            base[n]=m
        for b in [b for b in ep['branches'] if b['timestep']==0 and b['object']=='box']:
            for n in base:
                camera,rr,gt=prepare(source/b['source'],n in ('O1','O3'));m=rebased(base[n],camera.episode,1)
                camera=replace(camera,timestep=1,timestamp=.05)
                for r in rr:r['source_observation_ids']=[(s[0],1,s[2],s[3]) for s in r['source_observation_ids']]
                if n=='O1':m.update_records(camera.episode,1,rr,camera)
                else:m.update_oracle(camera.episode,1,rr,camera,pose_matrix(gt['actor_pose'][1]))
                delta=np.array(b['translation_world_m']);actor=gt['actor_pose'][1,:3]-delta
                predicted=transform(actor,m.objects[1].world_from_object@np.linalg.inv(base[n].objects[1].world_from_object))-actor
                out=root/b['condition']/n;out.mkdir(parents=True);save_records(out/'points.npz',m.active_records());write_json(out/'trace.json',m.trace[-1])
                rows.append(dict(mode=n,condition=b['condition'],physical_timestep=0,diagnostic_tick=1,physical_valid=b['physical_valid'],
                    imposed_delta_mm=float(np.linalg.norm(delta)*1000),pose_delta_error_mm=float(np.linalg.norm(predicted-delta)*1000),
                    **measured(m,gt,meshes,camera)))
        csv_write(root/'oracle_motion_metrics.csv',rows);write_json(root/'oracle_motion_manifest.json',dict(rows=len(rows),scope='same actual rendered warm-start t0 control; Oracle O1/O2/O3 only; no estimated metric replacement'))
        log.link_output(root/'oracle_motion_metrics.csv',kind='metrics')
if __name__=='__main__':main()
