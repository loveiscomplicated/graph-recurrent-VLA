"""Same-robot paired t=0 controls using finalized 09-C causal interventions.

Only t=0 branches are executed by this extension pilot: both maps start empty.
No subsequent/history state is used. Later-time forks require full causal
prefix replay (--full in benchmark plus an explicit map-state fork adapter).
"""
import argparse,copy,json,shutil
from pathlib import Path
import torch
from utils.experiment_logger import ExperimentLogger
from graph.integration_baseline.io import write_json,sha256
from graph.integration_baseline.geometry09b_probe import csv_write
from .benchmark import run_episode,BASE

def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);args=p.parse_args()
    m=json.loads((args.geometry/'geometry_manifest.json').read_text());capture=Path(m['config']['capture'])
    original=json.loads((capture/'capture_manifest.json').read_text())
    config=copy.deepcopy(m['config']);config.update(stage='paired_t0',full=False,
        branch_scope='t0 only; empty causal histories; same robot/calibration; no late-time fork claim')
    torch.set_num_threads(1)
    with ExperimentLogger(team='integration',config=config,seed=42,model='legacy_fps_paired_t0',job_type='evaluate',root=BASE/'paired_runs') as logger:
        root=logger.directory;print('LEGACY_PAIRED_RUN='+str(root),flush=True)
        (BASE/'latest_paired.txt').write_text(str(root)+'\n');frames=[];allrows=[];alledges=[];allfilters=[];identities=[]
        for ep in original['episodes']:
            for b in ep.get('branches',[]):
                if b['timestep']!=0:continue
                condition=b['condition'];own=root/'branch_inputs'/condition;dest=own/'collection'/f'seed{ep["seed"]}';dest.mkdir(parents=True,exist_ok=True)
                src=capture/'collection'/f'seed{ep["seed"]}'
                for name in ('sensor_contract.json','robot_meshes.npz','evaluation_meshes.npz'):shutil.copy2(src/name,dest/name)
                for name,digest in (('input',b['input_sha256']),('masks',b['masks_sha256']),('gt',b['gt_sha256'])):
                    source=src/b['source']/(name+'.npz')
                    if sha256(source)!=digest:raise ValueError('branch source hash changed')
                    shutil.copy2(source,dest/('f0000_'+name+'.npz'))
                branch=copy.deepcopy(ep);branch['frames']=[dict(timestep=0,stem='f0000',phase='reach',input_sha256=b['input_sha256'],mask_sha256=b['masks_sha256'],gt_sha256=b['gt_sha256'])]
                output=root/'conditions'/condition
                ff,rr,ee,fff=run_episode(output,own,branch,1,config)
                for frame in ff:
                    frame.update(folder=str(Path('conditions')/condition/frame['folder']),condition=condition,physical_valid=b['physical_valid'])
                frames+=ff;allrows.extend(dict(condition=condition,physical_valid=b['physical_valid'],**r) for r in rr)
                alledges.extend(dict(condition=condition,**r) for r in ee);allfilters.extend(fff)
                identities.append(dict(seed=ep['seed'],split=ep['split'],**b))
        write_json(root/'branch_identity.json',identities)
        write_json(root/'geometry_manifest.json',dict(config={**config,'source_geometry':str(args.geometry)},frames=frames))
        csv_write(root/'geometry_metrics.csv',allrows);csv_write(root/'edge_metrics.csv',alledges);csv_write(root/'robot_filter_metrics.csv',allfilters)
        for n in ('geometry_manifest.json','branch_identity.json','geometry_metrics.csv','edge_metrics.csv'):logger.link_output(root/n,kind='metrics')
if __name__=='__main__':main()
