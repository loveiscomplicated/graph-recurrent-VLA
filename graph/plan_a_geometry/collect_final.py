"""New blind diagnostic expert observations, only after freeze; no policy."""
import argparse,json,os
from pathlib import Path
from datetime import datetime
import numpy as np,torch
from graph.persistent_geometry.collect import collect_episode,inventory_sources
from graph.persistent_geometry.audit import validate_split
from graph.integration_baseline.io import write_json,sha256
from visual.runtime.grounding import GDino
from visual.runtime.tracking import SAM2Tracker
from utils.experiment_logger import ExperimentLogger

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text());freeze=json.loads(a.freeze.read_text())
    identity_path=Path(__file__).resolve().parents[3]/'model_identity.json'
    model_identity=json.loads(identity_path.read_text())
    for path,digest in model_identity['files'].items():
        if sha256(path)!=digest:raise ValueError('pretrained model/dependency identity changed')
    for path,h in freeze['frozen_files'].items():
        if sha256(path)!=h:raise ValueError('selection or source changed after freeze')
    assert freeze['final_seeds']==protocol['final_seeds']
    assert os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('WANDB_MODE')=='disabled'
    torch.set_num_threads(1);torch.manual_seed(42);np.random.seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    config=dict(task='09-D',stage='independent_final_capture',split=dict(train=[],validation=[],evaluation=protocol['final_seeds']),
        horizon=250,control_mode='pd_ee_delta_pose',branch_seeds=protocol['final_seeds'],branch_timesteps=[0,25,65],
        source_freeze=str(a.freeze),no_policy=True,gt_scope='separate evaluation/target sidecar only')
    with ExperimentLogger(team='integration',config=config,seed=42,model='task09d_blind_diagnostic_capture',job_type='evaluate',root=a.output) as log:
        root=log.directory;print('RUN_DIRECTORY='+str(root),flush=True)
        write_json(root/'pretrained_model_identity.json',model_identity)
        log.record(pretrained_model_identity_sha256=sha256(identity_path))
        detector=GDino(device='cuda');tracker=SAM2Tracker(device='cuda');episodes=[]
        for seed in protocol['final_seeds']:
            episodes.append(collect_episode(root,seed,'evaluation',detector,tracker,{},config))
            write_json(root/'capture_manifest.json',dict(config=config,episodes=episodes))
        validate_split(config['split'],{e['seed']:e['actual_reset_group'] for e in episodes})
        for path,digest in model_identity['files'].items():
            if sha256(path)!=digest:raise ValueError('pretrained model mutated during capture')
        write_json(root/'pretrained_model_protection.json',dict(status='PASS',checked_files=len(model_identity['files'])))
        inventory_sources(root);log.link_output(root/'capture_manifest.json',kind='metrics')
if __name__=='__main__':main()
