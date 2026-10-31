"""Inference-only R0 update3000 diagnostics on already frozen development cache."""
import argparse,os
from pathlib import Path
import numpy as np
import torch
from graph.plan_a_bc.cache import CachedDataset
from graph.plan_a_bc.model import CachedGraphPolicy,normalizer
from graph.plan_a_bc.common import read,write,sha,append_csv,checkpoint_load,npz,tensor_identity
from graph.integration_baseline.bc08_codec import ActionCodec,controller_commands
from utils.experiment_logger import ExperimentLogger


def run(base,new,output):
    assert os.environ.get('CONTAINER_ID')=='57640710'
    assert read(new/'evaluation_complete.json')['status']=='PASS'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1);cache=base/'full_cache';ds=CachedDataset(cache);norm=normalizer(cache);cfg=read(cache/'config.json')
    identity=read(new/'checkpoint_identity.json')['checkpoints']['R0_step3000'];assert sha(identity['path'])==identity['sha256']
    cp=checkpoint_load(identity['path']);codec=ActionCodec(cp['codec'])
    with ExperimentLogger(team='graph',model='Task10F_matched3000_cached_dev_diagnostics',seed=42,job_type='evaluate',
        root=output/'experiments',config=dict(parent_evaluation_uuid=read(new/'identity.json')['run_id'],
        parent_comparison_uuid='d7c26fee195e4f32a7ddb99a94877b72',training=False,checkpoint_sha256=identity['sha256'],
        existing_R1_diagnostics_reused=True)) as log:
        write(output/'identity.json',dict(run_id=log.run_id),immutable=True)
        log.link_output(identity['path'],checkpoint=True)
        model=CachedGraphPolicy(norm.schema,'E0',**cfg['model']).cuda();model.load_state_dict(cp['model'],strict=True);model.eval()
        assert tensor_identity(model.state_dict())==identity['model_sha256']
        rows=[]
        for eid,e in sorted(ds.episodes.items()):
            if e['split']!='validation':continue
            with np.load(e['supervision']['path']) as z:supervision=z['supervision_mask'].astype(bool)
            hidden=None;pred=[];target=[]
            with torch.inference_mode():
                for start in range(0,e['actions'],32):
                    g,s,y,valid=ds.batch([eid],start,32,'cuda');raw,hidden=model(g,s,norm,hidden);hidden=hidden.detach()
                    native=codec.decode(codec.tag(raw))[0].detach().cpu().numpy();selected=supervision[start:start+len(native)]
                    pred.append(native[selected]);target.append(y[0].detach().cpu().numpy()[selected])
            p=np.concatenate(pred);t=np.concatenate(target);assert len(p)==int(supervision.sum())
            a=controller_commands(p);b=controller_commands(t);dt=a[:,:3]-b[:,:3];dr=a[:,3:6]-b[:,3:6]
            cosine=float(a[0,:3]@b[0,:3]/(np.linalg.norm(a[0,:3])*np.linalg.norm(b[0,:3]))) if e['kind']=='recovery' else None
            row=dict(condition='R0',episode_id=eid,parent=e['parent'],kind=e['kind'],supervised_timesteps=len(p),
                controller_translation_mae_m=float(abs(dt).mean()),controller_translation_vector_error_m=float(np.linalg.norm(dt,axis=1).mean()),
                controller_rotation_component_mae_rad=float(abs(dr).mean()),gripper_mae_m=float(abs(a[:,6]-b[:,6]).mean()),
                first_expert_student_translation_cosine=cosine,first_expert_controller_translation_m=b[0,:3].tolist(),first_student_controller_translation_m=a[0,:3].tolist())
            rows.append(row);append_csv(output/'expert_command_errors.csv',row)
            npz(output/'predictions'/f'{eid}.npz',dict(native_pred_action7=p,native_expert_action7=t,supervised_state_t=np.flatnonzero(supervision)))
        summary={k:dict(episodes=sum(r['kind']==k for r in rows),controller_translation_mae_m=float(np.mean([r['controller_translation_mae_m'] for r in rows if r['kind']==k])),
            first_expert_student_translation_cosine=None if k=='nominal' else float(np.mean([r['first_expert_student_translation_cosine'] for r in rows if r['kind']==k]))) for k in ['nominal','recovery']}
        write(output/'summary.json',dict(status='PASS',checkpoint_sha256=identity['sha256'],summary=summary,episodes=len(rows),new_training=False),immutable=True)
        log.link_output(output/'summary.json',kind='metrics');print('OFFLINE_MATCHED_COMPLETE',summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.base,a.new,a.output)
