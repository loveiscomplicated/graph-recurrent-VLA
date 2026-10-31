"""Selected checkpoint expert-command diagnostics on frozen development graphs."""
import argparse,os
from pathlib import Path
import numpy as np,torch
from graph.plan_a_bc.cache import CachedDataset
from graph.plan_a_bc.model import CachedGraphPolicy,normalizer
from graph.plan_a_bc.common import read,write,sha,append_csv,checkpoint_load
from graph.integration_baseline.bc08_codec import ActionCodec,controller_commands
from utils.experiment_logger import ExperimentLogger


def run(root):
    assert os.getenv('CONTAINER_ID')=='57640710';root=root.resolve()
    frozen=read(root/'model_selection_frozen.json');cache=root/'full_cache';ds=CachedDataset(cache);norm=normalizer(cache);cfg=read(cache/'config.json')
    output=root/'offline_diagnostics';output.mkdir(exist_ok=False)
    with ExperimentLogger(team='graph',model='Task10F_selected_offline_expert_diagnostics',seed=42,config=dict(selection_sha256=sha(root/'model_selection_frozen.json')),
                          root=output/'experiments',job_type='evaluate') as log:
        allrows=[]
        for condition,receipt in frozen['selected'].items():
            log.link_output(receipt['path'],checkpoint=True)
            cp=checkpoint_load(receipt['path']);codec=ActionCodec(cp['codec']);model=CachedGraphPolicy(norm.schema,'E0',**cfg['model']).cuda();model.load_state_dict(cp['model']);model.eval()
            for eid,e in sorted(ds.episodes.items()):
                if e['split']!='validation':continue
                with np.load(e['supervision']['path']) as side:supervision=side['supervision_mask'].astype(bool)
                hidden=None;pred=[];target=[]
                with torch.inference_mode():
                    for start in range(0,e['actions'],32):
                        g,s,y,valid=ds.batch([eid],start,32,'cuda');raw,hidden=model(g,s,norm,hidden);hidden=hidden.detach()
                        native=codec.decode(codec.tag(raw))[0].detach().cpu().numpy();sel=supervision[start:start+len(native)]
                        pred.append(native[sel]);target.append(y[0].detach().cpu().numpy()[sel])
                p=np.concatenate(pred);t=np.concatenate(target);assert len(p)==int(supervision.sum())
                cpred=controller_commands(p);ctarget=controller_commands(t);dt=cpred[:,:3]-ctarget[:,:3];dr=cpred[:,3:6]-ctarget[:,3:6]
                first_direction=None
                if e['kind']=='recovery':
                    a=cpred[0,:3];b=ctarget[0,:3];first_direction=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a)>1e-8 and np.linalg.norm(b)>1e-8 else None
                row=dict(condition=condition,episode_id=eid,parent=e['parent'],kind=e['kind'],supervised_timesteps=len(p),
                         controller_translation_mae_m=float(np.abs(dt).mean()),controller_translation_vector_error_m=float(np.linalg.norm(dt,axis=1).mean()),
                         controller_rotation_component_mae_rad=float(np.abs(dr).mean()),gripper_mae_m=float(np.abs(cpred[:,6]-ctarget[:,6]).mean()),
                         first_expert_student_translation_cosine=first_direction,first_expert_controller_translation_m=ctarget[0,:3].tolist(),
                         first_student_controller_translation_m=cpred[0,:3].tolist())
                allrows.append(row);append_csv(output/'expert_command_errors.csv',row)
            del model;torch.cuda.empty_cache()
        summary={c:{k:dict(episodes=sum(r['condition']==c and r['kind']==k for r in allrows),
                           controller_translation_mae_m=float(np.mean([r['controller_translation_mae_m'] for r in allrows if r['condition']==c and r['kind']==k])))
                    for k in ['nominal','recovery']} for c in frozen['selected']}
        write(output/'summary.json',dict(status='PASS',selection_sha256=sha(root/'model_selection_frozen.json'),summary=summary),immutable=True)
        log.link_output(output/'summary.json',kind='metrics')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
