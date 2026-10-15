"""Offline reproduction, physical one-step probes and trace summaries."""
from utils.experiment_logger import ExperimentLogger
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from ..common import seed_all, write_json
from ..data import load_dataset, split_indices, PHASES
from ..metrics import action_metrics
from .diagnostics import load_predictor, qvel_dataset, ReplaySession, run_episode


def offline(checkpoint,data,geometry,qvel):
    predict,ck,model,norm,builder,sn=load_predictor(checkpoint,data)
    with ExperimentLogger(team='graph', config=ck['config'], seed=0, model=ck['model_name'], job_type='evaluate') as experiment:
        experiment.link_output(checkpoint, checkpoint=True)
        ids=split_indices(data,'test')
        with torch.inference_mode():
            pred=torch.cat([predict(geometry.select(i),qvel[i]) for i in ids.split(32)])
            metrics=action_metrics(pred,data['actions'][ids])
            phases={p:dict(samples=int((data['phase'][ids]==k).sum()),**action_metrics(pred[data['phase'][ids]==k],data['actions'][ids][data['phase'][ids]==k])) for k,p in enumerate(PHASES)}
            construction=[];state=[];network=[]
            for t in range(60):
                index=ids[(t*23)%len(ids):((t*23)%len(ids))+1];g=geometry.select(index)
                start=perf_counter();inputs=norm.apply(ck['model_name'],g,builder);end=perf_counter()
                shared=sn.apply(g,qvel[index],ck['state_mode']) if sn else None;end2=perf_counter()
                result=model(inputs,shared) if sn else model(inputs);end3=perf_counter()
                if t>=10:
                    construction.append((end-start)*1000);state.append((end2-end)*1000);network.append((end3-end2)*1000)
            latency=dict(graph_stats=inputs.stats if ck['model_name']!='flat' else None,geometry_feature_or_graph_ms=float(np.mean(construction)),shared_state_ms=float(np.mean(state)),network_ms=float(np.mean(network)),total_ms=float(np.mean(construction)+np.mean(state)+np.mean(network)),repeats=50,device='cpu',threads=4)
            distribution=dict(expert_mean=data['actions'][ids].mean(0).tolist(),predicted_mean=pred.mean(0).tolist(),expert_std=data['actions'][ids].std(0).tolist(),predicted_std=pred.std(0).tolist(),clipped_fraction=float((pred.abs()>1).float().mean()),gripper_sign_accuracy=float(((pred[:,6]>0)==(data['actions'][ids,6]>0)).float().mean()))
        report = dict(checkpoint=str(checkpoint),model=ck['model_name'],parameters=ck['parameters'],checkpoint_epoch=ck['epoch'],offline=metrics,phase_offline=phases,latency=latency,distribution=distribution)

        experiment.log(report, prefix='eval')
        return report


def probes(data,geometry,paths):
    source=json.loads(Path('outputs/graph/v2/diagnostics/expert.json').read_text())
    session=ReplaySession(data);records=[]
    try:
        with torch.inference_mode():
            for eid in (29,41,59):
                ep=next(e for e in data['manifest']['episodes'] if e['episode_id']==eid)
                align=int(data['timestep'][torch.where((data['episode_id']==eid)&(data['phase']==3))[0][0]])
                for stage,t in [('approach',15),('grasp',source['starts'][str(eid)]['grasp']),('alignment',align)]:
                    expert=run_episode(session,ep,prefix=t,expert=True,trace_limit=1)
                    baseline=expert['trace'][0]
                    for name,path in paths.items():
                        predictor,*_=load_predictor(path,data)
                        r=run_episode(session,ep,predictor,prefix=t,trace_limit=1)
                        if r['initial_geometry_sha256']!=expert['initial_geometry_sha256']:raise ValueError('unpaired probe')
                        row=r['trace'][0]
                        records.append(dict(model=name,episode_id=eid,stage=stage,prefix=t,initial_geometry_sha256=r['initial_geometry_sha256'],expert_action=baseline['prediction'],learner_action=row['prediction'],expert_realized_translation_m=baseline['realized_ee_translation_m'],learner_realized_translation_m=row['realized_ee_translation_m'],expert_realized_rotation_rad=baseline['realized_ee_rotation_rad'],learner_realized_rotation_rad=row['realized_ee_rotation_rad'],expert_after=baseline['after'],learner_after=row['after']))
    finally:session.close()
    return records


def summarize(paths):
    result={}
    for name,path in paths.items():
        r=json.loads(Path(path).read_text()); result[name]={}
        for stage,s in r['settings'].items():
            traces=[row for e in s['episodes'] for row in e['trace']]
            action=np.array([x['prediction'] for x in traces]);applied=np.array([x['applied_action'] for x in traces]);motion=np.array([x['realized_ee_translation_m'] for x in traces])
            result[name][stage]={k:s[k] for k in ('denominator','successes','stage_counts','failure_counts')}
            result[name][stage].update(mean_translation_command_norm=float(np.linalg.norm(action[:,:3],axis=1).mean()),mean_realized_translation_m=float(np.linalg.norm(motion,axis=1).mean()),mean_rotation_command=action[:,3:6].mean(0).tolist(),mean_gripper_command=float(action[:,6].mean()),intermediate_gripper_fraction=float((np.abs(action[:,6])<.5).mean()),clipped_component_fraction=float((action!=applied).mean()),actual100_truncations=sum(row['truncated'] and row['elapsed_steps']==100 for row in traces),final_transverse_mean_m=float(np.mean([e['final']['transverse_error_m'] for e in s['episodes']])))
    # Strict pair identity by setting, episode, prefix (excluding qvel which is
    # independently checked by one-step live replay tests).
    pairs={}
    for name,path in paths.items():
        r=json.loads(Path(path).read_text())
        for stage,s in r['settings'].items():
            for e in s['episodes']:
                key=(stage,e['episode_id'],e['prefix'])
                if key in pairs and pairs[key]!=e['initial_geometry_sha256']:raise ValueError(f'unpaired {key}')
                pairs[key]=e['initial_geometry_sha256']
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probes',action='store_true');p.add_argument('--summary',action='store_true');p.add_argument('--checkpoint',type=Path)
    args=p.parse_args();seed_all(0,4);data,geometry=load_dataset('outputs/graph/dataset.pt')
    paths={n:Path(f'outputs/graph/runs/{n}/best.pt') for n in ('flat','center','surface')}
    if args.probes:result=probes(data,geometry,paths)
    elif args.summary:result=summarize({n:f'outputs/graph/v2/diagnostics/{n}_v0.json' for n in paths})
    elif args.checkpoint:result=offline(args.checkpoint,data,geometry,qvel_dataset(data))
    else:result={n:offline(path,data,geometry,qvel_dataset(data)) for n,path in paths.items()}
    write_json(args.output,result);print(json.dumps(result,indent=2)[:10000])

if __name__=='__main__':main()
