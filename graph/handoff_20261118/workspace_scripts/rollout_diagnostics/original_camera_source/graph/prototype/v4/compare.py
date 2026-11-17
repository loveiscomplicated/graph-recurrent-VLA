"""Measured-only comparison and paired initialization/command checks."""
import argparse
import json
from pathlib import Path
import hashlib
import numpy as np
import torch
from ..common import write_json


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def rate(report, stage):
    setting = report.get('settings',{}).get(stage) if report else None
    return f"{setting['successes']}/{setting['denominator']}" if setting else 'N/A'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/graph/v4'));args=p.parse_args()
    rows=[];reference={};paired=True;history_consistent=True;checkpoint_audit=[];normalizers=None;baseline_rollout_equivalence=[]
    idle_latency=read(args.root/'comparison'/'latency.json') or {}
    for name in ['flat','surface']:
        for mode in ['current','action','state','both']:
            report=read(args.root/'evaluation'/f'{name}_{mode}_heldout.json')
            fresh=read(args.root/'evaluation'/f'{name}_{mode}_fresh.json')
            ablated=read(args.root/'evaluation'/f'{name}_{mode}_ablated.json')
            training=read(args.root/'training'/f'{name}_{mode}'/'training.json')
            checkpoint=args.root/'training'/f'{name}_{mode}'/'best.pt'
            if checkpoint.exists():
                ck=torch.load(checkpoint,weights_only=True)
                stats=(ck['normalizer'],ck['state_normalizer'])
                if normalizers is None:normalizers=stats
                same_stats=all(torch.equal(group[k],normalizers[i][k]) for i,group in enumerate(stats) for k in group)
                if not same_stats:raise AssertionError('normalizers differ between configurations')
                budget=ck['config']['epochs']==30 and ck['config']['seed']==0 and ck['config']['batch_size']==32 and ck['config']['learning_rate']==.001 and ck['config']['weight_decay']==1e-5
                completed=bool(ck.get('baseline_source')) or (training is not None and training['optimizer_steps']==10590 and training['smoke_max_batches'] is None and len(training['history'])==30)
                checkpoint_audit.append(dict(model=name,history=mode,full_budget=budget,completed=completed,normalizers_identical=same_stats,baseline_reused=bool(ck.get('baseline_source'))))
                if not budget or not completed:raise AssertionError(f'incomplete/unequal budget: {name}/{mode}')
                if ck.get('baseline_source') and report and len(report['settings'])==4:
                    old_path=Path(ck['baseline_source']).parents[2]/'diagnostics'/f'{name}_goal.json'
                    old=read(old_path)
                    if old:
                        checks=[]
                        for stage in ['full','grasp','transport','alignment']:
                            a,b=old['settings'][stage]['episodes'],report['settings'][stage]['episodes']
                            same_initials=[(e['episode_id'],e['prefix'],e['initial_geometry_sha256']) for e in a]==[(e['episode_id'],e['prefix'],e['initial_geometry_sha256']) for e in b]
                            same_successes=[e['success'] for e in a]==[e['success'] for e in b]
                            same_commands=[[t['applied_action'] for t in e['trace']] for e in a]==[[t['applied_action'] for t in e['trace']] for e in b]
                            checks.append(dict(stage=stage,initials_identical=same_initials,successes_identical=same_successes,all_applied_commands_identical=same_commands))
                            if not (same_initials and same_successes and same_commands):raise AssertionError('v2 baseline rollout changed')
                        baseline_rollout_equivalence.append(dict(model=name,source=str(old_path),conditions=checks))
            row=dict(encoder=name,history=mode,offline=report['offline'] if report else None,
                training_device=training.get('training_device','cpu') if training else 'cpu',resume=training.get('resume') if training else None,
                params=report['parameters'] if report else None,latency=report['latency'] if report else None,
                heldout_full=rate(report,'full'),fresh_full=rate(fresh,'full'),acquired=rate(report,'grasp'),
                transport=rate(report,'transport'),alignment=rate(report,'alignment'),
                offline_ablated=report.get('offline_repeated_current_zero_action') if report else None,
                ablated_full=rate(ablated,'full'),ablated_alignment=rate(ablated,'alignment'),
                selected_epoch=report['epoch'] if report else None,training_seconds=training['training_seconds'] if training else None,
                stages={},trace_diagnostics={})
            if f'{name}_{mode}' in idle_latency:row['latency']=idle_latency[f'{name}_{mode}']
            for label,r in [('heldout',report),('fresh',fresh),('ablated',ablated)]:
                if not r:continue
                for stage,setting in r['settings'].items():
                    key=(label,stage)
                    fingerprints=[(ep['episode_id'],ep['prefix'],ep['initial_geometry_sha256']) for ep in setting['episodes']]
                    if key in reference:paired &= reference[key]==fingerprints
                    else:reference[key]=fingerprints
                    row['stages'][f'{label}_{stage}']={k:v for k,v in setting.items() if k!='episodes'}
                    diagnostics=[]
                    for ep in setting['episodes']:
                        tr=ep['trace']
                        if label!='ablated':
                            for before,after in zip(tr,tr[1:]):
                                if r['history_length']:
                                    history_consistent &= after['history_actions'][-1]==before['applied_action'] and after['history_actions'][:-1]==before['history_actions'][1:]
                        actions=np.array([t['applied_action'] for t in tr]);raw=np.array([t['prediction'] for t in tr])
                        diagnostics.append(dict(episode_id=ep['episode_id'],failure=ep['failure_stage'],steps=ep['steps'],
                            action_delta_rms=ep['action_delta_rms'],gripper_switches=ep['gripper_switches'],
                            clipped_component_fraction=float(np.mean(actions!=raw)),
                            qvel_abs_mean=float(np.mean(np.abs([t['before']['qvel'] for t in tr]))),
                            gripper_state_mean=np.mean([t['before']['gripper_state'] for t in tr],axis=0).tolist(),
                            head_in_hole_mean_m=np.mean([t['before']['head_in_hole_m'] for t in tr],axis=0).tolist(),
                            grasp_distance_mean_m=float(np.mean([t['before']['grasp_distance_m'] for t in tr])),
                            maximum_prefix_error=ep['prefix_observation_max_error']))
                    row['trace_diagnostics'][f'{label}_{stage}']=diagnostics
            rows.append(row)
    preservation=read(args.root/'preservation.json')
    unchanged={path:Path(path).exists() and hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in (preservation or {}).items()}
    result=dict(rows=rows,checkpoint_audit=checkpoint_audit,baseline_rollout_equivalence=baseline_rollout_equivalence,paired_initializations=paired,learner_history_matches_applied_commands=history_consistent,
        preservation=dict(checked=len(unchanged),unchanged=sum(unchanged.values()),changed=[k for k,v in unchanged.items() if not v]))
    write_json(args.root/'comparison'/'results.json',result)
    lines=['# Graph prototype v4 measured results','',
        '| Encoder | History | Action MSE | Full SR held-out | Full SR fresh | Alignment SR | Params | Cached CPU ms |',
        '|---|---|---:|---|---|---|---:|---:|']
    for row in rows:
        mse=f"{row['offline']['action_mse']:.8f}" if row['offline'] else 'N/A'
        latency=f"{row['latency']['total_ms']:.3f}" if row['latency'] else 'N/A'
        lines.append(f"| {row['encoder']} | {row['history']} | {mse} | {row['heldout_full']} | {row['fresh_full']} | {row['alignment']} | {row['params'] or 'N/A'} | {latency} |")
    lines+=['','SR values are successes/evaluated episodes. Full held-out and fresh conditions are separate. Alignment SR is expert-prefix warm-start success, not full-task alignment reached. CPU timing excludes live GT extraction and simulator stepping.','',
        '| Encoder | History | Translation RMSE | Rotation RMSE | Gripper MAE | Selected epoch |','|---|---|---:|---:|---:|---:|']
    for r in rows:
        o=r['offline']
        lines.append(f"| {r['encoder']} | {r['history']} | "+(' | '.join(f"{o[k]:.8f}" for k in ['translation_rmse','rotation_rmse','gripper_mae']) if o else 'N/A | N/A | N/A')+f" | {r['selected_epoch'] or 'N/A'} |")
    lines+=['','Translation/rotation RMSE use normalized commanded components, not realized TCP motion.','',
        '| Encoder | History | Acquired SR | Transport SR | Ablated full SR | Ablated alignment SR | Ablated action MSE |','|---|---|---|---|---|---|---:|']
    for r in rows:
        loss=f"{r['offline_ablated']['action_mse']:.8f}" if r['offline_ablated'] else 'N/A'
        lines.append(f"| {r['encoder']} | {r['history']} | {r['acquired']} | {r['transport']} | {r['ablated_full']} | {r['ablated_alignment']} | {loss} |")
    lines+=['','Ablation repeats the current representation and zeros past actions, retaining validity masks. It can create OOD inputs; deterioration alone is not evidence of temporal reasoning.','',
        '| Encoder/history | Condition | Acquisition failure | Retention failure | Transport failure | Alignment failure | Insertion failure |','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        for condition,s in r['stages'].items():
            counts=s['failure_counts']
            lines.append(f"| {r['encoder']}/{r['history']} | {condition} | "+' | '.join(str(counts[k]) for k in ['acquisition','retention','transport','alignment','insertion'])+' |')
    lines+=['',f'Paired initialization checks: {paired}. Learner history/command checks: {history_consistent}.',f"Preserved prior output files: {sum(unchanged.values())}/{len(unchanged)}.",'','Component errors, stage counts, epochs, action changes, gripper switches, clipping and failed trace distributions are in `results.json`; full per-step states/actions are in evaluation JSON.']
    (args.root/'comparison'/'results.md').write_text('\n'.join(lines)+'\n')
    if not paired or not history_consistent or not all(unchanged.values()):raise AssertionError('paired/history/preservation audit failed')


if __name__=='__main__':main()
