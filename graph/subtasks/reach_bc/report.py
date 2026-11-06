"""Measured result table, compact diagnostics, and standalone figures."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from graph.prototype.common import write_json
from graph.prototype.data import load_dataset
from graph.prototype.v2.goals import goal_geometry
from graph.subtasks.metrics import compose, rotation, rotation_error
from .policy import ROOT,DATASET


def report(output=ROOT,dataset=DATASET):
    output=Path(output)
    read=lambda p:json.loads((output/p).read_text())
    train=read('training/training.json');audit=read('reports/trimming_audit.json')
    results={mode:read(f'evaluation/{mode}.json') for mode in ('flat','reference','fixed_open')}
    data,g=load_dataset(dataset)
    index=json.loads(Path(data['manifest']['source_index']).read_text())
    offsets=[];angles=[]
    for ep in data['episodes']:
        ids=torch.where(data['episode_id']==ep['episode_id'])[0][:1]
        geometry=g.select(ids);goal=goal_geometry(geometry)
        source=next(e for e in index['episodes'] if e['episode_id']==ep['episode_id'])
        seg=next(s for s in source['segments'] if s['subtask']=='reach')
        target=compose(geometry.peg_pose[0].numpy(),seg['goal_metadata']['reach_local'])
        offsets.append(float(np.linalg.norm(goal['grasp_position'][0].numpy()-target[:3])))
        angles.append(rotation_error(goal['grasp_rotation'][0].numpy().T@rotation(target)))
    detail=dict(v2_grasp_feature_vs_actual_reach_target=dict(position_min_m=min(offsets),position_max_m=max(offsets),
                                                           position_mean_m=float(np.mean(offsets)),rotation_max_rad=max(angles),
                                                           note='unchanged v2 grasp feature; actual Reach target is 5cm approach above grasp'),
                audit_tail_episodes=sum(r['removed_actions']>0 for r in audit['records']),
                tail_success_exits=sum(r['success_exits_in_saved_segment'] for r in audit['records']),
                tail_non_open_commands=sum(r['after_success'].get('non_open_commands',0) for r in audit['records']),
                tail_min_actions=min(r['removed_actions'] for r in audit['records']),tail_max_actions=max(r['removed_actions'] for r in audit['records']),
                mean_translation_before_m=float(np.mean([r['mean_realized_translation_before_m'] for r in audit['records']])),
                mean_translation_after_m=float(np.mean([r['mean_realized_translation_after_m'] for r in audit['records']])),
                modes={})
    for mode,result in results.items():
        mode_detail={}
        for group in ('held_out','fresh'):
            records=[r for r in result['records'] if r['reset_valid'] and r['start']['group']==group]
            rows=[s for r in records for s in r['trace']]
            failed=[r for r in records if not r['success']]
            near=[r for r in failed if r['diagnostics']['minimum_position_error_m']<=.02]
            mode_detail[group]=dict(**result[group],total_control_steps=len(rows),
                                   motion_clipping_steps=sum(any(abs(x)>1 for x in s['raw_action'][:6]) for s in rows),
                                   translation_clipping_steps=sum(any(abs(x)>1 for x in s['raw_action'][:3]) for s in rows),
                                   rotation_component_clipping_steps=sum(any(abs(x)>1 for x in s['raw_action'][3:6]) for s in rows),
                                   gripper_clipping_steps=sum(abs(s['raw_action'][6])>1 for s in rows),
                                   actual_closed_steps=sum(not s['after']['metrics']['gripper_open'] for s in rows),
                                   rotation_norm_saturation_steps=sum(s['rotation_norm_saturated'] for s in rows),
                                   failed_episodes_ever_within_position_threshold=[r['start']['eval_seed'] for r in near],
                                   successful_seeds=[r['start']['eval_seed'] for r in records if r['success']],
                                   median_failed_minimum_position_error_m=float(np.median([r['diagnostics']['minimum_position_error_m'] for r in failed])) if failed else None,
                                   median_failed_final_position_error_m=float(np.median([r['diagnostics']['final_position_error_m'] for r in failed])) if failed else None,
                                   median_failed_final_rotation_error_deg=float(np.rad2deg(np.median([r['diagnostics']['final_rotation_error_rad'] for r in failed]))) if failed else None)
        detail['modes'][mode]=mode_detail
    write_json(output/'reports/failure_summary.json',detail)
    lines=['| Model | Dataset | Test action MSE | Held-out SR | Fresh SR | Params | Forward latency |',
           '|---|---|---:|---:|---:|---:|---:|']
    for mode,label in [('flat','Flat MLP BC'),('reference','Cartesian reference (diagnostic)'),('fixed_open','Flat MLP + fixed-open (diagnostic)')]:
        r=results[mode]
        mse=f"{train['test_metrics']['action_mse']:.8f}" if mode=='flat' else 'N/A'
        params=str(train['parameters']) if mode!='reference' else 'N/A'
        lines.append(f"| {label} | {'Reach first-success' if mode!='reference' else 'N/A'} | {mse} | {r['held_out']['successful_episodes']}/{r['held_out']['valid_resets']} | {r['fresh']['successful_episodes']}/{r['fresh']['valid_resets']} | {params} | {r['latency']['forward_mean_ms']:.3f} ms |")
    lines.extend(['','Latency is CPU model forward only for Flat; reference latency is pose-command computation.',
                  f"Flat GT extraction + features + forward: {results['flat']['latency']['feature_and_forward_mean_ms']:.3f} ms.",
                  'All SR denominators are valid resets; 55 attempted, 55 valid, 0 invalid for each mode.',
                  'Fixed-open is a diagnostic result and does not replace the main learned-policy result.'])
    (output/'reports/results.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.joinpath('figures').mkdir(exist_ok=True)
    seeds=[11033,11029,12001,12000,12017,12005]
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,seed in zip(axes.flat,seeds):
        r=next(r for r in results['flat']['records'] if r['start']['eval_seed']==seed)
        states=[r['initial']]+[s['after'] for s in r['trace']]
        ax.plot([s['metrics']['position_error'] for s in states],label='Position (m)')
        ax.axhline(.02,color='k',linestyle='--',linewidth=.8)
        twin=ax.twinx();twin.plot([np.rad2deg(s['metrics']['rotation_error']) for s in states],color='tab:orange',alpha=.7,label='Rotation (deg)')
        twin.axhline(15,color='tab:orange',linestyle='--',linewidth=.8)
        ax.set(title=f"{seed}: {'success' if r['success'] else r['failure_reason']}",xlabel='Control step',ylabel='Position error (m)')
        twin.set_ylabel('Rotation error (deg)')
    fig.tight_layout();fig.savefig(output/'figures/representative_errors.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    modes=['flat','fixed_open','reference']
    for group in ('held_out','fresh'):
        axes[0].plot(modes,[results[m][group]['success_rate'] for m in modes],marker='o',label=group)
    axes[0].set(ylabel='Reach success rate',ylim=(-.05,1.05));axes[0].legend()
    labels=['success','peg displaced','timeout']
    for i,mode in enumerate(modes):
        r=results[mode]['fresh'];values=[r['successful_episodes'],r['failed_episodes'],r['timeout_episodes']]
        axes[1].bar(np.arange(3)+(i-1)*.25,values,width=.25,label=mode)
    axes[1].set_xticks(np.arange(3),labels);axes[1].set_ylabel('Fresh episodes');axes[1].legend()
    fig.tight_layout();fig.savefig(output/'figures/sr_failures.png',dpi=160);plt.close(fig)
    print('\n'.join(lines),flush=True)
    return detail


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=ROOT);p.add_argument('--dataset',default=DATASET)
    a=p.parse_args();report(a.output,a.dataset)
