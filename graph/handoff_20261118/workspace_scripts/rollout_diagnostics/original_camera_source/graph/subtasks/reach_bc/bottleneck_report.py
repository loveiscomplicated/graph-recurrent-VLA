"""Aggregate frozen-policy diagnosis and create standalone scientific figures."""
import argparse
import json
from pathlib import Path
import numpy as np
from graph.prototype.common import write_json
from .policy import ROOT
from .bottleneck import OUTPUT,MODES


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(minimum=float(a.min()),median=float(np.median(a)),p95=float(np.percentile(a,95)),maximum=float(a.max())) if len(a) else None


def report(output=OUTPUT):
    output=Path(output)
    read=lambda path:json.loads(Path(path).read_text())
    base=read(ROOT/'evaluation/flat.json')
    audit=read(output/'audit.json')
    if audit['partial'] or len(audit['records'])!=55:raise ValueError('audit incomplete')
    results={mode:read(output/f'{mode}.json') for mode in MODES}
    feedback=read(output/'feedback.json');switch=read(output/'reference_switch.json')
    groups={}
    for group in ('held_out','fresh'):
        records=[r for r in audit['records'] if r['group']==group]
        original=[r for r in base['records'] if r['start']['group']==group]
        first_under=[r for r in records if r['rows'][0]['actions']['original'][6]<1]
        first_over=[r for r in records if r['rows'][0]['actions']['original'][6]>=1]
        def early_stats(records):
            return dict(episodes=len(records),successes=sum(r['success'] for r in records),
                        second_step_finger_mean_action_delta=stats([r['rows'][1]['counterfactual_motion_delta']['finger_mean'] for r in records]),
                        second_step_finger_velocity_z_max=stats([max(abs(z) for z in r['rows'][1]['finger_qvel_z']) for r in records]))
        def qualifies(metric,checks):return all(check(metric) for check in checks)
        position=lambda m:m['position_error']<=.02
        orientation=lambda m:m['rotation_error']<=np.deg2rad(15)
        states=lambda r:[r['initial']['metrics']]+[s['after']['metrics'] for s in r['trace']]
        groups[group]=dict(first_gripper_under_one=early_stats(first_under),first_gripper_at_least_one=early_stats(first_over),
                           early_finger_velocity_above_one_z=sum(max(abs(z) for z in r['rows'][1]['finger_qvel_z'])>1 for r in records),
                           position_pass_episodes=sum(any(position(m) for m in states(r)) for r in original),
                           rotation_pass_episodes=sum(any(orientation(m) for m in states(r)) for r in original),
                           simultaneous_position_rotation_pass_episodes=sum(any(qualifies(m,[position,orientation]) for m in states(r)) for r in original),
                           first_command_alignment=stats([r['rows'][0]['command_target_alignment'] for r in records]),
                           episodes_any_command_away_from_target=sum(any(s['command_target_alignment'] is not None and s['command_target_alignment']<0 for s in r['rows']) for r in records),
                           workspace_exit_episodes=sum(r['diagnostics']['workspace_exit_steps']>0 for r in original),
                           early_ee_floor_counterfactual_effect_max=max(r['rows'][1]['counterfactual_motion_delta']['ee_floor_clip'] for r in records))
    gain={mode:dict(singular_gain=stats([r['results'][mode]['maximum_singular_gain'] for r in feedback['records']]),
                    spectral_radius=stats([r['results'][mode]['action_feedback_spectral_radius'] for r in feedback['records']])) for mode in feedback['records'][0]['results']}
    final=dict(groups=groups,maximum_replay_prediction_error=max(r['replay_and_prediction_max_error'] for r in audit['records']),
               baseline=dict(held_out=base['held_out'],fresh=base['fresh']),
               interventions={m:dict(held_out=r['held_out'],fresh=r['fresh']) for m,r in results.items()},
               feedback=gain,reference_switch=switch,offline=read(output/'offline.json'),weight_updates=read(output/'weight_updates.json'),
               all_intervention_initial_fingerprints_match=all(s['initial_state_fingerprint']==s['start']['initial_state_fingerprint'] for r in results.values() for s in r['records']))
    write_json(output/'summary.json',final)
    lines=['| Frozen checkpoint condition | Held-out | Fresh | Fresh peg displaced | Fresh timeout |',
           '|---|---:|---:|---:|---:|']
    for mode,r in [('original',base)]+list(results.items()):
        lines.append(f"| {mode} | {r['held_out']['successful_episodes']}/5 | {r['fresh']['successful_episodes']}/50 | {r['fresh']['failed_episodes']} | {r['fresh']['timeout_episodes']} |")
    (output/'results.md').write_text('\n'.join(lines)+'\n\nAll additional conditions are diagnostic input interventions, not new learned-policy main results.\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (output/'figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for col,seed in enumerate((11033,12000,12005)):
        r=next(r for r in audit['records'] if r['seed']==seed);rows=r['rows'][:20]
        ts=[s['t'] for s in rows]
        for i in range(2):axes[0,col].plot(ts,[s['finger_qvel_z'][i] for s in rows],label=f'finger {i}')
        axes[0,col].set(title=f'Seed {seed}',xlabel='Control state t',ylabel='Normalized finger velocity')
        axes[0,col].set_yscale('symlog',linthresh=1)
        if seed==12005:
            axes[0,col].set_ylim(-.22,.14)
            axes[0,col].set_yticks([-.2,-.1,0,.1])
            from matplotlib.ticker import ScalarFormatter
            axes[0,col].yaxis.set_major_formatter(ScalarFormatter())
        axes[0,col].legend()
        for mode in ('original','finger_mean','qvel_zero'):
            axes[1,col].plot(ts,[s['actions'][mode][5] for s in rows],label=mode)
        axes[1,col].set(xlabel='Control state t',ylabel='Predicted yaw component (normalized)')
        axes[1,col].set_yscale('symlog',linthresh=.1)
        axes[1,col].legend()
    fig.tight_layout();fig.savefig(output/'figures/finger_coupling.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    seeds=[r['seed'] for r in feedback['records']]
    for mode in ('original','finger_mean','qvel_zero','reference'):
        axes[0].plot([str(s) for s in seeds],[r['results'][mode]['maximum_singular_gain'] for r in feedback['records']],marker='o',label=mode)
    axes[0].axhline(1,color='k',linestyle='--',linewidth=.8);axes[0].set(ylabel='One-step action feedback gain',xlabel='Reset seed');axes[0].legend(fontsize=8)
    axes[0].tick_params(axis='x',labelrotation=45)
    names=['original']+list(MODES);result_list=[base]+[results[m] for m in MODES]
    axes[1].bar(np.arange(len(names)),[r['fresh']['successful_episodes'] for r in result_list])
    axes[1].set_xticks(np.arange(len(names)),names,rotation=35,ha='right');axes[1].set(ylabel='Fresh successes / 50')
    fig.tight_layout();fig.savefig(output/'figures/feedback_interventions.png',dpi=160);plt.close(fig)
    print('\n'.join(lines));print('groups',groups);print('gains',gain)
    return final


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    a=p.parse_args();report(a.output)
