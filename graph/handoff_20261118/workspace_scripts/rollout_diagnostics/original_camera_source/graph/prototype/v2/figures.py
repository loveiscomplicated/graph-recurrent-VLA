"""Scientific plots from saved measurements; never simulated performance data."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path('outputs/graph/v2/.mplconfig').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def trace_figure(reports,path):
    fig,axes=plt.subplots(4,2,figsize=(13,11),sharex='col')
    for col,setting in enumerate(['full','alignment']):
        for name,r in reports.items():
            episode=next(e for e in r['settings'][setting]['episodes'] if e['episode_id']==29)
            rows=episode['trace'];t=np.arange(len(rows));a=np.array([row['applied_action'] for row in rows]);state=[row['after'] for row in rows]
            axes[0,col].plot(t,[s['grasp_distance_m'] for s in state],label=name)
            axes[1,col].plot(t,[s['transverse_error_m'] for s in state])
            axes[2,col].plot(t,a[:,6]);axes[3,col].plot(t,np.linalg.norm(a[:,:3],axis=1))
        axes[0,col].set_title('Episode 29: '+setting+(' (expert prefix)' if col else ''))
        axes[3,col].set_xlabel('Learner steps')
    for row,label in enumerate(['EE–grasp residual (m)','Head transverse error (m)','Applied gripper command','Applied translation norm']):axes[row,0].set_ylabel(label)
    axes[0,0].legend();axes[2,0].set_ylim(-1.1,1.1);axes[2,1].set_ylim(-1.1,1.1)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/graph/v2'));args=p.parse_args()
    out=args.root/'figures';out.mkdir(parents=True,exist_ok=True)
    reports={}
    for name in ['flat','center','surface']:
        path=args.root/'diagnostics'/f'{name}_v0.json'
        reports[name]=json.loads(path.read_text())
    trace_figure(reports,out/'v0_action_traces.png')
    newer={name:json.loads((args.root/'diagnostics'/f'{name}_goal.json').read_text()) for name in ['flat','center','surface'] if (args.root/'diagnostics'/f'{name}_goal.json').exists()}
    if newer:trace_figure(newer,out/'v2_action_traces.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ['flat_duration','flat_proprio','flat_goal','center_goal','surface_goal']:
        path=args.root/'training'/name/'history.json'
        if path.exists():
            history=json.loads(path.read_text());x=[h['epoch'] for h in history]
            axes[0].plot(x,[h['train_mse'] for h in history],label=name)
            axes[1].plot(x,[h['val_mse'] for h in history],label=name)
    for ax,title in zip(axes,['Train','Validation']):ax.set(xlabel='Epoch',ylabel='Normalized action MSE',title=title,yscale='log')
    axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'loss_ablation.png',dpi=150);plt.close(fig)
    evaluations={name:json.loads((args.root/'evaluation'/f'{name}_goal.json').read_text()) for name in ['flat','center','surface'] if (args.root/'evaluation'/f'{name}_goal.json').exists()}
    if evaluations:
        fig,axes=plt.subplots(1,4,figsize=(13,4));names=list(evaluations)
        for ax,key,title in zip(axes,['action_mse','translation_rmse','rotation_rmse','gripper_mse'],['Action MSE','Translation RMSE','Rotation RMSE','Gripper MSE']):
            ax.bar(names,[evaluations[n]['offline'][key] for n in names],color=['steelblue','darkorange','seagreen'][:len(names)])
            ax.set_title(title);ax.set_ylabel('Normalized action units');ax.tick_params(axis='x',rotation=20)
        fig.tight_layout();fig.savefig(out/'errors_v2.png',dpi=150);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4));phases=['approach','grasp','transport','alignment','insertion']
        for name,r in evaluations.items():ax.plot(phases,[r['phase_offline'][p]['action_mse'] for p in phases],marker='o',label=name)
        ax.set(ylabel='Normalized action MSE',title='Held-out expert geometry-proxy phases (diagnostics only)');ax.legend();fig.tight_layout();fig.savefig(out/'phase_errors_v2.png',dpi=150);plt.close(fig)
    distribution=args.root/'diagnostics'/'distribution_v2.json'
    if distribution.exists():
        d=json.loads(distribution.read_text());fig,ax=plt.subplots(figsize=(8,4))
        for name,m in d['models'].items():
            values=np.sort([s['distance'] for s in m['per_state']]);ax.plot(values,np.arange(1,len(values)+1)/len(values),label=name)
        ax.axvline(d['expert_test']['p95'],c='black',ls='--',label='Expert test p95');ax.set(xlabel='Nearest train state distance',ylabel='Empirical CDF',xscale='log');ax.legend();fig.tight_layout();fig.savefig(out/'distribution_shift.png',dpi=150);plt.close(fig)
    print(out)

if __name__=='__main__':main()
