"""Local postprocessing for the separate mixed-control grasp diagnostic."""
import argparse
import csv
import html
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urlsplit, unquote

import cv2
import numpy as np

from graph.integration_baseline.io import write_json

CONDITIONS=('expert','model','replace_gripper','stop_arm')


def inference(values):
    e,m,g,a=(values[k]['outcome']['retained'] for k in CONDITIONS)
    if not e:return 'Expert control did not retain: no single-factor attribution.'
    if m:return 'Model close retained after normal approach under common full-close lift/hold.'
    if g and not a:return 'Only full-close gripper replacement rescued retention in this pair.'
    if a and not g:return 'Only zero-arm replacement rescued retention in this pair.'
    if a and g:return 'Either single replacement rescued; this pair does not rank their priority.'
    return 'Neither single replacement rescued; joint replacement (Expert) retained.'


def verify(run):
    report=json.loads((run/'results.json').read_text());c=report['protocol']
    if report['status']!='COMPLETE' or len(report['trials'])!=len(c['episodes'])*4:
        raise ValueError('incomplete experiment is not a completed comparison')
    if not report['weights_and_normalizer_unchanged'] or not all(v['status']=='PASS' for v in report['paired_checks'].values()):
        raise ValueError('weights/normalizer/pair contract failure')
    action_difference=hidden_difference=0.
    for spec in c['episodes']:
        ref=None
        for condition in CONDITIONS:
            name=f"{spec['split']}_{spec['episode_id']}_{condition}"
            folder=run/'trials'/name;t=json.loads((folder/'trial.json').read_text())
            trace=json.loads((folder/'trace.json').read_text());n=len(trace);s=t['close_start']
            if (n!=s+48 or t['hidden_at_close_reset'] or len(t['hidden_reset_events'])!=1 or
                t['physics_steps']!=n*5 or t['hidden_steps']!=n):raise ValueError('trial lifecycle failure')
            with np.load(folder/'actions.npz',allow_pickle=False) as a:
                applied=a['applied'].copy();pred=a['predicted'].copy();stages=a['stage'].copy()
                np.testing.assert_array_equal(a['controller_input'],applied)
            if applied.shape!=(n,7) or applied.dtype!=np.float32 or not np.isfinite(applied).all():raise ValueError('action shape/dtype/finiteness')
            expected=pred[s:s+8].copy()
            if condition in ('expert','stop_arm'):expected[:,:6]=0
            if condition in ('expert','replace_gripper'):expected[:,6]=-1
            np.testing.assert_array_equal(applied[s:s+8],expected)
            np.testing.assert_array_equal(applied[s+8:,6],-np.ones(40,np.float32))
            np.testing.assert_array_equal(stages[s:],['close']*8+['lift']*20+['hold']*20)
            for step,r in enumerate(trace):
                if r['timestep']!=step or (r['physics_steps_before'],r['physics_steps_after'])!=(step*5,(step+1)*5):
                    raise ValueError('one action per50ms contract')
            with np.load(folder/'hidden.npz',allow_pickle=False) as a:
                h=a['hidden'].copy();np.testing.assert_array_equal(a['timestep'],np.arange(n))
            if h.shape!=(n,1,1,144) or not np.isfinite(h).all():raise ValueError('GRU hidden trace')
            with np.load(folder/'observations.npz',allow_pickle=False) as a:
                obs={k:a[k][:s+1].copy() for k in a.files}
            with np.load(folder/'masks.npz',allow_pickle=False) as a:masks=a['masks'][:s+1].copy()
            if ref is None:ref=(s,applied[:s],pred[:s+1],h[:s+1],obs,masks,t['reset_group'])
            else:
                if s!=ref[0] or t['reset_group']!=ref[6]:raise ValueError('paired reset/time differs')
                np.testing.assert_array_equal(applied[:s],ref[1]);np.testing.assert_allclose(pred[:s+1],ref[2],atol=c['atol'],rtol=c['rtol'])
                np.testing.assert_allclose(h[:s+1],ref[3],atol=c['atol'],rtol=c['rtol'])
                action_difference=max(action_difference,float(np.max(np.abs(pred[:s+1]-ref[2]))))
                hidden_difference=max(hidden_difference,float(np.max(np.abs(h[:s+1]-ref[3]))))
                for k in obs:np.testing.assert_array_equal(obs[k],ref[4][k])
                np.testing.assert_array_equal(masks,ref[5])
                for step in range(s+1):
                    with np.load(folder/f'graph_{step:04d}.npz',allow_pickle=False) as a, np.load(
                        run/'trials'/f"{spec['split']}_{spec['episode_id']}_expert"/f'graph_{step:04d}.npz',allow_pickle=False) as b:
                        for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    counts=dict(json=0,npz=0,png=0,mp4=0,mp4_frames=0,links=0)
    for p in run.rglob('*'):
        if not p.is_file():continue
        if p.suffix=='.json':json.loads(p.read_text());counts['json']+=1
        elif p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as a:
                for k in a.files:a[k]
            counts['npz']+=1
        elif p.suffix=='.png':
            if cv2.imread(str(p)) is None:raise ValueError('image decode: '+str(p))
            counts['png']+=1
        elif p.suffix=='.mp4':
            cap=cv2.VideoCapture(str(p));n=0
            if not np.isclose(cap.get(cv2.CAP_PROP_FPS),20):raise ValueError('simulation video must be 20FPS')
            while True:
                ok,_=cap.read()
                if not ok:break
                n+=1
            cap.release()
            if not n:raise ValueError('empty video')
            counts['mp4']+=1;counts['mp4_frames']+=n
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href'):
                    u=urlsplit(value)
                    if not u.scheme and u.path:
                        if not (run/unquote(u.path)).is_file():raise ValueError('missing relative link: '+value)
                        counts['links']+=1
    Links().feed((run/'grasp_intervention.html').read_text())
    return dict(status='PASS',decoded=counts,actual_trial_contracts=len(report['trials']),
        live_prefix_observations_masks_robot_exact=True,close_hidden_action_within_fixed_tolerance=True,
        maximum_prefix_action_absolute_difference=action_difference,maximum_prefix_hidden_absolute_difference=hidden_difference,
        interventions_and_applied_commands_exact=True,physics5_control1=True,
        scope='artifact readback, paired prefixes and actual action/physics/hidden trace contracts')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    report=json.loads((run/'results.json').read_text());protocol=report['protocol']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sections=[];summary=[];csv_rows=[]
    for spec in protocol['episodes']:
        name=f"{spec['split']}_{spec['episode_id']}"
        out=run/name;out.mkdir(exist_ok=True)
        trials={k:next(t for t in report['trials'] if t['episode_id']==spec['episode_id'] and t['condition']==k) for k in CONDITIONS}
        traces={k:json.loads((run/'trials'/f'{name}_{k}'/'trace.json').read_text()) for k in CONDITIONS}
        rgb={}
        for k in CONDITIONS:
            with np.load(run/'trials'/f'{name}_{k}'/'observations.npz',allow_pickle=False) as a:rgb[k]=a['rgb'].copy()
        fig,axes=plt.subplots(5,1,figsize=(10,11),sharex=True)
        for k in CONDITIONS:
            t=trials[k];start=t['close_start'];trace=traces[k][start:]
            x=np.arange(len(trace))
            axes[0].plot(x,[(r['after']['peg_pose'][2]-t['anchor']['peg_pose'][2])*100 for r in trace],label=k)
            axes[1].plot(x,[(r['after']['ee_pose'][2]-t['anchor']['ee_pose'][2])*100 for r in trace])
            axes[2].plot(x,[r['after']['grasp'] for r in trace])
            axes[3].plot(x,[r['applied_action'][6] for r in trace])
            axes[4].plot(x,[sum(r['after']['gripper_qpos'])*1000 for r in trace])
            for r in trace:
                entry=dict(split=spec['split'],episode_id=spec['episode_id'],condition=k,timestep=r['timestep'],stage=r['stage'],
                    grasp=r['after']['grasp'],peg_z_m=r['after']['peg_pose'][2],tcp_z_m=r['after']['ee_pose'][2],
                    peg_lift_m=r['after']['peg_pose'][2]-t['anchor']['peg_pose'][2],
                    tcp_lift_m=r['after']['ee_pose'][2]-t['anchor']['ee_pose'][2],gripper_opening_m=sum(r['after']['gripper_qpos']))
                for label,key in [('applied','applied_action'),('predicted','predicted_action')]:
                    entry.update({f'{label}_{i}':v for i,v in enumerate(r[key])})
                entry.update({f'finger_{side}_force_norm_n':float(np.linalg.norm(v)) for side,v in enumerate(r['after']['finger_forces'])})
                csv_rows.append(entry)
        for ax,label in zip(axes,['Peg elevation (cm)','TCP elevation (cm)','Bilateral grasp','Applied gripper','Finger opening sum (mm)']):
            ax.set_ylabel(label);ax.grid(alpha=.2);ax.axvspan(-.5,7.5,color='orange',alpha=.12)
            ax.axvline(7.5,color='grey',linestyle=':');ax.axvline(27.5,color='grey',linestyle=':')
        axes[0].axhline(4,color='grey',ls='--',alpha=.5);axes[1].axhline(6,color='grey',ls='--',alpha=.5)
        axes[0].legend(ncol=2);axes[-1].set_xlabel('Control step relative to close start (20Hz): close0..7 / lift8..27 / hold28..47')
        fig.suptitle(name+' — actual post-action states, mixed control diagnostic');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=135);plt.close(fig)
        # Post-action RGB, not the pre-action frame from the rollout overlay.
        writer=cv2.VideoWriter(str(out/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1024,1024))
        if not writer.isOpened():raise RuntimeError('video encoding unavailable')
        def tile(k,j):
            t=trials[k];step=t['close_start']+j;r=traces[k][step]
            im=cv2.resize(rgb[k][step+1],(512,512),interpolation=cv2.INTER_NEAREST)
            cv2.putText(im,k+' / '+r['stage'],(8,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            elev=(r['after']['peg_pose'][2]-t['anchor']['peg_pose'][2])*100
            cv2.putText(im,f"after t={step} peg+{elev:.2f}cm grasp={int(r['after']['grasp'])}",(8,47),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            return im
        try:
            for j in range(48):
                frames=[tile(k,j) for k in CONDITIONS]
                im=np.vstack((np.hstack(frames[:2]),np.hstack(frames[2:])))
                writer.write(cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
            montage=np.vstack([np.hstack([tile(k,j) for k in CONDITIONS]) for j in (7,47)])
            cv2.imwrite(str(out/'close_and_hold.png'),cv2.cvtColor(montage,cv2.COLOR_RGB2BGR))
        finally:writer.release()
        finding=inference(trials);summary.append(dict(spec,interpretation=finding,
            outcomes={k:trials[k]['outcome'] for k in CONDITIONS}))
        cells=[]
        for k in CONDITIONS:
            t=trials[k];o=t['outcome'];folder=f'trials/{name}_{k}'
            cells.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [k,o['retained'],o['close_end_grasp'],
                f"{o['minimum_hold_peg_lift_m']*100:.2f}",f"{o['hold_contact_fraction']:.2f}",t['close_presence_mean']])+f'<td><a href="{folder}/trace.json">trace</a> · <a href="{folder}/actions.npz">actions</a> · <a href="{folder}/rollout.mp4">full video</a></td></tr>')
        sections.append(f'<h2>{name}</h2><p>{finding}</p><table><tr><th>Condition</th><th>Retained</th><th>Grasp after close8</th><th>Min hold peg lift(cm)</th><th>Hold contact fraction</th><th>Close presence</th><th>Raw</th></tr>'+''.join(cells)+f'</table><img src="{name}/comparison.png"><video controls src="{name}/comparison.mp4"></video><p>Post-action RGB; top: expert/model, bottom: replace_gripper/stop_arm.</p><img src="{name}/close_and_hold.png">')
    write_json(run/'paired_findings.json',dict(status=report['status'],scope=report['scope'],episodes=summary))
    with (run/'intervention_frames.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    (run/'grasp_intervention.html').write_text('<!doctype html><meta charset="utf-8"><title>Live close-window intervention</title>'
        '<style>body{font:15px system-ui;margin:30px auto;max-width:1150px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #aaa}img,video{width:100%;margin-top:14px}pre{white-space:pre-wrap}</style>'
        '<h1>Live close-window intervention</h1><p>MIXED CONTROL DIAGNOSTIC — NOT POLICY SUCCESS RATE. Four fixed Train/Validation seeds, four paired conditions, fixed checkpoint. '
        'Fresh real Expert approach and live Vision/GRU history in every trial. Close8 only is treated; lift20/hold20 use a common robot servo and gripper=-1. '
        'Later full closing can recover a weak close; inspect close-end contact separately. Presence does not prove semantic Vision accuracy.</p>'
        '<p>Retention requires all20 hold observations: peg lift≥4cm, TCP lift≥6cm, and installed bilateral grasp contact. No insertion evaluation.</p>'
        '<a href="results.json">Raw results</a> · <a href="protocol.json">Frozen protocol</a> · <a href="paired_findings.json">Findings</a> · '
        '<a href="intervention_frames.csv">Frame CSV</a>'+''.join(sections))
    result=verify(run);write_json(run/'verification.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
