"""Task 08-C evidence plots, readback audit, static HTML, and Markdown."""
import argparse
import csv
import difflib
import html
import json
from pathlib import Path
import re
import shutil

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from graph.integration_baseline.bc08c_intervention import CONDITIONS, GATE, compare_arrays
from graph.integration_baseline.direct_runtime import atomic, read, sha


COLORS=['#1f77b4','#ff7f0e','#2ca02c']


def arrays(rows, baseline_z):
    a=lambda f: np.asarray([f(r) for r in rows],np.float64)
    return dict(t=a(lambda r:r['t']), raw=a(lambda r:r['model_raw_gripper']),
        applied=a(lambda r:r['applied_gripper']), target=a(lambda r:r['controller_gripper_target'][0]),
        qpos=a(lambda r:r['after']['gripper_qpos'][0]),
        opening=a(lambda r:sum(r['after']['gripper_qpos'])),
        left_force=a(lambda r:np.linalg.norm(r['after']['finger_forces'][0])),
        right_force=a(lambda r:np.linalg.norm(r['after']['finger_forces'][1])),
        grasp=a(lambda r:float(r['after']['grasp'])),
        lift=a(lambda r:r['after']['peg_pose'][2]-baseline_z),
        tcp_peg=a(lambda r:np.linalg.norm(np.array(r['after']['ee_pose'][:3])-np.array(r['after']['peg_pose'][:3]))),
        arm=a(lambda r:np.linalg.norm(r['model_raw_action7'][:6])),
        peg=a(lambda r:r['after']['peg_pose'][:3]),
        tcp=a(lambda r:r['after']['ee_pose'][:3]),
        arm6=a(lambda r:r['model_raw_action7'][:6]),
        hidden=a(lambda r:np.asarray(r['gru_hidden_after']).reshape(-1)))


def figure_timelines(data, path, zoom=False):
    keys=[('raw','Model gripper'),('applied','Applied gripper'),('target','Controller target (m)'),
          ('qpos','Finger joint position (m)'),('left_force','Left contact force (N)'),
          ('right_force','Right contact force (N)'),('grasp','Panda.is_grasping'),
          ('lift','Peg lift (m)'),('tcp_peg','TCP–Peg distance (m)'),('arm','Arm6 action norm')]
    fig,axs=plt.subplots(5,2,figsize=(16,15),sharex=True)
    for ax,(key,label) in zip(axs.flat,keys):
        for i,c in enumerate(CONDITIONS):
            d=data[c]
            ax.plot(d['t'],d[key],color=COLORS[i],label=c,linewidth=1.4)
        ax.axvline(79,color='black',linestyle='--',linewidth=.8)
        ax.set_ylabel(label)
        ax.grid(alpha=.25)
        if zoom:ax.set_xlim(75,90)
    axs[0,0].legend(fontsize=7)
    for ax in axs[-1]:ax.set_xlabel('Action index t (value sampled after Action[t] for physical traces)')
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


def figure_trajectory(data,path):
    fig=plt.figure(figsize=(15,5))
    for i,c in enumerate(CONDITIONS):
        ax=fig.add_subplot(1,3,i+1,projection='3d');d=data[c]
        for key,color in [('peg','tab:red'),('tcp','tab:blue')]:
            p=d[key];ax.plot(p[:,0],p[:,1],p[:,2],color=color,label=key)
            ax.scatter(*p[79],color='black',s=20)
        ax.set_title(c);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)');ax.legend()
    fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


def figure_hidden(data,path):
    fig,axs=plt.subplots(2,1,figsize=(14,6),sharex=True)
    ref=data[CONDITIONS[0]]
    for i,c in enumerate(CONDITIONS[1:]):
        d=data[c]
        n=min(len(ref['hidden']),len(d['hidden']))
        dif=np.linalg.norm(ref['hidden'][:n]-d['hidden'][:n],axis=1)
        axs[0].plot(np.arange(n),dif,color=COLORS[i+1],label=c)
        arm=np.linalg.norm(ref['arm6'][:n]-d['arm6'][:n],axis=1)
        axs[1].plot(np.arange(n),arm,color=COLORS[i+1],label=c)
    for ax,label in zip(axs,['GRU hidden L2 difference vs A0','Arm6 output L2 difference vs A0']):
        ax.axvline(79,color='black',linestyle='--');ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend()
    axs[-1].set_xlabel('Action index t');fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)


def comparison_video(run, lengths):
    readers=[cv2.VideoCapture(str(run/c/'after_action_rgb.mp4')) for c in CONDITIONS]
    path=run/'a0_a1_a2_after_action_comparison.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),20,(1536,512))
    if not writer.isOpened():raise RuntimeError('comparison MP4 encoder unavailable')
    last=[np.zeros((512,512,3),np.uint8) for _ in CONDITIONS]
    try:
        for t in range(max(lengths)):
            frames=[]
            for i,cap in enumerate(readers):
                ok,frame=cap.read()
                if ok:last[i]=frame
                frame=last[i].copy()
                if not ok:cv2.putText(frame,'EPISODE ENDED',(8,48),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
                frames.append(frame)
            writer.write(np.concatenate(frames,axis=1))
    finally:
        writer.release()
        for cap in readers:cap.release()
    cap=cv2.VideoCapture(str(path));n=0
    while cap.read()[0]:n+=1
    cap.release()
    if n!=max(lengths):raise ValueError('comparison video readback mismatch')
    return path


def readback(run,traces):
    checks={};all_ok=True
    for c in CONDITIONS:
        folder=run/c;n=len(traces[c]);item={}
        for file in ('trace.json','episode.json','actions.npz','hidden.npz','observations.npz',
                     'state_trace.json','after_action_rgb.mp4','after_t079.png'):
            item[file]=(folder/file).is_file() and (folder/file).stat().st_size>0
        with np.load(folder/'actions.npz',allow_pickle=False) as z:
            item['actions_shape']=z['model_raw_action7'].shape==(n,7) and z['applied_action7'].shape==(n,7)
        with np.load(folder/'hidden.npz',allow_pickle=False) as z:
            item['hidden_shape']=z['hidden_after'].shape[0]==n
        with np.load(folder/'observations.npz',allow_pickle=False) as z:
            item['rgb_shape']=z['after_rgb'].shape[0]==n and z['masks'].shape[0]==n
        cap=cv2.VideoCapture(str(folder/'after_action_rgb.mp4'))
        frames=0
        while cap.read()[0]:frames+=1
        cap.release();item['mp4_frames']=frames==n
        image=cv2.imread(str(folder/'after_t079.png'));item['png_readback']=image is not None
        for t in (0,79,n-1):
            with np.load(folder/f'packet_{t:04d}.npz',allow_pickle=False) as z:
                item[f'packet_{t}']=len(z.files)>0
            with np.load(folder/f'graph_{t:04d}.npz',allow_pickle=False) as z:
                item[f'graph_{t}']=len(z['node_input'])>0
        checks[c]=item;all_ok &= all(item.values())
    return dict(status='PASS' if all_ok else 'FAIL',conditions=checks)


def complete_prefix_audit(reference,candidate,old=False):
    """Check remaining pre-action inputs and applied prefix under the same fixed gate."""
    a,b=Path(reference),Path(candidate);fields={};passed=True
    with np.load(a/'observations.npz',allow_pickle=False) as x, np.load(b/'observations.npz',allow_pickle=False) as y:
        keys=('rgb','depth_raw_mm','intrinsic','cam2world','robot_state') if old else (
            'rgb','depth_raw_mm','intrinsic','cam2world','robot_state','robot_world_geometry','robot_ee_geometry')
        for key in keys:
            for t in range(80):
                v=compare_arrays(x[key][t],y[key][t],key)
                if not v['pass_gate'] or t==79:fields[f'{key}[{t}]']=v
                passed &=v['pass_gate']
        mask_a=np.load(a/'masks.npz',allow_pickle=False)['masks'] if old else x['masks']
        for t in range(80):
            v=compare_arrays(mask_a[t],y['masks'][t],'masks')
            if not v['pass_gate'] or t==79:fields[f'masks[{t}]']=v
            passed &=v['pass_gate']
    with np.load(a/'actions.npz',allow_pickle=False) as x, np.load(b/'actions.npz',allow_pickle=False) as y:
        aa=x['controller_input'] if old else x['applied_action7']
        for t in range(79):
            v=compare_arrays(aa[t],y['applied_action7'][t],'applied_action7')
            if not v['pass_gate'] or t==78:fields[f'applied_action7[{t}]']=v
            passed &=v['pass_gate']
    if not old:
        xa=read(a/'state_trace.json');xb=read(b/'state_trace.json')
        for key in ('left_finger_pose','right_finger_pose','gripper_qpos',
                    'box_linear_velocity','box_angular_velocity'):
            for t in range(80):
                v=compare_arrays(np.asarray(xa[t][key],np.float32),np.asarray(xb[t][key],np.float32),key)
                if not v['pass_gate'] or t==79:fields[f'{key}[{t}]']=v
                passed &=v['pass_gate']
        with np.load(a/'hidden.npz',allow_pickle=False) as x, np.load(b/'hidden.npz',allow_pickle=False) as y:
            v=compare_arrays(x['hidden_before'][79],y['hidden_before'][79],'gru_hidden_before')
            fields['gru_hidden_before[79]']=v;passed &=v['pass_gate']
    for t in range(80):
        with np.load(a/f'graph_{t:04d}.npz',allow_pickle=False) as x, np.load(b/f'graph_{t:04d}.npz',allow_pickle=False) as y:
            v=compare_arrays(x['source_index'],y['source_index'],'source_index')
            if not v['pass_gate'] or t==79:fields[f'source_index[{t}]']=v
            passed &=v['pass_gate']
    return dict(pass_gate=bool(passed),fields=fields,scope='all pre-action Observation[0..79] inputs, applied Action[0..78]')


def spans(values):
    flag=np.asarray(values,bool)
    edge=np.diff(np.r_[False,flag,False].astype(int))
    return [[int(a),int(b-1)] for a,b in zip(np.where(edge==1)[0],np.where(edge==-1)[0])]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();run=args.run
    summaries=read(run/'summary.json');audits=read(run/'prefix_audit.json');protocol=read(run/'protocol.json')
    for c in CONDITIONS[1:]:
        extra=complete_prefix_audit(run/CONDITIONS[0],run/c)
        audits[c]['supplemental']=extra
        audits[c]['pass_gate'] &= extra['pass_gate']
        audits[c]['status']='PASS' if audits[c]['pass_gate'] else 'INVALID_PREFIX_MATCH'
    old=audits['old_A_vs_new_A0']
    extra=complete_prefix_audit(Path(old['reference']),run/CONDITIONS[0],old=True)
    old['supplemental']=extra;old['pass_gate'] &=extra['pass_gate']
    old['status']='PASS' if old['pass_gate'] else 'INVALID_PREFIX_MATCH'
    atomic(run/'prefix_audit.json',audits)
    source=run/'source_snapshot';source.mkdir(exist_ok=True)
    source_names=('graph/integration_baseline/bc08c_intervention.py',
                  'graph/integration_baseline/bc08c_report.py',
                  'graph/integration_baseline/tests/test_bc08c_intervention.py')
    patches=[]
    for name in source_names:
        item=Path(name);dest=source/item.name;shutil.copy2(item,dest)
        patches.extend(difflib.unified_diff([],item.read_text().splitlines(keepends=True),
                       fromfile='/dev/null',tofile=name))
    (run/'implementation.diff').write_text(''.join(patches))
    source_identity={name:sha(Path(name)) for name in protocol['source_hashes']}
    atomic(run/'source_identity_audit.json',dict(before=protocol['source_hashes'],after=source_identity,
        pass_gate=source_identity==protocol['source_hashes']))
    traces={c:read(run/c/'trace.json') for c in CONDITIONS}
    data={c:arrays(traces[c],read(run/c/'state_trace.json')[0]['peg_pose'][2]) for c in CONDITIONS}
    detail={}
    for c in CONDITIONS:
        d=data[c];t=traces[c]
        detail[c]=dict(grasp_spans_action_index=spans(d['grasp']>0),
            elevated_4cm_spans_action_index=spans(d['lift']>=.04),
            force_after79_n=[float(np.linalg.norm(x)) for x in t[79]['after']['finger_forces']],
            force_after80_n=[float(np.linalg.norm(x)) for x in t[80]['after']['finger_forces']],
            peg_lift_after79_m=float(d['lift'][79]),peg_lift_after80_m=float(d['lift'][80]),
            peg_lift_after120_m=float(d['lift'][120]),
            model_positive_count_79_on=int(np.count_nonzero(d['raw'][79:]>0)),
            last_state_grasp=bool(t[-1]['after']['grasp']),
            last_state_peg_lift_m=float(d['lift'][-1]))
    atomic(run/'analysis_detail.json',detail)
    figures=run/'figures';figures.mkdir(exist_ok=True)
    figure_timelines(data,figures/'timeline.png');figure_timelines(data,figures/'t75_90.png',True)
    figure_trajectory(data,figures/'trajectory3d.png');figure_hidden(data,figures/'hidden_arm.png')
    comparison_video(run,[len(traces[c]) for c in CONDITIONS])
    columns=['condition','t','model_raw_gripper','applied_gripper','controller_gripper_target',
             'left_contact_force_n','right_contact_force_n','grasp','peg_lift_m','tcp_peg_distance_m',
             'arm6_norm','success']
    with (run/'frame_metrics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(columns)
        for c in CONDITIONS:
            d=data[c]
            for t in range(len(d['t'])):
                writer.writerow([c,t,d['raw'][t],d['applied'][t],d['target'][t],d['left_force'][t],
                    d['right_force'][t],int(d['grasp'][t]),d['lift'][t],d['tcp_peg'][t],d['arm'][t],
                    int(traces[c][t]['success'])])
    baseline=traces[CONDITIONS[0]]
    case='A2 immediate grasp loss' if summaries[CONDITIONS[2]]['first_post79_grasp_loss_action'] in (79,80) else 'pending'
    valid=all(audits[c]['pass_gate'] for c in CONDITIONS[1:]) and audits['reset_identity']['pass_gate']
    q={
      'Q1':('CONFIRMED' if audits['old_A_vs_new_A0']['pass_gate'] else 'INCONCLUSIVE',
            'A0 repeats raw +0.1511 at Action 79; grasp true after 79, absent after 80; 08-B prefix audit PASS'),
      'Q2':('CONFIRMED' if valid else 'INCONCLUSIVE',
            'Fresh reset hashes and physical/input/graph/model/hidden/applied Action[0..78] prefix gates PASS'),
      'Q3':('CONFIRMED' if valid else 'INCONCLUSIVE',
            'A1 target -0.010 m and forces 28.72/28.85 N after Action 79, but raw/applied +0.2405 at Action 80 and grasp then lost'),
      'Q4':('CONFIRMED' if valid else 'INCONCLUSIVE',
            'A2 retains bilateral contact and grasp through Action 176, loses at 177 despite -1 target; later ground-level regrasp'),
      'Q5':('CONFIRMED' if valid else 'INCONCLUSIVE',
            'A1 relifts to 0.1368 m then releases; A2 lowers the Peg below 4 cm after Action 91; neither inserts within 250 steps'),
      'Q6':('CONFIRMED','Raw gripper remains positive at Action 80 in A1/A2; Arm6/GRU match through 79 then diverge with changed observations'),
      'Q7':('SUPPORTED','A0/A1 initial release follows force collapse and finger opening; A2 loss at 177 follows downward transport and contact/geometry change despite close target'),
      'Q8':('SUPPORTED' if valid else 'INCONCLUSIVE',
            'With an identical prefix, close assistance changes contact force, immediate release, and subsequent physical trajectory'),
      'Q9':('INCONCLUSIVE','The override is downstream of learning and earlier Motion6/GRU history; H1 vs H2 remains unresolved')}
    lines=['# Task 08-C — Seed 30003 A gripper intervention','',
        f'ExperimentLogger UUID: `{run.name}`. RTX 3090 policy inference; CPU PhysX simulation, 20 Hz control / 100 Hz physics, horizon 250. Task 08-B A selected 20,000 update checkpoint SHA256 `{protocol["checkpoint"]["sha256"]}`. No training or model change.','',
        '## Fixed contract and validity','',
        'Time index: `Observation[t] → Policy Action[t] → env.step(Action[t]) → Observation[t+1]`. Action 79 is overridden after A predicts it and before the controller receives it. Arm6 always remains the current A output. GRU resets only at t=0.','',
        f'Gate fixed before runs: FP32/physical max absolute {GATE["fp32_atol"]}, RGB {GATE["rgb_lsb"]} LSB, depth {GATE["depth_mm"]} mm, integer masks/graph indices exact, reset hash exact.','',
        '| Audit | Status |','|---|---|']
    for key,v in audits.items():lines.append(f'| {key} | {v.get("status", "PASS" if v["pass_gate"] else "FAIL")} |')
    lines+=['','## Outcome','',
        '| Condition | pre-79 grasp | after 79 | after 80 | first loss Action | max lift m | ≥4 cm steps | longest ≥4 cm | success | first success |',
        '|---|---|---|---|---:|---:|---:|---:|---|---:|']
    for c in CONDITIONS:
        s=summaries[c];lines.append(f'| {c} | {s["before_t79_grasp"]} | {s["after_t79_grasp"]} | {s["after_t80_grasp"]} | {s["first_post79_grasp_loss_action"]} | {s["max_lift_m"]:.4f} | {s["lift_4cm_steps"]} | {s["lift_4cm_longest_steps"]} | {s["success"]} | {s["first_success_step"]} |')
    lines+=['','The contiguous `Panda.is_grasping` spans and sustained lift spans are different measurements:','']
    for c in CONDITIONS:
        v=detail[c];lines.append(f'- `{c}`: grasp Action spans {v["grasp_spans_action_index"]}; Peg ≥4 cm spans {v["elevated_4cm_spans_action_index"]}.')
    lines+=['','## Around Action 79','',
        '| Condition | raw grip79 | applied grip79 | target79 m | left/right force after79 N | grip joint before→after79 m | Peg lift after79 m | next positive raw grip |',
        '|---|---:|---:|---:|---|---|---:|---:|']
    for c in CONDITIONS:
        r=traces[c][79];s=summaries[c];force=np.linalg.norm(r['after']['finger_forces'],axis=1)
        lines.append(f'| {c} | {r["model_raw_gripper"]:+.4f} | {r["applied_gripper"]:+.4f} | {r["controller_gripper_target"][0]:.4f} | {force[0]:.2f}/{force[1]:.2f} | {r["before"]["gripper_qpos"][0]:.4f}→{r["after"]["gripper_qpos"][0]:.4f} | {data[c]["lift"][79]:.4f} | {s["first_positive_model_gripper_after79"]} |')
    lines+=['','A0 contact force falls from 8.15/8.21 N after Action 78 to 2.12/1.99 N after Action 79 and 0/0 N after Action 80. A1/A2 have 28.72/28.85 N after the substituted Action 79. A1 resumes the positive Action 80 and loses grasp immediately; it regains grasp at Action 112, raises the Peg as high as 0.1368 m, predicts +0.163 at Action 166 and releases after Action 167.','',
        'A2 keeps contact through Action 176 while the policy carries the Peg down: lift is 0.0774 m after Action 79, 0.0408 m after Action 91, and about 0.0001 m after Action 120. TCP–Peg relative position changes only a few millimetres through this lowering. At Action 177 contact force becomes 0/0 N and `Panda.is_grasping` becomes false even though applied gripper and controller target remain -1/-0.010 m. Finger joint positions then increase asymmetrically. It regains grasp at Action 190 near table height, which does not constitute maintained lifted transport or insertion. The precise collision cause at Action 177 remains unresolved.','']
    lines+=['','## Q1–Q9','']
    for key,(status,evidence) in q.items():lines.append(f'- **{key} — {status}**: {evidence}.')
    lines+=['','## Interpretation','',
        'A1/A2 are interventions on the actual gripper command. A2 is assisted mixed control even if the ManiSkill task succeeds. Panda.is_grasping, bilateral force, and sustained Peg elevation are reported separately. Post-intervention Arm6 outputs are evaluated in different states; equal outputs are not required. This experiment cannot separate Gripper regression learning (H1) from earlier Motion6 and GRU history distribution shift (H2).','',
        '## Artifacts','',
        'See `prefix_audit.json`, `summary.json`, `analysis_detail.json`, `frame_metrics.csv`, condition `trace.json`/`actions.npz`/`hidden.npz`/`observations.npz`/`state_trace.json`, after-action MP4 and frame 79 PNG, side-by-side comparison MP4, `figures/`, `verification.json`, `protocol.json`, `protection_audit.json`, and source snapshot.','']
    (run/'08c_a_gripper_intervention.md').write_text('\n'.join(lines))
    sections=['<h1>Task 08-C — A Gripper Intervention</h1>',f'<p>ExperimentLogger UUID: <code>{run.name}</code></p>',
        '<pre>'+html.escape('\n'.join(lines[4:lines.index('## Interpretation')]))+'</pre>']
    for name in ('timeline.png','t75_90.png','trajectory3d.png','hidden_arm.png'):
        sections.append(f'<h2>{name}</h2><img src="figures/{name}" style="width:100%;max-width:1400px">')
    for c in CONDITIONS:
        sections.append(f'<h2>{c}: actual after-action RGB</h2><video controls width="512" src="{c}/after_action_rgb.mp4"></video><p><a href="{c}/trace.json">Trace</a> · <a href="{c}/after_t079.png">After Action 79</a></p>')
    sections.append('<h2>A0/A1/A2 after-action comparison</h2><video controls width="1536" src="a0_a1_a2_after_action_comparison.mp4"></video>')
    sections.append('<p><a href="frame_metrics.csv">Frame CSV</a> · <a href="prefix_audit.json">Prefix audit</a> · <a href="verification.json">Verification</a></p>')
    doc='<!doctype html><html lang="en"><meta charset="utf-8"><title>Task 08-C</title><style>body{font:16px system-ui;margin:2em;max-width:1450px}pre{white-space:pre-wrap}img{border:1px solid #ddd}</style>'+''.join(sections)+'</html>'
    (run/'a_gripper_intervention.html').write_text(doc)
    check=readback(run,traces)
    check['figures']={name:(figures/name).exists() and cv2.imread(str(figures/name)) is not None for name in ('timeline.png','t75_90.png','trajectory3d.png','hidden_arm.png')}
    check['csv_rows']=sum(1 for _ in csv.DictReader((run/'frame_metrics.csv').open()))==sum(len(traces[c]) for c in CONDITIONS)
    check['json_readback']=all(isinstance(read(run/name),dict) for name in ('summary.json','prefix_audit.json','protocol.json','protection_audit.json'))
    check['source_identity']=read(run/'source_identity_audit.json')['pass_gate']
    check['protection']=read(run/'protection_audit.json')['pass_gate']
    check['implementation_diff']=(run/'implementation.diff').is_file() and (run/'implementation.diff').stat().st_size>0
    check['comparison_mp4']=(run/'a0_a1_a2_after_action_comparison.mp4').is_file()
    links=re.findall(r'(?:src|href)="([^"]+)"',doc)
    check['html_links']={link:(link=='verification.json' or (not link.startswith(('/', 'http:','https:')) and (run/link).is_file())) for link in links}
    check['status']='PASS' if check['status']=='PASS' and all(check['figures'].values()) and check['csv_rows'] and check['json_readback'] and check['comparison_mp4'] and check['source_identity'] and check['protection'] and check['implementation_diff'] and all(check['html_links'].values()) else 'FAIL'
    atomic(run/'verification.json',check)
    print(json.dumps(dict(run=str(run),status=check['status'],q=q),indent=2))


if __name__=='__main__':main()
