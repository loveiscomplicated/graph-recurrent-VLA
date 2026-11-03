"""Analyze saved traces only; no policy/optimizer or selection changes."""
import argparse
import csv
import html
import math
from pathlib import Path
import shutil
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write,sha,now
from graph.graph_recurrent10g.eval_runner import csv_write


def summarize(root):
    cfg=read(root/'config.json');seedman=read(root/'final_seed_manifest.json')
    rows=[read(p) for p in sorted((root/'rollouts/final').glob('seed*/*/episode.json'))]
    assert len(rows)==seedman['episodes']==120
    assert {(r['seed'],r['condition']) for r in rows}=={(s,c) for s in seedman['parents'] for c in seedman['conditions']}
    traces={(r['seed'],r['condition']):read(root/r['folder']/'trace.json') for r in rows}
    paired=[];drift=[];action=[];prediction=[];controls=[];latency=[];anchors=[]
    for seed in seedman['parents']:
        for suffix in '012':
            rr={family:next(r for r in rows if r['seed']==seed and r['condition']==family+suffix) for family in 'SRJ'}
            paired.append(dict(seed=seed,mode=suffix,S_insertion=rr['S']['insertion'],R_insertion=rr['R']['insertion'],J_insertion=rr['J']['insertion'],
                S_minus_R=int(rr['S']['insertion'])-int(rr['R']['insertion']),S_minus_J=int(rr['S']['insertion'])-int(rr['J']['insertion']),
                **{f'{f}_{metric}':rr[f][metric] for f in 'SRJ' for metric in ['grasp','lift','alignment','actions','handoff_timestep','valid_anchor','completion_steps']}))
    for row in rows:
        tr=traces[row['seed'],row['condition']];student=[t for t in tr if t['control']=='B']
        for t in tr:
            if 'state_prediction_errors' in t:
                prediction.append(dict(seed=row['seed'],condition=row['condition'],timestep=t['timestep'],phase=t['phase'],
                    handoff_relative_step=t['handoff_relative_step'],missing_peg=t['missing_objects'][0],missing_hole=t['missing_objects'][1],
                    lateral_error_m=float(np.linalg.norm(t['peg_head_pos_at_hole'][1:])),
                    **t['state_prediction_errors']))
        for phase in sorted({t['phase'] for t in student}):
            ts=[t for t in student if t['phase']==phase]
            action.append(dict(seed=row['seed'],condition=row['condition'],phase=phase,frames=len(ts),
                mean_action_mae=float(np.mean([t['action_difference_mae'] for t in ts])),
                direction_error_frames=sum(t['action_direction_error'] for t in ts)))
        controls.append(dict(seed=row['seed'],condition=row['condition'],student_frames=len(student),
            arm_saturation_frames=sum(t['arm_saturation'] for t in student),
            premature_open_frames=sum(t['action'][6]>0 and t['pre']['grasp'] for t in student),
            premature_close_frames=sum(t['action'][6]<0 and t['pre']['tcp_peg_distance_m']>.03 and not t['pre']['grasp'] for t in student),
            grasp_loss_frames=sum(t['pre']['grasp'] and not t['grasp'] for t in student),
            geometry_missing_frames=sum(any(t['missing_objects']) for t in student),
            controller_failure=any(not np.isfinite(t['action']).all() for t in tr),failure_labels=';'.join(row['failure_labels'])))
        latency.append(dict(seed=row['seed'],condition=row['condition'],
            **{key:float(np.mean([t[key] for t in tr])) for key in ['inference_seconds','policy_seconds','gnn_encoder_readout_seconds','simulator_step_wall_seconds']},
            simulator_control_dt_seconds=.05,wallclock_realtime_claim=False))
        if row['handoff_timestep'] is not None:
            first=student[:10];base=float(np.linalg.norm(first[0]['pre']['peg_head_pos_at_hole'][1:]))
            for t in student:
                vector=np.asarray(t['peg_head_pos_at_hole']);pre=np.asarray(t['pre']['peg_head_pos_at_hole'])
                drift.append(dict(seed=row['seed'],condition=row['condition'],relative_step=t['handoff_relative_step'],
                    insertion=row['insertion'],lateral_error_m=float(np.linalg.norm(vector[1:])),axial_position_m=float(vector[0]),
                    initial_lateral_m=base,lateral_drift_m=float(np.linalg.norm(vector[1:]))-base,
                    arm_saturation=t['arm_saturation'],gripper_action=t['action'][6],action_x=t['action'][0],action_y=t['action'][1],action_z=t['action'][2],
                    predicted_tip_vector_error_mm=t.get('state_prediction_errors',{}).get('peg_tip_hole')))
            anchors.append(dict(seed=row['seed'],condition=row['condition'],handoff_timestep=row['handoff_timestep'],
                initial_lateral_m=base,axial_distance_m=float(first[0]['pre']['peg_head_pos_at_hole'][0]),
                first10_steps=len(first),first10_lateral_drift_m=float(np.linalg.norm(first[-1]['peg_head_pos_at_hole'][1:]))-base,
                insertion=row['insertion']))
    for name,data in [('paired_policy_comparison',paired),('phasewise_action_error',action),('state38_prediction_error',prediction),
         ('handoff_lateral_drift',drift),('handoff_anchor_diagnostics',anchors),('arm_saturation_gripper_analysis',controls),('latency_metrics',latency)]:
        if data:csv_write(root/(name+'.csv'),data)
        else:
            columns={'handoff_lateral_drift':['seed','condition','relative_step','insertion','lateral_error_m','axial_position_m','initial_lateral_m','lateral_drift_m','arm_saturation','gripper_action','action_x','action_y','action_z','predicted_tip_vector_error_mm'],
                     'handoff_anchor_diagnostics':['seed','condition','handoff_timestep','initial_lateral_m','axial_distance_m','first10_steps','first10_lateral_drift_m','insertion']}.get(name,['seed','condition'])
            with (root/(name+'.csv')).open('w',newline='') as stream:csv.DictWriter(stream,fieldnames=columns).writeheader()
    summary={c:{m:sum(bool(r[m]) for r in rows if r['condition']==c) for m in ['grasp','lift','alignment','insertion']} for c in seedman['conditions']}
    comparison={}
    for suffix in '012':
        selected=[r for r in paired if r['mode']==suffix];wins=sum(r['S_minus_R']>0 for r in selected);losses=sum(r['S_minus_R']<0 for r in selected);n=wins+losses
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
        comparison[suffix]=dict(S_gained_vs_R=wins,S_lost_vs_R=losses,exact_discordant_two_sided_nominal_p=p,
            interpretation='exploratory paired parent analysis; no multiplicity adjustment')
    write(root/'paired_summary.json',dict(functional=summary,paired=comparison,denominator_per_condition=12))
    write(root/'analysis_contract.json',dict(state_prediction_error_units='F1 errors API: position/shape/tip-vector mm; orientation deg',
        prediction_frame='pre-action, matching actual sensor graph; GT scoring only after student forward',
        lateral_drift='post-action hole-frame yz norm minus handoff pre-action yz norm, m',
        teacher_action_error='same-state GT shadow Expert diagnostic; not pure perception error',
        paired_unit='parent, not individual frames',completion_steps='actual actions until official success; handoff subtracts prefix'))
    fig=root/'figures';fig.mkdir(exist_ok=True)
    labels=seedman['conditions'];x=np.arange(len(labels));plt.figure(figsize=(12,4))
    for j,key in enumerate(['grasp','lift','alignment','insertion']):plt.bar(x+(j-1.5)*.2,[summary[c][key]/12 for c in labels],width=.2,label=key)
    plt.xticks(x,labels);plt.ylim(0,1.05);plt.ylabel('Success fraction (12 parents)');plt.legend();plt.tight_layout();plt.savefig(fig/'functional_success.png');plt.close()
    for suffix in '12':
        plt.figure(figsize=(9,5))
        for family,color in zip('SRJ',['tab:blue','tab:orange','tab:green']):
            rr=[r for r in drift if r['condition']==family+suffix]
            for seed in seedman['parents']:
                ts=[r for r in rr if r['seed']==seed]
                if ts:plt.plot([r['relative_step'] for r in ts],[1000*r['lateral_error_m'] for r in ts],color=color,alpha=.2)
            plt.plot([],[],color=color,label=family)
        plt.xlabel('Actions after H'+suffix);plt.ylabel('Peg-tip lateral error (mm)');plt.legend();plt.tight_layout();plt.savefig(fig/('H'+suffix+'_lateral_trajectory.png'));plt.close()
    plt.figure(figsize=(9,4))
    for j,suffix in enumerate('12'):
        for k,family in enumerate('SRJ'):
            yy=[1000*r['first10_lateral_drift_m'] for r in anchors if r['condition']==family+suffix]
            plt.scatter(np.full(len(yy),j*4+k),yy,label=family+' H'+suffix)
    plt.xticks([0,1,2,4,5,6],['S H1','R H1','J H1','S H2','R H2','J H2']);plt.ylabel('First10 lateral drift (mm)');plt.tight_layout();plt.savefig(fig/'first10_lateral_drift.png');plt.close()
    plt.figure(figsize=(9,5))
    for family,color in zip('SRJ',['tab:blue','tab:orange','tab:green']):
        ts=[r for r in drift if r['condition']==family+'2' and r['relative_step']<10]
        if ts:plt.plot([np.mean([r['action_y'] for r in ts if r['relative_step']==t]) for t in range(10)],label=family+' action Y',color=color)
    plt.xlabel('H2 first10 action index');plt.ylabel('Native normalized action Y');plt.legend();plt.tight_layout();plt.savefig(fig/'first10_actions.png');plt.close()
    plt.figure(figsize=(7,5));ts=[r for r in drift if r['predicted_tip_vector_error_mm'] is not None and r['relative_step']<10]
    plt.scatter([r['predicted_tip_vector_error_mm'] for r in ts],[r['lateral_drift_m']*1000 for r in ts],s=8,alpha=.5)
    plt.xlabel('F1 tip-vector prediction error (mm)');plt.ylabel('Control lateral drift (mm)');plt.tight_layout();plt.savefig(fig/'prediction_vs_control.png');plt.close()
    counts={}
    for family,path in [('S',root/'best.pt')]+[(k,Path(v['path'])) for k,v in read(root/'comparison_policy_manifest.json').items()]:
        weights=torch.load(path,map_location='cpu',weights_only=False)['model']
        counts[family]=dict(total=sum(v.numel() for v in weights.values()),controller=sum(v.numel() for k,v in weights.items() if k.startswith('policy.')),
            total_includes_buffers=True,input_dimension=38 if family=='S' else 398)
    selection=read(root/'development_selection.json');lines=[
        '# Task 10-I — Predicted State38 → New GRU → Action7','',
        'All preregistered 120 independent Final episodes completed. Primary comparison is S vs R; J is a practical joint-BC baseline.','',
        'S freezes the original F1 GNN, structured readout and 398→256→128→18 state head. Only a fresh 38→128 Projection, GRU144 and 144→128→7 head were trained. No GT object state or GT sidecar is a policy input.','',
        f"Training used seed42, 11,487 optimizer updates, batch2/sequence32, original Task10G schedule and Method B loss. S checkpoint selected at step {selection['selected_step']} using the original six Development seeds and rank rule, before Final access.", '',
        '| Condition | Grasp | Lift | Alignment | Official insertion |','|---|---:|---:|---:|---:|']
    for c in labels:lines.append('| '+c+' | '+' | '.join(str(summary[c][m])+'/12' for m in ['grasp','lift','alignment','insertion'])+' |')
    lines+=['','H1/H2 grasp and lift include Expert prefix achievements. H2 is the original align→insert boundary; it is not a near-contact anchor. Anchor-unreachable parents remain in the denominator.','',
        '## Paired interpretation','']
    for suffix in '012':
        cc=comparison[suffix];lines.append(f"Mode {suffix}: S gained {cc['S_gained_vs_R']} / lost {cc['S_lost_vs_R']} insertion successes against R; nominal exact paired p={cc['exact_discordant_two_sided_nominal_p']:.6g}.")
    sa,ra=summary['S0']['insertion'],summary['R0']['insertion']
    lines+=['',('S achieved higher observed autonomous insertion than R on these parents.' if sa>ra else
                'No autonomous insertion advantage over R was observed on these parents.' if sa==ra else
                'S achieved lower observed autonomous insertion than R on these parents.')]
    if sa>0:
        lines+=['S autonomous successes provide limited evidence that the frozen explicit-state path can support end-to-end control for some registered parents. They do not establish general superiority.']
    if summary['S2']['insertion']>0 and summary['S1']['insertion']==0:
        lines+=['S succeeded on some original H2 handoffs but no H1 handoffs. This supports some axial approach/insertion ability after Expert alignment, while alignment from the post-lift state remains unresolved.']
    lines+=['','First10 H2 lateral drift, in mm (parent mean):','']
    for family in 'SRJ':
        vals=[r['first10_lateral_drift_m']*1000 for r in anchors if r['condition']==family+'2']
        lines.append(f"- {family}: {np.mean(vals):.4f} mm across {len(vals)} valid anchors." if vals else f'- {family}: no valid H2 anchor; undefined.')
    lines+=['','F1 prediction errors at the actual H1/H2 handoff observation (parent means, mm):','',
        '| Anchor | Peg relative position | Hole relative position | Tip–hole vector |',
        '|---|---:|---:|---:|']
    for c in ['S1','S2']:
        vals=[r for r in prediction if r['condition']==c and r['handoff_relative_step']==0]
        if vals:
            lines.append('| '+c+' | '+' | '.join(f"{np.mean([r[k] for r in vals]):.4f}" for k in ['peg_position','hole_position','peg_tip_hole'])+' |')
    lines+=['','A difference on these 12 parents and one selected checkpoint does not establish general superiority. Similar low performance does not separate F1 prediction error, information loss through the bottleneck, BC fit, and history distribution effects. J also changes GNN training, so S–J is not a pure representation comparison. Prediction and control-error co-occurrence is descriptive, not a causal attribution.','',
        '## Reproducibility and scope','',
        'Normalizer was fitted solely on Train predicted State38, before Projection; quaternion std floor0.01, position/shape std floor1mm, Clean constants use mean[1,0]/std[1,1]. No additional normalizer experiments, head fine-tuning, GNN training, new expert collection, probe fitting or GT correction were performed.','',
        'Controller parameter counts and input dimensions: `'+str(counts)+'`. Frozen F1 parameters/buffers remained identical before/after BC. Original source/checkpoints/data identities are rechecked in completion verification.','',
        'Executed on Vast.ai58752018 RTX3090: CPU PhysX + CUDA renderer/vision/policy, control20Hz, physics100Hz, horizon250 including prefix. GPU PhysX and macOS CUDA were not executed. Reported wall inference latency is distinct from simulator dt; real-time20Hz is not claimed.','',
        'One rollout process runs at a time. tmux survives SSH disconnect but does not guarantee survival of instance stop/deletion/recycle. No Codex watcher/callback, Git commit or external publishing was used.','',
        '## Evidence and media','',
        '[Preflight](preflight_verification.json) · [Selection](development_selection.json) · [Paired results](paired_policy_comparison.csv) · [State prediction error](state38_prediction_error.csv) · [Handoff drift](handoff_lateral_drift.csv) · [Anchors](handoff_anchor_diagnostics.csv) · [Latency](latency_metrics.csv)','']
    for p in sorted(fig.glob('*.png')):lines+=['!['+p.stem+'](figures/'+p.name+')','']
    for c in labels:lines.append(f"[{c} representative video](rollouts/final/seed{seedman['representative_parent']}/{c}/rollout.mp4)")
    # Report concrete first observed failures without changing representative selection.
    for family in 'SRJ':
        failed=next((r for r in rows if r['condition']==family+'2' and not r['insertion']),None)
        if failed:lines.append(f"[{family} first H2 failure]({failed['folder']}/rollout.mp4)")
    name='10i_predicted_state38_graph_recurrent_bc.md';content='\n'.join(lines)+'\n';(root/name).write_text(content)
    docs=root/'docs/visual-graph-integration';docs.mkdir(parents=True,exist_ok=True)
    doc=content
    for prefix in ['figures/','rollouts/']:
        doc=doc.replace(']('+prefix,'](../../'+prefix)
    for p in root.glob('*.json'):doc=doc.replace(']('+p.name+')','](../../'+p.name+')')
    for p in root.glob('*.csv'):doc=doc.replace(']('+p.name+')','](../../'+p.name+')')
    (docs/name).write_text(doc)
    (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Task10-I</title><style>body{max-width:1100px;margin:40px auto;font-family:system-ui}pre{white-space:pre-wrap}img{max-width:100%}</style><pre>'+html.escape(content)+'</pre>'+''.join('<img src="figures/'+p.name+'">' for p in sorted(fig.glob('*.png'))))
    with ExperimentLogger(team='graph',model='Task10I_report',seed=42,config=cfg,root=root/'experiments',job_type='analyze') as log:
        for p in root.glob('*.csv'):log.link_output(p,kind='metrics')
        log.link_output(root/name,kind='reports');log.link_output(root/'report.html',kind='reports')
        write(root/'report_identity.json',dict(run_id=log.run_id,at=now(),parents=12,episodes=120))


def verify(root):
    assert read(root/'preflight_verification.json')['status']=='PASS'
    assert read(root/'train_S/result.json')['updates']==11487
    assert read(root/'train_S/result.json')['initial']['frozen']==read(root/'train_S/result.json')['final_frozen']
    for name,h in read(root/'source_checkpoint_audit.json')['protected'].items():assert sha(name)==h,name
    for name,h in read(root/'execution_source_identity.json')['hashes'].items():assert sha(root/'source'/name)==h,name
    rows=[read(p) for p in (root/'rollouts/final').glob('seed*/*/episode.json')];assert len(rows)==120
    for row in rows:
        folder=root/row['folder'];assert sha(folder/'trace.json')==row['trace_sha256'];assert sha(folder/'rollout.mp4')==row['video_sha256']
        cap=cv2.VideoCapture(str(folder/'rollout.mp4'));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert count==row['actions']
        ok,_=cap.read();assert ok
        cap.set(cv2.CAP_PROP_POS_FRAMES,count-1);ok,_=cap.read();assert ok;cap.release()
        tr=read(folder/'trace.json');assert len(tr)==row['actions'] and len(tr)<=250
        assert row['insertion']==(any(t['success'] for t in tr) and (row['condition'][-1:] not in ('1','2') or row['handoff_timestep'] is not None))
        for t in tr:
            head=np.asarray(t['peg_head_pos_at_hole']);radius=float(np.asarray(row['hole_radius']).reshape(-1)[0])
            assert bool(head[0]>=-.015 and abs(head[1])<=radius and abs(head[2])<=radius)==t['success']
    required=['preflight_verification.json','source_checkpoint_audit.json','state38_contract_verification.json',
        'offline_online_feature_equivalence.json','gradient_freeze_verification.json','sequence_hidden_verification.json',
        'training_smoke_verification.json','evaluation_smoke_verification.json','train_config.json',
        'state38_feature_cache_manifest.json','state38_normalizer.json','train_metrics.csv','validation_metrics.csv',
        'checkpoint_manifest.json','best.pt','last.pt','development_selection.json','final_seed_manifest.json',
        'final_functional_success.csv','handoff_success.csv','paired_policy_comparison.csv','phasewise_action_error.csv',
        'state38_prediction_error.csv','handoff_lateral_drift.csv','arm_saturation_gripper_analysis.csv','latency_metrics.csv',
        '10i_predicted_state38_graph_recurrent_bc.md','report.html','final_paired_verification.json']
    for name in required:assert (root/name).is_file() and (root/name).stat().st_size>0,name
    assert len(list(csv.DictReader((root/'train_metrics.csv').open())))==11487
    files={str(p.relative_to(root)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in root.rglob('*')
        if p.is_file() and not p.is_symlink() and not any(x in p.relative_to(root).parts for x in ['source','host_data','features','experiments','ledger'])
        and p.name not in ['pipeline.log','pipeline_status.json','heartbeat.json','artifact_index.json','verification.json','completion_audit.json']
        and not p.name.endswith('.lock')}
    for path,row in files.items():assert sha(root/path)==row['sha256'] and (root/path).stat().st_size==row['bytes']
    write(root/'artifact_index.json',dict(files=files,readback='all indexed SHA/size checked; videos decoded first/last only'))
    write(root/'verification.json',dict(status='PASS',episodes=120,official_success_crosscheck=True,
        protected_originals_unchanged=True,frozen_training_parameters_unchanged=True,source_unchanged=True,
        artifacts=len(files),video_endpoint_readback=True,full_video_decode=False))
    write(root/'completion_audit.json',dict(status='PASS',all_required_present=True,required=required,at=now(),
        artifact_index_sha256=sha(root/'artifact_index.json'),no_publishing=True,no_callback=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['analyze','verify'],required=True)
    a=p.parse_args();{'analyze':summarize,'verify':verify}[a.stage](a.root)
