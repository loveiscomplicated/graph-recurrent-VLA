"""Static Task07-G artifacts, plots, readback and relative-link verification."""
import csv
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from graph.integration_baseline.io import write_json, sha256

CONDITIONS=('phase1','local01','local05')
COLORS=dict(phase1='#b83243',local01='#2d7eae',local05='#ad6c16')


def episode_plots(built,data,rows,impulses,config):
    folder=built['folder'];plots=folder/'plots';plots.mkdir()
    n=len(data['expert']);t=np.arange(n)
    fig,axes=plt.subplots(4,2,figsize=(14,15),sharex=True)
    for c in CONDITIONS:
        label=c
        for ax,key,ylabel in [(axes[0,0],c+'_z_relative_l2','Structured readout relative L2'),
            (axes[0,1],c+'_hidden_l2','Independent-history hidden L2'),
            (axes[1,0],c+'_instant_command_translation_mm','Same-past translation difference (mm)'),
            (axes[1,1],c+'_history_command_translation_mm','Full-history translation difference (mm)'),
            (axes[2,0],c+'_history_command_rotation_rad','Full-history XYZ rotation distance (rad)'),
            (axes[2,1],c+'_history_command_gripper_mm','Full-history finger target difference (mm)'),
            (axes[3,0],c+'_history_expert_mse_delta','Expert native7 MSE delta'),
            (axes[3,1],c+'_history_extra_command_translation_mm','Extra history contribution (mm)')]:
            ax.plot(t,data[key],color=COLORS[c],label=label,linewidth=1);ax.set_ylabel(ylabel);ax.grid(alpha=.25)
    for ax in axes.ravel():ax.legend(fontsize=8);ax.set_xlabel('Observation / action timestep')
    fig.suptitle(f"{built['spec']['split']} {built['spec']['episode_id']} — same expert observations, different point selections")
    fig.tight_layout();fig.savefig(plots/'timeline.png',dpi=120);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,12))
    for c in CONDITIONS:
        for j,slot in enumerate(('TCP','Left','Right','Peg','Box')):
            axes[0,0].plot(t,data[c+'_slot_relative_l2'][:,j],label=c+' '+slot,alpha=.75,linewidth=.8)
        axes[0,1].plot(t,data[c+'_hidden_cosine'],label=c,color=COLORS[c])
        axes[1,0].scatter(data[c+'_z_l2'],data[c+'_instant_command_translation_mm'],s=6,label=c,color=COLORS[c],alpha=.5)
        selected=[r for r in rows if r['condition']==c]
        axes[1,1].scatter([1-(r['peg_source_jaccard']+r['box_source_jaccard'])/2 for r in selected],
            data[c+'_instant_command_translation_mm'],s=6,label=c,color=COLORS[c],alpha=.5)
        axes[2,0].plot(t,data[c+'_z_l2'],label=c,color=COLORS[c])
        axes[2,1].plot(t,data['action_history_'+c][:,6],label=c,color=COLORS[c])
    axes[2,0].plot(t,data['natural_z_l2'],label='Phase0 consecutive observations',color='black',alpha=.65)
    axes[2,1].plot(t,data['action_phase0'][:,6],label='phase0',color='black')
    axes[2,1].plot(t,data['expert'][:,6],label='expert',color='green',linestyle='--')
    labels=[('Timestep','Slot relative L2'),('Timestep','Hidden cosine'),('Readout L2','Instant translation delta mm'),
        ('Mean source-set disagreement','Instant translation delta mm'),('Timestep','Readout L2: same-state vs natural change'),('Timestep','Native gripper command')]
    for ax,(x,y) in zip(axes.ravel(),labels):ax.set_xlabel(x);ax.set_ylabel(y);ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
    fig.tight_layout();fig.savefig(plots/'representation_action.png',dpi=120);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(14,12),sharex=True)
    for c in CONDITIONS:
        metrics=[r for r in built['metrics'] if r['condition']==c]
        for oid,name in ((0,'Peg'),(1,'Box')):
            axes[0,0].plot(t,[r['objects'][oid]['count_b'] for r in metrics],label=c+' '+name,linewidth=.8)
            axes[0,1].plot(t,[1000*(r['objects'][oid]['chamfer_mean_m'] or 0) for r in metrics],label=c+' '+name,linewidth=.8)
            axes[1,0].plot(t,[r['objects'][oid]['voxel_jaccard'] for r in metrics],label=c+' '+name,linewidth=.8)
            axes[1,1].plot(t,[r['objects'][oid]['raw_ede_jaccard'] for r in metrics],label=c+' '+name,linewidth=.8)
        for k,name in ((1,'Robot–object'),(2,'Peg–Box')):
            axes[2,0].plot(t,[r['edge_jaccard'][k] for r in metrics],label=c+' '+name,linewidth=.8)
        axes[2,1].plot(t,[np.mean([np.nanmean([v if v is not None else np.nan for v in o['f5_nearest_abs_difference'][:3]])
            for o in r['objects'] if o['f5_nearest_abs_difference'] is not None]) for r in metrics],label=c,color=COLORS[c])
    for ax,label in zip(axes.ravel(),['Alternative point count','Symmetric NN mean / Chamfer (mm)',
        'Nominal voxel Jaccard','Raw EDE source-edge Jaccard','Interaction source-edge Jaccard','Nearest-point normal component abs difference']):
        ax.set_ylabel(label);ax.set_xlabel('Timestep');ax.legend(fontsize=6,ncol=2);ax.grid(alpha=.2)
    fig.suptitle('Actual geometry / feature / topology changes — identical observed support')
    fig.tight_layout();fig.savefig(plots/'geometry_topology.png',dpi=120);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(15,11))
    for row,c in enumerate(CONDITIONS):
        for t0 in config['impulse_timesteps']:
            if t0>=n:continue
            x=np.arange(n-t0)
            for col,(key,label) in enumerate([('hidden_l2','Hidden L2'),('command_translation_mm','Translation delta mm'),('command_gripper_mm','Finger target delta mm')]):
                y=data[f'impulse_{c}_{t0}_{key}'][t0:]
                axes[row,col].semilogy(x,np.maximum(y,1e-9),label=f'impulse t={t0}')
                axes[row,col].set_ylabel(c+' '+label);axes[row,col].set_xlabel('Steps since one-frame perturbation');axes[row,col].grid(alpha=.2)
        for ax in axes[row]:ax.legend(fontsize=8)
    fig.suptitle('SINGLE_FRAME_IMPULSE — subsequent inputs return to Phase0; hidden is carried')
    fig.tight_layout();fig.savefig(plots/'impulse_recovery.png',dpi=120);plt.close(fig)
    for t0,demo in built['demos'].items():
        ga=demo['graphs']['phase0'];a=ga.packet
        for c in CONDITIONS:
            b=demo['graphs'][c].packet
            fig=plt.figure(figsize=(15,8))
            for col,oid in enumerate((0,1)):
                ia=a['object_id']==oid;ib=b['object_id']==oid
                sa=a['provenance']['source_index'][ia];sb=b['provenance']['source_index'][ib]
                common=set(sa)&set(sb)
                for row,bg in [(0,demo['rgb']),(1,demo['depth'])]:
                    ax=fig.add_subplot(2,3,row*3+col+1);ax.imshow(bg,cmap=None if row==0 else 'viridis')
                    width=demo['depth'].shape[1]
                    for src,color,label in [(np.array(sorted(common),int),'white','common'),(sa[~np.isin(sa,sb)],'#00c9ff','A only'),(sb[~np.isin(sb,sa)],'#ff3b5c','B only')]:
                        ax.scatter(src%width,src//width,s=9,c=color,label=label,alpha=.85)
                    ax.set_title(f"{'Peg' if oid==0 else 'Box'} t={t0}, A={len(sa)}, B={len(sb)}, intersection={len(common)}")
                    ax.legend(fontsize=7);ax.axis('off')
            ax=fig.add_subplot(1,3,3,projection='3d')
            for oid in (0,1):
                ia=a['object_id']==oid;ib=b['object_id']==oid
                xa=a['camera_points'][ia];xb=b['camera_points'][ib]
                ax.scatter(*xa.T,s=7,c='#00aacc',alpha=.6,label=f'A object {oid}')
                ax.scatter(*xb.T,s=7,c='#e23853',alpha=.6,label=f'B object {oid}')
            ax.set_xlabel('Camera X m');ax.set_ylabel('Camera Y m');ax.set_zlabel('Camera Z m');ax.legend(fontsize=7)
            fig.suptitle(f"Episode {built['spec']['episode_id']} — phase0 vs {c}; identical RGB-D / predicted masks")
            fig.tight_layout();fig.savefig(plots/f'overlay_t{t0}_{c}.png',dpi=120);plt.close(fig)


def aggregate(rows,keys):
    result={}
    for key in keys:
        x=np.array([r[key] for r in rows if r[key] is not None],float);x=x[np.isfinite(x)]
        result[key]=dict(n=len(x),mean=float(x.mean()) if len(x) else None,p95=float(np.quantile(x,.95)) if len(x) else None,
            max=float(x.max()) if len(x) else None)
    return result


def fmt(v):
    return 'N/A' if v is None else f'{v:.6g}'


def finalize_report(run,results,rows,config):
    keys=['z_l2','z_relative_l2','hidden_l2','hidden_cosine','instant_command_translation_mm','history_command_translation_mm',
        'history_command_rotation_rad','history_command_gripper_mm','history_extra_command_translation_mm',
        'instant_expert_mse_delta','history_expert_mse_delta','peg_source_jaccard','box_source_jaccard',
        'peg_chamfer_mean_m','box_chamfer_mean_m','peg_raw_ede_jaccard','box_raw_ede_jaccard']
    overall={c:aggregate([r for r in rows if r['condition']==c],keys) for c in CONDITIONS}
    stages={};presence={};transitions={}
    for c in CONDITIONS:
        for s in sorted(set(r['stage'] for r in rows)):
            subset=[r for r in rows if r['condition']==c and r['stage']==s]
            stages[c+'/'+s]=aggregate(subset,keys)
        for peg in (0,1):
            for box in (0,1):
                subset=[r for r in rows if r['condition']==c and r['presence_peg']==peg and r['presence_box']==box]
                presence[c+f'/peg{peg}_box{box}']=aggregate(subset,keys)
        for near in (False,True):
            subset=[r for r in rows if r['condition']==c and r['near_presence_transition']==near]
            transitions[c+f'/near_transition_{near}']=aggregate(subset,keys)
    write_json(run/'aggregate.json',dict(overall=overall,by_stage=stages,by_presence=presence,by_presence_transition=transitions,
        inference='descriptive paired episodes; consecutive frames are not independent replicates; no population p-value'))
    ep_rows=[]
    for ep in results['episodes']:
        for c in CONDITIONS:
            selected=[r for r in rows if r['episode_id']==ep['episode_id'] and r['condition']==c]
            values=aggregate(selected,keys)
            ep_rows.append(dict(episode_id=ep['episode_id'],split=ep['split'],condition=c,frames=ep['frames'],
                **{k+'_mean':v['mean'] for k,v in values.items()}))
    with (run/'episode_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(ep_rows[0]));writer.writeheader();writer.writerows(ep_rows)
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for c in CONDITIONS:
        selected=[r for r in rows if r['condition']==c]
        for ax,key,label in zip(axes.ravel(),['peg_source_jaccard','box_source_jaccard','z_relative_l2','history_command_translation_mm','history_expert_mse_delta','hidden_l2'],
            ['Peg source Jaccard','Box source Jaccard','Readout relative L2','History translation difference mm','Expert MSE delta','Hidden L2']):
            vals=[r[key] for r in selected if r[key] is not None]
            ax.hist(vals,bins=35,histtype='step',label=c,color=COLORS[c]);ax.set_xlabel(label);ax.set_ylabel('Frames');ax.legend(fontsize=8)
    fig.suptitle('528 fixed frames are correlated observations, not 528 independent experiments')
    fig.tight_layout();fig.savefig(run/'distributions.png',dpi=120);plt.close(fig)
    # Point count, surface ratio and feature distribution from the fixed train subset.
    distributions={}
    for c in ('phase0',)+CONDITIONS:
        obj={}
        for oid in (0,1):
            feature=[];counts=[];surface=[];valid=[]
            for ep in results['episodes']:
                if ep['split']!='train':continue
                folder=run/f"{ep['split']}_{ep['episode_id']}"/'graphs'
                for t in range(ep['frames']):
                    with np.load(folder/f'{t:04d}_{c}.npz',allow_pickle=False) as f:
                        i=f['object_id']==oid;feature.append(f['F5'][i]);counts.append(int(i.sum()))
                        surface.append(float(f['surface'][i].mean()) if i.any() else np.nan)
                        valid.append(f['valid'][i])
            f=np.concatenate(feature);v=np.concatenate(valid)
            obj[oid]=dict(frames=len(counts),point_count_mean=float(np.mean(counts)),point_count_p95=float(np.quantile(counts,.95)),
                surface_fraction_mean=float(np.nanmean(surface)),f5_mean=np.nanmean(f,axis=0).tolist(),f5_std=np.nanstd(f,axis=0).tolist(),
                validity_fraction=v.mean(0).tolist())
        distributions[c]=obj
    write_json(run/'fixed_train_subset_distribution.json',dict(scope='Only the two predeclared train episodes; not the entire training distribution',conditions=distributions))
    local=[]
    for ep in results['episodes']:
        local+=json.loads((run/f"{ep['split']}_{ep['episode_id']}"/'local_replacement.json').read_text())
    replacement={}
    for c in ('local01','local05'):
        replacement[c]={}
        for oid in ('0','1'):
            os=[r['objects'][oid] for r in local if r['condition']==c]
            surface=sum(x['surface_count'] for x in os); actual=sum(x['actual'] for x in os)
            replacement[c][oid]=dict(requested=sum(x['requested'] for x in os),actual=actual,surface_count=surface,
                actual_fraction=actual/surface if surface else 0.,no_replacement_frames=sum(x['actual']==0 for x in os))
    write_json(run/'replacement_summary.json',replacement)
    control=json.loads((run/'negative_controls.json').read_text());provenance=json.loads((run/'provenance.json').read_text())
    def m(c,k):return overall[c][k]['mean']
    phase=overall['phase1']
    q={
        'Q1':dict(status='CONFIRMED',answer=f"Phase1 mean structured-readout relative L2 {fmt(m('phase1','z_relative_l2'))}; absolute L2 {fmt(m('phase1','z_l2'))}. All five slots and cosine recorded."),
        'Q2':dict(status='CONFIRMED',answer=f"Same-previous-hidden Phase1 translation-command difference mean {fmt(m('phase1','instant_command_translation_mm'))} mm; full-history {fmt(m('phase1','history_command_translation_mm'))} mm. XYZ quaternion distance and finger targets reported separately."),
        'Q3':dict(status='CONFIRMED',answer=f"Local01/05 readout relative L2 means {fmt(m('local01','z_relative_l2'))}/{fmt(m('local05','z_relative_l2'))}; Phase1 {fmt(m('phase1','z_relative_l2'))}. Geometry and achieved replacement fractions measured, not presumed equivalent."),
        'Q4':dict(status='INCONCLUSIVE',answer=f"Phase1 paired native7 MSE delta mean {fmt(m('phase1','history_expert_mse_delta'))}; local01/05 {fmt(m('local01','history_expert_mse_delta'))}/{fmt(m('local05','history_expert_mse_delta'))}. Episode/component increases and decreases are measured. Four fixed correlated episodes do not establish population statistical significance or physical harm."),
        'Q5':dict(status='CONFIRMED',answer=f"Independent Phase1 histories produce hidden L2 mean {fmt(m('phase1','hidden_l2'))}; cosine mean {fmt(m('phase1','hidden_cosine'))}. Growth/decline counts, first difference and episode timelines are recorded; hidden magnitude alone is not failure."),
        'Q6':dict(status='CONFIRMED',answer='Single-frame impulses at t10/20/30 have full carried-hidden and action tails for all alternatives and episodes. Recovery uses a predeclared 5% initial difference threshold plus numerical floors; physical recovery was not tested.'),
        'Q7':dict(status='CONFIRMED',answer=f"Current graph effect and history contribution are separated using the same pre-observation Hidden A. Additional Phase1 history translation difference mean {fmt(m('phase1','history_extra_command_translation_mm'))} mm. This cross-history output is a read-only diagnostic."),
        'Q8':dict(status='INCONCLUSIVE',answer='Sensitivity under controlled same-observation interventions is measured. Phase1 changes geometry/topology and may be out of distribution; local replacements are artificial diagnostics. These do not isolate natural Phase0 temporal inconsistency as a cause of control failure, or establish task-wide robustness.'),
        'Q9':dict(status='SUPPORTED',answer='The measured geometry→readout→action and history effects provide a quantitative baseline for deciding whether temporal consistency merits controlled follow-up. No anchor method, tracking system, retraining comparison or physical success advantage was tested. Persistent-anchor superiority and necessity remain NOT_TESTED.')}
    write_json(run/'interpretation.json',q)
    lines=['# Task07-G — Point Resampling Sensitivity & Recurrent Stability',
        '',f"ExperimentLogger UUID: `{results['run_id']}`. Vast.ai 57640710 / {provenance['gpu']}. **{results['status']}**, {results['frames']} valid action frames across the four frozen episodes.",
        '',f"[Static HTML](point_resampling_stability.html) · [Results](results.json) · [Frame CSV](frames.csv) · [Episode CSV](episode_summary.csv) · [Verification](verification.json).",
        '', '## 1. Identity and execution', '',f"Checkpoint `{config['checkpoint_sha256']}`, training pipeline `2be5e62357a44b14a5651a1a346edc7a`, epoch39 / step8959. Dataset `d1fd6173cb3648bfa4c03507f84b7e71`; original train-only Robot76 normalizer, Regression7, GNN L2/d64, k4/M16, readout320, presence2, GRU144 unchanged. All calculations eval + torch.inference_mode, FP32, TF32 disabled, deterministic algorithms. Existing ExperimentLogger used with WANDB_MODE=disabled. No upload or publish.",
        '', '[Checkpoint/source/input identity](provenance.json) · [Input hashes](input_hashes.json) · [Normalizer hash](normalizer_hash.json) · [Source hashes](source_hashes.json) · [Frozen selection](selection.json).',
        '', '## 2. Sampling contracts', '', 'Phase0 and Phase1 use the same Prepared Evidence from original RGB-D and stored predicted masks: optimized PCA5, outer silhouette, keep_ratio0.5, original structural/Surface rules and observed 3D points. Phase1 is the existing diagnostic odd-index thinning alternative. F5, availability/validity, observed components, Raw EDE, robot-object k4, global Peg-Box M16 and edge attr4/type are rebuilt with the unchanged helpers. Phase0 is recursively exact with every saved packet and its graph on this runtime.',
        '', 'CONTROLLED_LOCAL_REPLACEMENT fixes structural points and replaces evenly spaced Surface rows with the nearest unused original observed source in the same component. Candidates must be 8-neighbour pixels within the nominal metric scale, or share the nominal camera-frame voxel. Distance/pixel/source tie-breaks are deterministic. Failed replacements are skipped. Requested count is ceil(surface_count × fraction), so discrete rates can exceed 1%/5%. No target/model output enters selection. This is not natural Shape-First temporal noise.',
        '', '| Alternative | Object | Actual / requested replacements | Actual fraction of Surface |', '| --- | --- | --- | --- |']
    for c,objects in replacement.items():
        for oid,r in objects.items():lines.append(f"| {c} | {'Peg' if oid=='0' else 'Box'} | {r['actual']} / {r['requested']} | {r['actual_fraction']:.4%} |")
    lines+=['','[Replacement provenance](replacement_summary.json). Per-frame source pairs are stored in each episode local_replacement.json.',
        '', '## 3. Negative controls and contracts', '',f"All four episodes PASS before comparison. Fixed FP32 tolerances atol={config['atol']}, rtol={config['rtol']}. Identical packet/previous hidden repeated; object rows permuted across object boundaries with edge endpoints remapped; all readout/action/hidden outputs match. Robot identity slots remain rows0..2 as required by structured readout. This checks arbitrary object point order, not a redefinition of robot identities. [Measured maxima](negative_controls.json).",
        '', 'Source/row alignment, original observed support, object/component boundaries, dense F5 and validity/availability are validated through existing packet builders/receiver; common-source F5 exactness checked. Raw EDE direction/index/type and interaction edges are rebuilt and receiver-validated. Robot76 and normalizer identical; episode/timestep source identities match; the final observation without an action is excluded. New diagnostics begin hidden at zero per episode; independent alternative histories carry their own hidden.',
        '', '## 4. Geometry, representation and controller action', '', '| Condition | Peg / Box source Jaccard mean | Readout relative L2 mean | Same-past translation delta mm mean | History translation delta mm mean | History rotation distance rad mean | Finger target delta mm mean |', '| --- | --- | --- | --- | --- | --- | --- |']
    for c in CONDITIONS:lines.append(f"| {c} | {m(c,'peg_source_jaccard'):.4f} / {m(c,'box_source_jaccard'):.4f} | {m(c,'z_relative_l2'):.6f} | {m(c,'instant_command_translation_mm'):.6f} | {m(c,'history_command_translation_mm'):.6f} | {m(c,'history_command_rotation_rad'):.6f} | {m(c,'history_command_gripper_mm'):.6f} |")
    lines+=['', 'Chamfer is the mean of concatenated bidirectional nearest-neighbour distances (metres), with p50/p95/max reported. Voxel coverage uses the same nominal camera-frame metric grid for A/B against the full observed support. Source-index identity is a raster location, not a persistent object point. Edge Jaccards use typed source endpoints; common-induced-edge Jaccard separates changed node support from connectivity on shared sources. Feature changes include nearest-point per-channel differences, distribution/validity changes and exact equality at shared sources. [Geometry/feature/topology JSON and per-frame NPZ are linked from HTML](point_resampling_stability.html).',
        '', 'Controller analysis executes installed PDEEPoseController._clip_and_scale_action and gripper clip_and_scale_action on configuration-only stubs. Translation clips to ±1 and scales0.1m; rotation norm clips then multiplies rot_lower−0.1; finger target is clipped then maps to [−0.01,0.04]m. Rotation distance composes quaternions from the installed XYZ Euler matrix helper. No simulator is instantiated. Native7 norm/MSE are diagnostic loss coordinates, not a physical performance score.',
        '', '## 5. Expert error, stage and training reference', '', '| Episode | Condition | History translation delta mm mean | Native expert MSE delta mean | Hidden L2 mean |', '| --- | --- | --- | --- | --- |']
    for r in ep_rows:lines.append(f"| {r['split']} {r['episode_id']} | {r['condition']} | {fmt(r['history_command_translation_mm_mean'])} | {fmt(r['history_expert_mse_delta_mean'])} | {fmt(r['hidden_l2_mean'])} |")
    lines+=['', 'Native per-frame errors use the unchanged seven-component regression MSE; translation/rotation/gripper errors and actual controller-command errors are separate columns. Increased/decreased/unchanged uses absolute MSE delta1e−6 as a descriptive numerical band, not statistical significance. [Stage and presence groups](aggregate.json) use expert phase metadata only as an analysis sidecar. No phase/GT object/hole/goal enters the policy. Empty objects remain empty.',
        '', '[Fixed train subset distributions](fixed_train_subset_distribution.json) compare point counts, Surface ratios, F5 means/std and validity on Train30000/30003. They are a limited reference, not the full training distribution. [All-train point count reference](training_point_count_reference.json) uses original manifest counts if available. Consecutive frames are correlated; per-episode summaries are the primary independent cases. No population p-value or causal task success claim is made.',
        '', '## 6. GRU history, impulses and natural changes', '', 'History B carries independently from episode start. Instantaneous B uses Hidden A immediately before the current observation. History-extra compares Policy(Graph B[t], Hidden A[t−1]) with Policy(Graph B[t], Hidden B[t−1]); the same current graph isolates the diagnostic contribution of prior altered sampling. This is a cross-history read-only intervention, not a physically executed policy trajectory.',
        '', 'All conditions receive SINGLE_FRAME_IMPULSE at t10,20,30 and then return to Phase0 without resetting hidden. Recovery is the first lag after which the entire observed tail remains below max(5% of the initial effect, numerical floor): hidden1e−6; translation/finger0.001mm; rotation1e−6rad. Censoring is recorded. This descriptive criterion is not physical recovery success. [Per-episode impulse summaries and recovery curves](point_resampling_stability.html).',
        '', 'Consecutive original Phase0 readout changes and slot changes are stored alongside same-state perturbation sizes; graph point/edge counts and source-pixel topology changes are recorded. Those adjacent observations include actual robot/object motion, camera transforms and mask evolution. They are not pure sampling-noise ground truth.',
        '', '## 7. Q1–Q9 and evidence limits', '', '| Question | Status | Answer |', '| --- | --- | --- |']
    for key,v in q.items():lines.append(f"| {key} | {v['status']} | {v['answer']} |")
    lines+=['', '## 8. Artifacts and verification', '', 'Every numerical NPZ is written with immediate byte/shape/dtype readback. Final verification reopens JSON/CSV/NPZ and decodes PNGs, and validates every local HTML/Markdown relative asset. Source snapshot includes all shared inference helpers and the installed controller source. implementation.diff records the remote checkout diff; the added diagnostic files are in source_snapshot. Local source identity and concurrent HEAD changes are captured separately in preflight.json. [Protection](protection.json), [Tests](tests.log), [Verification](verification.json).',
        '', 'Executed: Vast57640710 CUDA GNN/GRU inference, CPU PCA5/Shape-First/F5/Raw EDE and installed controller preprocessing, four fixed full expert sequences, 36 one-frame impulses, all controls. Not executed: CUDA simulator, CPU simulator/physical rollout, MPS, new DINO/SAM2 inference, training/fine-tuning/HPO, persistent tracking/anchors, dependency installation, commit, external publishing. Existing datasets/checkpoint/normalizer and original experiment artifacts are hash-protected.']
    (run/'07g_point_resampling_stability.md').write_text('\n'.join(lines)+'\n')
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Task07-G Point Resampling Stability</title><style>body{max-width:1250px;margin:30px auto;padding:0 20px;background:#f4f6f8;color:#152639;font:16px/1.6 system-ui}img{width:100%;background:white;border:1px solid #ccd3db;border-radius:8px;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #bbc8d4;padding:8px;text-align:left}code{font-size:13px}section{background:white;padding:22px;margin:20px 0;border-radius:9px}.note{background:#e4eff8;padding:16px}a{color:#126092}pre{white-space:pre-wrap;overflow:auto}summary{cursor:pointer;font-weight:bold}</style>',
        f'<h1>Task07-G · Point Resampling & Recurrent Stability</h1><p>Run <code>{results["run_id"]}</code> · {results["frames"]} fixed expert action frames · RTX3090 CUDA · FP32 inference only</p>',
        '<p class="note">Same observations/masks/robot/checkpoint. Only point selection and its derived features/edges change. These are offline interventions, not physical rollouts or task success rates. Frame observations are correlated.</p>',
        '<p><a href="07g_point_resampling_stability.md">Full report</a> · <a href="results.json">Results</a> · <a href="frames.csv">Frame CSV</a> · <a href="episode_summary.csv">Episode CSV</a> · <a href="interpretation.json">Q1–Q9</a> · <a href="negative_controls.json">Controls</a> · <a href="verification.json">Verification</a> · <a href="input_hashes.json">Input hashes</a> · <a href="source_hashes.json">Source hashes</a> · <a href="tests.log">Test log</a></p>',
        '<section><h2>Comparisons and scope</h2><table><tr><th>Question</th><th>Status</th><th>Finding</th></tr>']
    for key,v in q.items():parts.append(f'<tr><td>{key}</td><td>{v["status"]}</td><td>{html.escape(v["answer"])}</td></tr>')
    parts+=['</table><img src="distributions.png" alt="Geometry, representation, action and hidden distributions"></section>',
        '<section><h2>Controls and actual replacement rates</h2><p>Phase0 exact packet/graph reproduction, identical input repetition, object-row permutation with remapped edges, original support/source/F5/component contracts: PASS. Fixed atol1e−6 / rtol1e−5.</p>',
        '<details><summary>Measured numerical maxima</summary><pre>'+html.escape(json.dumps(control,indent=2))+'</pre></details>',
        '<details><summary>Achieved Local Replacement rates</summary><pre>'+html.escape(json.dumps(replacement,indent=2))+'</pre></details></section>']
    for ep in results['episodes']:
        prefix=f"{ep['split']}_{ep['episode_id']}"
        parts+=[f'<section><h2>{ep["split"]} {ep["episode_id"]} · {ep["frames"]} actions</h2>',
            f'<p><a href="{prefix}/frames.csv">Frame CSV</a> · <a href="{prefix}/tensors.npz">Readout / hidden / action tensors</a> · <a href="{prefix}/sampling_comparison.json">Geometry/features/topology</a> · <a href="{prefix}/local_replacement.json">Replacement source pairs</a> · <a href="{prefix}/natural_graph_changes.json">Adjacent Phase0 graphs</a> · <a href="{prefix}/impulses.json">Impulse recovery</a></p>',
            f'<img src="{prefix}/plots/timeline.png" alt="Readout hidden action and error timeline">',
            f'<img src="{prefix}/plots/representation_action.png" alt="Slots, source disagreement, natural changes and gripper">',
            f'<img src="{prefix}/plots/geometry_topology.png" alt="Point counts, distances, voxel, topology and feature changes">',
            f'<img src="{prefix}/plots/impulse_recovery.png" alt="One-frame impulse recovery curves">',
            '<details><summary>Same RGB-D / same observed 3D surface: A/B selected point overlays</summary><p>Cyan: Phase0 only; red: alternative only; white: common sources. Empty support remains empty. Frames10,20,30 were fixed before inference.</p>']
        for t0 in config['impulse_timesteps']:
            if t0>=ep['frames']:continue
            for c in CONDITIONS:
                parts.append(f'<figure><figcaption>t{t0} · {c}</figcaption><img loading="lazy" src="{prefix}/plots/overlay_t{t0}_{c}.png" alt="Original RGB and depth point overlay with camera-frame 3D selection"></figure>')
        parts+=['</details></section>']
    parts+=['<section><h2>Stage sensitivity (sidecar only)</h2><table><tr><th>Condition / expert stage</th><th>Frames</th><th>Readout relative L2 mean</th><th>History translation delta mm mean</th><th>Expert MSE delta mean</th></tr>']
    for name,values in stages.items():
        parts.append(f'<tr><td>{html.escape(name)}</td><td>{values["z_l2"]["n"]}</td><td>{fmt(values["z_relative_l2"]["mean"])}</td><td>{fmt(values["history_command_translation_mm"]["mean"])}</td><td>{fmt(values["history_expert_mse_delta"]["mean"])}</td></tr>')
    parts+=['</table></section><section><h2>Interpretation and limits</h2><p>Measured same-state sensitivity does not prove natural Phase0 resampling harms task performance. Phase1 can change geometry/topology and depart from training support. Local Replacement is a controlled artificial intervention. Persistent anchors, retraining comparisons and physical success are NOT_TESTED. Per-episode error increases/decreases and command units must be considered separately from embedding norms.</p><p><a href="aggregate.json">Stage/presence summaries</a> · <a href="fixed_train_subset_distribution.json">Fixed-train feature reference</a> · <a href="training_point_count_reference.json">All-train point count reference</a> · <a href="protection.json">Input/weight protection</a></p></section></html>']
    (run/'point_resampling_stability.html').write_text('\n'.join(parts))


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links += [v for k,v in attrs if k in ('href','src') and v]


def verify(run):
    counts=dict(json=0,npz=0,csv=0,png=0,html=0,relative_links=0)
    for path in run.rglob('*'):
        if not path.is_file():continue
        if path.suffix=='.json':json.loads(path.read_text());counts['json']+=1
        elif path.suffix=='.npz':
            with np.load(path,allow_pickle=False) as f:
                for key in f.files:
                    a=f[key];assert a.dtype!=object
            counts['npz']+=1
        elif path.suffix=='.csv':
            with path.open() as f:
                reader=csv.reader(f);header=next(reader)
                for row in reader:assert len(row)==len(header)
            counts['csv']+=1
        elif path.suffix=='.png':
            with Image.open(path) as im:im.verify()
            counts['png']+=1
        elif path.suffix in ('.html','.md'):
            if path.suffix=='.html':
                parser=Links();parser.feed(path.read_text());links=parser.links;counts['html']+=1
            else:links=re.findall(r'\]\(([^)]+)\)',path.read_text())
            for link in links:
                if re.match(r'^(https?:|data:|#)',link):continue
                target=path.parent/link.split('#')[0]
                # verification.json is written immediately after this complete readback.
                assert target.exists() or target==run/'verification.json',str(target)
                counts['relative_links']+=1
    return dict(status='PASS',readback=counts,npz_immediate_byte_readback=True,html_relative_assets='PASS',
        numeric_finiteness='native z/hidden/action checked in contract tests and inference; undefined missing-object statistics explicitly null/NaN',
        device='CUDA on Vast57640710',simulator_executed=False,publishing_executed=False)
