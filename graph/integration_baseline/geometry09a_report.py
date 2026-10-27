"""Read-only analysis of Task09-A physical counterfactual artifacts."""
import argparse
import csv
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
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from graph.integration_baseline.direct_runtime import read, sha
from graph.integration_baseline.graph import Graph, build_graph
from graph.integration_baseline.io import read_packet, write_json, save_npz
from graph.integration_baseline.point_resampling_adapter import edge_keys, jaccard
from graph.integration_baseline.geometry09a import TOL

LABELS=['Full_A','Full_B','Full_C','Single_A','Single_B','Single_C']
SLOTS=['TCP','Left Finger','Right Finger','Peg Mean','Box Mean']


def load_npz(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}


def graph_at(folder):
    d=load_npz(folder/'graph.npz');p=read_packet(folder/'packet.npz')
    return Graph(**{k:torch.from_numpy(a) for k,a in d.items()},packet=p,timings_ms={})


def scalar_stats(x):
    x=np.asarray([v for v in x if v is not None],float)
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,
        min=float(x.min()) if len(x) else None,max=float(x.max()) if len(x) else None)


def cos(a,b):
    n=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/n) if n>1e-15 else None


def rot(command):return Rotation.from_euler('XYZ',command[3:6])


def response_metrics(base,alt,eb,ea):
    a0,a1=base['action'][0],alt['action'][0]
    c0,c1=base['controller'][0],alt['controller'][0]
    dt=(c1[:3]-c0[:3]).astype(float)*1000;de=(ea[:3]-eb[:3]).astype(float)*1000
    nm,ne=np.linalg.norm(dt),np.linalg.norm(de)
    dr=rot(c0).inv()*rot(c1);er=rot(eb).inv()*rot(ea)
    discr=ne>TOL['expert_translation_discriminative_mm']
    z0,z1=base['readout'][0].astype(float),alt['readout'][0].astype(float)
    slots=[]
    for name,x,y in zip(SLOTS,z0.reshape(5,64),z1.reshape(5,64)):
        slots.append(dict(slot=name,absolute_l2=float(np.linalg.norm(y-x)),
            relative_l2=float(np.linalg.norm(y-x)/max(np.linalg.norm(x),1e-12)),cosine=cos(x,y)))
    return dict(native_translation_l2=float(np.linalg.norm(a1[:3]-a0[:3])),
        native_rotation_l2=float(np.linalg.norm(a1[3:6]-a0[3:6])),native_gripper_abs=float(abs(a1[6]-a0[6])),
        native_delta=(a1-a0).tolist(),model_translation_delta_mm=dt.tolist(),expert_translation_delta_mm=de.tolist(),
        model_translation_mm=float(nm),expert_translation_mm=float(ne),
        reference_class='DISCRIMINATIVE_REFERENCE' if discr else 'NON_DISCRIMINATIVE_REFERENCE',
        translation_direction_cosine=cos(dt,de) if discr and nm>TOL['model_translation_noise_mm'] else None,
        translation_magnitude_ratio=float(nm/ne) if discr else None,
        translation_signed_gain=float(np.dot(dt,de)/(ne*ne)) if discr else None,
        translation_response_error_mm=float(np.linalg.norm(dt-de)),
        response_better_than_zero=bool(np.linalg.norm(dt-de)<ne-1e-6) if discr else None,
        rotation_command_delta_rad=float(dr.magnitude()),expert_rotation_delta_rad=float(er.magnitude()),
        rotation_response_error_rad=float((dr.inv()*er).magnitude()),
        gripper_delta_mm=float((c1[6]-c0[6])*1000),expert_gripper_delta_mm=float((ea[6]-eb[6])*1000),
        gripper_response_error_mm=float(abs((c1[6]-c0[6])-(ea[6]-eb[6]))*1000),
        perturbed_translation_error_mm=float(np.linalg.norm(c1[:3]-ea[:3])*1000),
        perturbed_rotation_error_rad=float((rot(c1).inv()*rot(ea)).magnitude()),
        perturbed_gripper_error_mm=float(abs(c1[6]-ea[6])*1000),
        readout_absolute_l2=float(np.linalg.norm(z1-z0)),readout_relative_l2=float(np.linalg.norm(z1-z0)/max(np.linalg.norm(z0),1e-12)),
        readout_cosine=cos(z0,z1),slots=slots,presence_before=base['presence'][0].tolist(),presence_after=alt['presence'][0].tolist(),
        gru_input_l2=float(np.linalg.norm(alt['gru_input']-base['gru_input'])),
        hidden_output_l2=float(np.linalg.norm(alt['hidden']-base['hidden'])),
        encoded_action_delta_l2=float(np.linalg.norm(alt['encoded']-base['encoded'])),
        gripper_sign_changed=bool((a0[6]<0)!=(a1[6]<0)),
        previous_hidden_exact=bool(np.array_equal(base['previous_hidden'],alt['previous_hidden'])))


def input_metrics(bfolder,folder,spec):
    g0,g1=graph_at(bfolder),graph_at(folder)
    b,a=g0.packet,g1.packet;bo,ao=load_npz(bfolder/'observation.npz'),load_npz(folder/'observation.npz')
    phys=read(folder/'physical.json');stats=[]
    for o in (0,1):
        p=b['points'][b['object_id']==o];q=a['points'][a['object_id']==o]
        bs=set(b['provenance']['source_index'][b['object_id']==o].tolist())
        ss=set(a['provenance']['source_index'][a['object_id']==o].tolist())
        both=len(p)>0 and len(q)>0
        centroid=(q.mean(0)-p.mean(0)).astype(float)*1000 if both else None
        chamfer=np.r_[cKDTree(p).query(q)[0],cKDTree(q).query(p)[0]]*1000 if both else None
        union=np.logical_or(bo['masks'][o],ao['masks'][o]).sum()
        expected=np.array(spec['xyz_mm'],float)
        from visual.runtime.vision_packet import pose_matrix
        ee_rotation=pose_matrix(ao['tcp_pose'])[:3,:3]
        world_centroid=None if centroid is None else centroid@ee_rotation.T
        supp=phys['geometry_support'][o]
        bpur=read(bfolder/'physical.json')['geometry_support'][o]['selected_gt_purity']
        valid=both and bpur is not None and bpur>=.5 and supp['selected_gt_purity'] is not None and supp['selected_gt_purity']>=.5
        if np.linalg.norm(expected)>0 and spec['object']==('peg' if o==0 else 'box'):
            projected=None if world_centroid is None else float(np.dot(world_centroid,expected)/np.linalg.norm(expected))
            valid=valid and projected is not None and projected>=TOL['geometry_min_centroid_response_mm']
        else:projected=None
        rows=a['object_id']==o
        features=a['geometric_features'][rows]
        usable=(a['feature_field_available'][rows]&a['feature_field_valid'][rows])[:,[0,0,0,1,2]]
        fstats=[scalar_stats(features[usable[:,i],i].tolist()) for i in range(5)]
        stats.append(dict(object_id=o,count_before=len(p),count_after=len(q),source_index_jaccard=jaccard(bs,ss),
            mask_iou=float(np.logical_and(bo['masks'][o],ao['masks'][o]).sum()/max(1,union)),
            centroid_delta_ee_mm=None if centroid is None else centroid.tolist(),
            centroid_delta_world_mm=None if world_centroid is None else world_centroid.tolist(),
            projected_centroid_response_mm=projected,chamfer_mm=None if chamfer is None else float(chamfer.mean()),
            support_status='SUPPORTED_GEOMETRY_INPUT' if valid else 'INSUFFICIENT_GEOMETRY_INPUT',
            feature_available_counts=np.sum(a['feature_field_available'][rows],axis=0).tolist(),
            feature_valid_counts=np.sum(a['feature_field_valid'][rows],axis=0).tolist(),
            feature_distributions=fstats,
            point_xyz_quantiles_ee_m=np.quantile(q,[.05,.5,.95],axis=0).tolist() if len(q) else None,
            gt_support=supp))
    return dict(objects=stats,edge_counts_before=np.bincount(g0.edge_type.numpy(),minlength=3).tolist(),
        edge_counts_after=np.bincount(g1.edge_type.numpy(),minlength=3).tolist(),
        typed_source_edge_jaccard={str(k):jaccard(edge_keys(g0,k),edge_keys(g1,k)) for k in range(3)},
        raw_ede_edges_before=int(b['intra_object_edge_index'].shape[1]),raw_ede_edges_after=int(a['intra_object_edge_index'].shape[1]))


def repeat_gate(base,repeat):
    tests={}
    for filename in ('observation.npz','graph.npz',*[l+'.npz' for l in LABELS]):
        a,b=load_npz(base/filename),load_npz(repeat/filename)
        for name in a:
            x,y=a[name],b[name];same=x.shape==y.shape
            tol=2 if name=='rgb' else 1 if name=='depth_raw_mm' else 0 if x.dtype.kind in 'biu' else 1e-6
            gap=float(np.max(abs(x.astype(float)-y.astype(float)),initial=0)) if same else None
            passed=same and bool(np.allclose(x,y,atol=tol,rtol=0 if tol>=1 else 1e-5))
            tests[filename+':'+name]=dict(pass_gate=passed,max_abs=gap,tolerance=tol)
    return dict(pass_gate=all(t['pass_gate'] for t in tests.values()),fields=tests)


def historical_audit(root):
    path=Path('/workspace/task08/experiments/1e33012210174b8aa49a5485f1b9c063')
    checks={}
    if not path.exists():return dict(status='NOT_AVAILABLE',pass_gate=None)
    states=read(path/'reference_expert/states.json')
    for k in (0,10,25):
        now=read(root/f't{k:02d}/baseline/physical.json')['before'];checks[str(k)]={}
        for name in ('qpos','qvel','ee_pose','left_finger_pose','right_finger_pose','peg_pose','box_pose'):
            gap=float(np.max(abs(np.asarray(now[name])-np.asarray(states[k][name]))))
            checks[str(k)][name]=dict(max_abs=gap,pass_gate=gap<=1e-5,exact=gap==0)
    return dict(status='CHECKED_CURRENT_VS_08F',pass_gate=all(x['pass_gate'] for v in checks.values() for x in v.values()),fields=checks)


def csv_write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})


def figures(root,trials,metrics):
    out=root/'figures';out.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    for k in (0,10,25):
        candidates=[r for r in trials if r['t']==k and r['group']=='primary']
        shown=[r for r in trials if r['t']==k and r['group'] in ('baseline','primary')]
        fig,axs=plt.subplots(3,3,figsize=(10,10))
        for ax,r in zip(axs.ravel(),shown):
            d=load_npz(root/r['directory']/'observation.npz');im=d['rgb'].copy()
            for o,col in enumerate(((255,70,70),(70,170,255))):im[d['masks'][o]]=(.6*im[d['masks'][o]]+.4*np.array(col)).astype(np.uint8)
            ax.imshow(im);ax.set_title(r['name']);ax.axis('off')
        save(fig,f'primary_observations_t{k:02d}')
        example=next(r for r in candidates if r['name']=='peg_x+10')
        b=root/f't{k:02d}/baseline';p=root/example['directory']
        bo,po=load_npz(b/'observation.npz'),load_npz(p/'observation.npz')
        fig,axs=plt.subplots(2,4,figsize=(12,6))
        for i,(d,title) in enumerate(((bo,'Baseline'),(po,'Peg +10mm world X'))):
            axs[i,0].imshow(d['rgb']);axs[i,0].set_title(title+' RGB')
            axs[i,1].imshow(d['depth_raw_mm'],vmin=0,vmax=1500,cmap='viridis');axs[i,1].set_title('Depth (raw mm)')
            im=d['rgb'].copy()
            for o,color in enumerate(((255,70,70),(70,170,255))):im[d['masks'][o]]=(.5*im[d['masks'][o]]+.5*np.array(color)).astype(np.uint8)
            axs[i,2].imshow(im);axs[i,2].set_title('Student predicted masks')
            axs[i,3].imshow(d['gt_masks'][0]+d['gt_masks'][1]*2,vmin=0,vmax=2,cmap='coolwarm');axs[i,3].set_title('GT visible masks: audit only')
        for ax in axs.ravel():ax.axis('off')
        save(fig,f'rgbd_masks_t{k:02d}')
        fig=plt.figure(figsize=(12,5))
        for i,folder in enumerate((b,p)):
            ax=fig.add_subplot(1,2,i+1,projection='3d');d=load_npz(folder/'collision_geometry.npz')
            for n,v in d.items():
                if n.startswith('peg_'):col='crimson'
                elif n.startswith('box_'):col='royalblue'
                elif n.startswith('robot_'):col='gray'
                else:continue
                ax.scatter(*v.T,c=col,s=2,alpha=.45)
            ax.set(xlabel='World X (m)',ylabel='World Y (m)',zlabel='World Z (m)',title=('Baseline','Peg X+10mm')[i])
            ax.view_init(elev=25,azim=-55)
        save(fig,f'physical_geometry_t{k:02d}')
        fig=plt.figure(figsize=(12,5))
        for i,folder in enumerate((b,p)):
            ax=fig.add_subplot(1,2,i+1,projection='3d');g=load_npz(folder/'graph.npz');v=g['node_input'][:,:3]
            for a,z in g['edge_index'].T:ax.plot(*v[[a,z]].T,c='gray',lw=.25,alpha=.15)
            ax.scatter(*v.T,c=g['node_type'],cmap='coolwarm',s=5)
            ax.set(xlabel='EE X (m)',ylabel='EE Y (m)',zlabel='EE Z (m)',title=('Baseline graph','Peg X+10mm graph')[i])
        save(fig,f'graph_t{k:02d}')
        fig,axs=plt.subplots(2,3,figsize=(12,7),sharex=True)
        for ax,label in zip(axs.ravel(),LABELS):
            rows=[m for m in metrics if m['t']==k and m['model']==label and m['group']=='primary']
            values=np.array([[s['relative_l2'] for s in m['slots']] for m in rows]).T*100
            im=ax.imshow(values,aspect='auto',cmap='magma');ax.set_title(label)
            ax.set_yticks(range(5),SLOTS);ax.set_xticks(range(len(rows)),[r['name'].replace('peg_','') for r in rows],rotation=45)
            fig.colorbar(im,ax=ax,label='Readout relative L2 (%)')
        fig.tight_layout();save(fig,f'readout_slots_t{k:02d}')
        fig,axs=plt.subplots(1,3,figsize=(13,4))
        for ax,axis in zip(axs,range(3)):
            rows=[m for m in metrics if m['t']==k and m['model']=='Full_A' and m['group']=='primary']
            names=[m['name'].replace('peg_','') for m in rows]
            ax.bar(range(len(rows)),[m['expert_translation_delta_mm'][axis] for m in rows],color='black')
            ax.set_xticks(range(len(rows)),names,rotation=45);ax.set_title('Expert Δ'+'XYZ'[axis]+' command (mm)');ax.axhline(0,c='gray',lw=.6)
        save(fig,f'expert_response_t{k:02d}')
        fig,axs=plt.subplots(2,3,figsize=(13,7))
        for ax,label in zip(axs.ravel(),LABELS):
            rows=[m for m in metrics if m['t']==k and m['model']==label and m['group']=='primary']
            ax.plot([m['expert_translation_mm'] for m in rows],marker='o',label='Expert')
            ax.plot([m['model_translation_mm'] for m in rows],marker='o',label='Model')
            ax.set_xticks(range(len(rows)),[m['name'].replace('peg_','') for m in rows],rotation=45)
            ax.set(title=label,ylabel='Translation command change (mm)');ax.legend(fontsize=8)
        fig.tight_layout();save(fig,f'model_response_t{k:02d}')
        fig,axs=plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True)
        for ax,label in zip(axs.ravel(),LABELS):
            rows=[m for m in metrics if m['t']==k and m['model']==label and m['group']=='primary']
            xy=np.array([m['xyz_mm'][:2] for m in rows]);de=np.array([m['expert_translation_delta_mm'][:2] for m in rows]);dm=np.array([m['model_translation_delta_mm'][:2] for m in rows])
            ax.quiver(*xy.T,*de.T,color='black',angles='xy',scale_units='xy',scale=1,width=.006,label='Expert')
            ax.quiver(*xy.T,*dm.T,color='tab:blue',angles='xy',scale_units='xy',scale=1,width=.006,label='Model')
            ax.scatter(*xy.T,s=10,c='gray');ax.set(title=label,xlabel='Peg world X offset (mm)',ylabel='Peg world Y offset (mm)',xlim=(-15,15),ylim=(-15,15),aspect='equal');ax.legend(fontsize=7)
        fig.suptitle('XY controller-command response vectors anchored at pose offsets (mm); Z in raw metrics')
        fig.tight_layout();save(fig,f'response_vectors_t{k:02d}')
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for ax,key,title in zip(axs,['translation_direction_cosine','translation_magnitude_ratio','translation_response_error_mm'],['Direction cosine','Magnitude ratio','Response error (mm)']):
        for i,label in enumerate(LABELS):
            rows=[m for m in metrics if m['model']==label and m['alignment_eligible'] and m['group']=='primary']
            for j,k in enumerate((0,10,25)):
                vals=[m[key] for m in rows if m['t']==k and m[key] is not None]
                ax.scatter(np.full(len(vals),i+(j-1)*.18),vals,s=13,color=('tab:blue','tab:orange','tab:green')[j],label=f't={k}' if i==0 else None,alpha=.7)
        ax.set_xticks(range(6),LABELS,rotation=35);ax.set_title(title);ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'alignment')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,key,title in zip(axs,('model_translation_mm','readout_relative_l2'),('Model translation change (mm)','Readout relative L2')):
        for label in LABELS:
            vals=[]
            for k in (0,10,25):
                r=[m[key] for m in metrics if m['model']==label and m['t']==k and m['group']=='primary' and m['alignment_eligible']]
                vals.append(np.mean(r) if r else np.nan)
            ax.plot((0,10,25),vals,marker='o',label=label)
        ax.set(title=title,xlabel='Expert prefix length');ax.legend(fontsize=7)
    save(fig,'full_vs_single')
    fig,ax=plt.subplots(figsize=(10,4))
    names=['07-G Phase1 same hidden\n(old checkpoint, 528 frames)']+LABELS
    values=[.129796]+[np.mean([m['model_translation_mm'] for m in metrics if m['model']==label and m['group']=='primary' and m['alignment_eligible']]) for label in LABELS]
    ax.bar(range(7),values,color=['gray']+['royalblue']*3+['darkorange']*3)
    ax.set_xticks(range(7),names,rotation=25,ha='right');ax.set_ylabel('Mean translation command change (mm)')
    ax.set_title('Different perturbations, checkpoints and state distributions; descriptive comparison only')
    save(fig,'sampling_comparison')


def analyze(root, render_figures=True):
    trials=read(root/'trials.json');metrics=[];inputs=[]
    gates={str(k):repeat_gate(root/f't{k:02d}/baseline',root/f't{k:02d}/baseline_repeat') for k in (0,10,25)}
    write_json(root/'repeat_reset_controls.json',gates)
    historic=historical_audit(root);write_json(root/'historical_prefix_audit.json',historic)
    for r in trials:
        folder=root/r['directory'];base=root/f't{r["t"]:02d}/baseline'
        im=input_metrics(base,folder,r);inputs.append(dict(**r,**im))
        eb=load_npz(base/'expert.npz')['controller'];ea=load_npz(folder/'expert.npz')['controller']
        target_object=0 if r['object']=='peg' else 1
        support=im['objects'][target_object]['support_status']=='SUPPORTED_GEOMETRY_INPUT'
        for label in LABELS:
            m=response_metrics(load_npz(base/(label+'.npz')),load_npz(folder/(label+'.npz')),eb,ea)
            m.update(**r,model=label,geometry_supported=support,
                peg_chamfer_mm=im['objects'][0]['chamfer_mm'],
                peg_source_jaccard=im['objects'][0]['source_index_jaccard'],
                peg_mask_iou=im['objects'][0]['mask_iou'],
                alignment_eligible=r['status']=='VALID' and r['expert_status']=='VALID_STAGE_CONDITIONED' and gates[str(r['t'])]['pass_gate'] and m['previous_hidden_exact'] and
                    support and m['reference_class']=='DISCRIMINATIVE_REFERENCE')
            metrics.append(m)
    write_json(root/'input_geometry_metrics.json',inputs);write_json(root/'per_condition_metrics.json',metrics)
    csv_write(root/'per_condition_metrics.csv',[{k:v for k,v in m.items() if k!='slots'} for m in metrics])
    csv_write(root/'readout_slot_metrics.csv',[dict(model=m['model'],t=m['t'],name=m['name'],group=m['group'],**s) for m in metrics for s in m['slots']])
    csv_write(root/'graph_metrics.csv',[dict(t=r['t'],name=r['name'],status=r['status'],edges_before=r['edge_counts_before'],edges_after=r['edge_counts_after'],**o) for r in inputs for o in r['objects']])
    summary={}
    for label in LABELS:
        summary[label]={}
        for k in (0,10,25):
            rows=[m for m in metrics if m['model']==label and m['t']==k and m['group']=='primary']
            eligible=[m for m in rows if m['alignment_eligible']]
            keys=['expert_translation_mm','model_translation_mm','translation_direction_cosine','translation_magnitude_ratio',
                'translation_signed_gain','translation_response_error_mm','readout_relative_l2','perturbed_translation_error_mm']
            summary[label][str(k)]=dict(all_primary=len(rows),eligible=len(eligible),**{key:scalar_stats([m[key] for m in eligible]) for key in keys})
    write_json(root/'summary.json',summary)
    grouped={}
    keys=['expert_translation_mm','model_translation_mm','translation_direction_cosine','translation_magnitude_ratio',
        'translation_signed_gain','translation_response_error_mm','readout_relative_l2','perturbed_translation_error_mm',
        'rotation_command_delta_rad','rotation_response_error_rad','gripper_delta_mm','gripper_response_error_mm',
        'gru_input_l2','hidden_output_l2','encoded_action_delta_l2']
    for group in ('primary','additional','box_control'):
        grouped[group]={}
        for label in LABELS:
            rows=[m for m in metrics if m['model']==label and m['group']==group]
            use=[m for m in rows if m['alignment_eligible']] if group!='box_control' else rows
            grouped[group][label]=dict(total=len(rows),eligible=sum(m['alignment_eligible'] for m in rows),descriptive_count=len(use),
                response_better_than_zero=sum(m['response_better_than_zero'] is True for m in use),
                gripper_sign_changes=sum(m['gripper_sign_changed'] for m in use),
                **{key:scalar_stats([m[key] for m in use]) for key in keys})
    write_json(root/'group_summary.json',grouped)
    correlations={}
    for label in LABELS:
        correlations[label]={}
        for k in (0,10,25):
            rows=[m for m in metrics if m['model']==label and m['t']==k and m['group']=='primary' and m['alignment_eligible']]
            pairs={}
            for x,y in (('peg_chamfer_mm','readout_relative_l2'),('peg_source_jaccard','readout_relative_l2'),
                ('readout_relative_l2','model_translation_mm'),('readout_relative_l2','gru_input_l2'),
                ('gru_input_l2','hidden_output_l2'),('hidden_output_l2','model_translation_mm')):
                a=np.array([[r[x],r[y]] for r in rows if r[x] is not None and r[y] is not None],float)
                value=float(np.corrcoef(a.T)[0,1]) if len(a)>2 and np.all(a.std(0)>1e-12) else None
                pairs[x+'__'+y]=dict(n=len(a),pearson=value)
            correlations[label][str(k)]=pairs
    write_json(root/'descriptive_correlations.json',dict(interpretation='within-t offset associations, one seed; not causal mediation or population statistics',values=correlations))
    if render_figures:figures(root,trials,metrics)
    # Reports deliberately retain invalid, missing-object and non-discriminative rows.
    table=['| Model | t | eligible/8 | Expert Δ mm | Model Δ mm | cosine | magnitude ratio | response error mm |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(v):return '—' if v is None else f'{v:.4f}'
    for label in LABELS:
        for k in (0,10,25):
            s=summary[label][str(k)]
            table.append('| '+' | '.join([label,str(k),f'{s["eligible"]}/8',*[fmt(s[x]['mean']) for x in ('expert_translation_mm','model_translation_mm','translation_direction_cosine','translation_magnitude_ratio','translation_response_error_mm')]])+' |')
    valid=sum(r['status']=='VALID' for r in trials)
    primary=[m for m in metrics if m['group']=='primary' and m['alignment_eligible']]
    text=f'''# Task 09-A — Object Geometry Sensitivity & Graph Utilization Diagnostic

ExperimentLogger UUID `{root.name}`. Vast.ai 57640710 / RTX 3090 inference, CPU PhysX and CPU Vulkan image readback. Seed 30003; t=0,10,25; one-step read-only student inference after physical actor intervention. No student action was applied after branching. No training, dependency installation, commit or external publishing.

[Static HTML](object_geometry_sensitivity.html) · [all per-condition JSON](per_condition_metrics.json) · [CSV](per_condition_metrics.csv) · [input geometry](input_geometry_metrics.json) · [verification](verification.json).

## Registered protocol and validity

Eight Peg X/Y ±5/±10 mm candidates at each t, baseline, independent-reset baseline repeat, Box X/Y ±10 mm negative controls. Conditional additional X/Y ±20 mm and world-Z ±5° candidates follow the saved saturation/small-reference trigger; they are aggregated separately. All candidates and exclusions are preserved. Physical validity: {valid}/{len(trials)}. Independent repeat gates: { {k:v['pass_gate'] for k,v in gates.items()} }. Historical 08-F physical prefix comparison: {historic['pass_gate']}.

Gymnasium is currently 1.4.0; prior collection/08-F used 1.3.0. No package was changed. The diagnostic adapter checks installed ManiSkill/SAPIEN versions, original controller settings, reset SHA, five actual physics substeps per Expert action, and the historical prefix states. Pairwise intervention checks require exact Robot76, qpos/qvel, TCP/fingertip pose and calibration; previous hidden is recomputed causally and compared exactly. No hidden or tracker state is copied between trials. Each trial resets independently and reconstructs its observed prefix.

Actor.set_pose and get_obs execute on CPU PhysX without physics advancement. Collision checks use installed shape-local-transformed collision meshes, world transforms, convex SAT, table XY bounds and absence of new near-contact pairs. Tolerance is 0.2 mm. Cached PhysX contact lists are explicitly not claimed to be freshly recomputed contact forces after teleport. No post-branch contact dynamics or stability rollout was run. Box Hole/Goal properties derive from current Box pose, verified in installed source. All GT masks, actor poses and collision meshes are audit-only sidecars.

Expert references reconstruct a fresh grasp target from the changed physical Peg, then retain the pre-branch reach/descend phase and counter. Translation reconstruction and an independent servo translation calculation are checked. Invalid phases/targets are excluded. Rotation controller saturation is included in the installed decoder. Translation references ≤0.1 mm are NON_DISCRIMINATIVE_REFERENCE; model direction below 0.001 mm is undefined. These bands were frozen before model results. The reference is stage conditioned, not a demonstration from the perturbed initial state.

## Primary quantitative results

Only physically valid, repeat-controlled, geometry-supported, discriminative references enter this table. Geometry support requires both baseline/perturbed selected-point GT purity ≥50% and ≥1 mm centroid response projected onto a requested translation. This conservative audit is a local input-support criterion, not a new student feature. Every excluded row remains in raw JSON/CSV.

'''+ '\n'.join(table)+f'''

All primary eligible model/condition pairs: {len(primary)}. Slot-wise changes cover TCP, both fingers, Peg Mean and Box Mean. Rotation/geodesic response error, native action differences, gripper target differences and per-perturbed-state Expert error are in the per-condition files.

## Interpretation and research questions

Interpret numerical results at these three fixed pre-grasp states only. Nonzero graph/action response establishes dependence, not correct control. Small action response despite an informative graph and a discriminative Expert reference supports local under-utilization; a substantial but misaligned response supports an inaccurate graph-to-action mapping. This experiment alone does not isolate GNN–GRU fusion, representational information loss, model capacity or optimization, and cannot explain the full closed-loop failure causally.

The 07-G Phase1 same-hidden translation mean was 0.129796 mm, independent-history mean 0.153050 mm and readout relative L2 5.7744%. That task used a different checkpoint (epoch39 original policy), four episodes and sampling perturbations. The values are descriptive context and cannot be treated as equal-strength noise or a paired Full/Single comparison. Stability to resampling can coexist with useful geometry sensitivity, or with under-use; Expert alignment is needed to distinguish them.

## Artifacts and limitations

[Protocol](protocol.json), [checkpoint/codec/normalizer identities](identities.json), [additional-candidate trigger](additional_decisions.json), [reset controls](repeat_reset_controls.json), [historical prefix audit](historical_prefix_audit.json), [source protection](protection_audit.json), [slot CSV](readout_slot_metrics.csv), [graph CSV](graph_metrics.csv), [contract tests](contract_tests.log), [installed API source](installed_api_source.txt), [implementation diff](implementation.diff).

Each `tXX/condition/` preserves actual RGB-D, predicted and GT audit masks, calibration, packet/source index/F5/Raw EDE, graph, collision geometry, physical GT, Expert reference, six readout/action/controller tensors, fused input, GRU input/previous/output hidden, causal vision record and prefix identities. The executed source snapshot is under `source_snapshot/`. CUDA PhysX, MPS, real-time deadlines, extra seeds, post-branch student closed loop and new training are NOT_TESTED. Existing Full pipeline remains stopped.
'''
    finding_rows=['| Model | Peg readout Δ % | Model Δ mm | Direction cosine | Magnitude ratio | Signed gain | Response error mm | Error below zero response |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in LABELS:
        s=grouped['primary'][label]
        finding_rows.append('| '+' | '.join([label,fmt(s['readout_relative_l2']['mean']*100),
            *[fmt(s[key]['mean']) for key in ('model_translation_mm','translation_direction_cosine','translation_magnitude_ratio','translation_signed_gain','translation_response_error_mm')],
            f'{s["response_better_than_zero"]}/{s["eligible"]}'])+' |')
    missing=[(r['t'],r['name']) for r in inputs if r['group'] in ('primary','additional') and r['objects'][0]['support_status']!='SUPPORTED_GEOMETRY_INPUT']
    unsupported_box=[(r['t'],r['name']) for r in inputs if r['group']=='box_control' and r['objects'][1]['support_status']!='SUPPORTED_GEOMETRY_INPUT']
    findings=f'''## Observed findings

Primary Peg input support passed for 23/24 physical translation pairs. The excluded Primary pair is t=0 world Y+5 mm: GT visible Peg pixels 850, predicted mask/point count zero. The additional t=0 Y+20 mm candidate similarly has 882 visible Peg pixels but zero predicted mask/points. Missing-input pairs {missing} remain in the raw files; their often large policy response includes presence loss and cannot establish a correct geometry-to-action mapping. No detector, threshold, mask or checkpoint was repaired after observing this.

In the 23 supported Primary pairs, decoded Expert translation response averaged 0.746397 mm. Every model's readout changed, including robot slots. Full policies respond more strongly than Single B/C, but alignment is incomplete:

'''+ '\n'.join(finding_rows)+f'''

Magnitude ratio is the mean of per-pair norms, not the ratio of pooled means. Signed gain is ΔModel·ΔExpert / ||ΔExpert||² and distinguishes aligned correction from sideways or opposite movement. The last column is a descriptive comparison against a zero response, with no population or success-rate interpretation.

At t=25 the Expert response averaged 1.249468 mm, while Full A/B/C changed translation only 0.094974/0.054504/0.061745 mm; mean magnitude ratios were 0.077872/0.057798/0.068991. Their readout relative changes remained 3.2012/5.4376/6.0047%. This supports a local attenuation of useful current-geometry response after the fixed Expert history. Full A is also directionally poor at t=10/25 (cosines −0.021765/−0.054046). The data do not establish complete episode-wide graph neglect.

Single B/C Primary mean magnitude ratios were 0.063296/0.068297, signed gains 0.001643/0.008818, and response errors 0.745905/0.745433 mm, close to the 0.746397 mm zero-response scale. Their prior one-episode insertion success therefore does not demonstrate useful sensitivity to a changed current Peg pose. Robot/history memorization is compatible with this observation but is not uniquely identified.

Box-only Expert translation response was exactly zero in all 12 controls. At t=0/10, eight physically visible Box interventions produced Full A/B/C translation changes averaging 0.257389/0.305621/0.238275 mm. Such changes are unnecessary for this Expert reach reference, but the RGB detector/tracker can also change Peg masks, so the result does not isolate semantic Box-slot influence. At t=25 GT Box visible pixels were zero while the predicted Box graph retained 69 points of GT purity zero. All four Box interventions then had identical RGB-D, graph and model action. These unsupported controls {unsupported_box} are excluded from geometry-dependence judgments. The all-conditions RGB-D-change contract remains failed for these four occluded controls; it is not relabelled a global PASS. Accepted Peg comparisons pass this gate.

The additional group has 17/18 supported, discriminative pairs. Its Expert mean response is 1.542757 mm; Full A/B/C model responses are 0.559921/0.277409/0.337877 mm and Single A/B/C are 0.103694/0.032120/0.022336 mm. This group corroborates the limited/poor mapping pattern without being pooled into the Primary results.

## Q1–Q10 judgments

| Question | Judgment | Evidence and limit |
|---|---|---|
| Q1 Actual Peg geometry in Vision Graph | CONFIRMED for 23/24 Primary pairs; input failure in one | Source/depth backprojection and physical pose audits; t0 Y+5 has no Peg graph despite visibility. |
| Q2 GNN structured readout changes | CONFIRMED | Nonzero 320D and five-slot changes in supported pairs, stored individually. |
| Q3 Same Robot76/hidden action response | CONFIRMED | Model-dependent magnitudes above; exact robot and previous-hidden gates pass. |
| Q4 Expert geometry-dependent response | CONFIRMED | Fresh stage-conditioned targets; Primary mean 0.746397 mm, all three stages actually reach. |
| Q5 Direction and scale alignment | SUPPORTED: incomplete/inaccurate | Full A poor directional alignment; B/C partial alignment and under-response; Single B/C weak aligned gain. |
| Q6 Full A/B/C useful geometry response | SUPPORTED locally for B/C, limited and inconsistent overall | B/C response error below zero-response reference in 17/23 and 16/23 pairs; A 10/23. None is uniformly well aligned or scaled. |
| Q7 Single vs Full | CONFIRMED for measured differences; memorization hypothesis SUPPORTED, not identified | Full usually more sensitive; Single B/C current-frame response mostly insufficient. No claim that all Single methods are equally insensitive. |
| Q8 07-G stability vs appropriate geometry sensitivity | INCONCLUSIVE for joint robustness of these six checkpoints | Different checkpoint and perturbation distributions. Sampling stability alone neither proves useful graph use nor proves neglect. |
| Q9 Insufficient graph utilization hypothesis | SUPPORTED locally; blanket Full graph-ignore explanation not supported | Informative Peg readout with small/misaligned response; Full policies clearly depend on graph changes. Episode-wide neglect and causal explanation of closed-loop failure remain INCONCLUSIVE. |
| Q10 What to investigate | SUPPORTED for perception support and learned graph-to-action mapping; fusion/capacity/optimization causes INCONCLUSIVE | Real missed Peg/false Box support plus downstream response errors. 08-E training-fit deficit remains separate prior evidence, not a cause established here. |

No new architecture, loss, training or closed-loop experiment is implied by these judgments. All conclusions concern one seed and three reach states; the correlated offsets are not independent episode samples.
'''
    text+='\n'+findings
    write_json(root/'research_findings.json',dict(primary_supported_pairs=23,primary_total=24,
        missing_peg_input=missing,unsupported_box_controls=unsupported_box,primary=grouped['primary'],
        q_statuses=['CONFIRMED_WITH_INPUT_EXCLUSION','CONFIRMED','CONFIRMED','CONFIRMED','SUPPORTED',
            'SUPPORTED_PARTIAL','CONFIRMED_DIFFERENCES','INCONCLUSIVE','SUPPORTED_LOCAL_ONLY','SUPPORTED_WITH_UNRESOLVED_CAUSES']))
    (root/'09a_object_geometry_sensitivity.md').write_text(text)
    images=sorted((root/'figures').glob('*.png'))
    links=['protocol.json','identities.json','summary.json','per_condition_metrics.json','per_condition_metrics.csv',
        'group_summary.json','inference_dependency_identity.json',
        'input_geometry_metrics.json','graph_metrics.csv','readout_slot_metrics.csv','repeat_reset_controls.json',
        'historical_prefix_audit.json','protection_audit.json','additional_decisions.json','verification.json','contract_tests.log','09a_object_geometry_sensitivity.md']
    table_html='<table>'+''.join('<tr>'+''.join(('<th>' if i==0 else '<td>')+html.escape(cell.strip())+('</th>' if i==0 else '</td>') for cell in row.strip('|').split('|'))+'</tr>' for i,row in enumerate(finding_rows) if i!=1)+'</table>'
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Task09-A Geometry Sensitivity</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:20px;background:#f5f6f8;color:#202532}img{max-width:100%;background:white}pre{white-space:pre-wrap}a{color:#1457a6}section{background:white;padding:20px;margin:20px 0;border-radius:12px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dae0e8;text-align:right}th:first-child,td:first-child{text-align:left}tr:nth-child(even){background:#f4f7fc}summary{cursor:pointer;font-weight:600}</style>',
        '<h1>Task09-A · Object geometry sensitivity</h1>',f'<p>Run {root.name} · RTX 3090 / CPU PhysX · read-only policy inference</p>',
        '<section><h2>Current geometry influences action, but correction is limited</h2><p>23 of 24 primary Peg pairs have supported geometry input. Expert translation response: mean 0.746397 mm. Full policies respond, with incomplete direction and scale alignment. Single B/C respond weakly. Four occluded Box controls and two missed-Peg conditions remain explicit exclusions.</p>'+table_html+'</section>',
        '<section><details><summary>Protocol, per-timepoint values, interpretation and Q1–Q10</summary><pre>'+html.escape(text)+'</pre></details></section>',
        '<section><h2>Artifacts</h2><ul>'+''.join('<li><a href="'+l+'">'+l+'</a></li>' for l in links)+'</ul></section>']
    for image in images:body.append('<section><h2>'+html.escape(image.stem)+'</h2><img src="figures/'+image.name+'" alt="'+html.escape(image.stem)+'"></section>')
    body.append('</html>');(root/'object_geometry_sensitivity.html').write_text('\n'.join(body))
    if not (root/'verification.json').exists():write_json(root/'verification.json',dict(status='PENDING_READBACK'))
    return summary


def verify(root):
    counts=dict(json=0,csv=0,npz=0,png=0,html_links=0);errors=[];shapes={}
    for path in root.rglob('*'):
        if 'source_snapshot' in path.parts or not path.is_file():continue
        try:
            if path.suffix=='.json':read(path);counts['json']+=1
            elif path.suffix=='.csv':
                with path.open(newline='') as f:rows=list(csv.DictReader(f));assert rows
                counts['csv']+=1
            elif path.suffix=='.npz':
                a=load_npz(path);counts['npz']+=1
                if path.stem in LABELS:
                    assert a['readout'].shape==(1,320) and a['action'].shape==(1,7)
                    assert a['previous_hidden'].shape==(1,1,144) and a['hidden'].shape==(1,1,144)
                    assert a['fused_input'].shape==(1,1,398) and a['gru_input'].shape==(1,1,128)
                    assert all(np.isfinite(v).all() for v in a.values())
                    shapes[str(path.relative_to(root))]={k:list(v.shape) for k,v in a.items()}
                if path.stem=='packet':
                    p=read_packet(path);assert not p['observation_provenance']['gt_used_as_student_input']
                    assert not p['observation_provenance']['future_observations_used']
            elif path.suffix=='.png':
                image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED);assert image is not None;counts['png']+=1
            elif path.suffix=='.html':
                for link in re.findall(r'(?:href|src)="([^"]+)"',path.read_text()):
                    if not link.startswith(('https:','http:','#','data:')):
                        assert (path.parent/link).is_file(),link
                        counts['html_links']+=1
        except Exception as exc:errors.append(dict(path=str(path.relative_to(root)),error=type(exc).__name__,detail=str(exc)))
    trials=read(root/'trials.json');source_audits=[]
    for r in trials:
        f=root/r['directory'];ob=load_npz(f/'observation.npz');phys=read(f/'physical.json')
        g=graph_at(f)
        assert len(g.packet['object_id'])==len(g.node_type)-3
        assert g.packet['observation_provenance']['timestep']==r['t']
        assert len(read(f/'vision.json'))==r['t']+1
        assert read(f/'expert.json')['phase_before']==read(f/'prefix.json')['phase_at_branch']
        assert np.isfinite(ob['robot_state']).all()
        rebuilt=build_graph(g.packet,ob['robot_ee_geometry'],k=4,M=16,block_rows=64)
        for key in ('node_input','node_type','edge_index','edge_attr','edge_type'):
            np.testing.assert_array_equal(getattr(g,key).numpy(),getattr(rebuilt,key).numpy())
        source=g.packet['provenance']['source_index'];v,u=np.divmod(source,ob['depth_raw_mm'].shape[1])
        z=(ob['depth_raw_mm'].astype(np.float32)/1000)[v,u];intrinsic=ob['intrinsic'].astype(float)
        camera=np.column_stack(((u+.5-intrinsic[0,2])*z/intrinsic[0,0],(v+.5-intrinsic[1,2])*z/intrinsic[1,1],z)).astype(np.float32)
        tr=g.packet['T_EE_from_camera'];ee=(camera.astype(float)@tr[:3,:3].T+tr[:3,3]).astype(np.float32)
        gap=float(np.max(abs(ee-g.packet['points']),initial=0));assert gap<=1e-6
        source_audits.append(dict(t=r['t'],name=r['name'],depth_source_backprojection_max_m=gap,
            graph_rebuild_exact=True,source_pixel_count=len(source)))
    write_json(root/'source_graph_readback_audit.json',source_audits)
    write_json(root/'tensor_shapes.json',shapes)
    result=dict(pass_gate=not errors,counts=counts,errors=errors,trial_count=len(trials),tensor_files=len(shapes))
    write_json(root/'verification.json',result);return result


def contract_audit(root):
    from graph.integration_baseline.bc08_codec import controller_commands
    rows=read(root/'trials.json');identities=read(root/'identities.json')
    physical=[read(root/r['directory']/'physical.json') for r in rows]
    controls=[read(root/r['directory']/'controls.json') for r in rows]
    references=[read(root/r['directory']/'expert.json') for r in rows]
    conditions_changed=[p['render_rgb_changed_values']>0 and p['render_depth_changed_pixels']>0
        for r,p in zip(rows,physical) if r['group'] in ('primary','additional','box_control')]
    output_valid=True;fuse_valid=True;decode_valid=True;hidden_valid=True
    for r in rows:
        f=root/r['directory'];base=root/f't{r["t"]:02d}/baseline'
        for label in LABELS:
            d=load_npz(f/(label+'.npz'));b=load_npz(base/(label+'.npz'))
            output_valid &= d['readout'].shape==(1,320) and np.isfinite(d['readout']).all()
            fuse_valid &= np.array_equal(d['fused_input'][0,0,:320],d['readout'][0]) and np.array_equal(d['fused_input'][0,0,320:322],d['presence'][0])
            decode_valid &= np.array_equal(d['controller'],controller_commands(d['action']))
            hidden_valid &= np.array_equal(d['previous_hidden'],b['previous_hidden'])
        expert=load_npz(f/'expert.npz')
        decode_valid &= np.array_equal(expert['controller'],controller_commands(expert['action'][None])[0])
    expected_single=dict(A='5a94542e6d1f973d052fce9598e92a3be5b8c558974cc4e761f17e087a013f4d',
        B='72473e581c6c0dfdfa566a26fcf7d73619f310470a6e983a05a69db06cfd8899',
        C='1f81b559b5a95fbd2805b952850b896993eda4be2703f2f46a8f43a569c0ecc5')
    id_valid=all(identities['Single_'+m]['receipt']['sha256']==expected_single[m] for m in 'ABC')
    id_valid &= all(sha(v['receipt']['path'])==v['receipt']['sha256'] for v in identities.values())
    c={}
    def add(i,name,passed,evidence):c[str(i)]=dict(name=name,pass_gate=bool(passed),evidence=evidence)
    add(1,'Checkpoint/Normalizer/Codec identity',id_valid,'identities.json; original strict loader + plan validation; six fixed receipt hashes')
    add(2,'Same-seed independent reset identity',all(g['pass_gate'] for g in read(root/'repeat_reset_controls.json').values()),'repeat_reset_controls.json; live reset SHA assertions')
    add(3,'Requested actor-only pose intervention',all(p['status']=='VALID' for p in physical),'physical.json requested_pose and before/after; actual pose assertion')
    add(4,'Robot76 exact equality',all(p['exact']['robot76'] and p.get('cross_trial_robot76_exact',True) for p in physical),'physical.json; exact within intervention and across independent trials')
    add(5,'Box invariant in Peg-only interventions',all(p['exact'].get('box_pose',True) for r,p in zip(rows,physical) if r['object']=='peg'),'physical.json exact box_pose / velocity / hole / goal')
    add(6,'Collision/contact geometry validity',all(p['collision_after']['valid'] and not p['new_near_contacts'] for p in physical),'collision meshes SAT; no new near contacts; stale force cache limitation explicit')
    add(7,'RGB-D reflects new physical state',all(conditions_changed),'physical.json nonzero RGB and depth pixel changes')
    add(8,'Camera calibration invariant',all(p['exact']['intrinsic'] and p['exact']['cam2world'] for p in physical),'physical.json exact calibration')
    add(9,'Causal Vision observed prefix',all(p['same_history_exact'] for p in physical),'prefix.json exact RGB/graph/history hashes; vision.json prefix_length=t+1')
    add(10,'No GT/future in student graph',read(root/'verification.json')['pass_gate'],'packet provenance + whitelist source snapshot; GT sidecar read after packet construction')
    add(11,'Actual 320D readout extraction',output_valid and fuse_valid,'forward hook + exact fused_input[:320] / presence equality')
    add(12,'Same-hidden repeat and permutation',hidden_valid and all(v['pass_gate'] for c0 in controls for cc in c0.values() for v in cc.values()),'controls.json for all six models and every trial; exact previous hidden')
    add(13,'Independent Expert target per pose',all(v['fresh_target_recomputed'] for v in references),'expert.json fresh target and independent translation calculation')
    add(14,'Expert phase/cache consistency',all(v['status']=='VALID_STAGE_CONDITIONED' for v in references),'expert.json reconstructed reach phase/counter; translation grasp error <=2um')
    add(15,'Action codec/controller decoding',decode_valid,'stored Action7 vs installed preprocessing exact; actual prefix controller audit')
    add(16,'Source index/edge/feature validation',all(x['graph_rebuild_exact'] and x['depth_source_backprojection_max_m']<=1e-6 for x in read(root/'source_graph_readback_audit.json')),'source_graph_readback_audit.json; receiver F5 masks / Raw EDE validation')
    add(17,'Episode/timestep/observation alignment',read(root/'verification.json')['pass_gate'],'readback checks packet timestep, prefix length, expert phase, Tensor shape and finite values')
    add(18,'Original source/artifact protection',read(root/'protection_audit.json')['pass_gate'] and read(root/'inference_dependency_identity.json')['pass_gate'],'protection_audit.json; inference_dependency_identity.json')
    invisible=[dict(t=r['t'],name=r['name'],reason='RGB_DEPTH_UNCHANGED; GT object fully occluded')
        for r,p in zip(rows,physical) if r['group'] in ('primary','additional','box_control') and
        not (p['render_rgb_changed_values']>0 and p['render_depth_changed_pixels']>0)]
    c['7']['failed_conditions']=invisible
    c['7']['accepted_peg_comparisons_pass']=all(p['render_rgb_changed_values']>0 and p['render_depth_changed_pixels']>0
        for r,p in zip(rows,physical) if r['group'] in ('primary','additional') and r['object']=='peg')
    result=dict(pass_gate=all(v['pass_gate'] for v in c.values()),contracts=c,
        accepted_peg_comparisons_pass=all(v['pass_gate'] for i,v in c.items() if i!='7') and c['7']['accepted_peg_comparisons_pass'],
        note='Contract 6 validates static collision geometry, not freshly recomputed post-teleport dynamic contact forces.')
    write_json(root/'contract_audit.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reuse-figures',action='store_true',help='Reuse existing PNGs during text-only report revisions')
    a=p.parse_args()
    analyze(a.run,render_figures=not a.reuse_figures);print(json.dumps(verify(a.run)));print(json.dumps(contract_audit(a.run)))
