"""Readback, static scientific figures and evidence-only Task09B reporting."""
import argparse
from collections import Counter
import csv
import hashlib
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
from torch import nn

from graph.integration_baseline.direct_runtime import read,sha
from graph.integration_baseline.io import write_json,read_packet
from graph.integration_baseline.geometry09b_probe import absolute,delta,features,csv_write
from graph.integration_baseline.geometry09b import target,TASK09A
from utils.experiment_artifacts import register_existing_output

LABELS=['Full_A','Full_B','Full_C','Single_A','Single_B','Single_C']
COLORS=['#2672ae','#dd7f21','#389662','#a34ba6','#b84343']


def load(path):
    with np.load(path,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}


def csv_rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def pair_rows(root):
    out=csv_rows(root/'per_pair.csv')
    strings={'model','probe','decoder','control','split','condition','group','validity','baseline_validity'}
    bools={'physical_valid','primary_valid','geometry09a_response_support'}
    for row in out:
        for k,v in list(row.items()):
            if k in strings:continue
            row[k]=v=='True' if k in bools else None if v=='' else float(v)
    return out


def aggregate_pairs(rows):
    def mean(key):
        x=[r[key] for r in rows if r[key] is not None]
        return float(np.mean(x)) if x else None
    result=dict(n=len(rows),**{key:mean(key) for key in ('true_norm_mm','predicted_norm_mm','delta_error_mm',
        'direction_cosine','signed_gain','delta_x_error_mm','delta_y_error_mm','delta_z_error_mm')})
    result['better_than_zero_pairs']=sum(r['delta_error_mm']<r['true_norm_mm'] for r in rows)
    return result


def summarize(root):
    metrics=read(root/'probe_metrics.json');pairs=pair_rows(root);samples=read(root/'samples.json')
    groups=[]
    for model in LABELS+['Shared']:
        ps=['P0','P1','P2','P3','P4'] if model!='Shared' else ['Constant','RawCentroid']
        for probe in ps:
            for decoder in (['linear','mlp'] if model!='Shared' else ['constant' if probe=='Constant' else 'direct']):
                for control in (['normal','shuffle'] if model!='Shared' else ['normal']):
                    for part in ('train','validation','final'):
                        for t in (None,0,10,25):
                            rows=[r for r in pairs if r['model']==model and r['probe']==probe and
                                r['decoder']==decoder and r['control']==control and r['split']==part and
                                r['primary_valid'] and (t is None or r['t']==t)]
                            groups.append(dict(model=model,probe=probe,decoder=decoder,control=control,
                                split=part,t=t,**aggregate_pairs(rows)))
    write_json(root/'delta_metrics.json',groups);csv_write(root/'delta_metrics.csv',groups)
    counts=[dict(split=part,validity=v,n=sum(r['split']==part and r['validity']==v for r in samples),
        physical_invalid=sum(r['split']==part and r['validity']==v and not r['physical_valid'] for r in samples))
        for part in ('train','validation','final') for v in
        ('VISIBLE_VALID','VISIBLE_MISSING','OCCLUDED','GEOMETRY_CONTAMINATED')]
    write_json(root/'validity_counts.json',counts)
    # Every sample, including invalid/missing/secondary, gets explicit absolute errors.
    sample_rows=[]
    tensor=load(root/'frozen_readouts320.npz');yy=tensor['target_peg_position_tcp']
    support=[]
    for part in ('train','validation','final'):
        mask=np.array([r['split']==part and r['group']=='baseline' and r['physical_valid'] and r['validity']=='VISIBLE_VALID' for r in samples])
        for k in range(3):
            v=yy[mask,k]*1000
            support.append(dict(split=part,axis='XYZ'[k],baseline_rows=len(v),minimum_mm=float(v.min()) if len(v) else None,maximum_mm=float(v.max()) if len(v) else None,std_mm=float(v.std()) if len(v) else None))
    write_json(root/'target_axis_support.json',support)
    secondary=[];presence_metrics=[]
    for pp in sorted((root/'probes').glob('*_predictions.npz')):
        key=pp.stem.removesuffix('_predictions');pred=load(pp)['prediction']
        for i,r in enumerate(samples):
            error=pred[i].astype(float)-yy[i].astype(float)
            sample_rows.append(dict(probe_identity=key,sample_id=r['sample_id'],split=r['split'],seed=r['seed'],t=r['t'],
                condition=r['condition'],group=r['group'],validity=r['validity'],box_validity=r['box_validity'],
                physical_valid=r['physical_valid'],absolute_error_mm=float(np.linalg.norm(error)*1000),
                target_x_m=float(yy[i,0]),target_y_m=float(yy[i,1]),target_z_m=float(yy[i,2]),
                prediction_x_m=float(pred[i,0]),prediction_y_m=float(pred[i,1]),prediction_z_m=float(pred[i,2])))
        presence=tensor['Full_B_presence2'].astype(bool)
        for part in ('train','validation','final'):
            for peg in (False,True):
                for box in (False,True):
                    mask=np.array([r['split']==part and r['physical_valid'] and r['group'] in ('baseline','primary') for r in samples])&(presence[:,0]==peg)&(presence[:,1]==box)
                    presence_metrics.append(dict(probe_identity=key,split=part,peg_present=peg,box_present=box,**absolute(yy[mask],pred[mask])))
        for group in ('primary','additional','box_control','repeat','baseline'):
            mask=np.array([r['split']=='final' and r['group']==group for r in samples])
            secondary.append(dict(probe_identity=key,group=group,**absolute(yy[mask],pred[mask])))
    csv_write(root/'per_sample_predictions.csv',sample_rows);write_json(root/'final_condition_groups.json',secondary);write_json(root/'presence_metrics.json',presence_metrics)
    # Descriptive episode macro means: perturbations never counted as episodes.
    episodes=csv_rows(root/'per_episode.csv');macro=[]
    for model in LABELS:
        for probe in ('P0','P1','P2','P3','P4'):
            for decoder in ('linear','mlp'):
                for part in ('train','validation','final'):
                    e=[r for r in episodes if r['model']==model and r['probe']==probe and r['decoder']==decoder and
                        r['control']=='normal' and r['split']==part and r['mean_mm']!='']
                    macro.append(dict(model=model,probe=probe,decoder=decoder,split=part,episodes=len(e),
                        mean_episode_error_mm=float(np.mean([float(r['mean_mm']) for r in e])) if e else None))
    write_json(root/'episode_macro_metrics.json',macro)
    return metrics,groups,pairs,counts


def metric(ms,model,probe,decoder,part='validation',control='normal',validity='VISIBLE_VALID'):
    return next(r for r in ms if r['model']==model and r['probe']==probe and r['decoder']==decoder and
        r['split']==part and r['control']==control and r['validity']==validity)


def dmetric(ds,model,probe,decoder,part='final',control='normal',t=None):
    return next(r for r in ds if r['model']==model and r['probe']==probe and r['decoder']==decoder and
        r['split']==part and r['control']==control and r['t']==t)


def figures(root,metrics,ds,pairs,counts):
    figdir=root/'figures';figdir.mkdir(exist_ok=True);data=load(root/'frozen_readouts320.npz')
    y=data['target_peg_position_tcp'];samples=read(root/'samples.json')
    valid=np.array([r['physical_valid'] and r['validity']=='VISIBLE_VALID' and r['group'] in ('baseline','primary') for r in samples])
    val=valid&(data['split']=='validation');final=valid&(data['split']=='final')
    paths=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout();fig.savefig(figdir/name,dpi=155,bbox_inches='tight');plt.close(fig);paths.append('figures/'+name)
    # Actual/predicted absolute XYZ, independent validation, all frozen checkpoints.
    fig,axs=plt.subplots(6,3,figsize=(12,18))
    for row,m in enumerate(LABELS):
        p=load(root/'probes'/f'{m}_P1_linear_normal_predictions.npz')['prediction']
        for ax,k in zip(axs[row],range(3)):
            ax.scatter(y[val,k]*1000,p[val,k]*1000,s=8,alpha=.5)
            lim=np.r_[y[val,k],p[val,k]]*1000;lo,hi=lim.min(),lim.max();ax.plot([lo,hi],[lo,hi],'k--',lw=1)
            ax.set(title=f'{m} P1 linear {"XYZ"[k]}',xlabel='Actual (mm)',ylabel='Predicted (mm)')
    save(fig,'actual_predicted_xyz.png')
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for ax,mask,title in zip(axs,[val,final],['Independent validation episodes','09A seed30003 final primary']):
        errors=[];labels=[]
        for m in LABELS:
            for dec in ('linear','mlp'):
                p=load(root/'probes'/f'{m}_P1_{dec}_normal_predictions.npz')['prediction']
                errors.append(np.linalg.norm(p[mask]-y[mask],axis=1)*1000);labels.append(m+'\n'+dec)
        ax.boxplot(errors,tick_labels=labels,showfliers=True);ax.tick_params(axis='x',rotation=60);ax.set(title=title,ylabel='Absolute position error (mm)')
    save(fig,'absolute_error_distribution.png')
    fig,axs=plt.subplots(2,3,figsize=(16,8))
    for ax,m in zip(axs.ravel(),LABELS):
        rr=[r for r in pairs if r['model']==m and r['probe']=='P1' and r['decoder']=='linear' and r['control']=='normal' and r['split']=='final' and r['primary_valid']]
        for k,col in zip('xyz',COLORS):
            ax.scatter([r[f'true_d{k}_mm'] for r in rr],[r[f'predicted_d{k}_mm'] for r in rr],s=22,label=k.upper(),color=col)
        ax.plot([-13,13],[-13,13],'k--',lw=1);ax.set(title=m+' P1 linear',xlabel='True delta (mm)',ylabel='Decoded delta (mm)');ax.legend()
    save(fig,'delta_true_predicted.png')
    fig,axs=plt.subplots(1,3,figsize=(16,5));x=np.arange(6)
    for j,probe in enumerate(('P0','P1','P2','P4')):
        vals=[dmetric(ds,m,probe,'linear') for m in LABELS]
        for ax,key in zip(axs,['delta_error_mm','direction_cosine','signed_gain']):
            ax.bar(x+(j-1.5)*.19,[v[key] if v[key] is not None else np.nan for v in vals],width=.19,label=probe,color=COLORS[j])
            ax.set_xticks(x,LABELS,rotation=35);ax.set_title('Final primary linear '+key)
    axs[1].set_ylim(-1,1);axs[1].text(.02,.02,'P0 cosine undefined: zero response',transform=axs[1].transAxes,fontsize=8); axs[2].axhline(1,ls='--',color='k');axs[0].legend();save(fig,'delta_alignment.png')
    fig,axs=plt.subplots(1,2,figsize=(15,5));x=np.arange(6)
    for ax,decoder in zip(axs,['linear','mlp']):
        for j,p in enumerate(('P0','P1','P2','P3','P4')):
            ax.bar(x+(j-2)*.16,[metric(metrics,m,p,decoder)['mean_mm'] for m in LABELS],width=.16,label=p,color=COLORS[j])
        ax.set_xticks(x,LABELS,rotation=35);ax.set(title='Validation '+decoder,ylabel='Position error (mm)');ax.legend(ncol=5,loc='upper center',bbox_to_anchor=(.5,1.16),fontsize=8)
    save(fig,'robot_readout_combined_raw.png')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,part in zip(axs,['validation','final']):
        for m in LABELS:
            ax.scatter(metric(metrics,m,'P4','linear',part)['mean_mm'],metric(metrics,m,'P1','linear',part)['mean_mm'],s=55,label=m)
        ax.set(title=part+' P4 centroid ridge vs P1 ridge',xlabel='Raw geometry decoded error (mm)',ylabel='Readout decoded error (mm)');ax.legend(fontsize=8)
    save(fig,'raw_vs_readout.png')
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,dec in zip(axs,['linear','mlp']):
        for p,col in zip(('P1','P2','P3'),COLORS):
            for series,style in [('Full','o-'),('Single','s--')]:
                ax.plot(range(3),[metric(metrics,f'{series}_{m}',p,dec)['mean_mm'] for m in 'ABC'],style,label=series+' '+p,color=col)
        ax.set(xticks=range(3),xticklabels=list('ABC'),title='Independent validation '+dec,ylabel='Position error (mm)');ax.legend(fontsize=8)
    save(fig,'full_vs_single.png')
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for ax,dec in zip(axs,['linear','mlp']):
        for part,style in [('train','o-'),('validation','s--'),('final','^:')]:
            ax.plot(range(6),[metric(metrics,m,'P1',dec,part)['mean_mm'] for m in LABELS],style,label=part)
        ax.set_xticks(range(6),LABELS,rotation=35);ax.set(title='P1 '+dec,ylabel='Position error (mm)');ax.legend()
    save(fig,'train_validation_final.png')
    fig,axs=plt.subplots(1,2,figsize=(15,5));cats=['VISIBLE_VALID','VISIBLE_MISSING','OCCLUDED','GEOMETRY_CONTAMINATED']
    for j,part in enumerate(('train','validation','final')):
        axs[0].bar(np.arange(4)+(j-1)*.24,[next(r['n'] for r in counts if r['split']==part and r['validity']==v) for v in cats],width=.24,label=part)
    axs[0].set_xticks(range(4),cats,rotation=20);axs[0].set_title('All states retained, including repeats/secondary');axs[0].legend()
    for m in LABELS:
        vals=[metric(metrics,m,'P1','linear','final',validity=v) for v in cats]
        axs[1].plot(range(4),[v['mean_mm'] if v['n'] else np.nan for v in vals],'o-',label=m)
    axs[1].set_xticks(range(4),cats,rotation=20);axs[1].set(ylabel='Position error (mm)',title='Final primary by peg validity');axs[1].legend(fontsize=8)
    save(fig,'geometry_validity.png')
    # Presence versus validity, data recorded outside decoder inputs.
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    pres=data['Full_B_presence2'].astype(bool)
    for ax,k,name in zip(axs,[0,1],['Peg','Box']):
        vv=[r['validity'] if k==0 else r['box_validity'] for r in samples]
        mat=np.array([[sum(vv[i]==cat and pres[i,k]==p for i in range(len(samples))) for p in [False,True]] for cat in cats])
        im=ax.imshow(mat,cmap='Blues',aspect='auto');ax.set(yticks=range(4),yticklabels=cats,xticks=[0,1],xticklabels=['absent','present'],title=name+' presence vs GT audit')
        for i in range(4):
            for j in range(2):ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='white' if mat[i,j]>mat.max()*.5 else 'black')
    save(fig,'presence_validity.png')
    old=read(TASK09A/'per_condition_metrics.json');fig,axs=plt.subplots(1,3,figsize=(16,5))
    for m in LABELS:
        gain=[dmetric(ds,m,'P1','linear',t=t)['signed_gain'] for t in (0,10,25)]
        axs[0].plot([0,10,25],gain,'o-',label=m)
        oldrows=[[r for r in old if r['model']==m and r['t']==t and r['group']=='primary' and r['alignment_eligible']] for t in (0,10,25)]
        axs[1].plot([0,10,25],[np.mean([r['translation_signed_gain'] for r in rr]) for rr in oldrows],'o-',label=m)
        axs[2].plot([0,10,25],[np.mean([r['readout_relative_l2'] for r in rr])*100 for rr in oldrows],'o-',label=m)
    for ax,title in zip(axs,['09B position decoder signed gain','09A policy action signed gain','09A readout change (%)']):ax.set(title=title,xlabel='Observation timestep',xticks=[0,10,25]);ax.legend(fontsize=8)
    save(fig,'task09a_timestep_response.png')
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    for ax,dec in zip(axs,['linear','mlp']):
        for control,style in [('normal','o-'),('shuffle','s--')]:
            ax.plot(range(6),[metric(metrics,m,'P1',dec,control=control)['mean_mm'] for m in LABELS],style,label=control)
        ax.set_xticks(range(6),LABELS,rotation=35);ax.set(title='Train-label shuffle, validation '+dec,ylabel='Position error (mm)');ax.legend()
    save(fig,'label_shuffle.png')
    write_json(root/'figure_manifest.json',paths);return paths


def verify(root):
    root=Path(root);errors=[];counts=Counter();samples=read(root/'samples.json');data=load(root/'frozen_readouts320.npz')
    def fail(p,e):errors.append(dict(path=str(p),error=type(e).__name__,detail=str(e)))
    for p in root.rglob('*'):
        if not p.is_file() or 'source_snapshot' in p.parts:continue
        try:
            if p.suffix=='.json':read(p);counts['json']+=1
            elif p.suffix=='.csv':assert csv_rows(p);counts['csv']+=1
            elif p.suffix=='.npz':
                d=load(p);assert d
                for k,v in d.items():assert not v.dtype.hasobject
                if p.name in ('features.npz','frozen_readouts320.npz'):
                    for k,v in d.items():
                        if v.dtype.kind in 'fc':assert np.isfinite(v).all(),k
                counts['npz']+=1
            elif p.suffix=='.png':assert cv2.imread(str(p),cv2.IMREAD_UNCHANGED) is not None;counts['png']+=1
            elif p.suffix=='.html':
                for link in re.findall(r'(?:href|src)="([^"]+)"',p.read_text()):
                    if not link.startswith(('http:','https:','#','data:')):assert (p.parent/link).is_file(),link;counts['html_links']+=1
        except Exception as e:fail(p,e)
    for csvname,jsonname in [('absolute_metrics.csv','probe_metrics.json'),('delta_metrics.csv','delta_metrics.json')]:
        try:
            actual=csv_rows(root/csvname);expected=read(root/jsonname);assert len(actual)==len(expected)
            for cr,jr in zip(actual,expected):
                for k,v in jr.items():
                    if v is None:assert cr[k]==''
                    elif isinstance(v,(int,float)) and not isinstance(v,bool):assert float(cr[k])==float(v)
                    else:assert cr[k]==str(v)
        except Exception as e:fail(csvname,e)
    source=[];repeat_audits=[]
    from graph.integration_baseline.graph import build_graph
    for i,r in enumerate(samples):
        try:
            folder=root/r['sample_id'];f=load(folder/'features.npz');src=Path(r['source_directory']) if r['split']=='final' else folder
            if not src.exists() and r['split']=='final':src=root.parent/TASK09A.name/f't{r["t"]:02d}'/r['condition']
            phys=read(src/'physical.json');state=phys['after']
            # Live labels used original float32 GT subtraction; imported09A labels used JSON float64 inputs.
            peg=np.asarray(state['peg_pose'],np.float32) if r['split']!='final' else state['peg_pose']
            tcp=np.asarray(state['ee_pose'],np.float32) if r['split']!='final' else state['ee_pose']
            yy=target(peg,tcp)
            rounding=float(np.max(abs(yy-target(state['peg_pose'],state['ee_pose']))));assert rounding<=5e-8
            np.testing.assert_array_equal(f['target_peg_position_tcp'],yy)
            for k,v in f.items():np.testing.assert_array_equal(v,data[k][i])
            assert sha(folder/'features.npz')==r['feature_sha256']
            ob=load(src/'observation.npz');packet=read_packet(src/'packet.npz');g=load(src/'graph.npz')
            assert sha(src/'graph.npz')==r['graph_sha256'];assert sha(src/'packet.npz')==r['packet_sha256']
            np.testing.assert_array_equal(ob['robot_state'],f['robot76'])
            assert packet['observation_provenance']['timestep']==r['t'] and not packet['observation_provenance']['gt_used_as_student_input']
            assert not packet['observation_provenance']['future_observations_used']
            rebuilt=build_graph(packet,ob['robot_ee_geometry'],k=4,M=16,block_rows=64)
            for k in g:np.testing.assert_array_equal(g[k],getattr(rebuilt,k).numpy())
            rows=packet['provenance']['source_index'];v,u=np.divmod(rows,ob['depth_raw_mm'].shape[1]);z=ob['depth_raw_mm'][v,u].astype(np.float32)/1000
            intr=ob['intrinsic'].astype(float);camera=np.column_stack(((u+.5-intr[0,2])*z/intr[0,0],(v+.5-intr[1,2])*z/intr[1,1],z)).astype(np.float32)
            tr=packet['T_EE_from_camera'];xyz=(camera.astype(float)@tr[:3,:3].T+tr[:3,3]).astype(np.float32)
            gap=float(np.max(abs(xyz-packet['points']),initial=0));assert gap<=1e-6
            if r['group']=='repeat':
                baseline=src.parent/'baseline';bg=load(baseline/'graph.npz');bf=load(root/samples[next(j for j,vv in enumerate(samples) if vv['seed']==r['seed'] and vv['t']==r['t'] and vv['group']=='baseline')]['sample_id']/'features.npz')
                for kk in g:np.testing.assert_array_equal(g[kk],bg[kk])
                for mm in LABELS:np.testing.assert_array_equal(f[mm+'_readout320'],bf[mm+'_readout320'])
                repeat_audits.append(dict(sample_id=r['sample_id'],graph_readout_exact=True))
            for m in LABELS:
                assert f[m+'_readout320'].shape==(320,);np.testing.assert_array_equal(f[m+'_readout320'].reshape(5,64),f[m+'_slots5x64'])
            source.append(dict(sample_id=r['sample_id'],graph_exact=True,source_backprojection_max_m=gap,label_frame_exact=True,float32_label_reference_gap_m=rounding))
        except Exception as e:fail(r['sample_id'],e)
    masks=load(root/'probe_training_masks.npz');train=masks['train'];val=masks['validation'];y=data['target_peg_position_tcp'].astype(float)
    predrows=[]
    for p in (root/'probes').glob('*_checkpoint.npz'):
        try:
            name=p.stem.removesuffix('_checkpoint');m,method,probe,dec,control=name.split('_');model=m+'_'+method
            cp=load(p);x=features(data,model,probe).astype(float)
            np.testing.assert_allclose(cp['target_mean'],y[train].mean(0),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(cp['target_std'],np.maximum(y[train].std(0),.001),rtol=1e-12,atol=1e-12)
            selection=read(p.with_name(name+'_selection.json'));assert selection['final_used'] is False
            if dec=='linear':assert selection['selected_alpha']==min(selection['candidates'],key=lambda row:row['val_mse'])['alpha']
            else:assert selection['selected_weight_decay']==min(selection['candidates'],key=lambda row:row['val_mse'])['weight_decay']
            np.testing.assert_allclose(cp['feature_mean'],x[train].mean(0),rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(cp['feature_std'],np.maximum(x[train].std(0),1e-6),rtol=1e-12,atol=1e-12)
            xx=(x-cp['feature_mean'])/cp['feature_std']
            if dec=='linear':pp=xx@cp['weight']+cp['bias']
            else:
                net=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,3));net.load_state_dict({k:torch.from_numpy(v) for k,v in cp.items() if k[0].isdigit()});net.eval()
                with torch.no_grad():pp=net(torch.from_numpy(xx.astype(np.float32))).numpy().astype(float)
            pred=pp*cp['target_std']+cp['target_mean'];saved=load(p.with_name(name+'_predictions.npz'))
            np.testing.assert_allclose(pred,saved['prediction'],rtol=1e-6,atol=1e-7)
            np.testing.assert_array_equal(saved['target'],y.astype(np.float32))
            for part,mask in [('train',train),('validation',val)]:
                measured=absolute(y[mask],saved['prediction'][mask])
                expected=metric(read(root/'probe_metrics.json'),model,probe,dec,part,control)
                for k in measured:
                    if measured[k] is not None:assert np.isclose(measured[k],expected[k],atol=1e-4,rtol=1e-5),(name,part,k)
            predrows.append(dict(checkpoint=name,prediction_reload=True,train_normalization_exact=True))
        except Exception as e:fail(p,e)
    pair_predictions={}
    pair_indices={(s['seed'],s['t'],s['condition']):i for i,s in enumerate(samples)}
    for r in pair_rows(root):
        try:
            key=f"{r['model']}_{r['probe']}_{r['decoder']}_{r['control']}"
            path=root/'probes'/(key+'_predictions.npz')
            if key not in pair_predictions:pair_predictions[key]=load(path)['prediction']
            pred=pair_predictions[key];indices=pair_indices
            i=indices[(int(r['seed']),int(r['t']),r['condition'])];b=indices[(int(r['seed']),int(r['t']),'baseline')]
            # Saved predictions are FP32, metrics calculated before FP32 export: tolerance0.01um.
            measured=delta(y[i]-y[b],pred[i].astype(float)-pred[b].astype(float))
            for k in ('delta_error_mm','predicted_norm_mm','true_norm_mm'):assert np.isclose(measured[k],r[k],atol=1e-4,rtol=1e-5)
            np.testing.assert_array_equal(data['robot76'][i],data['robot76'][b])
        except Exception as e:fail('pair '+key,e)
    write_json(root/'repeat_graph_readout_audit.json',repeat_audits)
    write_json(root/'source_graph_readback_audit.json',source);write_json(root/'decoder_reload_audit.json',predrows)
    result=dict(pass_gate=not errors,counts=dict(counts),samples=len(samples),decoder_checkpoints=len(predrows),errors=errors,
        numerical_checks='JSON/CSV/NPZ targets,train normalization,decoder reload,position and paired delta metrics; actual source pixel depth backprojection and graph rebuild',
        png_decode=True,html_relative_links=True)
    write_json(root/'verification.json',result);return result


def fmt(v):return '—' if v is None else f'{v:.3f}'


def markdown_html(text):
    """Render this generated report's headings, tables, links and paragraphs."""
    def inline(value):
        value=html.escape(value)
        value=re.sub(r'`([^`]+)`',r'<code>\1</code>',value)
        value=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',value)
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',value)
    lines=text.splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        heading=re.match(r'^(#{1,6}) (.*)',line)
        if heading:
            level=len(heading[1]);out.append(f'<h{level}>'+inline(heading[2])+f'</h{level}>');i+=1
        elif line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[v.strip() for v in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',v) for v in cells):table.append(cells)
                i+=1
            out.append('<table>')
            for ri,cells in enumerate(table):
                tag='th' if ri==0 else 'td';out.append('<tr>'+''.join(f'<{tag}>'+inline(v)+f'</{tag}>' for v in cells)+'</tr>')
            out.append('</table>')
        elif line.startswith('- '):
            out.append('<ul>')
            while i<len(lines) and lines[i].startswith('- '):out.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            out.append('</ul>')
        else:
            paragraph=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','- ')):
                paragraph.append(lines[i]);i+=1
            out.append('<p>'+inline(' '.join(paragraph))+'</p>')
    return '\n'.join(out)


def report(root,metrics,ds,counts,figs):
    c=read(root/'protocol.json');aud=read(root/'probe_fit_audit.json');ids=read(root/'identities.json')
    replay=[read(root/'collection'/f'seed{s}'/'replay_audit.json') for s in c['train']+c['validation']]
    exact=sum(r['stored_observation_robot_tcp_exact'] for r in replay);reset=sum(r['reset_identity_exact'] for r in replay)
    abs_table=['| Model | Decoder | P0 Robot | P1 Readout | P2 Combined | P3 Object | P4 Raw mean |','|---|---|---:|---:|---:|---:|---:|']
    delta_table=['| Model | Decoder | P0 Δerr | P1 Δerr | P1 cosine | P1 gain | P2 Δerr | P4 Δerr |','|---|---|---:|---:|---:|---:|---:|---:|']
    validation_delta=['| Model | Decoder | P0 Δerr | P1 Δerr | P1 cosine | P1 gain | P2 Δerr | P4 Δerr |','|---|---|---:|---:|---:|---:|---:|---:|']
    for m in LABELS:
        for dec in ('linear','mlp'):
            abs_table.append('|'+m+'|'+dec+'|'+'|'.join(fmt(metric(metrics,m,p,dec)['mean_mm']) for p in ('P0','P1','P2','P3','P4'))+'|')
            for part,tab in [('final',delta_table),('validation',validation_delta)]:
                dd={p:dmetric(ds,m,p,dec,part) for p in ('P0','P1','P2','P4')}
                tab.append('|'+m+'|'+dec+'|'+'|'.join(fmt(v) for v in (dd['P0']['delta_error_mm'],dd['P1']['delta_error_mm'],dd['P1']['direction_cosine'],dd['P1']['signed_gain'],dd['P2']['delta_error_mm'],dd['P4']['delta_error_mm']))+'|')
    checkpoint=['| Model | Original SHA256 |','|---|---|']+[f"|{m}|`{ids[m]['receipt']['sha256']}`|" for m in LABELS]
    validity=['| Split | Peg validity | States | Physical invalid |','|---|---|---:|---:|']+[f"|{r['split']}|{r['validity']}|{r['n']}|{r['physical_invalid']}|" for r in counts]
    details=[]
    for m in LABELS:
        for dec in ('linear','mlp'):
            tr=metric(metrics,m,'P1',dec,'train');va=metric(metrics,m,'P1',dec);fi=metric(metrics,m,'P1',dec,'final');sh=metric(metrics,m,'P1',dec,control='shuffle')
            details.append(f"|{m}|{dec}|{fmt(tr['mean_mm'])}|{fmt(va['mean_mm'])}|{fmt(fi['mean_mm'])}|{fmt(sh['mean_mm'])}|{fmt(va['p95_mm'])}|{fmt(va['maximum_mm'])}|{fmt(va['mae_x_mm'])}/{fmt(va['mae_y_mm'])}/{fmt(va['mae_z_mm'])}|{fmt(va['r2_x'])}/{fmt(va['r2_y'])}/{fmt(va['r2_z'])}|")
    # Conclusions remain quantitative; no posthoc universal accuracy threshold.
    findings={}
    for m in LABELS:
        findings[m]={dec:dict(validation_readout_mm=metric(metrics,m,'P1',dec)['mean_mm'],
            validation_robot_mm=metric(metrics,m,'P0',dec)['mean_mm'],validation_combined_mm=metric(metrics,m,'P2',dec)['mean_mm'],
            validation_raw_mm=metric(metrics,m,'P4',dec)['mean_mm'],final_delta=dmetric(ds,m,'P1',dec),
            validation_delta=dmetric(ds,m,'P1',dec,'validation')) for dec in ('linear','mlp')}
    write_json(root/'findings.json',findings)
    outcome=(root/'interpretation.md').read_text().split('\n\n')[0] if (root/'interpretation.md').exists() else ''
    head=f'''# Task 09-B — Frozen GNN Readout Geometry Decodability Probe

ExperimentLogger UUID `{root.name}`. Vast.ai57640710 / RTX3090 frozen GNN 추론, CPU PhysX 및 CPU Vulkan readback, CPU Ridge/MLP 진단 학습. 원본 Policy/GNN/GRU/Action Head 학습·수정, 기존 Full 재개, 새 Policy Closed-loop 평가, 설치·Commit·외부 Publishing은 수행하지 않았다.

{outcome}

[정적 HTML](readout_geometry_probe.html) · [절대 오차](absolute_metrics.csv) · [Paired Δ](delta_metrics.csv) · [Sample/GT Manifest](samples.json) · [Frozen 감사](frozen_encoder_audit.json) · [검증](verification.json)

## 연구 질문과 Task09-A 연결

09-A에서 실제 Peg 이동에 Readout이 변했지만 행동 반응은 작거나 방향이 부정확했다. 이번에는 **동결된 Readout에서 위치를 꺼낼 수 있는가**를 별도 작은 Decoder로 측정한다. Decoder가 성공해도 원래 Policy가 해당 정보를 사용했다는 뜻은 아니다. Raw Geometry와 Robot-only 대조, 독립 Validation Episode 및 동일 Robot State의 실제 Peg 이동을 함께 비교한다. 유효성 그룹 밖 실패를 GNN 정보 손실로 단정하지 않는다.

## 고정 Identity·데이터 출처·분리

Shape-First/PCA5/F5, Raw EDE, k4/M16, Edge-aware GNN L2/d64, Structured Readout320, Presence2, Robot76, GRU144 및 기존 Action Decoder를 유지했다. Full A35/B48/C38, Single A/B/C Update20000은09-A identities.json 및 원본 Receipt와 정확히 대조했다.

'''+ '\n'.join(checkpoint)+f'''

원래 Train96 Robot76 Normalizer Identity는 `4e607d9a0af71ca65134f71b187946c2c47640bbe9f5cf81fb4942903ec52446`. `frozen_readouts320.npz`는 **정규화 전 원본 Readout320과 Robot76**을 저장한다. Probe용 각 Feature mean/std와 Target mean/std는 Probe Train의 물리 유효 VISIBLE_VALID Primary/Baseline에서만 계산했다. 기존 Policy Normalizer와 새 Probe Standardizer를 혼동하지 않는다. P1에는 Robot76을 직접 제공하지 않지만 TCP/Fingertip Node Slot이 포함된다. P3 실패만으로 전체 GNN 실패를 주장하지 않는다.

Direct HDF523개를 실제 감사했으며 시간별 Peg Pose는 없다. Reset Hash나 Robot/Action으로 Label을 만들지 않았다. Frozen Split의 **원래 순서**에서 Train16(30003 제외), Validation6을 결과 전에 선택했다. 각 Episode t=0/10/25, Baseline + World X/Y±5/±10mm + 동일 Pose 반복으로22 Episode×30=660개 새 실제 관측을 수집했다. 기존09-A60개는 독립 **Probe 최종 평가**로만 읽었다. 같은 Episode에서 파생된 모든 상태는 같은 Split이다.

- Probe Train: `{c['train']}`
- Probe Validation: `{c['validation']}`
- Final: Seed30003 / 기존09-A Artifact. Probe Fit·정규화·Hyperparameter 선택에 미사용.

Full Encoder는 기존 Train96을 학습했으므로 Probe Train Episode가 Encoder에게 새롭다는 뜻은 아니다. Single Encoder는 Seed30003을 원래 BC 학습했으므로 Final은 **Decoder에게만 미사용**이며 Encoder에게 OOD Episode가 아니다. Validation6은 Probe Train과 Episode/Reset Group이 독립이며 Full의 기존 Validation Split에서 선택했다. 기존 Test12는 사용하지 않았다.

최초 시도 `99f1dd3ff9cb4b9e81088b80ad62483a`는 첫 Seed30002의 Reset Hash 불일치로 Sample 생성 전에 중단됐다. 다음 시도 `61864f9940f44f55a123c8ec6e528e9e`도 첫 Vision 호출에서 CUBLAS 결정론 환경변수 누락으로 중단됐으며 Sample은 없었다. `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 설정한 뒤 현재 실행을 시작했다. 이를 무시하거나 Label을 보간하지 않고 새 실행에서 현재 Live ScriptedExpert Action을 사용했다. 모든 수집은 **LIVE_DIAGNOSTIC**이다. 현재 실행의 원본 Reset Hash 일치는 {reset}/22, 전체26개 저장 Robot/TCP Observation의 Exact 일치는 {exact}/22다. `replay_audit.json`에 시점별 Robot/TCP/Expert Action 차이를 보존한다. 원본 Dataset과 같은 Observation이라고 주장하지 않는다. 물리 개입은09-A의 Actor.set_pose·충돌 메시 SAT·0.2mm 허용오차·새 근접 접촉 금지 계약을 재사용했다. 개입 뒤 Physics Step 없이 새 RGB-D를 렌더링하고 원래 Pose로 복원한 뒤 Expert를 진행했다. Teleport 뒤 Fresh Contact Force나 동적 안정성을 검증한 것은 아니다.

## Target·동결·수집 계약

Target은 `R_TCP.T @ (t_PegActorOrigin - t_TCP)`의 float32 XYZ, metre다. Pose는 world `xyz,qw,qx,qy,qz`; 보고 수치는 mm. Point Cloud Centroid, GT Grasp/Goal/Hole Pose를 Target/Feature로 쓰지 않았다. GT Segmentation과 Pose는 Sidecar의 Label·유효성 감사에만 사용했다. 각 Sample의 Robot76, TCP/Actor GT, Packet/source pixels/F5/Raw EDE, Graph Hash, 여섯 Readout320/Slot5×64/Presence2, 실제 시점과 Episode Identity를 저장했다.

원본 여섯 모델 전 Parameter `requires_grad=False`, `eval()`, Encoder는 `torch.inference_mode()`로만 실행했다. Encoder 내부 Gradient/grad_fn 없음, 원본 전체·GNN·후속 Policy(GRU/Action 포함) Weight Hash의 수집 전후 및 Probe 후 동일성은 `frozen_encoder_audit.json`, `post_probe_frozen_audit.json`에 있다. Decoder Optimizer에는 독립 Decoder Parameter만 들어간다. Probe 모듈은 원본 Policy 객체를 받지 않고 NumPy 배열만 읽는다.

현재 Frame 처리 전 Causal Tracker Memory를 Deep-copy해 각 Counterfactual에 같은 과거를 제공했다. Predictor Weight/Perception 알고리즘은 변경하지 않았다. 전체 수집 전 첫 Episode의 t10에서 독립 Tracker Reset 후 실제 Prefix RGB를 재처리해 Mask/Readout Exact Equality를 확인했고, 모든 t와 Episode의 Baseline 반복 및 매 Graph의 Readout 반복도 Exact였다. `collection_contract.json`과 계약 테스트 로그를 보존한다. Source Depth/Calibration 역투영, Graph 재구성, Label/Observation 시점·좌표계 Readback을 별도로 검사한다.

## Probe와 고정 학습 예산

| Probe | Feature | 차원 |
|---|---|---:|
| P0 | Raw Robot76 |76|
| P1 | Frozen Structured Readout |320|
| P2 | Readout + Robot |396|
| P3 | Peg/Box Mean Slot |128|
| P4 | 관측 Peg EE XYZ 평균 |3|

P4 평균은 부분 표면 Centroid이므로 GT Origin과 같다고 가정하지 않는다. 같은 Point Set의 min/max/bbox center/span15도 저장했지만 P4 학습 입력은 평균3만이다. 별도 무학습 RawCentroid 직접 예측과 Constant(Train Target 평균)를 기록했다. Presence2는 기록/분류용이며 Decoder 입력에 추가하지 않았다.

Linear는 SVD Linear/Ridge, alpha `{c['linear']['alphas']}`. Small MLP는1 Hidden64/ReLU/Output3, full-batch MSE, AdamW lr0.001, weight_decay `{c['mlp']['weight_decay']}`, 최대2000 Epoch, patience150, min_delta1e-8. 동일 Train Target z-score(mean/std, floor1mm)를 두 Decoder에 사용했다. Feature std floor1e-6. 고정 Seed42, Train Label Shuffle Seed43. Validation Target 정규화 MSE 최소 후보/Checkpoint를 선택하며 같은 값이면 먼저 나온 후보를 택한다. Final은 선택에 미사용이다. 6모델×5입력×2Decoder×정상/Shuffle=120개 선택 결과, Ridge6·MLP2 후보 외 Capacity Search 없음.

학습 Primary는 물리 유효 VISIBLE_VALID Baseline/±5/±10mm이며 Repeat은 제외했다. 실제 Fit Row는 Train {aud['train_rows']}, Validation {aud['validation_rows']}. Perturbation Row를 독립 Episode로 세지 않는다. Population p-value, Mutual Information, 새로운 Policy 성공률을 측정하지 않는다. Episode macro 결과는 `episode_macro_metrics.json`, 각 Episode는 `per_episode.csv`에 있다.

## 관측 유효성

결과 전 고정: GT Visible Peg Pixel<20이면 OCCLUDED; 보이지만 Point0이면 VISIBLE_MISSING; Point<10 또는 선택 Point의 GT Purity<0.5면 GEOMETRY_CONTAMINATED; 나머지 VISIBLE_VALID.09-A Purity0.5 기준과 비교 가능하며 최소 Pixel/Point 수는 새 Config에서 사전 고정했다. Box 상태는 별도 저장해 Peg Primary를 Box 오염 때문에 자동 제거하지 않았다.09-A의 이동 방향 Centroid≥1mm 지원 여부도 Pair CSV에 별도로 남겼다. Primary Delta는 Baseline/변경 양쪽 물리 유효 VISIBLE_VALID에서 계산한다. 아래 Count는 Repeat과09-A 추가/Box 대조도 포함한다.

'''+ '\n'.join(validity)+'''

Object Presence 두 Bit의 네 조합별 절대 오차도 `presence_metrics.json`에 분리했다. Presence는 선택 Point 수로 결정되는 Student 신호로, GT 관측 가능성과 같지 않다. 모든 Missing/Contaminated/Occluded Prediction을 보존했으며 `absolute_metrics.csv`의 그룹별 결과와 `per_pair.csv`에서 확인할 수 있다. 부재 상태의 P4는 zero mean feature이며 해석상 유효 Geometry가 아니다. GT 감사 Mask를 Student Mask로 대체하거나 수집 뒤 Threshold를 조정하지 않았다.

## 독립 Validation 절대 위치 복원

평균 Euclidean Error, mm. 전체 Median/p90/p95/Max/축별 MAE·R²는 JSON/CSV에 있다.

'''+ '\n'.join(abs_table)+'''

| Model | Decoder | P1 Train | P1 Validation | P1 Final | Shuffle Validation | Val p95 | Val Max | Val MAE X/Y/Z | Val R² X/Y/Z |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
'''+ '\n'.join(details)+'''

## 동일 Robot State에서 실제 위치 변화 복원

새 수집에는 GRU를 실행하지 않았다. GNN은 현재 Graph만의 함수이므로 현재 위치 Probe에 Recurrent Hidden 입력은 없다. Final09-A에서는 기존 감사로 Pair의 Previous Hidden 동일성도 확인돼 있다.

다음 Δ 오차는 `||(prediction_alt-prediction_base)-(target_alt-target_base)||`, mm다. Direction Cosine은 Zero Response에서 정의하지 않으며 Signed Gain은 실제 Δ 방향으로 투영한 크기 비율이다. P0는 모든 Pair에서 Robot76이 같아 예측 Δ가 정확히0이다. X/Y World 개입의 TCP XYZ 응답을 측정했으며 **독립적인 Z 방향 개입 실험은 아니다**. Z 절대 복원 R²는 자연 상태 다양성의 결과로만 해석한다. Baseline의 Split별 XYZ 범위·표준편차는 `target_axis_support.json`에 별도 보존한다.

### 독립 Validation Episode Pair

'''+ '\n'.join(validation_delta)+'''

### 09-A Seed30003 최종 Primary Pair

'''+ '\n'.join(delta_table)+'''

실제/예측 Δ Norm, X/Y/Z 성분 오차, 방향, Gain, Baseline 반복 무이동 응답, 각 t0/10/25 및 모든 Invalid/Additional/Box 조건은 `per_pair.csv`, `delta_metrics.json`과 Prediction NPZ에 있다. Final Primary와 Additional/Rotation/Box를 섞어 단일 수치로 보고하지 않았다. 추가/Box/Repeat의 절대 오차는 `final_condition_groups.json`, 모든 상태별 절대 오차·Target·Prediction은 `per_sample_predictions.csv`에도 보존한다.

## 판정과 해석

이 문서의 아래 판정은 연속 오차·P0/P4 대조와 독립 Episode 결과에 근거한 진단이다. Validation은 Hyperparameter 선택에도 사용돼 낙관 편향이 있을 수 있으며, 독립 Final도 한 Seed여서 모집단 일반화 수치를 확정하지 않는다. Small MLP가 실패해도 정보 소실의 수학적 증명이 아니며, Train Fit만으로 표현 충분성을 주장하지 않는다. Linear/MLP의 Feature 차원·용량 차이가 있으므로 엄밀한 정보 이론적 MI 비교가 아니다. Task09-A의 Action 실패 원인은 이번 Probe만으로 한 모듈로 확정할 수 없다.

판정별 근거와 Q1–Q10 답변은 이어지는 수치 검토에서 작성한다.

## Artifact·실행 범위

원격 실행은 `.git` 없는 격리 Source Copy여서 Logger의 Git Commit은 null일 수 있다. 원본 로컬 Git HEAD/Branch/Dirty와 배포 Identity는 `deployment_provenance.json`, 실제 실행 Source Hash는 `source_hashes.json`, 사후 보고서/추가 테스트 Hash는 `final_source_identity.json`에 구분했다.

원격 `/workspace/task09b/experiments/'''+root.name+'''/`, 로컬 `experiments/'''+root.name+'''/`. Config/Metadata, 원본 Checkpoint Receipt/Identity, Source/Input Hash·Snapshot·Diff, Split Manifest, Live Diagnostic GT/RGB-D/Mask/Packet/Graph, Frozen Readout/Slot/Robot, 120개 Probe Checkpoint·Selection·Prediction NPZ, Per-episode/Per-pair CSV, JSON, PNG, HTML, 계약 로그와 `verification.json`을 보존한다. 기존09-A와 Input/Checkpoint/Full 중지 Status 보호 Hash는 `protected_before.json`, `protection_audit.json`에 있다. 기존09-A 파일과 정책 가중치는 덮어쓰지 않았다.

실행: RTX3090 Frozen Encoder, CPU PhysX/CPU render, 실제 Predicted Vision, 독립22 Episode 수집, Linear/Small MLP 진단 학습. 미실행: CUDA PhysX, macOS/MPS Simulator, 새로운 Policy 학습/Closed-loop 성공 평가, Rotation Probe, GRU Hidden Probe, Z 개입, 추가 Seed 확장.07-G는 다른 Checkpoint의 Sampling 안정성 자료이고09-B의 위치 Decodability와 동일 Noise 강도로 비교하지 않는다.08-B/08-D의 Single B/C 성공도 원래 Policy의 현재 Geometry 활용을 입증하지 않는다.08-E/08-F의 Full Fit 부족·오류 누적 증거는 별도로 유지한다.
'''
    if (root/'interpretation.md').exists():
        head=head.replace('판정별 근거와 Q1–Q10 답변은 이어지는 수치 검토에서 작성한다.',(root/'interpretation.md').read_text())
    (root/'09b_readout_geometry_probe.md').write_text(head)
    sections=['<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Task09B frozen readout geometry probe</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 24px;background:#f7f8fa;color:#172531}img{width:100%;background:white}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:24px}a{color:#2166a1}section{margin:35px 0}table{display:block;overflow-x:auto;border-collapse:collapse;background:white;font-size:14px;margin:20px 0}td,th{border:1px solid #d7dfe8;padding:8px 10px}th{background:#e7eef5}code{background:#eef1f5;padding:1px 4px;overflow-wrap:anywhere}article h2{margin-top:40px}article p{max-width:1100px}</style>',
        f'<h1>Task09B Frozen Readout Geometry Probe</h1><p>Run {root.name} · independent16 train /6 validation episodes;09A final30003</p>',
        '<p><a href="09b_readout_geometry_probe.md">한국어 연구 보고서</a> · <a href="verification.json">Verification</a> · <a href="findings.json">Findings</a></p>']
    sections.append(markdown_html(outcome))
    for p in figs:sections.append(f'<section><h2>{html.escape(Path(p).stem)}</h2><img loading="lazy" src="{p}"></section>')
    links=['protocol.json','identities.json','split_manifest.json','source_data_audit.json','frozen_encoder_audit.json',
        'post_probe_frozen_audit.json','probe_fit_audit.json','protection_audit.json','collection_contract.json',
        'probe_metrics.json','absolute_metrics.csv','per_sample_predictions.csv','final_condition_groups.json','delta_metrics.json','delta_metrics.csv','per_episode.csv','per_pair.csv',
        'episode_macro_metrics.json','validity_counts.json','frozen_readouts320.npz','probe_training_masks.npz',
        'source_hashes.json','implementation.diff','contract_tests.log','decoder_reload_audit.json','source_graph_readback_audit.json']
    sections.append('<h2>Artifacts</h2><ul>'+''.join(f'<li><a href="{p}">{p}</a></li>' for p in links)+'</ul>')
    sections.append('<article id="report">'+markdown_html(head)+'</article></html>')
    (root/'readout_geometry_probe.html').write_text('\n'.join(sections))


def main(root):
    root=Path(root);metrics,ds,pairs,counts=summarize(root)
    figs=figures(root,metrics,ds,pairs,counts);report(root,metrics,ds,counts,figs)
    verification=verify(root);print(json.dumps(verification),flush=True)
    if not verification['pass_gate']:raise AssertionError('readback failed')
    for path in (root/'readout_geometry_probe.html',root/'09b_readout_geometry_probe.md',*list((root/'figures').glob('*.png'))):
        register_existing_output(root,path,kind='reports')
    write_json(root/'status.json',dict(status='REPORT_READY_FOR_REVIEW',run_id=root.name,verification=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
