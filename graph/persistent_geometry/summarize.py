"""Aggregate saved metrics without changing candidates or selection rules."""
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from graph.integration_baseline.io import write_json
from graph.integration_baseline.geometry09b_probe import csv_write
from .render import table,read_npz,COLORS


def numeric_mean(rows,key):
    values=[float(r[key]) for r in rows if r.get(key) not in ('',None)]
    return float(np.mean(values)) if values else None


def grouped(rows,keys,metrics):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row[k] for k in keys)].append(row)
    return [{**dict(zip(keys,key)),'frames':len(values),
             **{name:numeric_mean(values,name) for name in metrics},
             **{name+'_defined_n':sum(v.get(name) not in ('',None) for v in values) for name in metrics}}
            for key,values in sorted(groups.items())]


def source_paths(state):
    return {entry['stage']:Path(entry['directory']) for entry in state['completed']}


def collect_summary(state,destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    paths=source_paths(state)
    geometry_fields=('count','surface_precision_4mm','surface_distance_mean_mm','surface_distance_p95_mm',
        'common_both_visible_coverage_4mm','common_union_observed_coverage_4mm','source_object_purity',
        'fused_source_object_purity','source_robot_contamination','source_background_contamination',
        'source_wrong_object_contamination','ghost_count','estimated_correct','persistent_estimated',
        'normal_consistency','object_local_surface_drift_mean_mm','pose_translation_error_mm',
        'pose_rotation_error_deg','motion_update_error_mm','uncertainty_mean_mm',
        'newly_added','newly_added_correct','new_correct_previously_unseen_nodes',
        'duplicate_neighbour_fraction_1mm','low_confidence_error_mm','medium_confidence_error_mm','high_confidence_error_mm')
    geo=[];filters=[];boundary=[];splits=[];costs=[];missing=[];matched_support=[];unknown=[];depth=[]
    for label,geometry_stage,capture_stage,audit_stage in (
            ('development','development_geometry','development_capture','development_boundary_audit'),
            ('independent','independent_geometry','independent_capture','independent_boundary_audit')):
        root=paths[geometry_stage];capture=paths[capture_stage]
        rows=table(root/'geometry_metrics.csv')
        geo.extend(dict(dataset=label,**r) for r in grouped(rows,('split','variant','budget_mode','object_id'),geometry_fields))
        filt=table(root/'robot_filter_metrics.csv')
        fields=('robot_removal_recall','object_preservation','confirmed_false_removal','uncertain_object_fraction',
                'robot_contamination_before','robot_contamination_after','object_purity_before','object_purity_after',
                'contact_near_false_removal','object_missing_before','object_missing_after')
        # Boolean CSVs are converted explicitly; empty denominators remain empty.
        for r in filt:
            for key in ('object_missing_before','object_missing_after'):r[key]=int(r[key]=='True')
        filters.extend(dict(dataset=label,**r) for r in grouped(filt,('split','camera','object_id'),fields))
        edge=table(paths[audit_stage]/'graph_boundary_metrics.csv')
        edge=[r for r in edge if r['condition']=='baseline']
        boundary.extend(dict(dataset=label,**r) for r in grouped(edge,('split','variant','budget','object_id'),
            ('nodes','undirected_surface_edges','connected_components','mesh_unsupported_edges_4mm',
             'camera_free_space_crossing_edges','aperture_crossing_edges','aperture_full_rim_coverage_4mm',
             'aperture_observed_rim_coverage_4mm','edge_length_mean_mm','edge_length_p95_mm')))
        captured=json.loads((capture/'capture_manifest.json').read_text())
        for episode in captured['episodes']:
            splits.append(dict(dataset=label,seed=episode['seed'],split=episode['split'],
                frames=len(episode['frames']),actual_reset_group=episode['actual_reset_group'],
                original_reset_match=episode['original_reset_match'],branches=len(episode['branches']),
                physically_valid_branches=sum(b['physical_valid'] for b in episode['branches'])))
        frame_manifest=json.loads((root/'geometry_manifest.json').read_text())
        fusion=[]
        for frame in frame_manifest['frames']:
            fusion.extend(dict(seed=frame['seed'],split=frame['split'],**r) for r in
                json.loads((root/frame['folder']/'fusion_audit.json').read_text()))
            source=capture/'collection'/f'seed{frame["seed"]}'
            with np.load(source/f'f{frame["timestep"]:04d}_input.npz',allow_pickle=False) as a:
                for key in a.files:
                    if key.endswith('/depth_raw_mm'):
                        values=a[key];depth.append(dict(dataset=label,seed=frame['seed'],camera=key.split('/')[0],
                            invalid_fraction=float((values<=0).mean()),zero_pixels=int((values==0).sum()),
                            negative_pixels=int((values<0).sum()),pixels=values.size))
        for key in sorted({(r['split'],r['camera'],r['object_id']) for r in fusion}):
            rr=[r for r in fusion if (r['split'],r['camera'],r['object_id'])==key]
            total=sum(r['count'] for r in rr);rejected=sum(r['unknown_conflict'] for r in rr)
            unknown.append(dict(dataset=label,split=key[0],camera=key[1],object_id=key[2],
                observed_samples=total,unknown_conflict=rejected,unknown_conflict_fraction=rejected/total if total else None,
                scope='cross-view depth-supported conflicting object identity; robot-filter uncertainty is separate'))
        matched=[json.loads((root/f['folder']/'matched_count_contract.json').read_text()) for f in frame_manifest['frames']]
        matched_support.append(dict(dataset=label,frames=len(matched),
            both_objects_eligible=sum(m['eligible_for_nonempty_geometry_ranking'] for m in matched),
            peg_nonempty=sum(m['counts']['0']>0 for m in matched),box_nonempty=sum(m['counts']['1']>0 for m in matched),
            scope='exact matched counts chosen from practical G0–G4 predictions; primary GNN uses common cap instead'))
        for episode in captured['episodes']:
            rr=[f for f in frame_manifest['frames'] if f['seed']==episode['seed']]
            times=np.array([f['elapsed_seconds'] for f in rr])
            costs.append(dict(dataset=label,seed=episode['seed'],frames=len(rr),
                geometry_worker_frame_mean_seconds=float(times.mean()),geometry_worker_frame_p95_seconds=float(np.percentile(times,95)),
                capture_frame_mean_seconds=episode['frame_wall_mean_s'],capture_episode_seconds=episode['elapsed_seconds']))
        for split in sorted(set(r['split'] for r in rows)):
            for oid in ('0','1'):
                for variant in ('G0','G1','G2','G3','G4'):
                    rr=[r for r in rows if r['variant']==variant and r['object_id']==oid and r['split']==split and r['budget_mode']=='cap128_per_object']
                    missing.append(dict(dataset=label,split=split,variant=variant,object_id=oid,frames=len(rr),
                        empty_input_frames=sum(int(r['count'])==0 for r in rr),
                        GT_visible_and_empty=sum(int(r['count'])==0 and int(r['common_both_visible_reference'])>0 for r in rr)))
    shape=[];identities=[]
    models=json.loads((paths['diagnostic_training']/'model_manifest.json').read_text())['models']
    for model in models:
        identity=json.loads((paths['diagnostic_training']/model['directory']/'model_identity.json').read_text())
        identities.append(identity)
        for label,root in (('development',paths['diagnostic_training']),('independent',paths['frozen_final_application'])):
            a=read_npz(root/model['directory']/'frozen_features.npz')
            for split in sorted(set(a['split'])):
                selected=(a['split']==split)&a['physical_valid']&(a['condition']!='baseline_repeat')
                for oid in (0,1):
                    valid=selected&a['target_valid'][:,6+oid*15]
                    error=a['prediction36'][valid,6+oid*15:6+(oid+1)*15]-a['target36'][valid,6+oid*15:6+(oid+1)*15]
                    shape.append(dict(dataset=label,split=str(split),variant=model['variant'],encoder=model['encoder'],object_id=oid,
                        n=int(valid.sum()),visible_summary15_MAE_mm=float(abs(error).mean()*1000) if len(error) else None,
                        visible_centroid_error_mm=float(np.linalg.norm(error[:,:3],axis=1).mean()*1000) if len(error) else None,
                        visible_span_error_mm=float(np.linalg.norm(error[:,-3:],axis=1).mean()*1000) if len(error) else None))
    for name,rows in (('geometry_summary.csv',geo),('robot_filter_summary.csv',filters),('boundary_summary.csv',boundary),
                     ('split_episode_summary.csv',splits),('compute_costs.csv',costs),('input_missing_summary.csv',missing),('surface_decoder_metrics.csv',shape)):
        csv_write(destination/name,rows)
    csv_write(destination/'fusion_unknown_summary.csv',unknown)
    depth_summary=grouped(depth,('dataset','seed','camera'),('invalid_fraction','zero_pixels','negative_pixels','pixels'))
    csv_write(destination/'depth_invalidity_summary.csv',depth_summary)
    costs_stage=[]
    for stage,root in paths.items():
        metadata=json.loads((root/'metadata.json').read_text())
        seconds=(datetime.fromisoformat(metadata['finished_at'])-datetime.fromisoformat(metadata['started_at'])).total_seconds()
        costs_stage.append(dict(stage=stage,run_id=root.name,wall_seconds=seconds))
    summary=dict(stage_costs=costs_stage,models=identities,geometry=geo,robot_filters=filters,boundaries=boundary,
                 splits=splits,input_missing=missing,shape_decoder=shape,matched_count_support=matched_support,
                 fusion_unknown=unknown,depth_invalidity=depth_summary,
                 statistic='unweighted frame mean within split/object/variant; undefined distance/purity excluded with defined_n; zero-support coverage is zero')
    write_json(destination/'benchmark_summary.json',summary)
    return summary


def figures(summary,paths,destination):
    import matplotlib.pyplot as plt
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(14,7),dpi=140)
    fields=(('surface_distance_mean_mm','Mean surface distance (mm)'),
            ('common_both_visible_coverage_4mm','Current both-camera GT coverage'),('ghost_count','Historical ghosts >8mm'))
    for oid in (0,1):
        for ax,(field,title) in zip(axes[oid],fields):
            rr=[r for r in summary['geometry'] if r['dataset']=='independent' and r['budget_mode']=='cap128_per_object' and r['object_id']==str(oid)]
            names=[r['variant'] for r in rr]
            ax.bar(names,[r[field] if r[field] is not None else np.nan for r in rr],color=[COLORS[n] for n in names],alpha=.85)
            for i,r in enumerate(rr):
                if r[field] is None:ax.text(i,0,'no support',rotation=90,fontsize=7,ha='center',va='bottom')
            ax.set_title(('Peg' if oid==0 else 'Box')+': '+title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Independent episodes; common cap128/object; oracle conditions separate O1–O3; missing-input counts in CSV')
    fig.tight_layout();fig.savefig(destination/'independent_geometry_comparison.png');plt.close(fig)
    # Show absolute and same-robot movement restoration separately, with the
    # unobservable fully hidden interventions explicitly excluded from the
    # visible-pair panel and retained in the machine-readable metrics.
    absolute=table(paths['frozen_final_application']/'absolute_metrics.csv')
    pairs=table(paths['frozen_final_application']/'paired_delta_metrics.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,8),dpi=140)
    for oid in (0,1):
        for encoder,offset,color in (('E0',-.22,'#0072b2'),('E1',0.,'#e69f00'),('E2',.22,'#009e73')):
            values=[];movement=[]
            for name in ('G0','G1','G2','G3','G4'):
                rr=[r for r in absolute if r['variant']==name and r['encoder']==encoder and r['probe']=='P1_readout' and r['decoder']=='linear' and r['object_id']==str(oid) and r['split']=='evaluation' and r['condition_group']=='baseline']
                values.append(float(rr[0]['mean_mm']) if rr and rr[0]['mean_mm'] else np.nan)
                pp=[r for r in pairs if r['variant']==name and r['encoder']==encoder and r['probe']=='P1_readout' and r['decoder']=='linear' and r['object_id']==str(oid) and r['split']=='evaluation' and r['physical_valid']=='True' and r['target_moved']=='True' and r['visibility_stratum']=='BOTH_VISIBLE']
                movement.append(numeric_mean(pp,'delta_error_mm'))
            axes[oid,0].bar(np.arange(5)+offset,values,width=.22,color=color,label=encoder)
            axes[oid,1].bar(np.arange(5)+offset,[v if v is not None else np.nan for v in movement],width=.22,color=color,label=encoder)
        for ax in axes[oid]:ax.set_xticks(range(5),('G0','G1','G2','G3','G4'));ax.grid(axis='y',alpha=.2)
        axes[oid,0].set_title(('Peg' if oid==0 else 'Box')+' baseline position error(mm)')
        axes[oid,1].set_title(('Peg' if oid==0 else 'Box')+' visible moved-pair delta error(mm)')
    axes[0,0].legend();fig.suptitle('Frozen Ridge on GNN full readout; same development selection, newly captured final episodes')
    fig.tight_layout();fig.savefig(destination/'gnn_absolute_and_visible_delta.png');plt.close(fig)
    # Readout and object-slot geometry, using the same final frames. PCA is a
    # descriptive plot only; it is neither a fitted probe nor a selection input.
    fig,axes=plt.subplots(2,3,figsize=(13,7),dpi=140)
    for column,encoder in enumerate(('E0','E1','E2')):
        a=read_npz(paths['frozen_final_application']/'models'/('G4_'+encoder)/'frozen_features.npz')
        selected=(a['condition']=='baseline')
        for row,key in enumerate(('object_slots','readout')):
            z=a[key][selected].astype(float);centered=z-z.mean(0);u,s,_=np.linalg.svd(centered,full_matrices=False)
            xy=u[:,:2]*s[:2];ax=axes[row,column]
            scatter=ax.scatter(xy[:,0],xy[:,1],c=a['target6'][selected,0]*1000,s=5,cmap='viridis')
            ax.set_title('G4 '+encoder+' '+key);ax.set(xlabel='descriptive PC1',ylabel='PC2')
    fig.suptitle('Independent baseline frames; colour=GT Peg relative X(mm), evaluation overlay only')
    fig.tight_layout();fig.savefig(destination/'object_slots_and_readout.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),dpi=140)
    options=(('Constant','Shared','unlearned'),('RawCentroidDirect','G4','unlearned'),
             ('P0_robot76','Shared','linear'),('P4_centroid','G4','linear'),
             ('P5_geometry_summary','G4','linear'),('P3_object_slots','G4_E2','linear'),('P1_readout','G4_E2','linear'))
    for oid in (0,1):
        errors=[];movement=[]
        for probe,model,decoder in options:
            rr=[r for r in absolute if r['model']==model and r['probe']==probe and r['decoder']==decoder
                and r['object_id']==str(oid) and r['split']=='evaluation' and r['condition_group']=='baseline']
            errors.append(float(rr[0]['mean_mm']) if rr and rr[0]['mean_mm'] else np.nan)
            pp=[r for r in pairs if r['model']==model and r['probe']==probe and r['decoder']==decoder and r['object_id']==str(oid)
                and r['physical_valid']=='True' and r['target_moved']=='True' and r['visibility_stratum']=='BOTH_VISIBLE']
            movement.append(numeric_mean(pp,'delta_error_mm'))
        for ax,values,title in ((axes[oid,0],errors,'baseline absolute position'),(axes[oid,1],movement,'visible moved-pair delta')):
            ax.bar(range(len(options)),[v if v is not None else np.nan for v in values],color='#0072b2')
            ax.set_xticks(range(len(options)),[r[0].replace('P','') for r in options],rotation=30,ha='right',fontsize=7)
            ax.set(title=('Peg' if oid==0 else 'Box')+': '+title+' (mm)');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Frozen input/control/slot/readout probes; raw centroid includes partial-surface origin bias')
    fig.tight_layout();fig.savefig(destination/'probe_controls_absolute_delta.png');plt.close(fig)
    validation=table(paths['frozen_probes']/'per_episode_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),dpi=140)
    for oid,ax in enumerate(axes):
        selected=[r for r in validation if r['variant']=='G4' and r['probe']=='P1_readout' and r['decoder']=='linear'
            and r['object_id']==str(oid) and r['split']=='validation' and r['condition_group']=='baseline']
        seeds=sorted(set(r['seed'] for r in selected))
        for encoder,offset,color in (('E0',-.22,'#0072b2'),('E1',0,'#e69f00'),('E2',.22,'#009e73')):
            values=[next((float(r['mean_mm']) for r in selected if r['seed']==seed and r['encoder']==encoder),np.nan) for seed in seeds]
            ax.bar(np.arange(len(seeds))+offset,values,width=.22,label=encoder,color=color)
        ax.set_xticks(range(len(seeds)),seeds);ax.set(title=('Peg' if oid==0 else 'Box')+' validation episodes',ylabel='Position error(mm)');ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(destination/'validation_per_episode.png');plt.close(fig)
