"""Episode-macro aggregation; final data never chooses a model or threshold."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from graph.integration_baseline.io import write_json
from graph.integration_baseline.geometry09b_probe import csv_write


def table(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def numeric(x):
    try:
        y=float(x);return y if np.isfinite(y) else None
    except (ValueError,TypeError):return None


def macro(rows,group,metrics):
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r.get(k) for k in group)].append(r)
    output=[]
    for key,rr in sorted(grouped.items()):
        row=dict(zip(group,key));row['rows']=len(rr);row['episodes']=len({r['seed'] for r in rr})
        for metric in metrics:
            by=defaultdict(list)
            for r in rr:
                value=numeric(r.get(metric))
                if value is not None:by[r['seed']].append(value)
            values=[np.mean(v) for v in by.values()]
            row[metric]=float(np.mean(values)) if values else None
            row[metric+'_support_episodes']=len(values)
        output.append(row)
    return output


def geometry(root):
    rows=json.loads((root/'metrics.json').read_text());rows=[r for r in rows if r['condition']=='baseline']
    for r in rows:r['missing_rate']=int(r['missing_when_visible']) if r['gt_visible_points'] else None;r['visible_supported']=int(r['gt_visible_points']>0);r['false_presence_when_hidden']=int(r['count']>0) if not r['gt_visible_points'] else None
    ms=('count','surface_distance_mean_mm','surface_distance_median_mm','surface_distance_p95_mm','current_both_visible_coverage_4mm','boundary_coverage_4mm',
        'source_object_purity','source_robot_contamination','source_background_contamination','source_wrong_object_contamination',
        'geometry_outlier_fraction_8mm','normal_valid_fraction','missing_rate','false_presence_when_hidden','camera_wrist_sources')
    summary=macro(rows,('split','variant','budget','object_id'),ms)
    edge=[r for r in json.loads((root/'graphs.json').read_text()) if r['condition']=='baseline']
    graph=macro(edge,('split','variant','budget','object_id'),('nodes','undirected_surface_edges','connected_components','mesh_unsupported_edges_4mm','camera_free_space_crossing_edges','aperture_crossing_edges','aperture_observed_rim_coverage_4mm','aperture_full_rim_coverage_4mm'))
    gates=[r for r in json.loads((root/'gate_trace.json').read_text()) if r['condition']=='baseline']
    epochs=[]
    for seed in sorted({r['seed'] for r in gates}):
        rr=[r for r in gates if r['seed']==seed]
        before=sum(r['correct_wrist_before'] for r in rr);after=sum(r['correct_wrist_after'] for r in rr)
        wrong=sum(r['wrong_wrist_before'] for r in rr);wa=sum(r['wrong_wrist_after'] for r in rr)
        epochs.append(dict(seed=seed,split=rr[0]['split'],frames=len(rr),accepted=sum(r['accepted'] for r in rr),
            correct_wrist_before=before,correct_wrist_after=after,correct_retention=after/before if before else None,
            wrong_wrist_before=wrong,wrong_wrist_after=wa,wrong_rejection=1-wa/wrong if wrong else None,
            wrist_only_recovery=sum(r['wrist_only_recovery'] and r['correct_wrist_after']>0 for r in rr),
            wrist_only_candidate_observations=sum(r['wrist_only_recovery'] for r in rr),
            wrist_only_false_recovery=sum(r['wrist_only_recovery'] and r['correct_wrist_after']==0 for r in rr),
            main_gt_missing=sum(r['main_gt_visible']==0 for r in rr),
            predicted_main_missing=sum(r['main_mask_points']==0 for r in rr),
            identity_conflict=sum(r['identity_conflict_fraction']>0 for r in rr)))
    for episode_row in epochs:
        cap=[r for r in rows if r['budget']=='cap128' and r['object_id']==0 and r['seed']==episode_row['seed']]
        baseline={r['timestep']:r for r in cap if r['variant']=='C0_SF'}
        episode_row['wrist_only_graph_recovery']=sum(baseline[r['timestep']]['count']==0 and r['count']>0 and (r['source_object_purity'] or 0)>0 and r['camera_wrist_sources']>0 for r in cap if r['variant']=='C3_SF')
    filters=[r for r in json.loads((root/'filters.json').read_text()) if r['condition']=='baseline']
    filter_summary=macro(filters,('split','camera','object_id'),('object_preservation','confirmed_false_removal','uncertain_object_fraction','contact_near_pixels','contact_near_false_removal','robot_removal_recall','object_purity_before','object_purity_after'))
    matched=[r for r in rows if r['budget']=='matched_N' and r['variant']=='C3_SF']
    support=[dict(split=split,object_id=oid,frames=sum(r['split']==split and r['object_id']==oid for r in matched),
        eligible_nonempty_frames=sum(r['split']==split and r['object_id']==oid and r['count']>=6 for r in matched)) for split in sorted({r['split'] for r in matched}) for oid in (0,1)]
    return dict(geometry=summary,graph=graph,gate_per_episode=epochs,filter=filter_summary,matched_support=support,
        aggregation='episode macro means of same-condition per-frame metrics; surface p95 column is mean per-frame p95, not pooled point p95')


def probes(root):
    absolute=table(root/'per_episode_metrics.csv');delta=table(root/'paired_delta_metrics.csv')
    absolute=[r for r in absolute if r['condition_group']=='baseline']
    position=macro(absolute,('split','variant','encoder','probe','decoder','object_id'),('mean_mm','median_mm','p95_mm','mae_x_mm','mae_y_mm','mae_z_mm','r2_x','r2_y','r2_z'))
    delta=[r for r in delta if r['physical_valid']=='True' and r['target_moved']=='True']
    movement=macro(delta,('split','variant','encoder','probe','decoder','object_id'),('delta_error_mm','direction_cosine','signed_gain','true_norm_mm','predicted_norm_mm'))
    shape=[]
    for path in (root/'models').glob('*/frozen_features.npz'):
        with np.load(path,allow_pickle=False) as a:data={k:a[k].copy() for k in a.files}
        variant,encoder=path.parent.name.rsplit('_',1)
        for i in range(len(data['seed'])):
            if data['condition'][i]!='baseline':continue
            for oid in (0,1):
                valid=data['target_valid'][i,6+oid*15:6+(oid+1)*15]
                if not valid.any():continue
                target=data['target36'][i,6+oid*15:6+(oid+1)*15]
                prediction=data['prediction36'][i,6+oid*15:6+(oid+1)*15]
                raw=data['raw_geometry30'][i,oid*15:(oid+1)*15]
                shape.append(dict(seed=int(data['seed'][i]),split=str(data['split'][i]),variant=variant,encoder=encoder,object_id=oid,
                    diagnostic_summary_mae_mm=float(np.abs(prediction-target)[valid].mean()*1000),
                    raw_summary_mae_mm=float(np.abs(raw-target)[valid].mean()*1000)))
    surface=macro(shape,('split','variant','encoder','object_id'),('diagnostic_summary_mae_mm','raw_summary_mae_mm'))
    return dict(position=position,delta=movement,surface_summary=surface)


def assess(dev,final,finalprobes,protocol):
    limits=protocol['suitability'];rows=final['geometry'];graphs=final['graph'];tests=[]
    def get(v,oid,m):return next(r[m] for r in rows if r['variant']==v and r['object_id']==oid and r['budget']=='cap128')
    def add(name,value,limit,passed,scope):tests.append(dict(criterion=name,value=value,limit=limit,pass_=bool(passed) if value is not None else None,scope=scope))
    gain=get('C3_SF',0,'current_both_visible_coverage_4mm')-get('C0_SF',0,'current_both_visible_coverage_4mm')
    add('Peg coverage gain C3_SF-C0_SF',gain,limits['peg_coverage_minimum_absolute_gain'],gain>=limits['peg_coverage_minimum_absolute_gain'],'perception')
    wrist=get('C3_SF',1,'camera_wrist_sources');add('Box Wrist sources',wrist,0,wrist==0,'perception')
    ratio=get('C3_SF',1,'surface_distance_mean_mm')/get('C0_SF',1,'surface_distance_mean_mm')
    add('Box distance ratio C3/C0',ratio,limits['box_distance_maximum_ratio_to_C0'],ratio<=limits['box_distance_maximum_ratio_to_C0'],'geometry')
    missing=get('C3_SF',0,'missing_rate')-get('C0_SF',0,'missing_rate')
    add('Peg missing rate increase',missing,limits['peg_missing_maximum_absolute_increase'],missing<=limits['peg_missing_maximum_absolute_increase'],'perception')
    for oid in (0,1):
        value=get('C0_SF',oid,'source_object_purity')-get('C3_SF',oid,'source_object_purity')
        add(f'Object{oid} source purity drop',value,limits['purity_maximum_absolute_drop'],value<=limits['purity_maximum_absolute_drop'],'geometry')
        value=get('C0_SF',oid,'boundary_coverage_4mm')-get('C3_SF',oid,'boundary_coverage_4mm')
        add(f'Object{oid} visible boundary coverage drop',value,limits['boundary_maximum_absolute_drop'],value<=limits['boundary_maximum_absolute_drop'],'geometry')
    ret=[r['correct_retention'] for r in final['gate_per_episode'] if r['correct_retention'] is not None]
    value=float(np.mean(ret)) if ret else None
    add('Correct Wrist Peg gate retention',value,limits['gate_correct_wrist_source_minimum_retention'],value is not None and value>=limits['gate_correct_wrist_source_minimum_retention'],'perception')
    for r in final['filter']:
        value=r['object_preservation'];add(f"{r['camera']} object{r['object_id']} self-filter preservation",value,limits['self_filter_correct_object_minimum_preservation'],value is not None and value>=limits['self_filter_correct_object_minimum_preservation'],'perception')
        near=r['contact_near_false_removal'];value=1-near if near is not None else None
        add(f"{r['camera']} object{r['object_id']} finger-near confirmed preservation",value,limits['finger_near_correct_object_minimum_preservation'],value is not None and value>=limits['finger_near_correct_object_minimum_preservation'],'perception')
    for encoder in ('E0','E1'):
        for kind,metric,limit in (('position','mean_mm',limits['final_position_maximum_ratio_to_C0']),('delta','delta_error_mm',limits['final_delta_maximum_ratio_to_C0'])):
            rr=finalprobes[kind]
            for oid in ('0','1'):
                def error(v):
                    return next((r[metric] for r in rr if r['variant']==v and r['encoder']==encoder and r['probe']=='P1_readout' and r['decoder']=='linear' and r['object_id']==oid),None)
                baseline=error('C0_SF');val=error('C3_SF');value=val/baseline if baseline and val is not None else None
                add(f'{encoder} object{oid} frozen linear readout {kind} ratio C3/C0',value,limit,value is not None and value<=limit,'representation')
    perception_fail=any(t['pass_'] is False for t in tests if any(name in t['criterion'] for name in ('missing rate','source purity','Box distance ratio','Box Wrist sources')))
    coverage_ok=tests[0]['pass_']
    decision='GO' if all(t['pass_'] is True for t in tests) else 'CONDITIONAL' if coverage_ok and not perception_fail else 'NO-GO'
    return dict(decision=decision,criteria=tests,scope='geometry/GNN research suitability only; no policy success claim',thresholds_frozen_before_final=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    state=json.loads(a.state.read_text());runs={k:Path(v) for k,v in state['completed'].items()}
    dev=geometry(runs['development_geometry']);final=geometry(runs['final_geometry']);dp=probes(runs['probes']);fp=probes(runs['final_apply'])
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text());decision=assess(dev,final,fp,protocol)
    for name,result in [('development',dev),('final',final),('development_probes',dp),('final_probes',fp),('suitability',decision)]:
        write_json(a.output/(name+'.json'),result)
        if isinstance(result,dict):
            for key,rows in result.items():
                if isinstance(rows,list) and rows and isinstance(rows[0],dict):csv_write(a.output/(name+'_'+key+'.csv'),rows)
if __name__=='__main__':main()
