"""Post-inference paired parent summaries; no independent-test tuning here."""
import argparse
import csv
from pathlib import Path
from .evaluation_metrics import hierarchical_paired
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha

TEXT={'condition','episode','parent_id','camera','reference_kind','source','branch','phase'}


def csv_rows(path):
    rows=[]
    with path.open() as stream:
        for row in csv.DictReader(stream):
            converted={}
            for key,value in row.items():
                if value=='':converted[key]=None
                elif key in TEXT:converted[key]=value
                elif value in ('True','False'):converted[key]=value=='True'
                else:converted[key]=float(value)
            rows.append(converted)
    return rows


def run(output,paired_root,expected,partial=False,candidate='A',subset=False,evaluation_scope='development'):
    output.mkdir(parents=True,exist_ok=False)
    manifest=read(expected);entries={str(e['episode_id']):e for e in manifest['entries']}
    if not partial:assert read(paired_root/'pipeline_state.json')['status']=='PAIRED_NEW_DEVELOPMENT_GRAPH_COMPLETE'
    pixel=[];graph=[];both=[];receipts=[];phase_sources={}
    for path in sorted((paired_root/'episodes').glob('*/receipt.json')):
        if subset and path.parent.name not in entries:continue
        receipt=read(path);eid=receipt['episode'];e=entries[eid]
        assert receipt.get('candidate','A')==candidate
        assert receipt['frames']==e['actions']+1
        assert receipt['raw_sha256']==e['raw_sha256']
        assert all(sha(path.parent/p)==h for p,h in receipt['files'].items())
        pp=csv_rows(path.parent/'pixel_metrics.csv');gg=csv_rows(path.parent/'graph_metrics.csv')
        phasepath=Path(e['raw_path']).parent/'trace.json';phases=None
        if phasepath.exists():
            values=read(phasepath)
            if isinstance(values,list) and len(values)==e['actions'] and all(isinstance(v,dict) and 'phase' in v for v in values):
                phases=[v['phase'] for v in values]
                phase_sources[str(phasepath)]=sha(phasepath)
            elif (isinstance(values,list) and len(values)==e['actions']+1
                  and all(isinstance(v,dict) and 'expert_phase' in v for v in values)
                  and [v.get('timestep') for v in values]==list(range(e['actions']+1))):
                # Final collector stores the terminal observation too. Phase
                # labels are read only here, after complete mask inference.
                phases=[v['expert_phase'] for v in values]
                phase_sources[str(phasepath)]=sha(phasepath)
        for r in pp+gg:
            t=int(r['timestep'])
            if t==e['actions']:r['phase']='TERMINAL_OBSERVATION'
            elif phases is not None:r['phase']=phases[t]
        lookup={(r['condition'],int(r['timestep']),int(r['object_id']),r['camera']):r for r in pp}
        for r in gg:
            oid=int(r['object_id']);cameras=('base_camera','hand_camera') if oid==0 else ('base_camera',)
            sources=[lookup[r['condition'],int(r['timestep']),oid,c] for c in cameras]
            # Raw actor visibility and valid-depth surface support have different
            # semantics. Keep both instead of inheriting the old denominator.
            raw=sum(v['visible_pixels'] for v in sources) if r['reference_kind']=='EXACT_GT' else None
            r['raw_gt_visible_pixels']=raw
            r['raw_visible_correct_point_missing']=float(r['correct_points']==0) if raw is not None and raw>=8 else None
            r['raw_false_object_presence']=float(r['geometry_points']>0) if raw==0 else None
        pixel.extend(pp);graph.extend(gg)
        both.extend(read(path.parent/'paired_frame_metrics.json'))
        receipts.append(dict(path=str(path),sha256=sha(path),frames=receipt['frames'],exact_frames=receipt['exact_frames']))
    if not partial:
        assert len(receipts)==manifest['episodes'] and sum(r['frames'] for r in receipts)==manifest['frames']
    pixel_metrics=('precision','recall','iou','correct_mask_availability','correct_pixel_missing','false_presence')
    graph_metrics=('correct_representative_fraction','correct_source_coverage','correct_boundary_coverage',
        'robot_point_fraction','background_point_fraction','wrong_object_point_fraction',
        'peg_ends_coverage','box_observed_rim_coverage','raw_visible_correct_point_missing','raw_false_object_presence')
    summaries=[]
    for metric in pixel_metrics:summaries.extend(dict(scope='pixel',**r) for r in hierarchical_paired(pixel,metric,candidate=candidate))
    for metric in graph_metrics:summaries.extend(dict(scope='graph',**r) for r in hierarchical_paired(graph,metric,('object_id',),candidate=candidate))
    summaries.extend(dict(scope='paired_frame',**r) for r in hierarchical_paired(both,'both_camera_peg_correct_pixel_missing',(),candidate=candidate))
    phases=[]
    for metric in ('correct_mask_availability','precision','recall','iou'):
        phases.extend(hierarchical_paired(pixel,metric,('camera','object_id','phase','segment'),candidate=candidate))
    weighted=[]
    for scope,rows,metrics,keys in [('pixel',pixel,pixel_metrics,('camera','object_id')),
                                  ('graph',graph,graph_metrics,('object_id',))]:
        groups={tuple(r.get(k) for k in keys) for r in rows}
        for group in sorted(groups,key=str):
            for condition in ('B0',candidate):
                selected=[r for r in rows if tuple(r.get(k) for k in keys)==group and r['condition']==condition and r['reference_kind']=='EXACT_GT']
                for metric in metrics:
                    values=[r[metric] for r in selected if r.get(metric) is not None]
                    weighted.append(dict(zip(keys,group),scope=scope,condition=condition,metric=metric,
                        frame_weighted=sum(values)/len(values) if values else None,defined_frames=len(values),exact_frames=len(selected)))
                if scope=='pixel':
                    tp=sum(r['tp'] for r in selected);fp=sum(r['fp'] for r in selected);fn=sum(r['fn'] for r in selected)
                    weighted.append(dict(zip(keys,group),scope=scope,condition=condition,metric='pooled_pixels',
                        tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None))
    write(output/'paired_parent_metrics.json',summaries)
    write(output/'phase_metrics.json',phases)
    csv_write(output/'final_graph_reliability_metrics.csv',[{k:v for k,v in r.items() if k!='parent_values'} for r in summaries if r['scope']=='graph'])
    csv_write(output/'tri_model_ablation_metrics.csv',[{k:v for k,v in r.items() if k!='parent_values'} for r in summaries])
    csv_write(output/'per_camera_per_phase_results.csv',[{k:v for k,v in r.items() if k!='parent_values'} for r in phases])
    csv_write(output/'frame_weighted_and_pooled_metrics.csv',weighted)
    write(output/'reference_accounting.json',dict(total_evaluated_frames=sum(r['frames'] for r in receipts),
        exact_frames=sum(r['exact_frames'] for r in receipts),
        unsupported_frames=sum(r['frames']-r['exact_frames'] for r in receipts)))
    write(output/'scope.json',dict(status='PARTIAL_EXPLORATORY_SUMMARY' if partial else ('NEW_DEVELOPMENT_PAIRED_SUMMARY_COMPLETE' if evaluation_scope=='development' else 'FROZEN_INDEPENDENT_PAIRED_SUMMARY_COMPLETE'),
        conditions=['B0',candidate],episodes=len(receipts),parents=len({r['parent_id'] for r in pixel}),
        receipts=receipts,phase_sources=phase_sources,bootstrap_unit='physical_parent',bootstrap_replicates=2000,
        reference_primary='EXACT_GT only',duplicate_episodes_not_independent=True,
        empty_precision_imputed=False,raw_and_depth_visibility_separate=True,
        no_model_forwards=True,no_threshold_changes=True,independent_test=evaluation_scope!='development',
        evaluation_scope=evaluation_scope,explicit_expected_subset=subset,expected_sha256=sha(expected),
        source_sha256=sha(Path(__file__))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('paired_root',type=Path)
    p.add_argument('expected',type=Path);p.add_argument('--partial',action='store_true')
    p.add_argument('--candidate',default='A');p.add_argument('--subset',action='store_true')
    p.add_argument('--scope',default='development',choices=('development','internal','final_normal','final_stress'));a=p.parse_args()
    run(a.output,a.paired_root,a.expected,a.partial,a.candidate,a.subset,a.scope)
