"""Read-only paired strata, explicit frame/episode/parent aggregation.

Additional descriptive slices do not select a candidate or alter registered
gates. Visibility and expert phase are evaluation labels only.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from .aggregate_evaluation import csv_rows
from .matching_phase_audit import phase_clock,visible_bin
from .evaluation_metrics import hierarchical_paired
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha


def aggregation_units(rows,metric,keys,candidate):
    observations={}
    for r in rows:
        if r['condition'] not in ('B0',candidate):continue
        key=tuple(r.get(k) for k in keys)+(r['parent_id'],r['episode'],r['timestep'])
        per=observations.setdefault(key,{})
        assert r['condition'] not in per,'duplicate observation/condition'
        value=r.get(metric)
        per[r['condition']]=value if r['reference_kind']=='EXACT_GT' and value is not None and np.isfinite(value) else None
    groups=defaultdict(list)
    for key,per in observations.items():
        if set(per)=={'B0',candidate} and all(v is not None for v in per.values()):
            groups[key[:-3]].append((key[-3],key[-2],per['B0'],per[candidate]))
    output=[]
    for group,pairs in groups.items():
        episodes=defaultdict(list)
        for p,e,b,c in pairs:episodes[p,e].append([b,c])
        means={k:np.mean(v,axis=0) for k,v in episodes.items()};parents=defaultdict(list)
        for (p,e),value in means.items():parents[p].append(value)
        frame=np.mean([[b,c] for p,e,b,c in pairs],axis=0)
        episode=np.mean(list(means.values()),axis=0)
        parent=np.mean([np.mean(v,axis=0) for v in parents.values()],axis=0)
        for index,condition in enumerate(('B0',candidate)):
            output.append(dict(zip(keys,group),metric=metric,condition=condition,paired_defined_frames=len(pairs),
                paired_episodes=len(episodes),paired_parents=len(parents),frame_weighted=float(frame[index]),
                episode_macro=float(episode[index]),parent_macro=float(parent[index]),
                undefined_precision_never_imputed=True))
    return output


def run(root,paired,expected,criteria,candidate):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    assert read(paired/'pipeline_state.json')['status']=='PAIRED_NEW_DEVELOPMENT_GRAPH_COMPLETE'
    pixel=[];graph=[];inputs=[];accounting=[]
    for e in read(expected)['entries']:
        eid=str(e['episode_id']);folder=paired/'episodes'/eid;receipt=read(folder/'receipt.json')
        assert receipt['candidate']==candidate and receipt['raw_sha256']==e['raw_sha256']
        for name in ('pixel_metrics.csv','graph_metrics.csv'):assert sha(folder/name)==receipt['files'][name]
        tracepath=Path(e['raw_path']).parent/'trace.json';trace=read(tracepath) if tracepath.exists() else []
        clock=phase_clock(trace,e['actions']+1)
        workflow='corrective' if e['source']=='new_stage3' and e['branch'].startswith('t') else 'full_task_or_nominal'
        pp=csv_rows(folder/'pixel_metrics.csv');gg=csv_rows(folder/'graph_metrics.csv')
        lookup={(r['timestep'],r['condition'],r['camera'],r['object_id']):r for r in pp}
        assert len(lookup)==len(pp)
        for rows in (pp,gg):
            for r in rows:
                t=int(r['timestep']);r.update(dataset_stratum=e.get('stratum','internal'),workflow=workflow,
                    clock_part='t0' if t==0 else 'after_t0',expert_phase=clock[t]['phase'],collection_segment=r.get('segment'))
                exact=r['reference_kind']=='EXACT_GT'
                if rows is pp:v=r['visible_pixels'] if exact else None
                else:
                    cameras=('base_camera','hand_camera') if r['object_id']==0 else ('base_camera',)
                    v=sum(lookup[r['timestep'],r['condition'],c,r['object_id']]['visible_pixels'] for c in cameras) if exact else None
                    r['raw_visible_correct_point_missing']=float(r['correct_points']==0) if v is not None and v>=8 else None
                    r['raw_false_object_presence']=float(r['geometry_points']>0) if v==0 else None
                r['visible_support_bin']=visible_bin(v)
                r['presence_stratum']='GT_UNSUPPORTED' if v is None else 'negative_zero_visible' if v==0 else 'low_support_1_7' if v<8 else 'positive_visible_ge8'
        pixel.extend(pp);graph.extend(gg)
        inputs.append(dict(receipt=str(folder/'receipt.json'),sha256=sha(folder/'receipt.json'),
            trace=str(tracepath) if tracepath.exists() else None,trace_sha256=sha(tracepath) if tracepath.exists() else None))
        accounting.append(dict(episode=eid,parent_id=e['parent_id'],workflow=workflow,dataset_stratum=e.get('stratum','internal'),
            frames=receipt['frames'],exact_frames=receipt['exact_frames'],unsupported_frames=receipt['frames']-receipt['exact_frames']))
    rules=read(criteria)['bootstrap'];summaries=[];weighted=[]
    config=dict(source_sha256=sha(Path(__file__)),expected_sha256=sha(expected),criteria_sha256=sha(criteria),candidate=candidate,
        metrics_are_descriptive_not_new_gates=True,visibility_bins='0,1-7,8-31,32-127,128-511,512+',
        grouping='normal/stress separate; exact same eligible paired frames; episode means then physical-parent means',
        phase_labels_never_inference_inputs=True)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_stratified_paired_perception_metrics',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        for scope,rows,metrics,base in [
            ('pixel',pixel,('precision','recall','iou','correct_mask_availability','correct_pixel_missing','false_presence'),('dataset_stratum','camera','object_id')),
            ('graph',graph,('geometry_points','correct_representative_fraction','correct_source_coverage','correct_boundary_coverage',
                'robot_point_fraction','background_point_fraction','wrong_object_point_fraction','peg_ends_coverage',
                'box_observed_rim_coverage','raw_visible_correct_point_missing','raw_false_object_presence'),('dataset_stratum','object_id'))]:
            for family,extra in [('clock',('clock_part',)),('visible_support',('visible_support_bin',)),('presence',('presence_stratum',)),
                                 ('workflow',('workflow',)),('phase',('expert_phase','collection_segment'))]:
                keys=base+extra
                for metric in metrics:
                    values=hierarchical_paired(rows,metric,keys,candidate=candidate,bootstrap=rules['replicates'],seed=rules['seed'])
                    summaries.extend(dict(scope=scope,family=family,**r) for r in values)
                    weighted.extend(dict(scope=scope,family=family,**r) for r in aggregation_units(rows,metric,keys,candidate))
        write(root/'stratified_parent_metrics.json',summaries)
        csv_write(root/'stratified_parent_metrics.csv',[{k:v for k,v in r.items() if k!='parent_values'} for r in summaries])
        csv_write(root/'frame_episode_parent_metrics.csv',weighted)
        csv_write(root/'stratum_reference_accounting.csv',accounting)
        write(root/'scope.json',dict(status='POST_FORWARD_STRATIFIED_PAIRED_EVALUATION_COMPLETE',run_id=logger.run_id,**config,
            inputs=inputs,parents=read(expected)['parents'],frames=read(expected)['frames'],model_forwards=0,
            Wrist_Box_pixel_rows_diagnostic_only_not_Graph_or_primary=True,normal_stress_never_pooled_as_primary=True,
            whole_goal_complete=False,files={p.name:sha(p) for p in root.iterdir() if p.is_file()}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','paired','expected','criteria'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',required=True);run(**vars(p.parse_args()))
