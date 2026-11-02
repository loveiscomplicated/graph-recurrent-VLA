"""Build reviewable numeric appendices from finished, downloaded evidence only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter


METRICS = (
    'mask/base_camera/0/precision','mask/base_camera/0/recall',
    'mask/base_camera/1/precision','mask/base_camera/1/recall',
    'mask/hand_camera/0/precision','mask/hand_camera/0/recall',
    'mask/hand_camera/0/iou','mask/hand_camera/0/visible_object_recall',
    'graph/0/object_recall','graph/0/missing','graph/0/correct_fraction',
    'graph/0/robot_fraction','graph/0/wrong_object_fraction','graph/0/background_fraction',
    'graph/0/visible_coverage_4mm','graph/0/projected_boundary_coverage_4mm',
    'graph/0/surface_distance_mean_mm','graph/0/surface_distance_p95_mm',
    'graph/0/normal_valid_fraction','graph/0/false_presence',
    'graph/1/object_recall','graph/1/missing','graph/1/correct_fraction',
    'graph/1/robot_fraction','graph/1/wrong_object_fraction','graph/1/background_fraction',
    'graph/1/visible_coverage_4mm','graph/1/projected_boundary_coverage_4mm',
    'graph/1/surface_distance_mean_mm','graph/1/surface_distance_p95_mm',
    'graph/1/normal_valid_fraction','graph/1/false_presence',
)


def fmt(value):
    return 'UNSUPPORTED' if value is None else f'{value:.5f}'


def parent_mean(rows, value):
    """Frame -> Episode -> Parent; undefined values never become zeros."""
    episodes=defaultdict(list)
    for row in rows:
        v=value(row)
        if v is not None:episodes[row['parent_id'],row['episode']].append(v)
    parents=defaultdict(list)
    for (pid,_),values in episodes.items():parents[pid].append(float(np.mean(values)))
    return dict(value=float(np.mean([np.mean(v) for v in parents.values()])) if parents else None,
        parents=len(parents),episodes=len(episodes),frames=sum(map(len,episodes.values())))


def diagnostic_digest(read, output):
    assert read('global_missing_oracles/scope.json')['status']=='GLOBAL_GRAPH_MISSING_ORACLES_FINISHED'
    assert read('refinement_retention/scope.json')['status']=='RAW_REANCHOR_REFINEMENT_RETENTION_FINISHED'
    result={};macro=[]
    functions={
        'visible_object_recall':lambda r:float(r['correct']>0) if r['visible_reference_points']>=8 else None,
        'source_purity':lambda r:r['correct']/r['points'] if r['points'] else None,
        'all_node_coverage_proxy':lambda r:r['visible_coverage_4mm'] if r['visible_reference_points']>=8 else None,
        'projected_boundary_coverage':lambda r:r['projected_boundary_coverage_4mm'] if r['visible_reference_points']>=8 else None,
    }
    for label,oracle,native in [
        ('original_four_prefixes','finalization/oracles/representative_points.json','oracle_grouping/representative_points.json'),
        ('global_missing_prefixes','global_missing_oracles/oracles/representative_points.json','global_missing_oracles/grouped/representative_points.json')]:
        rows=read(oracle)+read(native)
        for condition in sorted({r['condition'] for r in rows}):
            for oid in (0,1):
                rr=[r for r in rows if r['condition']==condition and r['object_id']==oid]
                for name,fn in functions.items():macro.append(dict(scope=label,condition=condition,object_id=oid,metric=name,**parent_mean(rr,fn)))
        if label=='global_missing_prefixes':
            cases=read('global_missing_oracles/registered_missing_cases.json')['cases'];at_failure=[]
            for case in cases:
                rr=[r for r in rows if r['episode']==case['episode'] and r['timestep']==case['timestep'] and r['object_id']==case['object_id']]
                assert len(rr)==6
                for row in rr:at_failure.append(dict(case=case,condition=row['condition'],points=row['points'],correct=row['correct'],robot=row['robot'],
                    other=row['wrong_object'],background=row['background'],visible_reference_points=row['visible_reference_points'],
                    all_node_coverage_proxy=row['visible_coverage_4mm']))
            result['global_missing_failure_frames']=at_failure
    result['oracle_parent_macro']=macro
    edge=read('oracle_graph/edge_rim_metrics.json');edge_summary=[]
    for condition in sorted({r['condition'] for r in edge}):
        for oid in (0,1):
            rr=[r for r in edge if r['condition']==condition and r['object_id']==oid]
            item=dict(condition=condition,object_id=oid,frames=len(rr),
                surface_edges=sum(r['undirected_surface_edges'] for r in rr),
                free_space_crossings=sum(r['camera_free_space_crossing_edges'] for r in rr),
                aperture_crossings=sum(r['aperture_crossing_edges'] for r in rr),
                no_surface_edge_frames=sum(r['undirected_surface_edges']==0 for r in rr))
            if oid==1:
                for key in ('aperture_full_rim_coverage_4mm','aperture_observed_rim_coverage_4mm'):
                    item[key]=parent_mean(rr,lambda r,k=key:r[k])
            edge_summary.append(item)
    result['original_prefix_edge_rim']=edge_summary
    pooled=defaultdict(Counter)
    for row in read('approximate_reference/pixel_rows.json'):
        assert row['reference_kind']=='APPROXIMATE_REFERENCE_RGB_ULP_ONLY'
        pooled[row['partition'],row['condition'],row['camera'],row['object_id']].update(
            {k:row['metrics'][k] for k in ('true_positive','false_positive','false_negative','robot_pixels','background_pixels','wrong_object_pixels')})
    approximate=[]
    for (partition,condition,camera,oid),counts in sorted(pooled.items()):
        tp,fp,fn=[counts[k] for k in ('true_positive','false_positive','false_negative')]
        approximate.append(dict(partition=partition,condition=condition,camera=camera,object_id=oid,**counts,
            precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
            reference_kind='APPROXIMATE_REFERENCE_RGB_ULP_ONLY',aggregation='pooled pixel, not Parent macro'))
    result['separate_approximate_pooled_pixels']=approximate
    detection=[]
    t0=read('initialization_bottlenecks/t0_reconstruction_rows.json')
    for camera in ('base_camera','hand_camera'):
        for oid in (0,1):
            rr=[r for r in t0 if r['camera']==camera and r['object_id']==oid and r['raw_gt_pixels']>=8]
            chosen=[r for r in rr if r['chosen_box'] is not None]
            good=lambda r:r['chosen_box'] is not None and r['chosen_gt_iou']>=.5
            tp=sum(map(good,rr));fp=len(chosen)-tp
            detection.append(dict(camera=camera,object_id=oid,scope='all Development t0, GT>=8 pixels',
                definition='accepted detector initialization matched to target GT bbox at IoU>=0.5; localization, not semantic purity',
                visible_objects=len(rr),accepted_boxes=len(chosen),localized_true_boxes=tp,unmatched_boxes=fp,
                no_box=len(rr)-len(chosen),frame_weighted_recall=tp/len(rr) if rr else None,
                missed_visible_object_rate=1-tp/len(rr) if rr else None,
                accepted_box_false_discovery_rate=fp/len(chosen) if chosen else None,
                parent_macro_recall=parent_mean(rr,lambda r:float(good(r))),
                parent_macro_accepted_false_discovery=parent_mean(chosen,lambda r:float(not good(r))),
                tiny_rows=sum(r['camera']==camera and r['object_id']==oid and 0<r['raw_gt_pixels']<8 for r in t0)))
    result['t0_detection_localization_metrics']=detection
    result['refinement_retention']=read('refinement_retention/retention_counts.json')
    result['scope']=dict(model_forwards=0,GT_is_evaluation_only=True,oracle_not_population_estimate=True,
        all_node_coverage_can_credit_false_points=True,approximate_never_mixed_with_exact=True,no_candidate_retuning=True)
    (output/'additional_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')


def run(evidence,output):
    used={}
    def read(relative):
        p=evidence/relative;data=p.read_bytes();used[str(relative)]=hashlib.sha256(data).hexdigest()
        return json.loads(data)
    assert read('finalization/handoff.json')['status']=='SCIENTIFIC_EVALUATIONS_FINISHED_REPORT_REVIEW_PENDING'
    assert read('posthoc/report_evidence_ready.json')['status']=='REPORT_EVIDENCE_READY'
    assert read('oracle_graph/scope.json')['status']=='ORACLE_GRAPH_STRUCTURE_AUDITED'
    assert read('oracle_grouping/oracle_grouping_control.json')['status']=='NATIVE_GROUPED_GT_BOX_CONTROL_FINISHED'
    assert read('camera_contract/scope.json')['status']=='RECORDED_CAMERA_CONTRACT_PASS'
    assert read('approximate_reference/reference_accounting.json')['exact_reference_flags_unchanged']
    assert read('detector_outcomes/scope.json')['status']=='DETECTOR_AND_IDENTITY_OUTCOMES_FINISHED'
    assert read('development_temporal/scope.json')['status']=='DEVELOPMENT_TEMPORAL_COMPARISONS_FINISHED'
    output.mkdir(parents=True,exist_ok=False)
    chunks=['## 최종 paired 수치 부록\n',
        'B/C는 동일하게 지원되는 Parent 집합의 평균이다. Precision은 prediction이 없으면 미정의이므로 '
        '양 조건의 유효 frame 집합까지 같다는 뜻은 아니다. Δ CI는 Parent를 단위로 2,000회 '
        'bootstrap한 95% 구간이며, Development CI는 탐색적이다. Unsupported reference/precision을 '
        '성공이나 0으로 바꾸지 않는다. Graph presence와 coverage 정의는 본문을 따른다.\n']
    all_rows=[]
    for name,label in [('reanchor','Development'),('internal_test','Internal Test'),('validation','Shared Validation')]:
        rows=read('posthoc/statistics_'+name+'/matched_parent_macro.json')
        all_rows.extend(dict(partition=label,**r) for r in rows)
        lookup={r['metric']:r for r in rows if r['scope']=='all'}
        chunks.extend(['\n### '+label+'\n','| Metric | B | C | Δ | Parent bootstrap95% CI | paired Parent |\n|---|---:|---:|---:|---|---:|\n'])
        for metric in METRICS:
            r=lookup[metric];ci=r['parent_bootstrap_95ci'];interval='UNSUPPORTED' if ci is None else f'[{fmt(ci[0])}, {fmt(ci[1])}]'
            chunks.append(f"| {metric} | {fmt(r['baseline'])} | {fmt(r['candidate'])} | {fmt(r['delta'])} | {interval} | {r['paired_supported_parents']}/{r['total_parents']} |\n")
    (output/'selected_paired_tables.md').write_text(''.join(chunks))
    # Preserve all t0/later/phase/segment/support strata, including unsupported.
    with (output/'all_matched_strata.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    rng=np.random.default_rng(42);coverage=[]
    chunks=['## 올바른 source만의 공간 coverage\n',
        'GT Actor가 맞는 실제 representative만 사용한다. FP node에는 coverage credit을 주지 않는다. '
        'Raw-visible object recall과 valid-depth surface sample coverage의 분모는 다르다.\n\n',
        '| Partition | Object | Metric | B | C | Δ | Parent95% CI | Parent |\n|---|---|---|---:|---:|---:|---|---:|\n']
    for name in ('development','internal_test','validation'):
        rows=read('posthoc/valid_coverage_'+name+'/parent_metrics.json')
        for oid in (0,1):
            for metric in ('raw_visible_object_recall','valid_source_coverage_4mm','valid_source_boundary_coverage_4mm'):
                rr=[r for r in rows if r['object_id']==oid and r['metric']==metric and r['delta'] is not None]
                v=np.array([r['delta'] for r in rr]);ci=np.percentile(rng.choice(v,(2000,len(v)),replace=True).mean(1),[2.5,97.5]).tolist() if len(v) else None
                row=dict(partition=name,object_id=oid,metric=metric,parents=len(rr),baseline=float(np.mean([r['baseline'] for r in rr])) if rr else None,
                    candidate=float(np.mean([r['candidate'] for r in rr])) if rr else None,delta=float(v.mean()) if len(v) else None,parent_bootstrap_95ci=ci)
                coverage.append(row);interval='UNSUPPORTED' if ci is None else f'[{fmt(ci[0])}, {fmt(ci[1])}]'
                chunks.append(f"| {name} | {('Peg','Box')[oid]} | {metric} | {fmt(row['baseline'])} | {fmt(row['candidate'])} | {fmt(row['delta'])} | {interval} | {len(rr)} |\n")
    (output/'correct_source_coverage.md').write_text(''.join(chunks))
    (output/'correct_source_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    # Descriptive figure: independently tested challenger versus two cleaner
    # geometry variants. All-node coverage here is explicitly labelled a proxy.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes=('reanchor','depth_component','depth_component_both')
    maps={m:{r['metric']:r for r in read('posthoc/statistics_'+m+'/matched_parent_macro.json') if r['scope']=='all'} for m in modes}
    labels=['Baseline','Wrist reanchor','Main Peg comp.','Main Peg+Box comp.']
    def series(metric):return [maps['reanchor'][metric]['baseline']]+[maps[m][metric]['candidate'] for m in modes]
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    colors=['#757575','#3a70b2','#2e9668','#c98735']
    for ax,metric,title,scale in zip(axes[:2],('graph/0/visible_coverage_4mm','graph/0/background_fraction'),
        ('Peg all-node 4mm coverage (proxy)','Peg background representatives (%)'),(1,100)):
        ax.barh(labels,np.array(series(metric))*scale,color=colors);ax.invert_yaxis();ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    latency=[read('finalization/latency_reanchor/latency.json')['summaries']['all_60_frames']['baseline_end_to_end_ms']['p95']]
    latency += [read('finalization/latency_'+m+'/latency.json')['summaries']['all_60_frames']['candidate_end_to_end_ms']['p95'] for m in modes]
    axes[2].barh(labels,latency,color=colors);axes[2].invert_yaxis();axes[2].axvline(50,color='black',linestyle='--',label='20Hz period: 50ms');axes[2].set_title('Fixed-prefix perception p95 (ms)',fontsize=10);axes[2].legend(fontsize=8);axes[2].grid(axis='x',alpha=.2);axes[2].set_axisbelow(True)
    fig.suptitle('Development trade-offs; no production adoption',fontsize=12)
    fig.savefig(output/'development_tradeoffs.png',dpi=180);fig.savefig(output/'development_tradeoffs.pdf');plt.close(fig)
    diagnostic_digest(read,output)
    (output/'input_hashes.json').write_text(json.dumps(dict(status='READ_ONLY_REPORT_DIGEST_FINISHED',inputs=used,
        bootstrap_unit='Parent',bootstrap_samples=2000,no_new_candidate_selection=True),indent=2)+'\n')
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.evidence,a.output)
