"""Exploratory failure/support attribution from frozen metrics, without model changes."""
from common import *
import numpy as np


def main():
    masks=read(ROOT/'mask_metrics.json')['rows'];points=read(ROOT/'stagewise_metrics.json')['rows'];poses=read(ROOT/'graph_pose_metrics.json')['rows']
    order=('raw','filtered','shape_first','cross_view','pre_budget','final')
    lookup={(r['arm'],r['scene'],r['size'],r['object_id'],r['stage']):r for r in points}
    cases=list(dict.fromkeys((r['scene'],r['size'],r['object_id']) for r in points if r['arm']=='actual'))
    losses=[]
    for scene,size,oid in cases:
        rr=[lookup.get(('actual',scene,size,oid,s)) for s in order]
        if not all(r is not None for r in rr):continue
        raw=rr[0];ceiling=lookup.get(('gt_mask',scene,size,oid,'raw'));box=lookup.get(('gt_box',scene,size,oid,'raw'))
        start=raw['surface_coverage_4mm'];sensor=ceiling['surface_coverage_4mm'] if ceiling else None
        first=None
        if sensor is None:first='NO_ELIGIBLE_VISIBLE_REFERENCE'
        elif start is not None and sensor-start>.05:first='PREDICTED_RAW_MASK_SUPPORT_GAP'
        for before,after in zip(rr,rr[1:]):
            a=before['surface_coverage_4mm'];b=after['surface_coverage_4mm']
            if first is None and a is not None and b is not None and a-b>.05:first=after['stage']
        row={k:raw[k] for k in ('scene','parent','cohort','timestep','size','object_id')}
        row.update(native_GT_mask_ceiling=sensor,actual_raw_coverage=start,
            GT_box_raw_coverage=box['surface_coverage_4mm'] if box else None,
            first_coverage_drop_over_5pp=first or 'NO_DROP_OVER_FIXED_DIAGNOSTIC_THRESHOLD',
            rule='descriptive support loss, not proof of unique error cause or non-estimability')
        for stage,r in zip(order,rr):
            for k in ('point_count','correct_points','merged_source_count','surface_coverage_4mm','end_minus_correct_points','end_plus_correct_points','rim_correct_points','rim_reference_points'):
                if k in r:row[stage+'__'+k]=r[k]
        losses.append(row)
    failures=[r for r in masks if r['visibility']=='VISIBLE' and (not r['mask08'] or r['wrong_object'])]
    pose_failures=[r for r in poses if r['graph_status']!='COMPLETE' or not r['peg_estimable'] or
                   (r.get('peg_position_error_mm') is not None and r['peg_position_error_mm']>=5)]
    csvwrite(ROOT/'mask_failure_cases.csv',failures);csvwrite(ROOT/'pose_failure_cases.csv',pose_failures)
    csvwrite(ROOT/'first_information_loss.csv',losses);write(ROOT/'first_information_loss.json',dict(rows=losses))
    tracking=[]
    for arm in ('actual','gt_box'):
        for size in SIZES:
            a=[r for r in masks if r['arm']==arm and r['size']==size and r['cohort']=='tracking']
            by={(r['parent'],r['camera'],r['repeat'],r['object_id'],r['timestep']):r for r in a}
            deltas=[];lost=0;initial_ok=0
            for k,first in by.items():
                if k[-1]!=0:continue
                end=by.get((*k[:-1],20))
                if end is None or first['visibility']!='VISIBLE' or end['visibility']!='VISIBLE':continue
                if first['mask_iou'] is not None and end['mask_iou'] is not None:deltas.append(end['mask_iou']-first['mask_iou'])
                if first['mask08']:
                    initial_ok+=1;lost+=int(not end['mask08'])
            visible=[r for r in a if r['visibility']=='VISIBLE']
            tracking.append(dict(arm=arm,size=size,visible_object_frames=len(visible),
                mean_mask_iou=float(np.mean([r['mask_iou'] for r in visible if r['mask_iou'] is not None])) if visible else None,
                visible_mask08=sum(r['mask08'] for r in visible)/len(visible) if visible else None,
                paired_endpoint_iou_delta=float(np.mean(deltas)) if deltas else None,
                endpoint_pairs=len(deltas),initial_good_endpoints=initial_ok,lost_good_endpoints=lost,
                lost_good_endpoint_rate=lost/initial_ok if initial_ok else None))
    write(ROOT/'tracking_quality_summary.json',dict(rows=tracking));csvwrite(ROOT/'tracking_quality_summary.csv',tracking)
    profile=read(ROOT/'timing_graph_profiles.json')['rows'];stage_names=sorted({s['stage'] for r in profile for s in r.get('spans',[])})
    details=[]
    for size in SIZES:
        selected=[r for r in profile if r['size']==size and r.get('hashes_equal')]
        for stage in stage_names:
            values=[sum(s['exclusive_ms'] for s in r['spans'] if s['stage']==stage) for r in selected]
            details.append(dict(size=size,stage=stage,**stats(values)))
    write(ROOT/'graph_profile_summary.json',dict(rows=details));csvwrite(ROOT/'graph_profile_summary.csv',details)
    print('FROZEN_METRIC_DIAGNOSTICS',len(losses),len(failures),len(pose_failures),flush=True)


if __name__=='__main__':main()
