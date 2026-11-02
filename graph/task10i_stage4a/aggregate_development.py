"""Denominator-preserving episode/parent macro baseline stage metrics."""
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
from .run_pipeline import read,write

def mean(values):
    values=[v for v in values if v is not None];return float(np.mean(values)) if values else None

def run(root):
    per_episode=[];temporal=[];cases=[];geometry=[]
    for p in sorted((root/'perception').glob('*/stage_rows.json')):
        rows=read(p);groups=defaultdict(list)
        for r in rows:
            m=r['metrics'];n=m['input_gt_support'] if m else None
            stratum='GT_UNSUPPORTED' if n is None else 'absent' if n==0 else 'tiny_1_7' if n<8 else 'visible_8_31' if n<32 else 'visible_32_127' if n<128 else 'visible_128_plus'
            for visibility in (stratum,'all_supported' if m else 'GT_UNSUPPORTED'):
                # Unsupported rows occur once, with explicit unknown metrics.
                if visibility=='GT_UNSUPPORTED' and m is None and (r['camera'],r['object_id'],r['stage'],visibility,'dummy') in groups:continue
                groups[r['camera'],r['object_id'],r['stage'],visibility,'t0' if r['timestep']==0 else 'later'].append(r)
                if visibility==stratum and stratum=='GT_UNSUPPORTED':break
        for key,gg in groups.items():
            mm=[r['metrics'] for r in gg if r['metrics'] is not None]
            per_episode.append(dict(episode=gg[0]['episode'],parent_id=gg[0]['parent_id'],camera=key[0],object_id=key[1],stage=key[2],support_stratum=key[3],time_stratum=key[4],frames=len(gg),supported_frames=len(mm),
                precision=mean(m['precision'] for m in mm),recall=mean(m['recall'] for m in mm),iou=mean(m['iou'] for m in mm),
                output_support=mean(m['output_support'] for m in mm),
                missed_visible_rate=mean(float(m['output_support']==0) for m in mm if m['input_gt_support']>=8),
                no_correct_visible_rate=mean(float(m['true_positive']==0) for m in mm if m['input_gt_support']>=8),
                false_absent_presence=mean(float(m['output_support']>0) for m in mm if m['input_gt_support']==0),
                robot_fraction=mean(m['robot_pixels']/m['output_support'] if m['output_support'] else None for m in mm),
                wrong_object_fraction=mean(m['wrong_object_pixels']/m['output_support'] if m['output_support'] else None for m in mm),
                background_fraction=mean(m['background_pixels']/m['output_support'] if m['output_support'] else None for m in mm)))
        # Preserve exact cases and causal first disappearance; no detector call is
        # inferred on a frame where the baseline only propagated SAM2 memory.
        by=defaultdict(list)
        for r in rows:by[r['camera'],r['object_id'],r['timestep']].append(r)
        for (c,oid,t),gg in by.items():
            final=next(r for r in gg if r['stage']=='final_graph_sources');m=final['metrics']
            if m is None or (c=='hand_camera' and oid==1):continue
            status=None
            if m['input_gt_support']>=8 and m['true_positive']==0:
                stage=next((r['stage'] for r in gg if r['metrics']['true_positive']==0),None)
                status='DETECTION_PRESENT_BUT_SEGMENTATION_MISSING' if stage=='sam2' else 'SEGMENTATION_PRESENT_BUT_FILTERED' if stage in ('depth_valid','robot_confirmed_removed','robot_uncertain_removed','overlap_removed','gate_accepted') else 'VALID_MASK_BUT_GRAPH_MISSING'
                # Detection-present is not proven without initialization metadata.
                if stage=='sam2':status='VISIBLE_SAM2_EMPTY_OR_WRONG_REQUIRES_DETECTOR_ATTRIBUTION'
            elif m['input_gt_support']==0 and m['output_support']>0:status='TEMPORAL_GHOST_OR_WRONG_OBJECT_REQUIRES_RGB_REVIEW';stage='sam2'
            if status:cases.append(dict(episode=final['episode'],parent_id=final['parent_id'],camera=c,object_id=oid,timestep=t,status=status,first_zero_true_support_stage=stage,metrics=m))
        geometry.extend(read(p.parent/'representative_points.json'))
    fields=('precision','recall','iou','output_support','missed_visible_rate','no_correct_visible_rate','false_absent_presence','robot_fraction','wrong_object_fraction','background_fraction')
    keys=('camera','object_id','stage','support_stratum','time_stratum')
    parent_groups=defaultdict(list)
    for r in per_episode:parent_groups[(r['parent_id'],*(r[k] for k in keys))].append(r)
    parent_rows=[]
    for key,rr in parent_groups.items():parent_rows.append(dict(parent_id=key[0],**dict(zip(keys,key[1:])),episodes=len(rr),**{f:mean(r[f] for r in rr) for f in fields}))
    macro=defaultdict(list)
    for r in parent_rows:macro[tuple(r[k] for k in keys)].append(r)
    parent_macro=[dict(zip(keys,key),parents=len(rr),**{f:mean(r[f] for r in rr) for f in fields}) for key,rr in sorted(macro.items())]
    write(root/'development_episode_macro.json',per_episode);write(root/'development_parent_macro.json',parent_macro);write(root/'development_parent_rows.json',parent_rows)
    write(root/'failure_case_inventory.json',dict(scope='all preregistered development source frames; supported labels only; review pending',cases=cases))
    write(root/'development_representative_points.json',geometry)
    geometry_episodes=defaultdict(list)
    for r in geometry:
        support=r.get('visible_reference_points')
        stratum='GT_UNSUPPORTED' if r['reference_kind']!='EXACT_GT' else 'absent' if support==0 else 'tiny_1_7' if support is not None and support<8 else 'visible_8_plus'
        geometry_episodes[r['episode'],r['parent_id'],r['object_id'],stratum,'t0' if r['timestep']==0 else 'later'].append(r)
    geometry_rows=[]
    for key,rr in geometry_episodes.items():
        gg=[r for r in rr if r['reference_kind']=='EXACT_GT'];n=lambda r:r['points']
        row=dict(episode=key[0],parent_id=key[1],object_id=key[2],support_stratum=key[3],time_stratum=key[4],frames=len(rr),supported_frames=len(gg),
            point_count=mean(r['points'] for r in gg),representative_purity=mean(r['correct']/n(r) if n(r) else None for r in gg),
            representative_robot_fraction=mean(r['robot']/n(r) if n(r) else None for r in gg),
            representative_wrong_object_fraction=mean(r['wrong_object']/n(r) if n(r) else None for r in gg),
            representative_background_fraction=mean(r['background']/n(r) if n(r) else None for r in gg),
            graph_missing_rate=mean(float(r['points']==0) for r in gg if r.get('visible_reference_points',0)>=8),
            graph_no_correct_point_rate=mean(float(r['correct']==0) for r in gg if r.get('visible_reference_points',0)>=8),
            false_object_presence=mean(float(r['points']>0) for r in gg if r.get('visible_reference_points')==0))
        for field in ('surface_distance_mean_mm','surface_distance_p95_mm','visible_coverage_4mm','projected_boundary_coverage_4mm','normal_valid_fraction'):row[field]=mean(r.get(field) for r in gg)
        geometry_rows.append(row)
    geom_fields=[k for k in geometry_rows[0] if k not in ('episode','parent_id','object_id','support_stratum','time_stratum','frames','supported_frames')] if geometry_rows else []
    parents=defaultdict(list)
    for r in geometry_rows:parents[r['parent_id'],r['object_id'],r['support_stratum'],r['time_stratum']].append(r)
    parent_geometry=[dict(parent_id=k[0],object_id=k[1],support_stratum=k[2],time_stratum=k[3],**{f:mean(r[f] for r in rr) for f in geom_fields}) for k,rr in parents.items()]
    macro_geometry=defaultdict(list)
    for r in parent_geometry:macro_geometry[r['object_id'],r['support_stratum'],r['time_stratum']].append(r)
    geometry_macro=[dict(object_id=k[0],support_stratum=k[1],time_stratum=k[2],parents=len(rr),**{f:mean(r[f] for r in rr) for f in geom_fields}) for k,rr in sorted(macro_geometry.items())]
    write(root/'geometry_episode_macro.json',geometry_rows);write(root/'geometry_parent_rows.json',parent_geometry);write(root/'geometry_parent_macro.json',geometry_macro)
    write(root/'aggregate_scope.json',dict(status='BASELINE_STAGE_AUDIT_AGGREGATED',stage_rows_are_pixel_or_attached_source_metrics=True,geometry_points_are_representatives=True,
        unsupported_reference_rows_retained=True,spatial_surface_metrics='representative purity; sampled collision mesh distance; visible/projected boundary coverage; excludes physical rim measurement',phase_metrics='PENDING',temporal_mask_metrics='PENDING',detector_attribution='PENDING',candidate_selection=False,validation_accessed=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
