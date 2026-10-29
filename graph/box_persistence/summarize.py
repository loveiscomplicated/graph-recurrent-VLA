"""Preregistered episode-macro geometry gate; no tuning or fitting."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from graph.integration_baseline.io import write_json
from graph.integration_baseline.geometry09b_probe import csv_write


def read_csv(path):
    with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
    for r in rows:
        for k,v in r.items():
            if v=='':r[k]=None
            elif v in ('True','False'):r[k]=v=='True'
            else:
                try:r[k]=float(v)
                except (TypeError,ValueError):pass
    return rows


def average(rows,key):
    a=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(a)) if a else None


def macro(rows,group):
    numeric={k for r in rows for k,v in r.items() if isinstance(v,(float,int)) and not isinstance(v,bool)}-set(group)-{'seed','timestep'}
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in group)].append(r)
    result=[]
    for key,rr in groups.items():
        per=defaultdict(list)
        for r in rr:per[r['seed']].append(r)
        row=dict(zip(group,key));row.update(episodes=len(per),frames=len(rr))
        for k in sorted(numeric):
            values=[average(ep,k) for ep in per.values()];values=[v for v in values if v is not None]
            row[k]=float(np.mean(values)) if values else None
            row[k+'_defined_episodes']=len(values)
        result.append(row)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--motion',type=Path,required=True);p.add_argument('--visible-motion',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(__file__).with_name('protocol.json').read_text());gate=cfg['gate']
    rows=sum([read_csv(p) for p in sorted(a.geometry.glob('geometry/seed*/metrics.csv'))],[])
    edges=sum([read_csv(p) for p in sorted(a.geometry.glob('geometry/seed*/edges.csv'))],[])
    for r in edges:
        r['false_bridge_fraction']=r['mesh_unsupported_edges_4mm']/r['undirected_surface_edges'] if r['undirected_surface_edges'] else 0.
    motion=read_csv(a.motion/'motion_metrics.csv')
    for r in motion:r['control_arm']='original_causal_prefix'
    if a.visible_motion:
        extra=read_csv(a.visible_motion/'motion_metrics.csv')
        for r in extra:r['control_arm']='baseline_initialized_t0_same_physical_state'
        motion.extend(extra)
    stats=macro(rows,('split','variant','budget'));edge_stats=macro(edges,('split','variant','budget'))
    write_json(a.output/'geometry_macro.json',stats);csv_write(a.output/'geometry_macro.csv',stats)
    write_json(a.output/'edges_macro.json',edge_stats);csv_write(a.output/'edges_macro.csv',edge_stats)
    allmotion=sum([read_csv(p) for p in sorted(a.geometry.glob('geometry/seed*/motion.csv'))],[])
    motion_gt=dict(episodes=len({r['seed'] for r in allmotion}),frames=len(allmotion),
        max_translation_from_initial_mm=max(r['translation_from_initial_mm'] for r in allmotion),
        max_rotation_from_initial_rad=max(r['rotation_from_initial_rad'] for r in allmotion),contact_force='NOT_TESTED')
    write_json(a.output/'normal_box_motion.json',motion_gt)
    tracking=[]
    for path in a.geometry.glob('geometry/seed*/*_tracking.json'):
        rr=json.loads(path.read_text());name=path.stem.removesuffix('_tracking');seed=int(path.parent.name[4:])
        tracking.append(dict(seed=seed,variant=name,frames=len(rr),
            mean_pose_center_error_mm=average(rr,'evaluation_pose_center_error_mm'),
            max_pose_center_error_mm=max([r['evaluation_pose_center_error_mm'] for r in rr if 'evaluation_pose_center_error_mm' in r],default=None),
            max_pose_rotation_error_rad=max([r['evaluation_pose_rotation_error_rad'] for r in rr if 'evaluation_pose_rotation_error_rad' in r],default=None),
            static_violation_frames=sum(r['static_violation'] for r in rr),unobservable_frames=sum(r['state']=='UNOBSERVABLE_UNKNOWN' for r in rr),
            registration_failure_frames=sum(r['state']=='TRACKING_FAILURE_OR_MOTION_UNKNOWN' for r in rr),
            rejected_points=sum(r['rejected_count'] for r in rr),accepted_points=sum(r['accepted_count'] for r in rr)))
    csv_write(a.output/'tracking_per_episode.csv',tracking)
    decisions=[]
    for candidate in ('B1','B2'):
        tests=[]
        def check(name,value,limit,direction='max',budget=None):
            tests.append(dict(criterion=name,budget=budget,value=value,limit=limit,
                status='NOT_TESTED' if value is None else 'PASS' if (value<=limit if direction=='max' else value>=limit) else 'FAIL'))
        for mode in ('native','cap128','matched_N'):
            r=next(s for s in stats if s['split']=='validation' and s['variant']==candidate and s['budget']==mode)
            b=next(s for s in stats if s['split']=='validation' and s['variant']=='B0' and s['budget']==mode)
            e=next(s for s in edge_stats if s['split']=='validation' and s['variant']==candidate and s['budget']==('cap128' if mode=='native' else mode))
            for key,threshold in [('surface_distance_mean_mm','mean_distance_max_mm'),('surface_distance_p95_mm','p95_distance_max_mm'),
                ('estimated_ghost_fraction','estimated_ghost_fraction_maximum'),('ghost_nodes','estimated_ghost_mean_nodes_maximum')]:check(key,r[key],gate[threshold],budget=mode)
            for key,threshold in [('surface_distance_mean_mm','mean_distance_max_increase_mm'),('surface_distance_p95_mm','p95_distance_max_increase_mm')]:
                check(key+'_increase',r[key]-b[key] if r[key] is not None and b[key] is not None else None,gate[threshold],budget=mode)
            check('source_purity',r['source_object_purity'],gate['purity_minimum'],'min',mode)
            check('source_purity_drop',b['source_object_purity']-r['source_object_purity'],gate['purity_maximum_drop'],budget=mode)
            check('historical_coverage_gain',r['historical_coverage']-b['historical_coverage'],gate['historical_coverage_minimum_gain'],'min',mode)
            check('current_coverage_drop',b['current_visible_coverage']-r['current_visible_coverage'],gate['current_coverage_maximum_drop'],budget=mode)
            check('current_rim_drop',b['current_rim_coverage']-r['current_rim_coverage'] if b['current_rim_coverage'] is not None and r['current_rim_coverage'] is not None else None,gate['observed_rim_coverage_maximum_drop'],budget=mode)
            if mode!='native':
                check('aperture_crossing_edges',e['aperture_crossing_edges'],gate['aperture_crossing_edges_maximum'],budget=mode)
                check('false_bridge_fraction',e['false_bridge_fraction'],gate['false_bridge_fraction_maximum'],budget=mode)
        visible=[r for r in motion if r['mode']==candidate and r['phase']=='intervention' and r['physical_valid'] and r['imposed_delta_mm']>0 and not r['condition'].startswith(('box_full','box_partial')) and r['gt_visible_pixels']>=6 and r.get('prior_nodes',0)>=6]
        reappear=[r for r in motion if r['mode']==candidate and r['phase']=='reappearance_represented_snapshot' and r['physical_valid'] and r['surface_mean_mm'] is not None]
        check('visible_motion_detection',float(np.mean([r['detected'] for r in visible])) if visible else None,gate['motion_visible_detection_minimum'],'min')
        check('reappearance_surface_error_mm',average(reappear,'surface_mean_mm'),gate['reappearance_error_maximum_mm'])
        hidden=json.loads((a.motion/'motion_manifest.json').read_text())['hidden_negative_control_pass']
        if a.visible_motion:hidden &= json.loads((a.visible_motion/'motion_manifest.json').read_text())['hidden_negative_control_pass']
        check('hidden_sensor_negative_control',int(hidden),1,'min')
        failed=any(x['status']=='FAIL' for x in tests);missing=any(x['status']=='NOT_TESTED' for x in tests)
        decisions.append(dict(candidate=candidate,decision='NO-GO' if failed else 'CONDITIONAL' if missing else 'GO',criteria=tests))
    decision=dict(scope='development/validation geometry feasibility only; no independent Final or policy claim',
        threshold_source='preregistered_protocol.json; unchanged thresholds',candidates=decisions,
        eligible_for_GNN=[d['candidate'] for d in decisions if d['decision'] in ('GO','CONDITIONAL')],
        overall='NO-GO' if all(d['decision']=='NO-GO' for d in decisions) else 'CONDITIONAL',
        B_SAFE='NOT_EXECUTED; no post-result rule tuning',independent_final='NOT_TESTED',
        gnn_comparison='NOT_EXECUTED' if all(d['decision']=='NO-GO' for d in decisions) else 'PENDING')
    write_json(a.output/'gate_decision.json',decision)
    csv_write(a.output/'motion_controls.csv',motion)
    print(json.dumps({k:v for k,v in decision.items() if k!='candidates'},indent=2),flush=True)
if __name__=='__main__':main()
