"""Post-freeze diagnostics; no result is fed back into selection."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from .core import distances, angle, PROTOCOL, SEEDS, random_order, require_frozen, freeze


def csv_write(path, rows):
    path=Path(path);rows=list(rows); assert rows
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.with_suffix(path.suffix+'.tmp').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    tmp=path.with_suffix(path.suffix+'.tmp')
    if path.exists():
        assert path.read_bytes()==tmp.read_bytes(),'immutable CSV changed: '+str(path);tmp.unlink()
    else:tmp.replace(path)


def stats(v):
    v=np.array(v,dtype=float)
    return dict(mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9)),
                p95=float(np.quantile(v,.95)),maximum=float(v.max()))


def coverage(out,rows,sets,scale,split):
    require_frozen(out)
    results=[];groups=[];nearest=[]
    for name,selected in sets.items():
        d,g=distances(rows,selected,scale); n=d.min(1);idx=d.argmin(1)
        results.append(dict(cohort=split,dataset=name,n=len(rows),**stats(n)))
        for j in range(5):
            groups.append(dict(cohort=split,dataset=name,group='G'+str(j+1),
                definition='independently nearest within group',**stats(g[:,:,j].min(1))))
            groups.append(dict(cohort=split,dataset=name,group='G'+str(j+1),
                definition='group distance to total-distance nearest parent',**stats(g[np.arange(len(rows)),idx,j])))
        nearest += [dict(cohort=split,dataset=name,id=r['id'],nearest_parent=selected[int(k)]['id'],distance=float(v))
                    for r,k,v in zip(rows,idx,n)]
    return results,groups,nearest


def monte_carlo(out,training,holdout,base,sets,scale):
    require_frozen(out);remaining=[r['id'] for r in training if r['id'] not in {b['id'] for b in base}]
    byid={r['id']:r for r in training}; rng=np.random.Generator(np.random.PCG64(SEEDS['monte_carlo']))
    rows=[];samples=[]; summaries={}
    matrices={}
    for label,pop in [('selection',training),('holdout',holdout)]:
        all_d,_=distances(pop,training,scale);bd,_=distances(pop,base,scale)
        matrices[label]=(all_d,bd.min(1))
    index={r['id']:i for i,r in enumerate(training)}
    for repeat in range(PROTOCOL['monte_carlo_repeats']):
        chosen=rng.choice(sorted(remaining),19,replace=False).tolist();samples.append(chosen)
        for cohort,(matrix,baseline) in matrices.items():
            v=np.minimum(baseline,matrix[:,[index[k] for k in chosen]].min(1))
            rows.append(dict(repeat=repeat,cohort=cohort,**stats(v)))
    for cohort,pop in [('selection',training),('holdout',holdout)]:
        scores=np.array([r['mean'] for r in rows if r['cohort']==cohort])
        measured={}
        for name in ['R','D']:
            d,_=distances(pop,sets[name],scale);v=float(d.min(1).mean())
            measured[name]=dict(score=v,percentile_lower_is_better=float(100*np.mean(scores<=v)),
                                fraction_random_worse_or_equal=float(np.mean(scores>=v)))
        summaries[cohort]=dict(mean=float(scores.mean()),median=float(np.median(scores)),variance=float(scores.var(ddof=1)),
            p5=float(np.quantile(scores,.05)),p95=float(np.quantile(scores,.95)),datasets=measured)
    csv_write(out/'random_monte_carlo_metrics.csv',rows)
    freeze(out/'random_monte_carlo_sampling_trace.json',dict(seed=SEEDS['monte_carlo'],samples=samples,
        conditional_on_common44=True,R_reselected=False))
    return summaries


def fields(row):
    d=row['descriptor'];r={}
    for key,unit in [('peg_xyz','m'),('g1','m'),('g3p','m'),('g4','m'),('g5','image_fraction'),('pixel_extent','px')]:
        r.update({f'{key}_{i}_{unit}':v for i,v in enumerate(d[key])})
    r.update(peg_yaw_rad=d['peg_yaw_rad'],peg_hole_angle_rad=d['peg_hole_angle_rad'],
             boundary_margin_px=d['boundary_margin_px'],tcp_peg_distance_m=d['tcp_peg_distance_m'],hole_radius_m=d['hole_radius_m'])
    return r


def distribution(out,populations,fit_rows):
    reference=populations['candidate300'];rf={k:np.array([fields(r)[k] for r in reference]) for k in fields(reference[0])}
    summary=[];shift=[];boundary=[]
    extkeys=['peg_xyz_0_m','peg_xyz_1_m','peg_xyz_2_m','g4_0_m','g4_1_m']
    thresholds={k:np.quantile([fields(r)[k] for r in fit_rows],[.05,.95]) for k in extkeys}
    extreme={}
    for name,rows in populations.items():
        for key,ref in rf.items():
            v=np.array([fields(r)[key] for r in rows])
            summary.append(dict(population=name,feature=key,n=len(v),minimum=float(v.min()),
                std=float(v.std()),p5=float(np.quantile(v,.05)),**stats(v)))
            shift.append(dict(population=name,reference='candidate300',feature=key,
                wasserstein=float(wasserstein_distance(ref,v)),KS=float(ks_2samp(ref,v).statistic),
                inference='descriptive dependent samples; not a hypothesis-test p-value'))
        margins=np.array([r['descriptor']['boundary_margin_px'] for r in rows])
        boundary.append(dict(population=name,n=len(rows),**stats(margins),minimum=float(margins.min()),
            **{f'fraction_below_{t}px':float(np.mean(margins<t)) for t in PROTOCOL['boundary_thresholds_px']}))
        flags=[any(fields(r)[k]<v[0] or fields(r)[k]>v[1] for k,v in thresholds.items()) for r in rows]
        extreme[name]=dict(count=sum(flags),fraction=float(np.mean(flags)),n=len(rows))
    # Rotation-aware orientation shift: fixed pair-angle distribution, not unwrapped yaw alone.
    from .core import angle
    for name,rows in populations.items():
        q=[r['descriptor']['g2'] for r in rows];rq=[r['descriptor']['g2'] for r in reference]
        a=angle(q,q)[np.triu_indices(len(q),1)];b=angle(rq,rq)[np.triu_indices(len(rq),1)]
        shift.append(dict(population=name,reference='candidate300',feature='SO3_pair_angle_rad',
            wasserstein=float(wasserstein_distance(a,b)),KS=float(ks_2samp(a,b).statistic),
            inference='rotation-aware descriptive pair distribution; pairs not independent'))
    csv_write(out/'initial_distribution_summary.csv',summary)
    csv_write(out/'distribution_shift_metrics.csv',shift)
    csv_write(out/'boundary_margin_comparison.csv',boundary)
    freeze(out/'extreme_distribution_audit.json',dict(fitted_on='existing_train33 + training220',
        thresholds={k:v.tolist() for k,v in thresholds.items()},populations=extreme))
    return boundary,extreme


def duplicates(out,pairs,states,scales):
    records=[];counts={}
    keys=['peg_pose','hole_pose','tcp_pose','peg_half','hole_radius','qpos','qvel']
    def qangle(a,b):return float(angle(np.array(a)[None,3:],np.array(b)[None,3:])[0,0])
    for label,left,right,block_exact in pairs:
        count=dict(exact_same_parent=0,exact_distinct_parent=0,near_distinct_parent=0,pairs=len(left)*len(right))
        metrics=[]
        for a in left:
            best=None
            for b in right:
                sa,sb=states[a['id']],states[b['id']]
                positions={k:float(np.linalg.norm(sa[k][:3].astype(float)-sb[k][:3])) for k in ['peg_pose','hole_pose','tcp_pose']}
                angles={k:qangle(sa[k],sb[k]) for k in ['peg_pose','hole_pose','tcp_pose']}
                half=float(np.max(np.abs(sa['peg_half'].astype(float)-sb['peg_half'])))
                radius=abs(float(np.asarray(sa['hole_radius']).reshape(-1)[0])-float(np.asarray(sb['hole_radius']).reshape(-1)[0]))
                robot=float(np.max(np.abs(sa['qpos'].astype(float)-sb['qpos'])))
                exact=all(np.array_equal(sa[k],sb[k]) for k in keys)
                # q/-q equivalent exact rotations also count as identical physical states.
                equivalent=(all(v==0 for v in positions.values()) and all(v==0 for v in angles.values()) and half==radius==robot==0
                            and np.array_equal(sa['qvel'],sb['qvel']))
                exact=exact or equivalent
                near=(max(positions.values())<=.001 and max(angles.values())<=np.deg2rad(1) and half<=.0001 and radius<=.0001 and robot<=.001)
                score=max(max(positions.values())/.001,max(angles.values())/np.deg2rad(1),half/.0001,radius/.0001,robot/.001)
                row=dict(comparison=label,left=a['id'],right=b['id'],same_parent=a['id']==b['id'],exact=exact,near=near,
                    peg_translation_m=positions['peg_pose'],hole_translation_m=positions['hole_pose'],tcp_translation_m=positions['tcp_pose'],
                    peg_angle_rad=angles['peg_pose'],hole_angle_rad=angles['hole_pose'],tcp_angle_rad=angles['tcp_pose'],
                    half_size_max_m=half,hole_radius_diff_m=radius,robot_qpos_max=robot,threshold_max_ratio=score)
                if best is None or score<best['threshold_max_ratio']:best=row
                if exact or near:
                    records.append(dict(row,record_type='exact_or_near'))
                    if exact: count['exact_same_parent' if row['same_parent'] else 'exact_distinct_parent']+=1
                    elif near:count['near_distinct_parent']+=1
                    if block_exact and exact:raise ValueError('Identical initial physical state crosses split: '+str(row))
            if best:records.append(dict(best,record_type='nearest_physical_match_for_each_left'))
        counts[label]=count
    csv_write(out/'near_duplicate_audit.csv',records)
    freeze(out/'physical_duplicate_summary.json',dict(status='PASS',comparisons=counts,
        near_rule=PROTOCOL['near_duplicate'],exact='all peg/hole/TCP poses (q/-q equivalent), shape, qpos/qvel identical',
        near_automatic_removal=False))
    return counts
