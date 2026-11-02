"""Pure post-inference reliability metrics, preserving undefined denominators."""
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree


def pixel_metrics(prediction,target,robot=None,other=None):
    prediction=np.asarray(prediction,bool);target=np.asarray(target,bool)
    if prediction.shape!=target.shape:raise ValueError('unaligned masks')
    tp=int((prediction&target).sum());fp=int((prediction&~target).sum());fn=int((~prediction&target).sum())
    n=int(prediction.sum());visible=int(target.sum())
    precision=tp/n if n else None
    iou=tp/(tp+fp+fn) if tp+fp+fn else None
    result=dict(tp=tp,fp=fp,fn=fn,predicted_pixels=n,visible_pixels=visible,
        precision=precision,recall=tp/visible if visible else None,iou=iou,
        correct_mask_availability=float(n>0 and precision>=.9 and iou>=.5) if visible>=8 else None,
        correct_pixel_missing=float(tp==0) if visible>=8 else None,
        false_presence=float(n>0) if visible==0 else None)
    if robot is not None and other is not None:
        robot=np.asarray(robot,bool);other=np.asarray(other,bool)
        if robot.shape!=target.shape or other.shape!=target.shape:raise ValueError('unaligned actor labels')
        result.update(robot_pixels=int((prediction&robot).sum()),wrong_object_pixels=int((prediction&other).sum()),
            background_pixels=int((prediction&~target&~robot&~other).sum()))
    return result


def correct_source_coverage(points,correct,reference,radius=.004):
    points=np.asarray(points).reshape(-1,3);correct=np.asarray(correct,bool)
    reference=np.asarray(reference).reshape(-1,3)
    if len(correct)!=len(points):raise ValueError('point provenance length')
    if not np.isfinite(points).all() or not np.isfinite(reference).all():raise ValueError('nonfinite geometry')
    if not len(reference):return None
    valid=points[correct]
    return float((cKDTree(valid).query(reference)[0]<radius).mean()) if len(valid) else 0.


def hierarchical_paired(rows,metric,group_keys=('camera','object_id'),bootstrap=2000,seed=42,candidate='A'):
    """Paired available values -> episode means -> parent means -> paired CI.

    Input requires each physical observation only once per condition. Missing
    reference values and undefined precision never become success or zero.
    """
    if candidate=='B0' or not candidate.isidentifier():raise ValueError('distinct named challenger required')
    observations={}
    for row in rows:
        key=tuple(row.get(k) for k in group_keys)+(row['parent_id'],row['episode'],row['timestep'])
        condition=row['condition']
        if condition not in ('B0',candidate):continue
        per=observations.setdefault(key,{})
        if condition in per:raise ValueError('duplicate physical observation/condition')
        per[condition]=row.get(metric) if row.get('reference_kind','EXACT_GT')=='EXACT_GT' else None
    episodes=defaultdict(list)
    for key,values in observations.items():
        if set(values)=={'B0',candidate} and all(v is not None and np.isfinite(v) for v in values.values()):
            episodes[key[:-1]].append((values['B0'],values[candidate]))
    parents=defaultdict(list)
    for key,values in episodes.items():
        parents[key[:-1]].append(np.mean(values,axis=0))
    groups=defaultdict(list)
    for key,values in parents.items():
        groups[key[:-1]].append((key[-1],np.mean(values,axis=0)))
    result=[];rng=np.random.default_rng(seed)
    for group,values in sorted(groups.items(),key=lambda x:str(x[0])):
        array=np.stack([v for p,v in values]);delta=array[:,1]-array[:,0]
        samples=delta[rng.integers(0,len(delta),(bootstrap,len(delta)))].mean(axis=1)
        result.append(dict(zip(group_keys,group),metric=metric,baseline=float(array[:,0].mean()),
            candidate=float(array[:,1].mean()),delta=float(delta.mean()),
            ci95=np.quantile(samples,[.025,.975]).tolist(),paired_parents=len(values),
            parent_values=[dict(parent_id=p,baseline=float(a[0]),candidate=float(a[1])) for p,a in values],
            aggregation='paired frame availability; episode mean; physical parent mean; parent bootstrap'))
    return result
