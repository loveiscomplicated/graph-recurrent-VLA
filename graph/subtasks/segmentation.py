"""Ordered geometric milestones, distinct from unavailable expert phase labels."""
import numpy as np
from .conditions import ConditionContext
from .dataset import state_row
from .specs import NAMES,SubtaskSpec


def runs(mask):
    indices=np.flatnonzero(mask)
    if not len(indices): return []
    groups=np.split(indices,np.where(np.diff(indices)>1)[0]+1)
    return [[int(g[0]),int(g[-1])+1] for g in groups]


def evaluate(data,config):
    specs={n:SubtaskSpec(n,config) for n in NAMES}
    context=ConditionContext(); metrics=[]; starts=[]; successes=[]; failures=[]
    for t in range(len(data['actions'])+1):
        m=specs['reach'].compute_metrics(state_row(data,t))
        context.update(m,t,config,specs['align'].align_raw(m))
        metrics.append(m)
        starts.append({n:specs[n].check_start(m,context) for n in NAMES})
        successes.append({n:specs[n].check_success(m,context) for n in NAMES})
        failures.append({n:specs[n].failure_reasons(m,context) for n in NAMES})
    return metrics,starts,successes,failures


def segment_episode(data,config):
    metrics,starts,successes,failures=evaluate(data,config)
    T=len(data['actions']); masks={n:np.asarray([s[n] for s in successes]) for n in NAMES}
    close=np.flatnonzero(data['actions'][:,-1]<0)
    close_t=int(close[0]) if len(close) else None
    boundaries={}; reasons={}; candidates={}
    # Reach is closest approach within the FIRST valid open reach run, before
    # close. This estimates a geometric waypoint; it is not an oracle label.
    reach_mask=masks['reach'].copy()
    if close_t is not None: reach_mask[close_t+1:]=False
    candidates['reach']=runs(reach_mask)
    if candidates['reach'] and close_t is not None:
        a,b=candidates['reach'][0]
        endpoint=min(range(a,b),key=lambda t:(metrics[t]['position_error'],t))
        if endpoint>0:
            boundaries['reach']=(0,endpoint)
            reasons['reach']='closest approach in first open reach-success run before close command'
    previous=boundaries.get('reach',(None,None))[1]
    for name in ('grasp','transport'):
        begin=previous
        eligible=np.flatnonzero(masks[name])
        eligible=eligible[eligible>begin] if begin is not None else np.array([],dtype=int)
        if name=='grasp' and close_t is not None: eligible=eligible[eligible>close_t]
        candidates[name]=runs(masks[name])
        if len(eligible):
            endpoint=int(eligible[0]); boundaries[name]=(begin,endpoint);previous=endpoint
            reasons[name]='first stable real contact after close' if name=='grasp' else 'first coarse pre-insertion geometric milestone; not planner phase'
        else: previous=None
    # Last stable alignment run BEFORE FIRST entrance crossing avoids early
    # refinement success that immediately exits. No lowering threshold to fill gaps.
    entry=next((t for t,m in enumerate(metrics) if previous is not None and t>previous and m['insertion_depth']>0),T+1)
    align_mask=masks['align'].copy()
    align_mask[:previous+1 if previous is not None else len(align_mask)]=False
    align_mask[entry:]=False
    candidates['align']=runs(align_mask)
    if previous is not None and candidates['align']:
        endpoint=candidates['align'][-1][0]
        boundaries['align']=(previous,endpoint);previous=endpoint
        reasons['align']='last sustained surface-clear pre-entry alignment run; geometric partition'
    else: previous=None
    candidates['insert']=runs(masks['insert'])
    if previous is not None:
        eligible=np.flatnonzero(masks['insert']);eligible=eligible[eligible>previous]
        if len(eligible):
            boundaries['insert']=(previous,int(eligible[0]))
            reasons['insert']='first live evaluate success following valid alignment'
    result=[]
    for n in NAMES:
        boundary=boundaries.get(n); i,j=boundary if boundary else (None,None)
        start_valid=bool(boundary and starts[i][n]);success=bool(boundary and successes[j][n])
        failed=bool(boundary and (j-i>=config['timeout_steps'] or
            any([reason for reason in failures[t][n] if reason!='timeout'] for t in range(i,j+1))))
        valid=bool(boundary and start_valid and success and not failed and data['record']['replay_valid'])
        reason=reasons.get(n,'missing prerequisite or no valid ordered condition boundary')
        if failed: reason+='; failure predicate encountered'
        if boundary and not start_valid: reason+='; start condition invalid'
        if boundary and not success: reason+='; endpoint success condition invalid'
        if not data['record']['replay_valid']: reason+='; replay observation mismatch'
        result.append(dict(subtask=n,start_t=i,end_t=j,start_valid=start_valid,success_reached=success,
            success=success,status='detected' if valid else 'unresolved',
            termination_reason='success' if valid else 'unresolved',boundary_reason=reason,
            boundary_status='geometry_estimate' if valid else 'unresolved',
            candidate_runs=candidates.get(n,[]),multiple_candidates=len(candidates.get(n,[]))>1,
            failure_encountered=failed,bc_eligible=valid and n=='reach'))
    diagnostics={n:dict(first_success=int(np.flatnonzero(masks[n])[0]) if masks[n].any() else None,
        success_states=int(masks[n].sum()),success_runs=runs(masks[n]),
        exits_after_success=sum(1 for a,b in runs(masks[n]) if b<=T),
        start_states=sum(s[n] for s in starts)) for n in NAMES}
    return result,metrics,diagnostics
