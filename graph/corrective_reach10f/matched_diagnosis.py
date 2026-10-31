"""Read-only, post-action Task10-F trace diagnosis; no simulator dependency."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
    assert read(path) == value


def csvwrite(path, rows):
    assert rows
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    assert len(list(csv.DictReader(Path(path).open()))) == len(rows)


def contact_category(names):
    text = '|'.join(names).lower()
    if 'table' in text: return 'ROBOT_TABLE'
    if all('panda' in n for n in names): return 'ROBOT_SELF'
    if 'finger' in text and ('peg' in text or 'box' in text): return 'FINGER_PEG_BOX'
    if 'peg' in text or 'box' in text: return 'OTHER_ROBOT_PEG_BOX'
    return 'OTHER_EXTERNAL'


def safety_predicates(row, protocol):
    d = row['physical_details']; out = []
    for v in d['shape_violations']:
        out.append('SHAPE_CLEARANCE_' + contact_category(v['pair']))
    for c in d['substep_contact_violations']:
        out.append('CONTACT_' + contact_category(c['names']))
    if d['joint_limit_violation']: out.append('JOINT_POSITION_LIMIT')
    if d['max_qvel'] >= protocol['max_joint_velocity_rad_s']: out.append('JOINT_VELOCITY_LIMIT')
    for k, v in d['object_translation_m'].items():
        if v >= .002: out.append('OBJECT_TRANSLATION_' + k)
    for k, v in d['object_rotation_rad'].items():
        if v >= .01: out.append('OBJECT_ROTATION_' + k)
    if not row['physical_safety'] and not out:
        out.append('OTHER_SAFETY_NOT_RESOLVABLE_FROM_TRACE')
    assert bool(out) == (not row['physical_safety'])
    return sorted(set(out))


def longest(values):
    best = n = 0
    for value in values:
        n = n + 1 if value else 0
        best = max(best, n)
    return best


def ranges(rows, predicate):
    groups = []
    for r in rows:
        t = r['post_state_t']
        if predicate(r):
            if groups and groups[-1][1] == t - 1: groups[-1][1] = t
            else: groups.append([t, t])
    return groups


def diagnose(folder, checkpoint_label, protocol):
    episode = read(folder/'episode.json'); trace = read(folder/'trace.json')
    assert trace and len(trace) == episode['actions']
    first_approach = None; stable = 0; completion = None; timeline = []
    for i, r in enumerate(trace):
        assert r['timestep'] == i and abs(r['simulator_time_s'] - (i+1)/20) < 1e-10
        ap = r['approach_position_error_m'] < .010
        ar = r['approach_orientation_error_rad'] < .05
        if ap and ar and first_approach is None: first_approach = i+1
        pp = r['target_position_error_m'] < .004
        pr = r['target_orientation_error_rad'] < .03
        opening = r['opening_m'] >= .065
        peg = r['peg_translation_m'] < .002
        student = r['segment'].startswith('student')
        ordered = first_approach is not None
        no_safety = student and ordered and pp and pr and opening and peg
        stable = stable+1 if no_safety and r['physical_safety'] else 0
        if stable >= 3 and completion is None: completion = i+1
        prev = trace[i-1] if i else None
        predicates = safety_predicates(r, protocol['safety'])
        timeline.append(dict(episode=folder.name, checkpoint=checkpoint_label, condition=episode['condition'], kind=episode['kind'],
            seed=episode['seed'], offset=episode['requested_offset_mm'], action_t=i, pre_state_t=i, post_state_t=i+1,
            simulator_time_s=r['simulator_time_s'], segment=r['segment'],
            position_error_mm=r['target_position_error_m']*1000, orientation_error_rad=r['target_orientation_error_rad'],
            approach_position_error_mm=r['approach_position_error_m']*1000, approach_orientation_error_rad=r['approach_orientation_error_rad'],
            pre_position_error_mm=None if prev is None else prev['target_position_error_m']*1000,
            pre_orientation_error_rad=None if prev is None else prev['target_orientation_error_rad'],
            position_error_delta_mm=None if prev is None else (r['target_position_error_m']-prev['target_position_error_m'])*1000,
            orientation_error_delta_rad=None if prev is None else r['target_orientation_error_rad']-prev['target_orientation_error_rad'],
            approach_position_pass=ap, approach_orientation_pass=ar, approach_simultaneous_pass=ap and ar,
            approach_seen=ordered, preclose_position_pass=pp, preclose_orientation_pass=pr, preclose_simultaneous_pass=pp and pr,
            gripper_opening_mm=r['opening_m']*1000, gripper_pass=opening, peg_motion_pass=peg,
            physical_safety=r['physical_safety'], safety_predicates=predicates,
            no_safety_ordered_candidate=no_safety, official_stable_count=stable,
            tcp_world_position_m=r['tcp_pose'][:3], tcp_quaternion_wxyz=r['tcp_pose'][3:]))
    assert first_approach == episode['first_approach_t']
    assert completion == episode['completion_t']
    official = completion is not None and all(r['physical_safety'] for r in trace) and episode['disturbance_ok']
    assert official == episode['corrective_reach_success']
    recovery = [r for r in timeline if r['segment'].startswith('student')]
    assert recovery
    first = recovery[0]; pre = timeline[first['action_t']-1] if first['action_t'] else None
    pos_min = min(recovery, key=lambda r:r['position_error_mm'])
    rot_min = min(recovery, key=lambda r:r['orientation_error_rad'])
    approach_min = min(recovery, key=lambda r:r['approach_position_error_mm'])
    approach_rot_min = min(recovery, key=lambda r:r['approach_orientation_error_rad'])
    everpos = any(r['preclose_position_pass'] for r in recovery)
    everrot = any(r['preclose_orientation_pass'] for r in recovery)
    ordered_pose = [r['approach_seen'] and r['preclose_simultaneous_pass'] for r in recovery]
    categories = []
    if not everpos: categories.append('POSITION_FAILURE')
    if not everrot: categories.append('ORIENTATION_FAILURE')
    if first_approach is None: categories.append('ORDER_FAILURE')
    if longest([r['no_safety_ordered_candidate'] for r in recovery]) < 3: categories.append('STABILITY_FAILURE')
    if not all(r['gripper_pass'] for r in recovery): categories.append('GRIPPER_FAILURE')
    if not episode['physical_safety']: categories.append('PHYSICAL_SAFETY_FAILURE')
    if len(categories)>1: categories.append('MULTIPLE_FAILURES')
    pos_frames = [r for r in recovery if r['preclose_position_pass']]
    safety_rows = [r for r in timeline if not r['physical_safety']]
    last = recovery[-1]
    summary = dict(episode=folder.name,checkpoint=checkpoint_label,condition=episode['condition'],kind=episode['kind'],seed=episode['seed'],
        offset=episode['requested_offset_mm'],reset_sha256=episode['reset_sha256'],official_success=official,
        disturbance_ok=episode['disturbance_ok'],disturbance_steps=episode['disturbance_steps'],actual_offset_mm=episode['actual_offset_mm'],
        aborted_before_intervention=episode['kind']!='nominal' and episode['disturbance_steps']==0 and not episode['physical_safety'],
        recovery_start_t=episode['recovery_start_t'],student_actions=len(recovery),last_post_state_t=last['post_state_t'],
        handoff_pre_position_error_mm=None if pre is None else pre['position_error_mm'],
        handoff_pre_orientation_error_rad=None if pre is None else pre['orientation_error_rad'],
        handoff_pre_safe=None if pre is None else pre['physical_safety'],
        first_student_post_position_error_mm=first['position_error_mm'],first_student_post_orientation_error_rad=first['orientation_error_rad'],
        final_position_error_mm=last['position_error_mm'],minimum_position_error_mm=pos_min['position_error_mm'],minimum_position_post_t=pos_min['post_state_t'],
        final_orientation_error_rad=last['orientation_error_rad'],minimum_orientation_error_rad=rot_min['orientation_error_rad'],minimum_orientation_post_t=rot_min['post_state_t'],
        minimum_approach_position_error_mm=approach_min['approach_position_error_mm'],approach_first_post_t=first_approach,
        minimum_approach_position_post_t=approach_min['post_state_t'],
        final_approach_position_error_mm=last['approach_position_error_mm'],
        minimum_approach_orientation_error_rad=approach_rot_min['approach_orientation_error_rad'],minimum_approach_orientation_post_t=approach_rot_min['post_state_t'],
        all_observed_minimum_position_error_mm=min(r['position_error_mm'] for r in timeline),
        all_observed_minimum_orientation_error_rad=min(r['orientation_error_rad'] for r in timeline),
        approach_seen_before_student=first_approach is not None and first_approach<=first['action_t'],
        handoff_pre_approach_position_error_mm=None if pre is None else pre['approach_position_error_mm'],
        handoff_pre_approach_orientation_error_rad=None if pre is None else pre['approach_orientation_error_rad'],
        final_position_predicate_pass=last['preclose_position_pass'],final_orientation_predicate_pass=last['preclose_orientation_pass'],
        orientation_ever_in_tolerance=everrot,
        failure_classification_semantics='POSITION/ORIENTATION_FAILURE: criterion never attained during student phase; endpoint failures are separate columns. STABILITY_FAILURE: 3-step complete candidate absent, including upstream failures.',
        preclose_pose_first_post_t=next((r['post_state_t'] for r in recovery if r['preclose_simultaneous_pass']),None),
        preclose_ordered_pose_first_post_t=next((r['post_state_t'] for r in recovery if r['preclose_simultaneous_pass'] and r['approach_seen']),None),
        preclose_position_frames=len(pos_frames),preclose_orientation_frames=sum(r['preclose_orientation_pass'] for r in recovery),
        position_only_pass_frames=sum(r['preclose_position_pass'] and not r['preclose_orientation_pass'] for r in recovery),
        orientation_only_pass_frames=sum(r['preclose_orientation_pass'] and not r['preclose_position_pass'] for r in recovery),
        orientation_at_position_pass_rad=[r['orientation_error_rad'] for r in pos_frames],
        approach_position_only_pass_frames=sum(r['approach_position_pass'] and not r['approach_orientation_pass'] for r in recovery),
        approach_orientation_only_pass_frames=sum(r['approach_orientation_pass'] and not r['approach_position_pass'] for r in recovery),
        simultaneous_pose_max_consecutive=longest(ordered_pose),no_safety_max_consecutive=longest([r['no_safety_ordered_candidate'] for r in recovery]),
        diagnostic_without_safety=longest([r['no_safety_ordered_candidate'] for r in recovery])>=3,
        official_max_consecutive=max(r['official_stable_count'] for r in recovery),
        gripper_final_mm=last['gripper_opening_mm'],gripper_minimum_mm=min(r['gripper_opening_mm'] for r in recovery),
        physical_safety=episode['physical_safety'],first_safety_failure_post_t=None if not safety_rows else safety_rows[0]['post_state_t'],
        first_safety_failure_segment=None if not safety_rows else safety_rows[0]['segment'],
        safety_observed_frames=len(safety_rows),safety_continued_observation='CENSORED_AT_FIRST_FAILURE' if safety_rows else 'NO_FAILURE_OBSERVED',
        observed_unsafe_after_pose_success=any(r['preclose_simultaneous_pass'] and not r['physical_safety'] for r in timeline),
        termination_reason=('PHYSICAL_SAFETY_ABORT_BEFORE_INTERVENTION' if episode['kind']!='nominal' and episode['disturbance_steps']==0 else 'PHYSICAL_SAFETY_ABORT') if not episode['physical_safety'] else ('REACH_SUCCESS' if official else 'HORIZON_EXHAUSTED'),
        target_error_reduction_mm=None if episode['target_error_reduction_m'] is None else episode['target_error_reduction_m']*1000,
        recovery_overshoot_mm=None if episode['recovery_error_overshoot_m'] is None else episode['recovery_error_overshoot_m']*1000,
        moving_toward_target_intervals=ranges(recovery,lambda r:r['position_error_delta_mm'] is not None and r['position_error_delta_mm'] < -1e-6),
        moving_away_target_intervals=ranges(recovery,lambda r:r['position_error_delta_mm'] is not None and r['position_error_delta_mm'] > 1e-6),
        first_increase_after_decrease_post_t=next((r['post_state_t'] for i,r in enumerate(recovery) if i>0 and r['position_error_delta_mm'] is not None and r['position_error_delta_mm']>1e-6 and any(x['position_error_delta_mm'] is not None and x['position_error_delta_mm'] < -1e-6 for x in recovery[:i])),None),
        maximum_rebound_from_running_min_mm=max(r['position_error_mm']-min(x['position_error_mm'] for x in recovery[:i+1]) for i,r in enumerate(recovery)),
        failure_categories=categories,trace_sha256=sha(folder/'trace.json'),video_path=str(folder/'rollout.mp4'),video_sha256=sha(folder/'rollout.mp4'))
    events = []
    for r in trace:
        predicates = safety_predicates(r,protocol['safety'])
        if predicates:
            d=r['physical_details']
            events.append(dict(action_t=r['timestep'],post_state_t=r['timestep']+1,segment=r['segment'],predicates=predicates,
                shape_violations=d['shape_violations'],substep_contact_violations=d['substep_contact_violations'],
                object_translation_m=d['object_translation_m'],object_rotation_rad=d['object_rotation_rad'],max_qvel=d['max_qvel'],
                joint_position_limit_violation=d['joint_limit_violation'],
                preclose_pose_pass=r['target_position_error_m']<.004 and r['target_orientation_error_rad']<.03))
    safety = dict(episode=folder.name,checkpoint=checkpoint_label,first_failure_post_t=summary['first_safety_failure_post_t'],
        handoff_pre_safe=summary['handoff_pre_safe'],events=events,
        transient_or_persistent='NOT_AVAILABLE_AFTER_EARLY_ABORT' if events else 'NO_OBSERVED_FAILURE',
        full_contact_substep_index_and_force='NOT_AVAILABLE; only threshold-violating contacts flattened by original evaluator; impulse is not force',
        nominal_expert_same_final_seed='NOT_TESTED; original nominal safety validation seed30000 is a separate condition')
    return summary,timeline,safety


def run(source,output,additional=None):
    output.mkdir(parents=True,exist_ok=False)
    protocol=read(source/'final_evaluation/evaluation_protocol.json')
    sources=[(source/'final_evaluation/rollouts',{'R0':'R0_step0','R1':'R1_step3000'})]
    if additional: sources.append((additional/'rollouts',{'R0':'R0_step3000'}))
    allsummary=[];alltimeline=[];allsafety=[];inventory=[]
    for folder,labels in sources:
        for path in sorted(folder.glob('*/episode.json')):
            ep=read(path); summary,timeline,safety=diagnose(path.parent,labels[ep['condition']],protocol)
            allsummary.append(summary)
            if ep['kind']=='handoff':alltimeline.extend(timeline);allsafety.append(safety)
            inventory.append(dict(episode=str(path.parent),checkpoint=labels[ep['condition']],trace_sha256=sha(path.parent/'trace.json'),episode_sha256=sha(path)))
    csvwrite(output/'episode_diagnostics.csv',allsummary)
    handoff=[r for r in allsummary if r['kind']=='handoff']
    csvwrite(output/'handoff_failure_breakdown.csv',handoff)
    csvwrite(output/'handoff_stepwise_metrics.csv',alltimeline)
    csvwrite(output/'reach_predicate_timeline.csv',alltimeline)
    write(output/'physical_safety_breakdown.json',dict(episodes=allsafety,
        failure_predicate_counts=dict(Counter(p for e in allsafety for v in e['events'] for p in v['predicates'])),
        semantics='Unchanged Safety v2. Each trace terminates at the first failing control boundary; persistence after abort is not observable.'))
    write(output/'trace_inventory.json',inventory)
    write(output/'diagnosis_complete.json',dict(status='PASS',episodes=len(allsummary),handoff=len(handoff),
        semantics='All errors are stored post-action GT evaluation metrics. Pre-action handoff is preceding row. Independent minimum position and orientation never imply success.',
        original_success_recomputed_exact=True,substep_forces='NOT_AVAILABLE',new_simulator_executions=0))
    for label in sorted(set(r['checkpoint'] for r in handoff)):
        rows=[r for r in handoff if r['checkpoint']==label]
        print(label,dict(episodes=len(rows),position_min_below4mm=sum(r['minimum_position_error_mm']<4 for r in rows),
            orientation_min_below03=sum(r['minimum_orientation_error_rad']<.03 for r in rows),
            approach=sum(r['approach_first_post_t'] is not None for r in rows),
            preclose=sum(r['preclose_pose_first_post_t'] is not None for r in rows),
            without_safety=sum(r['diagnostic_without_safety'] for r in rows),
            safety_fail=sum(not r['physical_safety'] for r in rows)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--additional',type=Path)
    a=p.parse_args();run(a.source,a.output,a.additional)
