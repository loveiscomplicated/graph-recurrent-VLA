"""Freeze explicit strata before the 768/1024 comparison, including old audits."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict


def main(args):
    screen=json.loads(Path(args.screen).read_text());config=json.loads(Path(args.protocol).read_text())
    historical=json.loads(Path(args.historical).read_text());lookup=defaultdict(list)
    for x in historical:
        if x['object_id']==0 and x['episode'].endswith('full_task'):
            lookup[x['parent_id'],x['timestep']].append(x)
    groups=defaultdict(list)
    for row in screen['rows']:
        if row['status']!='PASS':continue
        prior=lookup[row['parent'],row['timestep']]
        prior_failure=any(x['near_finger_pixels']>=5 and
            x['near_finger_confirmed_false_removed']+x['near_finger_uncertain_false_removed']>=.1*x['near_finger_pixels'] for x in prior)
        failure=prior_failure or any((x['correct_removed']>=16 and x['correct_removed']>=.01*x['correct_raw']) or
            (x['robot_raw']>=64 and x['robot_remaining']>=64 and x['robot_remaining']>=.2*x['robot_raw']) for x in row['items'])
        success=any(x['robot_raw']>=128 and x['robot_removed']>=.9*x['robot_raw'] and
            x['correct_removed']<=max(16,.01*x['correct_raw']) for x in row['items'])
        group='failure' if failure else 'success' if success else 'ordinary'
        row.update(stratum=group,prior_exact_audit=prior,prior_recorded_failure=prior_failure)
        groups[group].append(row)
    selected=[];limits=config['stratification'];cap=limits['per_stratum_cap'];per=limits['per_parent_cap']
    for group in ('failure','success','ordinary'):
        candidates=sorted(groups[group],key=lambda r:(not r['prior_recorded_failure'],
            hashlib.sha256(('task10i-filter-ablation:'+r['scene']).encode()).hexdigest()))
        by=defaultdict(list)
        for r in candidates:by[r['parent']].append(r)
        for index in range(per):
            for parent in list(by):
                if index<len(by[parent]) and len([r for r in selected if r['stratum']==group])<cap:
                    selected.append(by[parent][index])
    cases=dict(description='Frozen 512 screen strata; 768/1024 actual masks; selected before variants',
        scenarios=[dict(scene=x['scene'],parent=x['parent'],timestep=x['timestep'],cohort=x['cohort'],
            stratum=x['stratum'],screen=x['items'],prior_exact_audit=x['prior_exact_audit'],prior_recorded_failure=x['prior_recorded_failure']) for x in selected],
        scenes=[x['scene'] for x in selected],sizes=config['evaluation_sizes'],modes=config['modes'],mask_arm='actual',mask_repeat=0,
        warmups_per_mode=config['warmups_per_mode'],timed_repeats_per_mode=config['timed_repeats_per_mode'],
        torch_threads=config['torch_threads'],truth_barrier=config['truth_barrier'],
        screen_sha256=hashlib.sha256(Path(args.screen).read_bytes()).hexdigest(),
        historical_sha256=hashlib.sha256(Path(args.historical).read_bytes()).hexdigest(),
        stratum_available={k:len(groups[k]) for k in ('failure','success','ordinary')})
    Path(args.output).write_text(json.dumps(cases,indent=2)+'\n')
    print('available',cases['stratum_available'])
    for group in ('failure','success','ordinary'):
        subset=[x for x in selected if x['stratum']==group]
        print(group,'selected',len(subset),'parents',len({x['parent'] for x in subset}),[x['scene'] for x in subset])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--screen',required=True);p.add_argument('--historical',required=True)
    p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);main(p.parse_args())
