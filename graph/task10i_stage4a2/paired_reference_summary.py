"""Common-candidate ideal/practical reference intervention; diagnostic only."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha


def summarize(rows):
    result={}
    for origin in ('ideal','practical'):
        positives=[r for r in rows if r['positive']];negatives=[r for r in rows if not r['positive']]
        result[origin+'_TPR']=sum(r[origin+'_accept'] for r in positives)/len(positives) if positives else None
        result[origin+'_FPR']=sum(r[origin+'_accept'] for r in negatives)/len(negatives) if negatives else None
    result.update(pairs=len(rows),positives=sum(r['positive'] for r in rows),negatives=sum(not r['positive'] for r in rows),
        oracle_recovers_correct_acceptance=sum(r['positive'] and r['ideal_accept'] and not r['practical_accept'] for r in rows),
        oracle_prevents_wrong_acceptance=sum(not r['positive'] and not r['ideal_accept'] and r['practical_accept'] for r in rows),
        oracle_loses_correct_acceptance=sum(r['positive'] and not r['ideal_accept'] and r['practical_accept'] for r in rows),
        oracle_adds_wrong_acceptance=sum(not r['positive'] and r['ideal_accept'] and not r['practical_accept'] for r in rows))
    return result


def run(root,source):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    config=dict(source_sha256=sha(Path(__file__)),pairs_sha256=sha(source/'dinov2_positive_negative_pairs.csv'),
        threshold=.6,threshold_selected=False,common_candidate_pairs=True,GT_oracle_diagnostic=True,
        actual_B0_reference_bank_used=False,independent_test=False)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_paired_reference_intervention',seed=42,
            root=root/'experiments',job_type='analysis',config=config) as logger:
        groups=defaultdict(dict)
        with (source/'dinov2_positive_negative_pairs.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['reference_origin'] not in ('ORACLE_IDEAL_T0','HYPOTHETICAL_B0_T0'):continue
                key=tuple(row[k] for k in ('parent_id','mode','reference_camera','reference_timestep','target','candidate_index','time_relation'))
                origin='ideal' if row['reference_origin']=='ORACLE_IDEAL_T0' else 'practical'
                assert origin not in groups[key]
                groups[key][origin]=row
        paired=[]
        for key,group in groups.items():
            if set(group)!= {'ideal','practical'}:continue
            a,b=group['ideal'],group['practical'];assert a['intended_positive']==b['intended_positive']
            paired.append(dict(parent_id=a['parent_id'],mode=a['mode'],time_relation=a['time_relation'],
                cross_camera=a['cross_camera']=='True',positive=a['intended_positive']=='True',
                contaminated=b['contaminated_reference']=='True',reference_purity=float(b['reference_purity']),
                ideal_similarity=float(a['similarity']),practical_similarity=float(b['similarity']),
                ideal_accept=float(a['similarity'])>=.6,practical_accept=float(b['similarity'])>=.6,
                candidate_index=int(a['candidate_index']),reference_camera=a['reference_camera'],target=a['target']))
        summaries=[];grouped=defaultdict(list);rng=np.random.default_rng(1011202644)
        for row in paired:grouped[row['mode'],row['time_relation'],row['cross_camera'],row['contaminated']].append(row)
        for key,rows in grouped.items():
            result=dict(zip(('mode','time_relation','cross_camera','contaminated'),key),**summarize(rows))
            parents=defaultdict(list)
            for r in rows:parents[r['parent_id']].append(r)
            by_parent=[summarize(v) for v in parents.values()]
            for metric in ('TPR','FPR'):
                values=np.asarray([[r['ideal_'+metric],r['practical_'+metric]] for r in by_parent
                    if r['ideal_'+metric] is not None and r['practical_'+metric] is not None])
                if len(values):
                    delta=values[:,0]-values[:,1]
                    bootstrap=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1)
                    result.update({metric+'_parent_count':len(values),metric+'_ideal_parent_macro':float(values[:,0].mean()),
                        metric+'_practical_parent_macro':float(values[:,1].mean()),metric+'_ideal_minus_practical':float(delta.mean()),
                        metric+'_paired_parent_CI95':np.quantile(bootstrap,[.025,.975]).tolist()})
            summaries.append(result)
        csv_write(root/'paired_reference_intervention.csv',summaries);csv_write(root/'common_candidate_pairs.csv',paired)
        write(root/'scope.json',dict(status='PAIRED_REFERENCE_INTERVENTION_COMPLETE',run_id=logger.run_id,
            pairs=len(paired),unique_parents=len({r['parent_id'] for r in paired}),
            input_config=config,model_forwards=0,no_future_reference=True,actual_runtime_gate_performance=False,
            limitation='diagnostic raw-mask crop vectors; runtime uses geometry-clean crops and additional gates',
            whole_goal_complete=False))
        logger.link_output(root/'scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('source',type=Path)
    a=p.parse_args();run(a.root,a.source)
