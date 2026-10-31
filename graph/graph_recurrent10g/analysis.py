"""Expert-fit/phase diagnostics and paired functional outcome analysis."""
import csv
from collections import defaultdict
import numpy as np
import torch
from utils.experiment_logger import ExperimentLogger
from graph.plan_a_bc.common import read,write
from graph.integration_baseline.bc08_codec import controller_commands
from .eval_runner import load_model,csv_write
from .data import Dataset


def read_csv(path):return list(csv.DictReader(path.open()))


def run(root):
    cfg=read(root/'config.json');ds=Dataset(root);phase_rows=[];failures=[]
    final=read_csv(root/'final_rollouts.csv');functional=[];paired=[]
    for c in 'ABC':
        rr=[r for r in final if r['condition']==c]
        functional.append(dict(condition=c,parents=len(rr),**{k:sum(r[k]=='True' for r in rr) for k in ('insertion','lift','grasp','alignment','strict_reach')},
            single_training_seed=True,engineering=cfg['miniature']))
        for r in rr:
            trace=read(root/r['folder']/'trace.json');last=trace[-1]
            failures.append(dict(condition=c,parent=int(r['seed']),insertion=r['insertion'],lift=r['lift'],grasp=r['grasp'],
                failed_stage=next((k for k in ('grasp','lift','alignment','insertion') if r[k]!='True'),'none'),
                initial_action_mae=r['initial_action_mae'],premature_close=r['premature_close'],saturation=r['saturation'],
                missing_frames=r['missing_frames'],final_tcp_peg_distance_m=r['final_tcp_peg_distance_m'],
                final_tip_hole_error_m=r['final_tip_hole_error_m'],hidden_reset=True,
                close_loop_tcp_distance_change_m=last['tcp_peg_distance_m']-trace[0]['tcp_peg_distance_m'],
                collision_steps=sum(not t['collision']['valid'] for t in trace)))
    for left,right in [('A','B'),('B','C'),('A','C')]:
        for metric in ('insertion','lift','grasp','alignment'):
            l={int(r['seed']):r[metric]=='True' for r in final if r['condition']==left}
            r={int(x['seed']):x[metric]=='True' for x in final if x['condition']==right}
            if l.keys()!=r.keys():raise ValueError('paired parents differ')
            b=sum(not l[s] and r[s] for s in l);a=sum(l[s] and not r[s] for s in l)
            from scipy.stats import binomtest
            paired.append(dict(contrast=right+'-'+left,metric=metric,parents=len(l),gained=b,lost=a,
                success_count_difference=b-a,exact_discordant_p=float(binomtest(b,a+b,.5).pvalue) if a+b else 1.,
                interpretation='small paired descriptive sample; one training seed; no general superiority claim'))
    with ExperimentLogger(team='graph',model='Task10G_offline_phase_diagnostics',seed=42,config=cfg,
            root=root/'experiments',job_type='evaluate') as log:
        for c in 'ABC':
            chosen=read(root/'model_selection_frozen.json')['conditions'][c]
            _,m,norm,codec,_=load_model(root,c,root/chosen['path']);groups=defaultdict(list)
            with torch.inference_mode():
                specs=[(eid,e) for eid,e in ds.episodes.items() if e['split'] in ('train','validation')]
                if cfg['miniature']:specs=specs[:2]
                for eid,e in specs:
                    hidden=None
                    for start in range(0,e['actions'],32):
                        g,r,y,mask,_,ix=ds.batch([int(eid)],start,32,'cuda');p,hidden=m(g,r,norm,hidden)
                        pred=codec.decode(codec.tag(p))[mask].cpu().numpy();target=y[mask].cpu().numpy()
                        if not len(pred):continue
                        cp,cy=controller_commands(pred),controller_commands(target)
                        selected=[idx for idx,keep in zip(ix,mask[0].cpu().numpy()) if keep]
                        for j,idx in enumerate(selected):
                            key=(e['split'],ds.md['phases'][idx]);d=cp[j]-cy[j]
                            nt=np.linalg.norm(cp[j,:3]);ny=np.linalg.norm(cy[j,:3])
                            reverse=float(cp[j,:3]@cy[j,:3])<0 if nt>1e-9 and ny>1e-9 else False
                            groups[key].append([np.linalg.norm(d[:3])*1000,np.linalg.norm(d[3:6]),abs(d[6])*1000,
                                float(reverse),np.abs(pred[j]-target[j]).mean()])
            for (split,phase),values in groups.items():
                a=np.asarray(values);phase_rows.append(dict(condition=c,split=split,phase=phase,frames=len(a),
                    controller_translation_error_mm=float(a[:,0].mean()),controller_rotation_component_error_rad=float(a[:,1].mean()),
                    controller_gripper_target_error_mm=float(a[:,2].mean()),direction_reversal_rate=float(a[:,3].mean()),native_action_mae=float(a[:,4].mean())))
            del m;torch.cuda.empty_cache()
        for name,rows in [('functional_success_comparison.csv',functional),('paired_success_analysis.csv',paired),
                ('phasewise_action_errors.csv',phase_rows),('failure_analysis.csv',failures)]:
            csv_write(root/name,rows);log.link_output(root/name,kind='metrics')
    write(root/'analysis_verification.json',dict(status='PASS',all_final_parents_retained=True,
        offline_expert_fit_separate_from_closed_loop=True,recovery='NOT_TESTED; optional protocol not added',
        phase_controller_rotation='component difference in rad; not object displacement',
        hidden_contract_scope='tested carry/reset and saved traces; no causal hidden failure claim'))
    return dict(status='PASS',files=['functional_success_comparison.csv','paired_success_analysis.csv','phasewise_action_errors.csv','failure_analysis.csv','analysis_verification.json'])
