"""Autonomous CPU pipeline. Resume revalidates frozen inputs and artifacts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

from utils.experiment_logger import ExperimentLogger
from graph.initial_pool10i.mirror.verify import read,sha,inventory,write
from .core import PROTOCOL,SEEDS,canonical_hash,partition,fit,greedy,random_order,freeze,require_frozen
from .load_and_verify import verify_inputs,load_existing,load_candidates,preserve,POOL,HANDOFF,AUDIT,F1,G
from .evaluate import csv_write,coverage,monte_carlo,distribution,duplicates

STAGES=['INPUT_AND_SOURCE_AUDIT','EXISTING_PARENT_DESCRIPTOR_VERIFICATION','CANDIDATE_POOL_SPLIT_FREEZE',
 'FEATURE_DISTANCE_PROTOCOL_FREEZE','COMMON_RANDOM44_SELECTION','R_ADDITIONAL_RANDOM19','D_COVERAGE_GREEDY19',
 'R_D_SELECTION_MANIFEST_FREEZE','SELECTION_POOL_COVERAGE_ANALYSIS','DIAGNOSTIC_HOLDOUT_COVERAGE_ANALYSIS',
 'DISTRIBUTION_SHIFT_AND_OVERLAP_AUDIT','RESERVE_ORDER_FREEZE','STAGE3_COLLECTION_PLAN_EXPORT',
 'FINAL_VERIFICATION','REPORT_GENERATION','COMPLETED']


def link_result(logger,path,kind):
    """A resumed logger must preserve the immutable original run's sidecar."""
    path=Path(path)
    sidecar=path.with_name(path.name+'.tracking.json')
    logger.link_output(path,kind=kind,checkpoint=sidecar.exists())


def execute(out,logger):
    out=Path(out)
    if (out/'run_identity.json').exists():
        primary_run_id=read(out/'run_identity.json')['run_id']
        logger.record(source_run_id=primary_run_id,resume=True)
    else:
        primary_run_id=logger.run_id
        freeze(out/'run_identity.json',dict(run_id=primary_run_id))
    source=Path(__file__).resolve().parents[2]
    source_files={str(p.relative_to(source)):dict(bytes=p.stat().st_size,sha256=sha(p))
                  for base in [source/'graph/task10i_stage2',source/'graph/initial_pool10i/mirror',source/'utils']
                  for p in base.rglob('*') if p.is_file() and p.suffix in ('.py','.md')}
    freeze(out/'execution_source_identity.json',dict(files=source_files,python=platform.python_version(),numpy=np.__version__))
    done=[]
    def stage(name):
        if done:freeze(out/'stages'/f'{len(done):02d}_{done[-1]}.json',dict(status='PASS',stage=done[-1]))
        write(out/'progress.json',dict(status='RUNNING',stage=name,completed=done.copy(),run_id=logger.run_id))
        done.append(name);print(name,flush=True)
    stage(STAGES[0]);receipt=verify_inputs(out)
    freeze(out/'stage2_input_identity.json',dict(pool=str(POOL),handoff=str(HANDOFF),audit=str(AUDIT),
        f1=str(F1),task10g=str(G),source_sha256=canonical_hash(source_files),instance_id=57987772,
        manifest_sha256=sha(POOL/'candidate_pool_manifest.json'),raw_state_only=True))
    stage(STAGES[1]);existing,es=load_existing(out);candidates,cs=load_candidates(out);states={**es,**cs}
    train0=[r for r in existing if r['role']=='existing_train'];val0=[r for r in existing if r['role']=='existing_validation']
    stateids=sorted(r['id'] for r in existing+candidates)
    arrays={k:np.stack([np.asarray(states[i][k]).reshape(-1) if k=='hole_radius' else states[i][k] for i in stateids])
            for k in ['peg_pose','tcp_pose','hole_pose','peg_half','hole_radius','qpos','qvel','intrinsic','extrinsic_cv']}
    statepath=out/'analysis_states.npz'
    if statepath.exists():
        with np.load(statepath,allow_pickle=False) as z:
            np.testing.assert_array_equal(z['parent_id'],stateids)
            for k,a in arrays.items():np.testing.assert_array_equal(z[k],a)
    else:np.savez_compressed(statepath,parent_id=np.array(stateids),**arrays)
    stage(STAGES[2]);split=partition([r['id'] for r in candidates]);freeze(out/'candidate_pool_split_manifest.json',split)
    byid={r['id']:r for r in existing+candidates}
    for role,ids in [('validation',split['validation']),('holdout',split['holdout']),('training',split['training'])]:
        for cid in ids:byid[cid]['role']=role
    take=lambda ids:[byid[i] for i in ids]
    train=take(sorted(split['training']));fitrows=train0+train;fitids=[r['id'] for r in fitrows]
    stage(STAGES[3]);freeze(out/'feature_schema.json',PROTOCOL)
    freeze(out/'distance_protocol_frozen.json',PROTOCOL)
    freeze(out/'selection_protocol_frozen.json',dict(seeds=SEEDS,stages=STAGES,train_counts=[33,44,19],
        validation_counts=[7,9],split_counts=[20,60,220],freeze_before_holdout=True,
        common_random='uniform without replacement; independent PCG64 permutations',
        random_R='uniform 19 from training minus common44; overlap with D allowed',
        greedy='training220 demand only; continuation beyond 19 supplies D reserve',
        distribution_diagnostics_fixed_before_selection=True))
    freeze(out/'random_seed_manifest.json',SEEDS)
    scale=fit(fitrows,fitids);freeze(out/'feature_normalization.json',scale)
    stage(STAGES[4]);common_order=random_order(split['training'],SEEDS['common']);common=common_order[:44]
    base=train0+take(common)
    stage(STAGES[5]);r_order=random_order(set(split['training'])-set(common),SEEDS['random']);rtail=r_order[:19]
    stage(STAGES[6]);trace=greedy(train,base,scale)
    assert trace==greedy(train,base,scale),'independent greedy repetition differs'
    dtail=[r['id'] for r in trace[:19]];R=base+take(rtail);D=base+take(dtail)
    val=val0+take(split['validation'][:9]);hold=take(split['holdout'])
    stage(STAGES[7]);files=[]
    def selected_csv(name,rows):
        csv_write(out/name,[dict(rank=i+1,parent_id=r['id'],seed=r['seed'],origin=r['kind'],role=r['role'],expert_status=r['expert_status'])
                            for i,r in enumerate(rows)]);files.append(name)
    selected_csv('common_random44.csv',take(common));selected_csv('random_R_selected.csv',R)
    selected_csv('diversity_D_selected.csv',D);selected_csv('shared_validation_selected.csv',val)
    freeze(out/'sampling_trace.json',dict(common=common_order,random=r_order));files.append('sampling_trace.json')
    freeze(out/'coverage_selection_order.json',dict(trace=trace));files.append('coverage_selection_order.json')
    for name,rows in [('dataset_R_parent_manifest.json',R),('dataset_D_parent_manifest.json',D),('shared_validation_parent_manifest.json',val)]:
        freeze(out/name,dict(status='FROZEN_SELECTION_PLAN',parents=rows,count=len(rows),
            expert_status='NOT_TESTED_FOR_NEW_PARENTS',trajectories_generated=False,
            feature_protocol_sha256=canonical_hash(PROTOCOL),normalization_sha256=sha(out/'feature_normalization.json')))
        files.append(name)
    files+=['candidate_pool_split_manifest.json','feature_schema.json','feature_normalization.json',
            'distance_protocol_frozen.json','selection_protocol_frozen.json','random_seed_manifest.json']
    freeze(out/'selection_identity.json',dict(status='FROZEN',files={n:sha(out/n) for n in files},
        selection_order_sha256=canonical_hash(dtail),expert_status='NOT_TESTED'))
    require_frozen(out)
    sets=dict(existing33=train0,base77=base,R=R,D=D)
    stage(STAGES[8]);selection,groups1,near1=coverage(out,train,sets,scale,'selection')
    csv_write(out/'random_vs_diversity_coverage.csv',selection)
    stage(STAGES[9]);held,groups2,near2=coverage(out,hold,sets,scale,'holdout')
    csv_write(out/'coverage_holdout_metrics.csv',held)
    csv_write(out/'feature_group_coverage.csv',groups1+groups2);csv_write(out/'nearest_set_distances.csv',near1+near2)
    mc=monte_carlo(out,train,hold,base,sets,scale)
    stage(STAGES[10]);sr={r['id'] for r in R};sd={r['id'] for r in D};sv={r['id'] for r in val};sh=set(split['holdout'])
    assert len(sr)==len(sd)==96 and len(sv)==16
    assert len(sr&sd)==77+len(set(rtail)&set(dtail))
    assert not (sr|sd)&sv and not (sr|sd|sv)&sh
    assert len(set(split['validation'])|set(split['training'])|sh)==300
    assert len({r['seed'] for r in candidates+existing})==340
    overlap=dict(status='PASS',R_train=96,D_train=96,validation=16,shared=len(sr&sd),
        R_exclusive=len(sr-sd),D_exclusive=len(sd-sr),tail_overlap=sorted(set(rtail)&set(dtail)),
        train_validation_overlap=0,train_holdout_overlap=0,validation_holdout_overlap=0,
        existing_train_preserved=33,existing_validation_preserved=7,common44_identity=True,
        validation_cohort_excluded_from_training=True,no_final_test_used=True,seed_and_id_collisions=0,
        new_collection_union=len((sr|sd|sv)-{r['id'] for r in existing}))
    freeze(out/'selection_overlap_audit.json',overlap)
    dup=duplicates(out,[('new_vs_existing_train',candidates,train0,False),
        ('new_vs_existing_validation',candidates,val0,True),('R_vs_D',R,D,False),
        ('R_train_vs_validation',R,val,True),('D_train_vs_validation',D,val,True),
        ('R_train_vs_holdout',R,hold,True),('D_train_vs_holdout',D,hold,True)],states,scale)
    populations=dict(existing33=train0,existing_validation7=val0,candidate300=candidates,training220=train,
        common44=take(common),R_tail19=take(rtail),D_tail19=take(dtail),R=R,D=D,validation16=val)
    boundary,extreme=distribution(out,populations,fitrows)
    stage(STAGES[11]);commonreserve=random_order(set(split['training'])-set(common),SEEDS['common_reserve'])
    rreserve=random_order(set(split['training'])-set(common)-set(rtail),SEEDS['random_reserve'])
    dreserve=[r['id'] for r in trace[19:]]
    for name,ids in [('validation_reserve.csv',split['validation'][9:]),('common_reserve.csv',commonreserve),
                     ('random_reserve.csv',rreserve),('coverage_reserve.csv',dreserve)]:selected_csv(name,take(ids))
    reserve_protocol=dict(version=1,expert_executed=False,replacement_executed=False,
        queues=dict(validation=split['validation'][9:],common=commonreserve,R=rreserve,D=dreserve),
        rules=[
          'Freeze Stage2 initial manifests permanently; append Stage3 attempts and replacement events to a new version.',
          'Attempt each physical parent once, reuse its successful demonstration wherever shared; preserve all failures.',
          'A failed parent is globally unavailable. Skip failed IDs in every queue with an explicit skip event.',
          'Resolve failures in order: validation, common, R tail, D tail; within role initial slot rank then parent ID.',
          'Validation uses only its 11 reserved IDs; no cross-cohort borrowing. Exhaustion blocks collection.',
          'Common replacement must be absent from current union of R and D, so both retain exactly 44 new common slots.',
          'R replacement skips IDs already in current R. IDs in D alone may be shared and counted once.',
          'D replacement skips IDs already in current D. IDs in R alone may be shared and counted once.',
          'For each skip/assignment log role, failed parent, reserve rank, parent ID, reason, attempt evidence, version hash.',
          'Do not rerun greedy after attrition. Exhaustion blocks; no arbitrary seed substitutes.',
          'After replacements re-audit counts/splits/physical duplicates and actual coverage using frozen distance; report attrition.'
        ],historical_parent_failure='If an existing approved parent becomes invalid, block and report; do not silently replace fixed33/7.')
    freeze(out/'stage3_collection_reserve_protocol.json',reserve_protocol)
    stage(STAGES[12]);union=sorted((sr|sd|sv)-{r['id'] for r in existing})
    freeze(out/'stage3_parent_collection_union.json',dict(count=len(union),parents=[dict(**byid[i],
        memberships=[name for name,s in [('R',sr),('D',sd),('validation',sv)] if i in s]) for i in union],
        expert_status='NOT_TESTED',actions_available=False))
    episodes=read(out/'existing_episode_preservation_manifest.json')['episodes']
    from collections import Counter
    structure={}
    for s in ['train','validation']:
        subset=[e for e in episodes if e['split']==s]
        kept=[e for e in subset if e['visibility_pass']]
        structure[s]=dict(original_logical=len(subset),preserved_logical=len(kept),
            original_sources=dict(Counter(e['source'] for e in subset)),preserved_sources=dict(Counter(e['source'] for e in kept)),
            original_branches=dict(Counter(e['branch'] for e in subset)),preserved_branches=dict(Counter(e['branch'] for e in kept)),
            target_logical=272 if s=='train' else 48,new_parent_count=63 if s=='train' else 9,
            logical_deficit=len(subset)-len(kept))
    templates={}
    for e in episodes:
        if not e['visibility_pass']:templates.setdefault(e['parent_id'],[]).append(e)
    oldtrain=sorted([p for p,eps in templates.items() if eps[0]['split']=='train'])
    oldval=sorted([p for p,eps in templates.items() if eps[0]['split']=='validation'])
    assert len(oldtrain)==63 and len(oldval)==9
    allocations=[]
    for dataset,newids,oldids in [('R',common+rtail,oldtrain),('D',common+dtail,oldtrain),
                                  ('validation',split['validation'][:9],oldval)]:
        for cid,oldid in zip(newids,oldids):
            eps=templates[oldid]
            allocations.append(dict(dataset=dataset,new_parent=cid,replaced_visibility_failed_parent=oldid,
                planned_logical_episodes=len(eps),full_task_count=sum(e['source']=='full_task' for e in eps),
                corrective_source_count=sum(e['source']!='full_task' for e in eps),
                branch_templates=[dict(original_episode_id=e['episode_id'],branch=e['branch'],source=e['source'],
                    original_collection_spec=e['collection_spec']) for e in eps],expert_status='NOT_TESTED'))
    assert sum(a['planned_logical_episodes'] for a in allocations if a['dataset']=='R')==190
    assert sum(a['planned_logical_episodes'] for a in allocations if a['dataset']=='D')==190
    assert sum(a['planned_logical_episodes'] for a in allocations if a['dataset']=='validation')==25
    freeze(out/'stage3_demonstration_plan.json',dict(structure=structure,
        full_task='attempt one full-task successful demonstration for each new parent; no new actions exist yet',
        corrective='deterministic sorted failed-parent slot templates; Common44 shares first44 templates; each tail19 inherits remaining19',
        per_parent_allocations=allocations,
        shared_tail_rule='If same physical tail parent inherits different templates in R and D, collect union of required branches once and preserve per-dataset inclusion; no implicit extra logical episodes.',
        branch_mapping=[dict(parent_id=e['parent_id'],episode_id=e['episode_id'],branch=e['branch'],source=e['source'],split=e['split'])
                        for e in episodes if not e['visibility_pass']],
        supervision='Collector: recovery true, prefix/disturbance false. Task10G BC: prefix may be enabled only after exact approved nominal actions/robot/timestep comparison; disturbance always false; recovery true; segment order monotonic and T+1 observations/T actions.',
        masks_and_collection_protocol_sources=['graph/corrective_reach10f/recovery_collect.py','graph/graph_recurrent10g/data.py'],
        final_evaluation='use separately frozen seeds outside all 300 candidates and historical final/test parents',
        new_expert_success='NOT_TESTED',corrective_generation='NOT_EXECUTED'))
    summary=dict(overlap=overlap,selection=selection,holdout=held,monte_carlo=mc,boundary=boundary,extreme=extreme,
                 duplicates=dup,policy_performance='NOT_TESTED',expert_success='NOT_TESTED')
    freeze(out/'selection_comparison_summary.json',summary)
    stage(STAGES[13]);preserve(out);require_frozen(out)
    # Report uses only post-freeze measurements. The report is included in final artifact verification below.
    stage(STAGES[14]);from .report import generate
    generate(out,byid,states,sets,split,summary,primary_run_id)
    for name in ['selection_comparison_summary.json','10i_stage2_diversity_aware_dataset_selection.md',
                 '10i_stage2_diversity_aware_dataset_selection.html','gallery.html']:
        link_result(logger,out/name,'reports' if name.endswith(('.md','.html')) else 'metrics')
    logger.record(task='Task10-I Stage2',dataset_hash=sha(out/'selection_identity.json'),
        source_identity_sha256=sha(out/'execution_source_identity.json'),expert_executed=False,simulator_executed=False)
    logger.log(dict(parents_R=96,parents_D=96,shared_validation=16,new_collection_union=len(union)))
    freeze(out/'stages'/'15_REPORT_GENERATION.json',dict(status='PASS',stage='REPORT_GENERATION'))
    immutable={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*'))
               if p.is_file() and p.name not in ['progress.json','run.lock','artifact_index.json','stage3_input_receipt.json',
               'final_verification.json','pipeline.log'] and 'experiments' not in p.relative_to(out).parts}
    freeze(out/'artifact_index.json',dict(files=immutable,scope='scientific artifacts, evidence, tests, figures and report; operational files excluded'))
    for rel,info in immutable.items():
        p=out/rel;assert sha(p)==info['sha256'] and p.stat().st_size==info['bytes']
        if p.suffix=='.json':read(p)
        elif p.suffix=='.csv':
            import csv
            with p.open() as f:assert list(csv.DictReader(f))
        elif p.suffix=='.png':
            from PIL import Image
            with Image.open(p) as im:im.verify()
        elif p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:
                for key in z.files:
                    if z[key].dtype.kind in 'fiu':assert np.isfinite(z[key]).all()
    handoff_names=['dataset_R_parent_manifest.json','dataset_D_parent_manifest.json','shared_validation_parent_manifest.json',
        'stage3_parent_collection_union.json','stage3_collection_reserve_protocol.json','existing_episode_preservation_manifest.json',
        'selection_identity.json','artifact_index.json','stage3_demonstration_plan.json']
    freeze(out/'stage3_input_receipt.json',dict(status='READY_FOR_STAGE3_COLLECTION_PLAN_ONLY',stage2_complete=True,
        run_id=primary_run_id,files={n:sha(out/n) for n in handoff_names},R_train=96,D_train=96,validation=16,
        new_parent_union=len(union),expert_status='NOT_TESTED',auto_start_stage3=False))
    freeze(out/'final_verification.json',dict(status='PASS',artifact_count=len(immutable),
        artifact_index_sha256=sha(out/'artifact_index.json'),stage3_receipt_sha256=sha(out/'stage3_input_receipt.json'),
        source_preservation='PASS',selection_readback='PASS',independent_reselection='EXACT_MATCH',all_required_counts='PASS'))
    done.append('COMPLETED')
    write(out/'progress.json',dict(status='COMPLETED',stages=done,run_id=logger.run_id))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'progress.json').exists() and read(out/'progress.json')['status']=='COMPLETED':
        require_frozen(out)
        for rel,info in read(out/'artifact_index.json')['files'].items(): assert sha(out/rel)==info['sha256']
        print('Already completed; immutable readback PASS');return
    os.environ['WANDB_MODE']='disabled'
    with ExperimentLogger(team='graph',config=PROTOCOL,seed=SEEDS['split'],model='initial_state_selection',
                          job_type='analysis',root=out/'experiments') as logger:
        try:execute(out,logger)
        except BaseException as exc:
            write(out/'failure.json',dict(status='BLOCKED',error=repr(exc),traceback=traceback.format_exc()))
            p=read(out/'progress.json') if (out/'progress.json').exists() else {}
            write(out/'progress.json',dict(**{k:v for k,v in p.items() if k!='status'},status='BLOCKED'))
            raise


if __name__=='__main__':main()
