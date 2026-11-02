"""Frozen challenger: Internal once, then untouched synchronous Normal/Stress.

This entrypoint never tunes or selects a challenger. It requires a concrete
registered source/config freeze and cannot publish, recache training data or
start policy training. Statistical failures remain results; contract failures
stop the pipeline with partial artifacts intact.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .final_capture import freeze_barrier
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,freeze,partition,protocol,design,collector_source):
    from utils.experiment_logger import ExperimentLogger
    barriers=freeze_barrier(freeze,None,None,scope='internal')
    config=read(freeze/'candidate_config_frozen.json')
    assert config['continue_to_final_without_retuning_after_internal'] is True
    assert config['allow_independent_final_capture'] is True
    assert config['new_final_benchmark_protocol_sha256']==sha(protocol)
    assert config['final_collection_design_sha256']==sha(design)
    assert config['collector_source_root']==str(collector_source)
    assert sha(collector_source/'graph/task10i_stage4a2/final_capture.py')==read(design)['collector_source_sha256']
    forward_source=Path(config['forward_source_root'])
    candidate=config['candidate'];assert candidate in ('P','C_KEEP','BC_KEEP','BC_MAIN_BOX','BC_WB_HANDOFF')
    source=Path(__file__).resolve().parents[2]
    criteria=freeze/'adoption_criteria_frozen.json'
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(freezes=barriers,partition_sha256=sha(partition),protocol_sha256=sha(protocol),design_sha256=sha(design),
            coordinator_source_sha256=sha(Path(__file__)),candidate=candidate,forward_source_root=str(forward_source),
            collector_source_root=str(collector_source),candidate_config_sha256=sha(freeze/'candidate_config_frozen.json'))
        frozen=root/'source_config_frozen.json'
        if frozen.exists():assert read(frozen)==identity
        else:write(frozen,identity)
        signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        python=[sys.executable,'-B','-u','-m'];jobs=[]
        def add(name,module,args,output,cwd=source,retry_oom=False):
            jobs.append((name,python+[module,*map(str,args)],output,cwd,retry_oom))
        def live(name,expected,condition,output):
            options=['--condition',condition]
            if condition!='B0':
                if config['geometry_cache']:options+=['--geometry-cache']
                if config['lazy_matching']:options+=['--lazy-matching']
            add(name,'graph.task10i_stage4a2.frozen_inference',[output,expected,*options],output,forward_source,True)
        internal=root/'internal';ie=internal/'expected_counts.json'
        add('internal_manifest','graph.task10i_stage4a2.evaluation_inputs',['manifest',ie,partition,'--mode','internal','--freeze',freeze],ie)
        live('internal_B0',ie,'B0',internal/'baseline');live('internal_challenger',ie,candidate,internal/'candidate')
        add('internal_reference','graph.task10i_stage4a2.development_reference',[internal/'references',partition,'--partition-key','internal_holdout','--freeze',freeze],internal/'references')
        add('internal_masks','graph.task10i_stage4a2.evaluation_inputs',['combine',internal/'masks',ie,'--baseline',internal/'baseline',
            '--challenger',internal/'candidate','--candidate',candidate,'--require-baseline-isolation'],internal/'masks')
        add('internal_graph','graph.task10i_stage4a2.paired_evaluation',[internal/'paired',internal/'masks',internal/'references',ie,
            '--candidate',candidate,'--resume-existing'],internal/'paired')
        add('internal_consistency','graph.task10i_stage4a2.live_graph_consistency',[internal/'live_graph_consistency.json',ie,
            internal/'paired',internal/'masks','--candidate',candidate],internal/'live_graph_consistency.json')
        add('internal_summary','graph.task10i_stage4a2.aggregate_evaluation',[internal/'summary',internal/'paired',ie,
            '--candidate',candidate,'--scope','internal'],internal/'summary')
        add('internal_registered','graph.task10i_stage4a2.registered_evaluation',[internal/'registered',internal/'paired',ie,criteria,
            '--candidate',candidate,'--scope','internal'],internal/'registered')
        capture=root/'new_benchmark'
        add('new_capture','graph.task10i_stage4a2.final_capture',[capture,protocol,design,freeze],capture,collector_source)
        final=root/'final';fe=final/'expected_counts.json'
        add('final_manifest','graph.task10i_stage4a2.evaluation_inputs',['manifest',fe,capture/'independent_benchmark_manifest.json','--mode','all'],fe)
        for mode in ('normal','stress'):
            add('final_manifest_'+mode,'graph.task10i_stage4a2.evaluation_inputs',['manifest',final/(mode+'_expected_counts.json'),
                capture/'independent_benchmark_manifest.json','--mode',mode],final/(mode+'_expected_counts.json'))
        live('final_B0',fe,'B0',final/'baseline');live('final_challenger',fe,candidate,final/'candidate')
        add('final_runtime','graph.task10i_stage4a2.latency_report',[final/'runtime',fe,final/'baseline',final/'candidate',criteria,
            '--candidate',candidate],final/'runtime')
        add('final_reference','graph.task10i_stage4a2.evaluation_inputs',['references',final/'references',fe,
            '--baseline',final/'baseline','--challenger',final/'candidate'],final/'references')
        add('final_matching_audit','graph.task10i_stage4a2.matching_outcomes',[final/'matching',fe,final/'baseline',final/'candidate',
            final/'references','--candidate',candidate],final/'matching')
        add('final_masks','graph.task10i_stage4a2.evaluation_inputs',['combine',final/'masks',fe,
            '--baseline',final/'baseline','--challenger',final/'candidate','--candidate',candidate,'--require-baseline-isolation'],final/'masks')
        add('final_temporal_masks','graph.task10i_stage4a2.temporal_mask_diagnostics',[final/'temporal',final/'masks',final/'references',
            fe,criteria,'--candidate',candidate],final/'temporal')
        add('final_graph','graph.task10i_stage4a2.paired_evaluation',[final/'paired',final/'masks',final/'references',fe,
            '--candidate',candidate,'--resume-existing'],final/'paired')
        add('final_consistency','graph.task10i_stage4a2.live_graph_consistency',[final/'live_graph_consistency.json',fe,
            final/'paired',final/'masks','--candidate',candidate],final/'live_graph_consistency.json')
        for mode in ('normal','stress'):
            expected=final/(mode+'_expected_counts.json')
            add('final_summary_'+mode,'graph.task10i_stage4a2.aggregate_evaluation',[final/('summary_'+mode),final/'paired',expected,
                '--candidate',candidate,'--subset','--scope','final_'+mode],final/('summary_'+mode))
            add('final_registered_'+mode,'graph.task10i_stage4a2.registered_evaluation',[final/('registered_'+mode),final/'paired',expected,
                criteria,'--candidate',candidate,'--scope','final_'+mode],final/('registered_'+mode))
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_frozen_independent_evaluation_pipeline',seed=42,
                root=root/'coordinator_experiments',job_type='evaluation',config=identity) as logger:
            for name,command,output,cwd,retry_oom in jobs:
                stage=root/'stages'/(name+'.json')
                if stage.exists():
                    receipt=read(stage);assert receipt['signature']==signature and receipt['command']==command
                    assert all(sha(root/p)==h for p,h in receipt['files'].items());continue
                assert freeze_barrier(freeze,None,None,scope='internal')==barriers
                # CPU adapters/aggregators create a new directory. If a crash
                # preceded the coordinator commit, retain that uncommitted
                # output and regenerate from verified immutable upstream data.
                # Temporal inference, simulator collection and paired Graphs
                # instead resume through their own hash/prefix contracts.
                regenerate_modules={'graph.task10i_stage4a2.evaluation_inputs',
                    'graph.task10i_stage4a2.aggregate_evaluation','graph.task10i_stage4a2.registered_evaluation',
                    'graph.task10i_stage4a2.latency_report','graph.task10i_stage4a2.matching_outcomes',
                    'graph.task10i_stage4a2.temporal_mask_diagnostics'}
                if command[4] in regenerate_modules and output.is_dir():
                    output.rename(output.with_name(output.name+'.uncommitted_'+str(time.time_ns())))
                for attempt in range(2 if retry_oom else 1):
                    log=root/(name+f'_attempt{attempt}.log')
                    write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,attempt=attempt,command=command,
                        cwd=str(cwd),pid=os.getpid(),run_id=logger.run_id,whole_goal_complete=False))
                    with log.open('w') as stream:result=subprocess.run(command,cwd=cwd,stdout=stream,stderr=subprocess.STDOUT)
                    if not result.returncode:break
                    if not retry_oom or attempt or 'out of memory' not in log.read_text().lower():
                        raise RuntimeError('independent stage failed; preserve source/results: '+str(log))
                assert output.exists()
                artifacts=[output] if output.is_file() else [p for p in output.rglob('*') if p.is_file()]
                write(stage,dict(status='PASS',signature=signature,command=command,cwd=str(cwd),
                    files={str(p.relative_to(root)):sha(p) for p in artifacts}))
            write(root/'pipeline_state.json',dict(status='FROZEN_INTERNAL_AND_NEW_FINAL_EVALUATION_ARTIFACTS_COMPLETE',
                run_id=logger.run_id,candidate=candidate,source_or_threshold_retuned=False,
                decision_and_latency_report_pending=True,whole_goal_complete=False,Stage4B_started=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','freeze','partition','protocol','design','collector_source'):p.add_argument(name,type=Path)
    a=p.parse_args()
    try:run(a.root,a.freeze,a.partition,a.protocol,a.design,a.collector_source)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True,
            source_or_threshold_retuned=False,whole_goal_complete=False))
        raise
