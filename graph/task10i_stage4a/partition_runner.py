"""Frozen-candidate paired evaluation on a preregistered, previously unopened partition."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from .run_pipeline import read, write, sha


def run(root,freeze,partition):
    from utils.experiment_logger import ExperimentLogger
    frozen=read(freeze);source=Path(__file__).resolve().parents[2]
    for name,value in frozen.get('cpu_thread_environment',{}).items():assert os.environ.get(name)==value,(name,value)
    for name,digest in frozen['inference_files'].items():assert sha(source/name)==digest,name
    if partition=='independent_validation':assert frozen['stage']=='FINAL_CANDIDATE_FROZEN'
    else:assert frozen['stage']=='DEVELOPMENT_CANDIDATE_FROZEN'
    root.mkdir(parents=True,exist_ok=False)
    proof=Path('/workspace/task10i_stage4a_forward_equivalence_20261101_v1/run')
    protocol=read(Path('/workspace/task10i_stage4a_20261031_v5/run/protocol.json'))
    assert not set(protocol['development'])&set(protocol[partition])
    refroot=root/'reference';refroot.mkdir();write(refroot/'protocol.json',protocol)
    base=root/'baseline';base.mkdir();write(base/'protocol.json',protocol)
    write(base/'pipeline_state.json',dict(status='RUNNING',stage='PREREQUISITE_REFERENCE',partition=partition))
    (base/'references').symlink_to(refroot/'references',target_is_directory=True)
    mode_args=['--mode',frozen['mode']] if frozen['mode'].startswith('depth_component') else []
    if frozen.get('exact_current_frame_memo'):mode_args.append('--robot-memo')
    stages=[
      ('RAW_GT_REFERENCE',['-m','graph.task10i_stage4a.development_reference',str(refroot),str(proof),'--partition',partition]),
      ('GT_DISCONTINUITY_REPAIR',['-m','graph.task10i_stage4a.repair_reference',str(root/'repaired'),str(refroot)]),
      ('CAUSAL_BASELINE',['-m','graph.task10i_stage4a.development_perception',str(base),str(proof),'--partition',partition]),
      ('EXACT_REFERENCE_EVALUATION',['-m','graph.task10i_stage4a.reevaluate_references',str(root/'corrected'),str(base),str(root/'repaired')]),
      ('FROZEN_CANDIDATE',['-m','graph.task10i_stage4a.paired_candidate',str(root/'candidate'),str(base),str(root/'corrected'),'--partition',partition,*mode_args]),
      ('PAIRED_PARENT_MACRO',['-m','graph.task10i_stage4a.paired_summary',str(root/'comparison'),str(root/'corrected'),str(root/'candidate')]),
      ('PAIRED_TEMPORAL',['-m','graph.task10i_stage4a.temporal_pair',str(root/'temporal'),str(base),str(root/'corrected'),str(root/'candidate')]),
    ]
    with ExperimentLogger(team='integration',model='Task10I_Stage4A_frozen_partition_evaluation',seed=42,
         root=root/'experiments',job_type='evaluation',config=dict(partition=partition,freeze=str(freeze),freeze_sha256=sha(freeze),candidate=frozen['candidate'],threshold_retuning=False)) as log:
        write(root/'run_identity.json',dict(run_id=log.run_id,pid=os.getpid(),freeze_sha256=sha(freeze)))
        stage='INITIALIZATION'
        try:
            for stage,args in stages:
                write(root/'pipeline_state.json',dict(status='RUNNING',stage=stage,partition=partition,run_id=log.run_id,pid=os.getpid()))
                with (root/(stage.lower()+'.log')).open('x') as output:
                    subprocess.run([sys.executable,'-B','-u',*args],stdout=output,stderr=subprocess.STDOUT,check=True)
                print(partition,stage,'FINISHED',flush=True)
            rs=read(root/'corrected/reevaluation_summary.json')
            receipt=dict(status='FROZEN_PARTITION_EVALUATION_FINISHED',partition=partition,run_id=log.run_id,
                frames=rs['frames'],exact_frames=rs['exact_frames'],unsupported_frames=rs['frames']-rs['exact_frames'],
                complete_registered_partition=True,no_candidate_changes_after_freeze=True,
                independent_validation_accessed=partition=='independent_validation',freeze_sha256=sha(freeze))
            write(root/'partition_summary.json',receipt)
            for n in ('partition_summary.json','comparison/paired_macro.json','comparison/reference_support.json','temporal/scope.json'):
                log.link_output(root/n,kind='metrics')
            write(root/'pipeline_state.json',dict(status='SUBSTAGE_FINISHED',stage='FROZEN_PARTITION_EVALUATION_FINISHED',run_id=log.run_id))
        except BaseException as exc:
            write(root/'pipeline_state.json',dict(status='FAILED',stage=stage,error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('freeze',type=Path)
    p.add_argument('--partition',choices=('internal_test','independent_validation'),required=True)
    a=p.parse_args();run(a.root,a.freeze,a.partition)
