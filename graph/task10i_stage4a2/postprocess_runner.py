"""Autonomous exact-reference/Graph completion barrier and read-only summary."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from graph.task10i_stage4a.run_pipeline import read,write,sha


def main(root,paired,expected):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    identity={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')}
    write(root/'source_identity.json',identity)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_new_development_summary',seed=42,
         root=root/'experiments',job_type='evaluation',config=dict(paired_root=str(paired),expected=str(expected))) as logger:
        deadline=time.monotonic()+24*3600
        while True:
            state=read(paired/'pipeline_state.json')
            if state['status']=='PAIRED_NEW_DEVELOPMENT_GRAPH_COMPLETE':break
            if state['status']=='FAILED':raise RuntimeError('paired evaluator failed')
            pid=state.get('pid')
            command=Path(f'/proc/{pid}/cmdline')
            if pid is None or not command.exists() or not any(module in command.read_bytes() for module in
                (b'graph.task10i_stage4a2.paired_evaluation',b'graph.task10i_stage4a2.resume_evaluation')):
                raise RuntimeError('paired evaluator handle missing without terminal completion receipt')
            write(root/'pipeline_state.json',dict(status='WAITING_FOR_LIVE_PAIRED_EVALUATOR',
                dependency_pid=pid,completed_episodes=state.get('completed_episodes'),pid=os.getpid(),run_id=logger.run_id))
            if time.monotonic()>deadline:raise TimeoutError('paired completion deadline')
            time.sleep(10)
        command=[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.aggregate_evaluation',
                 str(root/'summary'),str(paired),str(expected)]
        with (root/'aggregation.log').open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            write(root/'pipeline_state.json',dict(status='FAILED',stage='AGGREGATION',returncode=result.returncode))
            raise RuntimeError('aggregation failed; inspect preserved log')
        scope=read(root/'summary/scope.json')
        assert scope['status']=='NEW_DEVELOPMENT_PAIRED_SUMMARY_COMPLETE'
        assert scope['episodes']==read(expected)['episodes']
        logger.link_output(root/'summary/scope.json',kind='metrics')
        write(root/'pipeline_state.json',dict(status='NEW_DEVELOPMENT_SUMMARY_COMPLETE',run_id=logger.run_id,
            internal_opened=False,final_opened=False,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('paired',type=Path)
    p.add_argument('expected',type=Path);a=p.parse_args();main(a.root,a.paired,a.expected)
