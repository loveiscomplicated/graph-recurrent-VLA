"""New Development hypothesis: uncertainty requests correction, keeps current mask.

Preserve the failed suppression experiment. Do not use GT to decide which
frames retain a mask. Serialize after queued capture engineering and repeat
live/fault engineering before unmodified historical windows.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,dependency):
    root.mkdir(parents=True,exist_ok=False)
    write(root/'source_identity.json',{str(p):sha(p) for p in Path(__file__).parent.glob('*') if p.is_file()})
    deadline=time.monotonic()+24*3600
    while not (dependency/'scope.json').exists():
        state=dependency/'pipeline_state.json'
        if state.exists() and read(state)['status']=='FAILED':raise RuntimeError('capture engineering failed')
        live=[]
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                command=p.read_bytes()
                if b'graph.task10i_stage4a2.final_capture' in command and str(dependency).encode() in command:live.append(p.parent.name)
            except OSError:pass
        if not live:raise RuntimeError('capture engineering dependency missing')
        if time.monotonic()>deadline:raise TimeoutError('GPU serialization deadline')
        write(root/'pipeline_state.json',dict(status='WAITING_FOR_LIVE_CAPTURE_ENGINEERING',pids=live,pid=os.getpid()))
        time.sleep(10)
    assert read(dependency/'scope.json')['status']=='BENCHMARK_COLLECTOR_ENGINEERING_COMPLETE'
    commands=[('contracts',[sys.executable,'-B','-m','unittest','graph.task10i_stage4a2.test_contracts','-v'])]
    for name,engineering in [('engineering',True),('historical',False)]:
        command=[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.integration_ablation',str(root/name),
            '--uncertain-output-policy','preserve_current']
        if engineering:command.append('--engineering')
        commands.append((name,command))
    for name,command in commands:
        write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,pid=os.getpid(),command=command))
        with (root/(name+'.log')).open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            write(root/'pipeline_state.json',dict(status='FAILED',stage=name,returncode=result.returncode,partial_preserved=True))
            raise RuntimeError(name+' failed; preserved output')
        if name!='contracts':
            scope=read(root/name/'scope.json');assert scope['uncertain_output_policy']=='preserve_current'
    write(root/'pipeline_state.json',dict(status='KEEP_CURRENT_HISTORICAL_CONTROL_COMPLETE',
        independent_test=False,candidate_selected=False,whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('dependency',type=Path)
    a=p.parse_args()
    try:run(a.root,a.dependency)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,
            message=str(exc),partial_preserved=True,whole_goal_complete=False))
        raise
