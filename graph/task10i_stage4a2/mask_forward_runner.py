"""Finite prerequisite wait, engineering barrier and autonomous Development run."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,partition,prerequisite):
    root.mkdir(parents=True,exist_ok=False)
    write(root/'source_identity.json',{str(p):sha(p) for p in Path(__file__).parent.glob('*.py')})
    deadline=time.monotonic()+1800
    while not (prerequisite/'scope.json').exists():
        live=[]
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                if b'graph.task10i_stage4a2.sam2_controls' in p.read_bytes():live.append(p.parent.name)
            except (OSError,PermissionError):pass
        if not live:raise RuntimeError('SAM controls stopped without completion proof')
        if time.monotonic()>deadline:raise TimeoutError('SAM controls wait deadline')
        write(root/'pipeline_state.json',dict(status='WAITING_FOR_LIVE_SAM_CONTROLS',pids=live))
        time.sleep(10)
    assert read(prerequisite/'scope.json')['status']=='SAM2_ORACLE_MEMORY_CONTROLS_COMPLETE'
    commands=[('contracts',[sys.executable,'-B','-m','unittest','graph.task10i_stage4a2.test_contracts','-v']),
        ('engineering',[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.development_inference',
                        str(root/'engineering'),str(partition),'--engineering']),
        ('development',[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.development_inference',
                        str(root/'development'),str(partition)])]
    for name,command in commands:
        for attempt in range(2):
            write(root/'pipeline_state.json',dict(status='RUNNING',stage=name,attempt=attempt,command=command))
            with (root/f'{name}_attempt{attempt}.log').open('w') as stream:
                result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode==0:break
            # An OOM retry starts a fresh process and episode prefix. Other
            # failures remain terminal instead of repeatedly hiding a defect.
            text=(root/f'{name}_attempt{attempt}.log').read_text()
            if attempt==1 or 'out of memory' not in text.lower():
                write(root/'pipeline_state.json',dict(status='FAILED',stage=name,attempt=attempt,
                    returncode=result.returncode,whole_goal_complete=False))
                raise RuntimeError(f'{name} failed; inspect preserved log')
        if name!='contracts':
            state=read(root/name/'pipeline_state.json')
            assert state['status']=='NEW_DEVELOPMENT_B0_A_INFERENCE_COMPLETE'
    write(root/'pipeline_state.json',dict(status='NEW_DEVELOPMENT_INFERENCE_COMPLETE',
        next_stage='EXACT_REFERENCE_AND_PAIRED_PIXEL_GRAPH_EVALUATION',whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('partition',type=Path)
    p.add_argument('prerequisite',type=Path);a=p.parse_args();run(a.root,a.partition,a.prerequisite)
