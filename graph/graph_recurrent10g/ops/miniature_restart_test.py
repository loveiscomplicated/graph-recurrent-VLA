"""Independent engineering harness, run inside tmux; never calls any agent."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);source=Path(sys.argv[2]);sys.path.insert(0,str(source))
from graph.plan_a_bc.common import read,write,sha
from graph.graph_recurrent10g.orchestrator import verify_all_stages
assert os.getenv('TMUX')
assert read(root/'pipeline_state.json')['status']=='ENGINEERING_INTERRUPTED'
write(root/'intentional_restart_snapshot.json',read(root/'pipeline_state.json'))
args=[sys.executable,'-u','-B','-m','graph.graph_recurrent10g.orchestrator','--root',str(root)]
with (root/'restart_attempt_interrupted.log').open('w') as log:
    p=subprocess.Popen(args,cwd=source,stdout=log,stderr=subprocess.STDOUT)
    deadline=time.monotonic()+180
    while True:
        if p.poll() is not None:raise RuntimeError('orchestrator exited before interruption fixture')
        state=read(root/'pipeline_state.json')
        if state.get('current_stage')=='DEVELOPMENT_CLOSED_LOOP' and state['stages']['DEVELOPMENT_CLOSED_LOOP']['status']=='RUNNING':
            time.sleep(3)
            p.send_signal(signal.SIGTERM);code=p.wait(timeout=45)
            assert code!=0
            break
        if time.monotonic()>deadline:raise TimeoutError('development fixture never started')
        time.sleep(.2)
interrupted=read(root/'pipeline_state.json');assert interrupted['status']=='BLOCKED'
write(root/'interrupted_running_stage_snapshot.json',interrupted)
with (root/'restart_attempt_complete.log').open('w') as log:
    p=subprocess.Popen(args,cwd=source,stdout=log,stderr=subprocess.STDOUT)
    code=p.wait()
if code:raise RuntimeError('resumed miniature failed; inspect log and preserved stage evidence')
verify_all_stages(root)
text=(root/'restart_attempt_complete.log').read_text()
assert 'SKIP verified PARALLEL_TRAIN_A_B_C' in text
write(root/'miniature_restart_verification.json',dict(status='PASS',completed_stage_skip=True,
    running_development_stage_interrupted_nonzero=True,restarted_same_orchestrator=True,
    final_marker_exists=(root/'COMPLETED').is_file(),checkpoint_resume_proof='resume_verification.json',
    duplicate_lock_rejection='separate actual concurrent CLI negative control',harness_sha256=sha(Path(__file__))))
print('MINIATURE RESTART AND END-TO-END PASS',flush=True)
