"""Autonomous fail-fast subprocess pipeline; no agent callbacks or retries."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from graph.plan_a_bc.common import read,write,sha,now

STAGES=[
    ('TRAIN_FEATURE_CACHE_VERIFY','prepare','verify'),
    ('FULL_BC_TRAINING','train',None),
    ('DEVELOPMENT_EVALUATION','evaluate','development'),
    ('CHECKPOINT_SELECTION_FREEZE','evaluate','select'),
    ('INDEPENDENT_FINAL_EVALUATION','evaluate','final'),
    ('H1_H2_HANDOFF_EVALUATION','evaluate','handoff'),
    ('PAIRED_COMPARISON_REPORT','report','analyze'),
    ('ARTIFACT_VERIFICATION','report','verify')]


class Runner:
    def __init__(self,root):
        self.root=root;self.child=None;self.stop=threading.Event();self.history=[]

    def heartbeat(self):
        while not self.stop.wait(10):
            write(self.root/'heartbeat.json',dict(at=now(),pid=os.getpid(),
                child_pid=self.child.pid if self.child else None))

    def stage(self,name,command):
        record=dict(stage=name,status='RUNNING',started_at=now(),command=command)
        self.history.append(record)
        write(self.root/'pipeline_status.json',dict(status='RUNNING',pid=os.getpid(),stages=self.history))
        folder=self.root/'logs';folder.mkdir(exist_ok=True)
        with (folder/(name+'.log')).open('xb') as log:
            print('STAGE_START',name,now(),flush=True)
            self.child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=self.root/'source')
            record['child_pid']=self.child.pid
            write(self.root/'pipeline_status.json',dict(status='RUNNING',pid=os.getpid(),stages=self.history))
            code=self.child.wait();self.child=None
        record.update(exit_code=code,completed_at=now(),log='logs/'+name+'.log',log_sha256=sha(folder/(name+'.log')),
                      status='PASS' if code==0 else 'FAILED')
        write(self.root/'pipeline_status.json',dict(status='RUNNING' if code==0 else 'FAILED',pid=os.getpid(),stages=self.history))
        if code:raise subprocess.CalledProcessError(code,command)
        print('STAGE_COMPLETE',name,now(),flush=True)

    def fail(self,exc):
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try:self.child.wait(timeout=30)
            except subprocess.TimeoutExpired:self.child.kill();self.child.wait()
        record=dict(status='FAILED',at=now(),error=repr(exc),traceback=traceback.format_exc(),
                    stages=self.history,last_checkpoint=str(self.root/'train_S/last.pt'))
        write(self.root/'pipeline_status.json',record);write(self.root/'FAILED',record)


def run(root):
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'PIPELINE_COMPLETE').exists() and not (root/'FAILED').exists()
    assert read(root/'preflight_verification.json')['status']=='PASS'
    cfg=read(root/'config.json')
    assert os.getenv('CONTAINER_ID')==str(cfg['runtime']['instance'])
    for name,h in read(root/'execution_source_identity.json')['hashes'].items():assert sha(root/'source'/name)==h,name
    runner=Runner(root)
    def interrupted(signum,frame):raise InterruptedError('pipeline signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    write(root/'heartbeat.json',dict(at=now(),pid=os.getpid(),child_pid=None))
    thread=threading.Thread(target=runner.heartbeat,daemon=True);thread.start()
    try:
        for name,module,stage in STAGES:
            command=[sys.executable,'-B','-u','-m','graph.predstate10i.'+module,'--root',str(root)]
            if stage:command+=['--stage',stage]
            runner.stage(name,command)
        assert read(root/'verification.json')['status']=='PASS'
        assert read(root/'completion_audit.json')['status']=='PASS'
        write(root/'pipeline_status.json',dict(status='PIPELINE_COMPLETE',completed_at=now(),stages=runner.history))
        receipt={name:sha(root/name) for name in ['pipeline_status.json','verification.json',
            'completion_audit.json','artifact_index.json','checkpoint_manifest.json','best.pt','last.pt',
            '10i_predicted_state38_graph_recurrent_bc.md','report.html']}
        for name,h in receipt.items():assert sha(root/name)==h
        write(root/'pipeline_receipt.json',dict(status='PASS',files=receipt,readback=True))
        write(root/'PIPELINE_COMPLETE',dict(completed_at=now(),verification_sha256=sha(root/'verification.json'),
            completion_audit_sha256=sha(root/'completion_audit.json'),receipt_sha256=sha(root/'pipeline_receipt.json')))
        print('PIPELINE_COMPLETE',now(),flush=True)
    except BaseException as exc:
        runner.fail(exc);raise
    finally:
        runner.stop.set();thread.join(timeout=15)


def fixture(root):
    """Execute real child exit-code/failure-marker and lock behavior before launch."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix='pipeline-fixture-',dir=root) as td:
        temp=Path(td);(temp/'source').mkdir();runner=Runner(temp)
        runner.stage('success',[sys.executable,'-c','print("child started")'])
        try:runner.stage('failure',[sys.executable,'-c','raise SystemExit(7)'])
        except subprocess.CalledProcessError as exc:
            assert exc.returncode==7;runner.fail(exc)
        else:raise AssertionError('failure did not propagate')
        assert read(temp/'pipeline_status.json')['status']=='FAILED'
        assert (temp/'FAILED').exists() and not (temp/'PIPELINE_COMPLETE').exists()
        owner=(temp/'lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=subprocess.run([sys.executable,'-c',
            'import fcntl,sys;f=open(sys.argv[1],"a");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)',str(temp/'lock')],capture_output=True)
        assert result.returncode!=0
    write(root/'pipeline_engine_verification.json',dict(status='PASS',child_failure_exit_code=7,
        no_complete_on_failure=True,atomic_manifest=True,duplicate_lock_rejected=True,automatic_retry=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--fixture',action='store_true')
    a=p.parse_args();fixture(a.root) if a.fixture else run(a.root)
