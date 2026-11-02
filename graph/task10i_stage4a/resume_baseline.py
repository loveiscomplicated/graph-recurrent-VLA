"""Prepared disk-failure recovery. No deletion, reset or implicit job restart."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .run_pipeline import read,write,sha,STAGE3


def main(root,previous,exact_baseline):
    # Caller must first inspect the former coordinator/worker handles. This
    # command refuses any still-live original development-perception process.
    process=subprocess.run(['ps','-eo','args'],capture_output=True,text=True,check=True).stdout.splitlines()
    if any(' -m graph.task10i_stage4a.development_perception '+str(previous) in line for line in process):raise RuntimeError('previous perception worker is still live')
    if shutil.disk_usage(root.parent).free<768*1024**2:raise RuntimeError('insufficient disk headroom; recover storage before model initialization')
    root.mkdir(parents=True,exist_ok=False)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from utils.experiment_logger import ExperimentLogger
        with ExperimentLogger(team='integration',model='Task10I_Stage4A_verified_disk_recovery',seed=42,root=root/'experiments',job_type='evaluation',config=dict(previous_baseline_root=str(previous),fresh_UUID=True,reexecute_only_uncommitted_episodes=True)) as log:
            write(root/'run_identity.json',dict(run_id=log.run_id,pid=os.getpid(),previous_root=str(previous)))
            try:
                source=Path(__file__).resolve().parents[2];old=read(previous/'source_freeze.json')['files']
                changed=[k for k,v in old.items() if not k.startswith('graph/task10i_stage4a/') and sha(source/k)!=v];assert not changed,changed
                write(root/'protocol.json',read(previous/'protocol.json'))
                write(root/'reuse_manifest.json',dict(previous_baseline_root=str(previous),previous_uuid=read(previous/'run_identity.json')['run_id'],new_uuid=log.run_id,compression='logical-byte SHA verified; original plain JSON or .json.gz fallback'))
                (root/'references').symlink_to(previous/'references',target_is_directory=True)
                (root/'perception').mkdir();(root/'traces').mkdir();reused=[]
                for p in sorted((previous/'perception').glob('*/receipt.json')):
                    r=read(p);eid=str(r['episode_id'])
                    assert sha(p.parent/'masks.npz')==r['masks_sha256'] and sha(p.parent/'stage_rows.json')==r['rows_sha256']
                    (root/'perception'/eid).symlink_to(p.parent,target_is_directory=True)
                    (root/'traces'/eid).symlink_to(previous/'traces'/eid,target_is_directory=True)
                    reused.append(eid)
                write(root/'reused_complete_episodes.json',reused)
                write(root/'pipeline_state.json',dict(status='RUNNING',stage='REMAINING_UNCOMMITTED_BASELINE_EPISODES',run_id=log.run_id,pid=os.getpid(),reused_episodes=len(reused)))
                with (root/'perception_resume.log').open('x') as out:subprocess.run([sys.executable,'-B','-u','-m','graph.task10i_stage4a.development_perception',str(root),str(exact_baseline)],stdout=out,stderr=subprocess.STDOUT,check=True)
                from .aggregate_development import run as aggregate
                aggregate(root);log.link_output(root/'development_perception_summary.json',kind='metrics')
                write(root/'pipeline_state.json',dict(status='SUBSTAGE_FINISHED',stage='RECOVERED_DEVELOPMENT_BASELINE_FINISHED',completed_goal=False,run_id=log.run_id))
            except BaseException as exc:
                write(root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),run_id=log.run_id));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('previous',type=Path);p.add_argument('exact_baseline',type=Path);a=p.parse_args();main(a.root,a.previous,a.exact_baseline)
