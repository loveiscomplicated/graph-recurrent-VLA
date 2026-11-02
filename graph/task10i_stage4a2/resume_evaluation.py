"""Resume verified current-only Graph evaluation across a supervised OOM retry.

The previous coordinator stopped on a transient FAILED inference child although
the finite-retry supervisor was live. Never restart the inference from here.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import time
from .paired_evaluation import evaluate_episode,initialize_worker
from graph.task10i_stage4a.run_pipeline import read,write,sha


def compatibility(previous,source):
    old=read(previous/'source_identity.json')
    old_source=previous.parent/'source';checks=[]
    changed=[]
    for p,digest in old['files'].items():
        assert sha(old_source/p)==digest
        if sha(source/p)!=digest:changed.append(p)
        checks.append(dict(path=p,old_sha256=digest,new_sha256=sha(source/p)))
    path='graph/task10i_stage4a2/paired_evaluation.py'
    assert set(changed)<= {path},changed
    def function(p):
        return next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='evaluate_episode')
    before=function(old_source/path);after=function(source/path)
    if ast.dump(before)!=ast.dump(after):
        # Only the engineering-prefix reference counter was corrected. For all
        # complete reused episodes, reference length equals evaluated length.
        removed=[]
        for node in list(after.body):
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('seg','exact') and isinstance(node.value,ast.Subscript) and isinstance(node.value.slice,ast.Slice):
                removed.append(node.targets[0].id);after.body.remove(node)
        assert sorted(removed)==['exact','seg']
        assert ast.dump(before)==ast.dump(after),'evaluation computation changed beyond prefix count correction'
    return dict(status='PASS',files=checks,changed_files=changed,
        evaluation_AST_equal_after_removing_engineering_reference_length_clips=True,
        reused_scope='complete original reference-length episodes only; no engineering or partial episode reuse')


def live_supervisor(supervisor):
    state=read(supervisor/'pipeline_state.json')
    if state.get('status')!='RUNNING':return False
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            command=path.read_bytes()
            if b'graph.task10i_stage4a2.mask_forward_runner' in command and str(supervisor).encode() in command:return True
        except OSError:pass
    return False


def run(root,previous,inference,references,expected,supervisor):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve().parents[2]
    proof=compatibility(previous,source);write(root/'source_compatibility.json',proof)
    manifest=read(expected);entries=manifest['entries'];byid={str(e['episode_id']):e for e in entries}
    assert read(previous/'source_identity.json')['expected_sha256']==sha(expected)
    write(root/'source_identity.json',dict(previous_root=str(previous),source_sha256=sha(Path(__file__)),
        evaluator_sha256=sha(Path(__file__).with_name('paired_evaluation.py')),expected_sha256=sha(expected)))
    done=[];submitted=set();pending={}
    for path in sorted((previous/'episodes').glob('*/receipt.json')):
        receipt=read(path);eid=receipt['episode'];e=byid[eid]
        assert receipt['frames']==e['actions']+1 and receipt['exact_frames']<=receipt['frames']
        assert receipt['raw_sha256']==e['raw_sha256']
        assert receipt['inference_receipt_sha256']==sha(inference/'perception'/eid/'receipt.json')
        assert receipt['reference_receipt_sha256']==sha(references/'references'/eid/'receipt.json')
        assert all(sha(path.parent/p)==h for p,h in receipt['files'].items())
        (root/'episodes').mkdir(exist_ok=True)
        (root/'episodes'/eid).symlink_to(path.parent,target_is_directory=True)
        done.append(receipt);submitted.add(eid)
    write(root/'verified_reuse.json',dict(episodes=len(done),receipts=done))
    deadline=time.monotonic()+24*3600
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_resumed_paired_Graph',seed=42,
         root=root/'experiments',job_type='evaluation',config=dict(previous=str(previous),verified_reused=len(done))) as logger:
        try:
            with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
                while len(done)<len(entries):
                    for e in entries:
                        eid=str(e['episode_id'])
                        if eid in submitted or len(pending)>=3:continue
                        if (inference/'perception'/eid/'receipt.json').exists() and (references/'references'/eid/'receipt.json').exists():
                            pending[pool.submit(evaluate_episode,e,inference,references,root)]=eid;submitted.add(eid)
                    for future,eid in list(pending.items()):
                        if future.done():
                            done.append(future.result());del pending[future]
                            write(root/'progress.json',done);print('resumed paired',len(done),'/',len(entries),eid,flush=True)
                    write(root/'pipeline_state.json',dict(status='RUNNING',pid=os.getpid(),run_id=logger.run_id,
                        completed_episodes=len(done),active_episodes=list(pending.values()),expected_episodes=len(entries)))
                    for dependency in (inference,references):
                        state=read(dependency/'pipeline_state.json')
                        if state['status']=='FAILED':
                            if dependency==inference and live_supervisor(supervisor):continue
                            raise RuntimeError('terminal dependency failure: '+str(dependency))
                    if time.monotonic()>deadline:raise TimeoutError('paired evaluation deadline')
                    if len(done)<len(entries):time.sleep(5)
            assert len(done)==manifest['episodes'] and sum(r['frames'] for r in done)==manifest['frames']
            write(root/'pipeline_state.json',dict(status='PAIRED_NEW_DEVELOPMENT_GRAPH_COMPLETE',run_id=logger.run_id,
                episodes=len(done),frames=sum(r['frames'] for r in done),exact_frames=sum(r['exact_frames'] for r in done),
                aggregation_pending=True,whole_goal_complete=False))
        except BaseException as exc:
            write(root/'pipeline_state.json',dict(status='FAILED',error=type(exc).__name__,message=str(exc),
                completed_episodes=len(done),partial_outputs_preserved=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','previous','inference','references','expected','supervisor'):p.add_argument(name,type=Path)
    a=p.parse_args();run(a.root,a.previous,a.inference,a.references,a.expected,a.supervisor)
