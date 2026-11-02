"""Live optimization check on the Development sequence with actual past Re-ID."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .frozen_inference import array_digest
from graph.task10i_stage4a.run_pipeline import read,write,sha


def wait_for(root,path,status,module,argument):
    deadline=time.monotonic()+24*3600
    while True:
        if path.exists():
            value=read(path)
            if status is None or value.get('status')==status:return
            if value.get('status')=='FAILED':raise RuntimeError('dependency failed: '+str(path))
        live=[]
        for p in Path('/proc').glob('[0-9]*/cmdline'):
            try:
                args=p.read_bytes().split(b'\0')
                if module.encode() in args and str(argument).encode() in args:live.append(p.parent.name)
            except OSError:pass
        if not live:raise RuntimeError('dependency has no live coordinator and no completed receipt')
        write(root/'pipeline_state.json',dict(status='WAITING_FOR_LIVE_DEPENDENCY',path=str(path),pids=live,pid=os.getpid()))
        if time.monotonic()>deadline:raise TimeoutError('finite optimization-control deadline')
        time.sleep(10)


def comparisons(events):
    return [(e['timestep'],e['camera'],e['object_id'],p['post_rank'],p['similarity'],p['negative_similarity'],p['reid_pass'])
            for e in events for p in e['proposals'] if p['similarity'] is not None]


def run(root,engineering,development,paired,coordinator):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    identity=dict(source={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
        selection='historical Development sequence containing the only actual past-reference comparison; no Internal/Final',
        dependency=str(engineering),geometry_cache=True,lazy_matching=True,GT_forward_input=False)
    write(root/'source_config_frozen.json',identity)
    eid='10i_stage3_10i_s1_0909_full_task'
    entry=next(e for e in read(development/'expected_counts.json')['entries'] if str(e['episode_id'])==eid)
    expected=root/'expected_counts.json';write(expected,dict(parents=1,episodes=1,frames=entry['actions']+1,entries=[entry]))
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_lazy_actual_ReID_equivalence',seed=42,
            root=root/'experiments',job_type='evaluation',config=identity) as logger:
        wait_for(root,engineering/'pipeline_state.json','LIVE_WRAPPER_COMPATIBILITY_COMPLETE',
                 'graph.task10i_stage4a2.live_engineering_runner',engineering)
        subprocess.run([sys.executable,'-B','-m','unittest','graph.task10i_stage4a2.test_contracts',
                        'graph.task10i_stage4a2.test_evaluation_inputs','-v'],check=True)
        command=[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.frozen_inference',str(root/'candidate'),
            str(expected),'--condition','BC_KEEP','--geometry-cache','--lazy-matching','--engineering']
        for attempt in range(2):
            write(root/'pipeline_state.json',dict(status='RUNNING',command=command,attempt=attempt,pid=os.getpid()))
            log=root/f'live_attempt{attempt}.log'
            with log.open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            if result.returncode==0:break
            if attempt or 'out of memory' not in log.read_text().lower():raise RuntimeError('preserved lazy control failure')
        actual=root/'candidate/perception'/eid;previous=development/'perception'/eid
        with np.load(actual/'BC_KEEP.npz') as a,np.load(previous/'BC_KEEP.npz') as b:
            assert np.array_equal(a['mask_bits'],b['mask_bits']),'optimized live candidate changed masks'
        before_events=read(previous/'BC_KEEP/events.json');after_events=read(actual/'events.json')
        assert comparisons(before_events)==comparisons(after_events) and len(comparisons(after_events))==1
        old_features=read(previous/'BC_KEEP/feature_calls.json');new_features=read(actual/'feature_calls.json')
        assert sum(r['model_forward'] for r in new_features)<=sum(r['model_forward'] for r in old_features)
        # The unchanged C and BC masks are byte equal on all Development frames.
        # Compare the optimized live Graph against the fully built C Graph too.
        wait_for(root,paired/'episodes'/eid/'receipt.json',None,
                 'graph.task10i_stage4a2.correction_pipeline',coordinator)
        gr=read(paired/'episodes'/eid/'receipt.json')
        assert gr['candidate']=='C_KEEP' and all(sha(paired/'episodes'/eid/p)==h for p,h in gr['files'].items())
        with np.load(previous/'C_KEEP.npz') as a,np.load(previous/'BC_KEEP.npz') as b:assert np.array_equal(a['mask_bits'],b['mask_bits'])
        for frame in read(actual/'graph_fingerprints.json'):
            with np.load(paired/'episodes'/eid/'C_KEEP'/f't{frame["timestep"]:04d}'/'graph.npz') as z:
                assert all(array_digest(z[k])==v for k,v in frame['tensors'].items()),'optimized Graph changed'
        write(root/'scope.json',dict(status='LAZY_ACTUAL_REID_EQUIVALENCE_COMPLETE',run_id=logger.run_id,
            episode=eid,frames=entry['actions']+1,mask_byte_equal=True,Graph_five_tensors_equal=True,
            actual_comparisons_exact_equal=comparisons(after_events),
            previous_feature_forwards=sum(r['model_forward'] for r in old_features),
            optimized_feature_forwards=sum(r['model_forward'] for r in new_features),
            old_receipt_sha256=sha(previous/'receipt.json'),new_receipt_sha256=sha(actual/'receipt.json'),
            GT_forward_input=False,latency_is_engineering_only=True,Internal_and_Final_opened=False,whole_goal_complete=False))
        write(root/'pipeline_state.json',dict(status='LAZY_ACTUAL_REID_EQUIVALENCE_COMPLETE',whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','engineering','development','paired','coordinator'):p.add_argument(name,type=Path)
    a=p.parse_args()
    try:run(a.root,a.engineering,a.development,a.paired,a.coordinator)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True))
        raise
