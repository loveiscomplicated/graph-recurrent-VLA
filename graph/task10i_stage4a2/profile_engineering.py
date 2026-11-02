"""Verify observer-light timing wrappers on actual late handoff and Graphs."""
import argparse
from pathlib import Path
import subprocess
import sys
import numpy as np
from .lazy_feature_control import wait_for
from .frozen_inference import array_digest
from graph.task10i_stage4a.run_pipeline import read,write,sha


def run(root,dependency,baseline,previous,gpu_dependency=None):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    eid='10i_stage3_10i_s1_0250_full_task';candidate='BC_WB_HANDOFF'
    entry=next(e for e in read(baseline/'expected_counts.json')['entries'] if str(e['episode_id'])==eid)
    expected=root/'expected_counts.json';write(expected,dict(parents=1,episodes=1,frames=16,entries=[dict(entry,actions=15,engineering_only=True)]))
    config=dict(source={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')},
        expected_sha256=sha(expected),GT_forward_input=False,Internal_and_Final_access=False,
        timing='module boundaries, no raw logits serialized; engineering CPU contention, not final latency')
    write(root/'source_config_frozen.json',config)
    if gpu_dependency is not None:
        wait_for(root,gpu_dependency/'pipeline_state.json','LIVE_PROFILE_WRAPPER_HANDOFF_GRAPH_EQUIVALENT',
            'graph.task10i_stage4a2.profile_engineering',gpu_dependency)
    wait_for(root,dependency/'development/pipeline_state.json','DEVELOPMENT_SCOPED_CORRECTION_INFERENCE_COMPLETE',
             'graph.task10i_stage4a2.main_box_pipeline',dependency)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_profile_late_handoff_equivalence',seed=42,
            root=root/'experiments',job_type='evaluation',config=config) as logger:
        command=[sys.executable,'-B','-u','-m','graph.task10i_stage4a2.frozen_inference',str(root/'live'),str(expected),
            '--condition',candidate,'--engineering','--geometry-cache','--lazy-matching']
        with (root/'live.log').open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        folder=root/'live/perception'/eid;source=dependency/'development/perception'/eid
        with np.load(folder/(candidate+'.npz')) as a,np.load(source/(candidate+'.npz')) as b:
            assert np.array_equal(a['mask_bits'],b['mask_bits'][:16]),'profile wrapper changed causal masks'
        with np.load(folder/'internal_B0.npz') as a,np.load(baseline/'perception'/eid/'B0.npz') as b:
            assert np.array_equal(a['mask_bits'],b['mask_bits'][:16]),'profile candidate changed original baseline memory'
        expected_handoffs=read(source/candidate/'baseline_handoff_events.json')
        actual_handoffs=read(folder/'baseline_handoff_events.json')
        assert actual_handoffs==[h for h in expected_handoffs if h['timestep']<=15]
        assert len(actual_handoffs)==1 and actual_handoffs[0]['accepted'] and actual_handoffs[0]['timestep']==15
        checks=[]
        for frame in read(folder/'graph_fingerprints.json'):
            t=frame['timestep'];label='C_KEEP' if t<15 else 'B0'
            path=previous/'episodes'/eid/label/f't{t:04d}'/'graph.npz'
            with np.load(path) as z:assert all(array_digest(z[k])==v for k,v in frame['tensors'].items()),('profile Graph changed',t)
            checks.append(dict(timestep=t,reference=str(path),reference_sha256=sha(path),five_tensors_equal=True))
        write(root/'scope.json',dict(status='LIVE_PROFILE_WRAPPER_HANDOFF_GRAPH_EQUIVALENT',run_id=logger.run_id,
            **config,mask_frames_equal=16,graph_frames_equal=16,actual_handoff_exact_equal=True,checks=checks,
            internal_original_baseline_frames_equal=16,
            fair_latency_executed=False,whole_goal_complete=False))
        write(root/'pipeline_state.json',dict(status='LIVE_PROFILE_WRAPPER_HANDOFF_GRAPH_EQUIVALENT',whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','dependency','baseline','previous'):p.add_argument(name,type=Path)
    p.add_argument('--gpu-dependency',type=Path)
    a=p.parse_args()
    try:run(a.root,a.dependency,a.baseline,a.previous,a.gpu_dependency)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True))
        raise
