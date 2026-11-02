"""Actual shared-input prefix invariance across different physical futures.

No GT, actions, phase labels or object poses are read. Only inputs consumed by
the current RGB-D/robot geometry pipeline define a shared causal prefix.
"""
import argparse
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import h5py
import numpy as np
from .lazy_feature_control import wait_for
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS

FIELDS=['timestamp','robot_state','tcp_pose','robot_world_geometry']+[
    camera+'/'+name for camera in CAMERAS for name in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl')]


def run(root,expected,full,scoped,paired,candidate='BC_MAIN_BOX'):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    config=dict(source_sha256=sha(Path(__file__)),expected_sha256=sha(expected),input_fields=FIELDS,
        scoped_candidate=candidate,
        GT_or_phase_or_future_actions_used=False,comparison='different same-parent branches; longest exact common sensor/robot prefix only')
    write(root/'protocol.json',config)
    wait_for(root,scoped/'development/pipeline_state.json','DEVELOPMENT_SCOPED_CORRECTION_INFERENCE_COMPLETE',
        'graph.task10i_stage4a2.main_box_pipeline',scoped)
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_actual_causal_prefix_invariance',seed=42,
            root=root/'experiments',job_type='evaluation',config=config) as logger:
        import torch
        from graph.plan_a_bc.geometry import CurrentGeometry
        torch.set_num_threads(1)
        schema=read('/workspace/task10f1_gt_pretrain_20261030/normalizer.json')['schema']
        groups=defaultdict(list)
        for e in read(expected)['entries']:groups[e['parent_id']].append(e)
        results=[];tested=0;frames=0;alias_graph_pairs=0
        for parent,entries in groups.items():
            for a,b in combinations(entries,2):
                ae,be=str(a['episode_id']),str(b['episode_id'])
                assert sha(a['raw_path'])==a['raw_sha256'] and sha(b['raw_path'])==b['raw_sha256']
                count=0
                with h5py.File(a['raw_path']) as aa,h5py.File(b['raw_path']) as bb:
                    for t in range(min(a['actions'],b['actions'])+1):
                        if not all(np.array_equal(aa[k][t],bb[k][t],equal_nan=True) for k in FIELDS):break
                        count+=1
                record=dict(parent_id=parent,episode_a=ae,episode_b=be,exact_shared_prefix_frames=count,
                    future_or_length_differs=a['raw_sha256']!=b['raw_sha256'],conditions=[])
                if count:
                    tested+=1;frames+=count
                    for condition,source in [('B0',full),('C_KEEP',full),('BC_KEEP',full),(candidate,scoped/'development')]:
                        left=source/'perception'/ae;right=source/'perception'/be
                        for folder in (left,right):
                            r=read(folder/'receipt.json');assert all(sha(folder/p)==h for p,h in r['files'].items())
                        with np.load(left/(condition+'.npz')) as x,np.load(right/(condition+'.npz')) as y:
                            assert np.array_equal(x['mask_bits'][:count],y['mask_bits'][:count]),('causal prefix mask mismatch',parent,ae,be,condition)
                        record['conditions'].append(dict(condition=condition,prefix_masks_byte_equal=True))
                    # Original FPS intentionally hashes the episode identity.
                    # Different branch IDs are different RNG inputs, not a
                    # future leak. Never rewrite their primary Graphs. For the
                    # first three supported pairs, control this RNG input too
                    # in a separate, GT-free engineering alias only.
                    record['original_cross_episode_Graph_equality_required']=False
                    record['reason']='legacy_fps.adapter.frame_seed includes actual episode ID'
                    if alias_graph_pairs<3:
                        alias='engineering_prefix_'+str(alias_graph_pairs)
                        controls=[]
                        for condition,input_root in [('C_KEEP',full),(candidate,scoped/'development')]:
                            pair_masks=[]
                            for eid in (ae,be):
                                with np.load(input_root/'perception'/eid/(condition+'.npz')) as z:
                                    pair_masks.append(np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool))
                            geometries=[CurrentGeometry(Path('/workspace/task10g_f1_abc_20261031/full/inputs'),schema) for _ in range(2)]
                            with h5py.File(a['raw_path']) as aa,h5py.File(b['raw_path']) as bb:
                                for t in sorted({0,min(1,count-1),count-1}):
                                    values=[]
                                    for side,raw in enumerate((aa,bb)):
                                        cameras,data,surfaces=geometries[side].from_h5(raw,t,alias)
                                        graph,_,_=geometries[side].build(cameras,data,surfaces,dict(zip(CAMERAS,pair_masks[side][t])))
                                        values.append({k:getattr(graph,k).numpy() for k in ('node_input','node_type','edge_index','edge_attr','edge_type')})
                                    assert all(np.array_equal(values[0][k],values[1][k]) for k in values[0]),('fixed RNG prefix Graph mismatch',ae,be,t,condition)
                                    controls.append(dict(condition=condition,timestep=t,five_tensors_equal=True))
                        record['fixed_RNG_alias_engineering_Graph_controls']=controls;alias_graph_pairs+=1
                results.append(record)
        assert tested>0,'no empirical shared-prefix support; cannot pass invariance'
        write(root/'causal_prefix_suffix_invariance.json',dict(status='ACTUAL_SHARED_PREFIX_INVARIANCE_PASS',run_id=logger.run_id,
            **config,pairs=results,pairs_with_exact_prefix=tested,total_pair_prefix_frames=frames,
            pairs_are_correlated_not_independent_samples=True,GT_forward_input=False,whole_goal_complete=False))
        write(root/'pipeline_state.json',dict(status='ACTUAL_SHARED_PREFIX_INVARIANCE_PASS',whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','expected','full','scoped','paired'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',default='BC_MAIN_BOX')
    a=p.parse_args()
    try:run(a.root,a.expected,a.full,a.scoped,a.paired,a.candidate)
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True))
        raise
