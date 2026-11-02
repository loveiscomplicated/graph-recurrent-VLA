"""Verified artifact adapters for frozen Internal/Final paired evaluation.

No model or selection rule runs here. Source RGB-D is referenced, never copied.
GT conversion is a separate post-inference operation from mask combination.
"""
import argparse
from pathlib import Path
import shutil
import numpy as np
from graph.task10i_stage4a.run_pipeline import read,write,sha,STAGE3


def manifest(output,source,mode,freeze=None):
    if mode=='internal':
        from .final_capture import freeze_barrier
        assert freeze is not None
        freeze_barrier(freeze,None,None,scope='internal')
        partition=read(source);parents=set(partition['internal_holdout'])
        assert not parents&set(partition['development'])
        entries=sorted([e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()
                        if e['parent_id'] in parents],key=lambda e:str(e['episode_id']))
        assert {e['parent_id'] for e in entries}==parents and len(parents)==30
    else:
        receipt=read(source);assert receipt['status']=='SYNCHRONOUS_BENCHMARK_CAPTURE_COMPLETE'
        assert mode in ('normal','stress','all')
        entries=[e for e in receipt['entries'] if mode=='all' or e['stratum']==mode]
    write(output,dict(parents=len({e['parent_id'] for e in entries}),episodes=len(entries),
        frames=sum(e['actions']+1 for e in entries),entries=entries,scope=mode,
        source=str(source),source_sha256=sha(source),model_or_GT_values_read=False))


def combine_masks(output,expected,baseline,challenger,candidate,require_baseline_isolation=False):
    output.mkdir(parents=True,exist_ok=False)
    assert read(baseline/'pipeline_state.json')['status']=='FROZEN_LIVE_CONDITION_COMPLETE'
    assert read(challenger/'pipeline_state.json')['status']=='FROZEN_LIVE_CONDITION_COMPLETE'
    assert read(baseline/'pipeline_state.json')['condition']=='B0'
    assert read(challenger/'pipeline_state.json')['condition']==candidate and candidate!='B0'
    spec=read(expected)
    for e in spec['entries']:
        eid=str(e['episode_id']);folder=output/'perception'/eid;folder.mkdir(parents=True)
        sources=[]
        for condition,root in [('B0',baseline),(candidate,challenger)]:
            source=root/'perception'/eid;receipt=read(source/'receipt.json')
            assert receipt['condition']==condition and receipt['raw_sha256']==e['raw_sha256']
            assert receipt['frames']==e['actions']+1
            assert all(sha(source/p)==h for p,h in receipt['files'].items())
            with np.load(source/(condition+'.npz')) as z:
                assert z['mask_bits'].shape==(e['actions']+1,8192) and z['mask_bits'].dtype==np.uint8
            shutil.copyfile(source/(condition+'.npz'),folder/(condition+'.npz'))
            assert sha(folder/(condition+'.npz'))==sha(source/(condition+'.npz'))
            sources.append(dict(condition=condition,receipt=str(source/'receipt.json'),sha256=sha(source/'receipt.json'),
                live_graph_fingerprints=str(source/'graph_fingerprints.json'),graph_sha256=sha(source/'graph_fingerprints.json')))
        write(folder/'upstream_receipts.json',sources)
        live_internal=challenger/'perception'/eid/'internal_B0.npz'
        if require_baseline_isolation:assert live_internal.exists(),'live candidate baseline isolation proof missing'
        if live_internal.exists():
            with np.load(live_internal) as x,np.load(folder/'B0.npz') as y:
                assert np.array_equal(x['mask_bits'],y['mask_bits']),'candidate contaminated its simultaneously running original baseline memory'
            write(folder/'baseline_memory_isolation.json',dict(status='PASS',all_frames_byte_equal=True,
                source=str(live_internal),source_sha256=sha(live_internal)))
        write(folder/'receipt.json',dict(episode=eid,parent_id=e['parent_id'],frames=e['actions']+1,
            raw_sha256=e['raw_sha256'],candidate=candidate,GT_forward_input=False,
            files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
    write(output/'expected_counts.json',spec)
    write(output/'pipeline_state.json',dict(status='PAIRED_LIVE_MASK_INPUTS_VERIFIED',candidate=candidate,
        episodes=spec['episodes'],frames=spec['frames'],GT_files_opened=False))


def synchronous_references(output,expected,baseline,challenger):
    # Enforce prediction completion *before* opening any GT pixels or poses.
    for root in (baseline,challenger):assert read(root/'pipeline_state.json')['status']=='FROZEN_LIVE_CONDITION_COMPLETE'
    output.mkdir(parents=True,exist_ok=False)
    spec=read(expected)
    for e in spec['entries']:
        eid=str(e['episode_id']);source=Path(e['raw_path']).parent;rr=read(source/'receipt.json')
        assert rr['synchronous_capture'] and rr['exact_frames']==rr['frames']==e['actions']+1
        assert rr['raw_sha256']==e['raw_sha256']
        assert all(sha(source/p)==h for p,h in rr['files'].items())
        folder=output/'references'/eid;folder.mkdir(parents=True)
        for name in ('segmentation.npz','meshes.npz'):shutil.copyfile(source/name,folder/name)
        with np.load(source/'gt_geometry.npz') as z:
            actor_pose=np.stack([z['raw_peg_pose'],z['physics_box_pose']],axis=1)
        assert actor_pose.shape==(e['actions']+1,2,7) and np.isfinite(actor_pose).all()
        np.savez_compressed(folder/'geometry_reference.npz',actor_pose=actor_pose)
        with np.load(folder/'segmentation.npz') as z:
            assert z['exact_frame'].all() and z['actor_segmentation'].shape==(e['actions']+1,2,128,128)
        write(folder/'receipt.json',dict(episode=eid,parent_id=e['parent_id'],total_frames=e['actions']+1,
            exact_frames=e['actions']+1,raw_sha256=e['raw_sha256'],object_ids=rr['object_ids'],robot_ids=rr['robot_ids'],
            actor_ids=rr['actor_ids'],reference_kind='EXACT_GT',method='synchronous collector with before/after render/physics readback',
            capture_receipt=str(source/'receipt.json'),capture_sha256=sha(source/'receipt.json'),
            files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
    write(output/'pipeline_state.json',dict(status='SYNCHRONOUS_EVALUATION_REFERENCES_VERIFIED',
        episodes=spec['episodes'],frames=spec['frames'],all_exact=True))


if __name__=='__main__':
    from utils.experiment_logger import ExperimentLogger
    p=argparse.ArgumentParser();p.add_argument('operation',choices=('manifest','combine','references'))
    p.add_argument('output',type=Path);p.add_argument('expected_or_source',type=Path)
    p.add_argument('--baseline',type=Path);p.add_argument('--challenger',type=Path);p.add_argument('--candidate')
    p.add_argument('--mode');p.add_argument('--freeze',type=Path)
    p.add_argument('--require-baseline-isolation',action='store_true');a=p.parse_args()
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_evaluation_input_'+a.operation,
            seed=42,root=a.output.parent/'adapter_experiments',job_type='evaluation',
            config=dict(operation=a.operation,source_sha256=sha(Path(__file__)),input_sha256=sha(a.expected_or_source))) as logger:
        if a.operation=='manifest':manifest(a.output,a.expected_or_source,a.mode,a.freeze)
        elif a.operation=='combine':combine_masks(a.output,a.expected_or_source,a.baseline,a.challenger,a.candidate,a.require_baseline_isolation)
        else:synchronous_references(a.output,a.expected_or_source,a.baseline,a.challenger)
