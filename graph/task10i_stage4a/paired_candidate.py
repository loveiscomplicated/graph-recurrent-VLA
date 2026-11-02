"""Causal RGB-D correction over frozen baseline masks; paired evaluator is separate."""
import argparse
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .run_pipeline import CAMERAS, STAGE3, read, write, sha


def run(root, baseline, corrected, partition='development', refine=True, mode='reanchor', robot_memo=False, reuse=None):
    from utils.experiment_logger import ExperimentLogger
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    from graph.plan_a_bc.geometry import CurrentGeometry
    from .reliable_reanchor import WristPegReanchor
    from .stagewise import trace_geometry
    from .reevaluate_references import updated_points
    from graph.persistent_geometry.metrics import sample_meshes
    from graph.plan_a_bc.geometry import validate_graph
    torch.set_num_threads(1)
    torch.manual_seed(42)
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    root.mkdir(parents=True, exist_ok=False)
    protocol = read(baseline / 'protocol.json')
    parents = set(protocol[partition])
    entries = sorted([e for e in read(STAGE3 / 'stage4_recache_input_manifest.json')['unique_sources'].values()
                      if e['parent_id'] in parents], key=lambda e: str(e['episode_id']))
    runtime_root = Path('/workspace/task10g_f1_abc_20261031/full')
    runtime = read(runtime_root / 'runtime_paths.json')
    schema = read(Path('/workspace/task10f1_gt_pretrain_20261030/normalizer.json'))['schema']
    candidate_name='MAIN_PEG_BOX_DOMINANT_DEPTH_COMPONENT' if mode=='depth_component_both' else 'MAIN_PEG_DOMINANT_DEPTH_COMPONENT' if mode=='depth_component' else 'RGBD_MAIN_SUPPORTED_WRIST_PEG_REANCHOR_V2' if refine else 'V1_UNREFINED_ABLATION'
    spec = dict(candidate=candidate_name, mode=mode, exact_current_robot_memo=robot_memo,
                cpu_thread_environment={k:__import__('os').environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
                evaluation_partition=partition, interval=5, topk=8, refine=refine,
                shared_surface_depth_agreement_m=.004, witness_minimum=6, witness_fraction=.6,
                unchanged_original_wrist_gate=True, current_only=True, box_main_only=True,
                GT_student_input=False, production_adoption=False,
                baseline_masks='same cached prefix-causal online adapter outputs, current frame only',
                source_sha256=sha(Path(__file__).with_name('reliable_reanchor.py')))
    if mode=='depth_component':spec.update(component_radius_m=.012,minimum_dominant_fraction=.8,minimum_points=6,source_sha256=sha(Path(__file__).with_name('depth_component.py')))
    if mode=='depth_component_both':spec.update(component_radius_m=.012,minimum_dominant_fraction=.8,minimum_points=6,object_ids=[0,1],source_sha256=sha(Path(__file__).with_name('depth_box_component.py')))
    write(root / 'protocol.json', protocol)
    write(root / 'candidate_spec.json', spec)
    with ExperimentLogger(team='integration', model='Task10I_Stage4A_paired_RGBD_reanchor', seed=42,
                          job_type='evaluation', root=root / 'experiments', config=spec) as log:
        write(root / 'run_identity.json', dict(run_id=log.run_id, source_sha256=sha(__file__)))
        detector = GDino(device='cuda', model_id=runtime['dino']) if mode=='reanchor' else None
        tracker = SAM2Tracker(device='cuda', checkpoint=runtime['sam']) if mode=='reanchor' else None
        progress = []
        try:
            for e in entries:
                eid = str(e['episode_id'])
                if reuse is not None and (reuse/'perception'/eid/'receipt.json').exists():
                    rs=read(reuse/'candidate_spec.json')
                    assert rs['source_sha256']==spec['source_sha256'] and rs['candidate']==spec['candidate']
                    oldfolder=reuse/'perception'/eid;rc=read(oldfolder/'receipt.json')
                    assert rc['raw_sha256']==e['raw_sha256'] and sha(oldfolder/'masks.npz')==rc['masks_sha256'] and sha(oldfolder/'stage_rows.json')==rc['rows_sha256']
                    (root/'perception').mkdir(exist_ok=True);(root/'traces').mkdir(exist_ok=True)
                    (root/'perception'/eid).symlink_to(oldfolder,target_is_directory=True)
                    (root/'traces'/eid).symlink_to(reuse/'traces'/eid,target_is_directory=True)
                    progress.append(dict(rc,reused_verified_complete_episode=True,original_receipt=str(oldfolder/'receipt.json')))
                    write(root/'progress.json',progress);continue
                old = baseline / 'perception' / eid
                br = read(old / 'receipt.json')
                assert sha(old / 'masks.npz') == br['masks_sha256']
                assert sha(e['raw_path']) == e['raw_sha256']
                cr = read(corrected / 'perception' / eid / 'receipt.json')
                rrpath = Path(cr['corrected_reference'])
                rr = read(rrpath)
                assert rr['raw_sha256'] == e['raw_sha256']
                assert sha(rrpath.parent / 'segmentation.npz') == rr['segmentation_sha256']
                with np.load(rrpath.parent / 'segmentation.npz', allow_pickle=False) as z:
                    seg, exact = z['actor_segmentation'].copy(), z['exact_frame'].copy()
                with np.load(baseline / 'references' / eid / 'geometry_reference.npz', allow_pickle=False) as z:
                    poses = z['actor_pose'].copy()
                meshes = sample_meshes(baseline / 'references' / eid / 'meshes.npz', samples_per_object=10000)
                prior_rows = read(corrected / 'perception' / eid / 'stage_rows.json')
                prior_points = read(corrected / 'perception' / eid / 'representative_points.json')
                by_t = {t: [r for r in prior_rows if r['timestep'] == t] for t in range(e['actions']+1)}
                points_by_t = {t: [r for r in prior_points if r['timestep'] == t] for t in range(e['actions']+1)}
                geometry = CurrentGeometry(runtime_root / 'inputs', schema)
                if robot_memo:
                    from .perframe_cache import install
                    install(geometry)
                if mode=='reanchor':policy = WristPegReanchor(detector, tracker, geometry, refine=refine)
                elif mode=='depth_component':
                    from .depth_component import MainPegDepthComponent
                    policy=MainPegDepthComponent(geometry)
                else:
                    from .depth_box_component import MainObjectDepthComponents
                    policy=MainObjectDepthComponents(geometry)
                policy.reset(eid)
                folder = root / 'perception' / eid
                folder.mkdir(parents=True)
                bits, costs, rows, points, changed_count = [], [], [], [], 0
                write(root / 'pipeline_state.json', dict(status='RUNNING', stage='PAIRED_CAUSAL_CANDIDATE',
                      episode=eid, completed_episodes=len(progress), run_id=log.run_id))
                try:
                    with h5py.File(e['raw_path'], 'r') as f, np.load(old / 'masks.npz', allow_pickle=False) as stored:
                        for t in range(e['actions']+1):
                            cameras, data, surfaces = geometry.from_h5(f, t, eid)
                            base = dict(zip(CAMERAS, np.unpackbits(stored['mask_bits'][t]).reshape(2,2,128,128).astype(bool)))
                            tick = time.perf_counter()
                            out, event = policy.process(cameras, data, surfaces, base)
                            torch.cuda.synchronize()
                            costs.append((time.perf_counter()-tick)*1000)
                            if mode=='reanchor':assert np.array_equal(out['base_camera'], base['base_camera'])
                            else:assert np.array_equal(out['hand_camera'],base['hand_camera'])
                            if mode!='depth_component_both':assert np.array_equal(out['base_camera'][1],base['base_camera'][1])
                            assert np.array_equal(out['hand_camera'][1],base['hand_camera'][1])
                            packed = np.packbits(np.stack([out[c] for c in CAMERAS]).ravel())
                            bits.append(packed)
                            trace = root / 'traces' / eid / f't{t:04d}'
                            if np.array_equal(packed, stored['mask_bits'][t]):
                                trace.parent.mkdir(parents=True, exist_ok=True)
                                trace.symlink_to(baseline / 'traces' / eid / f't{t:04d}', target_is_directory=True)
                                rows.extend(by_t[t])
                                points.extend(points_by_t[t])
                            else:
                                changed_count += 1
                                # Reference arrays enter only post-decision evaluation.
                                refs = {c: seg[t, i] if exact[t] else None for i, c in enumerate(CAMERAS)}
                                graph, _, new_rows = trace_geometry(geometry, cameras, data, surfaces, out, refs,
                                      [rr['object_ids']['peg'], rr['object_ids']['box']], rr['robot_ids'], trace)
                                validate_graph(graph)
                                for r in new_rows:
                                    r.update(parent_id=e['parent_id'], branch=e['branch'], source=e['source'], terminal_observation=t==e['actions'])
                                rows.extend(new_rows)
                                if exact[t]:
                                    with np.load(trace / 'support.npz', allow_pickle=False) as arrays:
                                        points.extend(updated_points(arrays, read(trace / 'trace.json'), refs, rr,
                                                                      cameras, meshes, poses, t, e))
                                else:
                                    points.extend(dict(episode=eid, parent_id=e['parent_id'], timestep=t,
                                          object_id=oid, points=int((graph.node_type==oid+3).sum()),
                                          reference_kind='GT_UNSUPPORTED', correct=None, robot=None,
                                          wrong_object=None, background=None) for oid in (0,1))
                    np.savez_compressed(folder / 'masks.npz', mask_bits=np.asarray(bits), correction_ms=np.asarray(costs))
                    np.savez_compressed(folder / 'correction_masks.npz',
                          raw_proposal_bits=np.asarray(policy.proposal_masks, np.uint8).reshape(-1,2048),
                          refined_proposal_bits=np.asarray(policy.validated_proposal_masks, np.uint8).reshape(-1,2048),
                          active_raw_bits=np.asarray([x[1] for x in policy.active_raw_masks], np.uint8).reshape(-1,2048),
                          active_raw_timestep=np.asarray([x[0] for x in policy.active_raw_masks], np.int64))
                    write(folder / 'events.json', policy.events)
                    write(folder / 'stage_rows.json', rows)
                    write(folder / 'representative_points.json', points)
                    receipt = dict(episode_id=eid, parent_id=e['parent_id'], frames=len(bits),
                          raw_sha256=e['raw_sha256'], masks_sha256=sha(folder / 'masks.npz'),
                          rows_sha256=sha(folder / 'stage_rows.json'), corrected_reference=str(rrpath),
                          baseline_receipt=str(old / 'receipt.json'), changed_frames=changed_count,
                          terminal_included=True, causal_episode_camera_reset=True, student_GT_used=False,
                          correction_ms_mean=float(np.mean(costs)), correction_ms_p95=float(np.percentile(costs,95)))
                    write(folder / 'receipt.json', receipt)
                    progress.append(receipt)
                    write(root / 'progress.json', progress)
                    print('candidate', partition, len(progress), '/', len(entries), eid, 'changed', changed_count, flush=True)
                finally:
                    policy.close()
            from .aggregate_development import run as aggregate
            aggregate(root)
            write(root / 'candidate_summary.json', dict(status='PAIRED_CANDIDATE_FINISHED', partition=partition,
                  episodes=len(progress), parents=len(parents), frames=sum(r['frames'] for r in progress),
                  changed_frames=sum(r['changed_frames'] for r in progress), rows=progress,
                  correction_timing_is_not_end_to_end_latency=True))
            log.link_output(root / 'candidate_summary.json', kind='metrics')
            write(root / 'pipeline_state.json', dict(status='SUBSTAGE_FINISHED', stage='PAIRED_CANDIDATE_FINISHED', run_id=log.run_id))
        except BaseException as exc:
            write(root / 'pipeline_state.json', dict(status='FAILED', error=type(exc).__name__, message=str(exc), run_id=log.run_id))
            raise


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root', type=Path);p.add_argument('baseline', type=Path);p.add_argument('corrected', type=Path)
    p.add_argument('--partition', choices=('development','internal_test','independent_validation'), default='development')
    p.add_argument('--unrefined', action='store_true')
    p.add_argument('--mode',choices=('reanchor','depth_component','depth_component_both'),default='reanchor')
    p.add_argument('--robot-memo',action='store_true');p.add_argument('--reuse-completed',type=Path)
    a=p.parse_args();run(a.root,a.baseline,a.corrected,a.partition,not a.unrefined,a.mode,a.robot_memo,a.reuse_completed)
