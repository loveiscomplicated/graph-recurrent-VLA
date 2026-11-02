"""New-host verification only, using the unchanged frozen baseline.run forward.

The observer runs after CurrentGeometry.build returns. References are never
passed to detector, tracker, geometry, or sampling. No dataset job is resumed.
"""
import argparse
import fcntl
import os
from pathlib import Path
import time

import numpy as np
import torch

from .run_pipeline import CAMERAS, STAGE3, read, sha, write


def run(root, previous, original):
    from utils.experiment_logger import ExperimentLogger
    from graph.plan_a_bc.geometry import CurrentGeometry
    from graph.plan_a_bc.cache import GRAPH_KEYS
    from .baseline import run as frozen_baseline

    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    with (root / 'runner.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = dict(previous_root=str(previous), original_baseline=str(original),
                      partition='preregistered_development_only', validation_accessed=False,
                      tolerance=0, research_resumed=False, production_changes=False,
                      observer='post-build, no RNG calls, no output modification')
        with ExperimentLogger(team='integration', model='Task10I_Stage4A_new_host_forward_equivalence',
                              seed=42, job_type='evaluation', root=root / 'experiments', config=config) as log:
            write(root / 'run_identity.json', dict(run_id=log.run_id, pid=os.getpid(),
                  source_root=str(source), instance_id=58576054, executable=os.sys.executable))
            started = time.perf_counter()
            observer_rows = []
            original_build = CurrentGeometry.build
            try:
                frozen = read(previous / 'source_freeze.json')['files']
                changed = [p for p, digest in frozen.items() if sha(source / p) != digest]
                assert not changed, changed
                write(root / 'source_freeze.json', dict(files=frozen,
                      added_verification_helper_sha256=sha(Path(__file__)),
                      frozen_files_checked=len(frozen), frozen_file_changes=changed))
                protocol = read(previous / 'protocol.json')
                write(root / 'protocol.json', protocol)
                entries = read(STAGE3 / 'stage4_recache_input_manifest.json')['unique_sources'].values()
                entry = min((e for e in entries if e['source'] == 'existing_full_task'
                             and e['parent_id'] in set(protocol['development'])),
                            key=lambda e: int(e['episode_id']))
                eid = str(entry['episode_id'])
                assert entry['split'] == 'train'
                assert sha(entry['raw_path']) == entry['raw_sha256']
                prior = previous / 'perception' / eid
                receipt = read(prior / 'receipt.json')
                assert sha(prior / 'masks.npz') == receipt['masks_sha256']
                assert sha(prior / 'stage_rows.json') == receipt['rows_sha256']
                old_masks = np.load(prior / 'masks.npz', allow_pickle=False)['mask_bits']
                expected_settings = read(previous / 'model_identity.json')['settings']
                settings = dict(deterministic=torch.are_deterministic_algorithms_enabled(),
                                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                                cudnn_tf32=torch.backends.cudnn.allow_tf32)
                assert settings == expected_settings, (settings, expected_settings)
                assert torch.__version__ == read(previous / 'model_identity.json')['torch_version']
                runtime = read(Path('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json'))
                model_identity = read(previous / 'model_identity.json')
                assert sha(runtime['sam']) == model_identity['sam_sha256']
                for p, digest in model_identity['dino_files'].items():
                    assert sha(Path(runtime['dino']) / p) == digest, p
                write(root / 'input_identity.json', dict(episode_id=eid, parent_id=entry['parent_id'],
                      raw_path=entry['raw_path'], raw_sha256=entry['raw_sha256'], actions=entry['actions'],
                      evaluated_timesteps=list(range(entry['actions'])),
                      terminal_timestep_not_evaluated=entry['actions'],
                      reason='Original F1 graph cache contains action frames only; no terminal graph',
                      model_identity=model_identity, settings=settings,
                      previous_receipt=receipt, original_proof_sha256=sha(original / 'baseline_forward_equivalence.json')))
                (root / 'actual_frames').mkdir()

                def observe(self, cameras, data, surfaces, masks):
                    graph, points, gate = original_build(self, cameras, data, surfaces, masks)
                    t = int(data['timestep'])
                    packed = np.packbits(np.stack([masks[c] for c in CAMERAS]).ravel())
                    expected_mask = np.unpackbits(old_masks[t]).reshape(2, 2, 128, 128).astype(bool)
                    actual_mask = np.stack([masks[c] for c in CAMERAS]).astype(bool)
                    per_camera_object = {c: {name: int(np.count_nonzero(actual_mask[i, j] != expected_mask[i, j]))
                                         for j, name in enumerate(('peg', 'box'))} for i, c in enumerate(CAMERAS)}
                    path = previous / 'traces' / eid / f't{t:04d}' / 'support.npz'
                    arrays = {k: getattr(graph, k).numpy().copy() for k in GRAPH_KEYS}
                    differences = {}
                    bitwise = {}
                    with np.load(path, allow_pickle=False) as target:
                        for k in GRAPH_KEYS:
                            a, b = arrays[k], target[k]
                            same_schema = a.shape == b.shape and a.dtype == b.dtype
                            differences[k] = int(np.count_nonzero(a != b)) if same_schema else -1
                            bitwise[k] = same_schema and a.tobytes() == b.tobytes()
                    previous_gate = read(path.parent / 'trace.json')['gate']
                    np.savez_compressed(root / 'actual_frames' / f't{t:04d}.npz', mask_bits=packed, **arrays)
                    observer_rows.append(dict(timestep=t,
                          camera_object_mask_pixel_differences=per_camera_object,
                          mask_packed_bytes_equal=bool(np.array_equal(packed, old_masks[t])),
                          graph_difference_values=differences, graph_bytes_equal=bitwise,
                          gate_equal=gate == previous_gate, previous_support_sha256=sha(path)))
                    return graph, points, gate

                # The pinned forward function remains unchanged; retain its own
                # original F1-cache comparison as an independent verification.
                CurrentGeometry.build = observe
                write(root / 'pipeline_state.json', dict(status='RUNNING',
                      stage='FORWARD_EQUIVALENCE_ONLY', run_id=log.run_id, pid=os.getpid()))
                frozen_baseline(root)
                f1_result = read(root / 'baseline_forward_equivalence.json')
                assert len(observer_rows) == entry['actions']
                paired_pass = all(r['mask_packed_bytes_equal'] and r['gate_equal']
                                  and all(r['graph_bytes_equal'].values()) for r in observer_rows)
                summary = dict(status='PASS' if f1_result['status'] == 'PASS' and paired_pass else 'FAIL',
                      run_id=log.run_id, instance_id=58576054, episode_id=eid,
                      parent_id=entry['parent_id'], frames=len(observer_rows),
                      f1_original_comparison=f1_result['status'],
                      previous_stage4a_comparison='PASS' if paired_pass else 'FAIL',
                      mask_different_frames=sum(not r['mask_packed_bytes_equal'] for r in observer_rows),
                      graph_different_frames=sum(not all(r['graph_bytes_equal'].values()) for r in observer_rows),
                      gate_different_frames=sum(not r['gate_equal'] for r in observer_rows),
                      torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0), settings=settings,
                      scope='one full historical action-frame sequence, both cameras, both objects, all five Graph tensors',
                      validation_accessed=False, terminal_included=False, research_resumed=False,
                      wall_seconds=time.perf_counter() - started, rows=observer_rows)
                write(root / 'new_host_forward_equivalence.json', summary)
                for name in ('source_freeze.json', 'input_identity.json', 'baseline_forward_equivalence.json',
                             'new_host_forward_equivalence.json'):
                    log.link_output(root / name, kind='metrics')
                log.log(dict(frames=len(observer_rows), mismatched_mask_frames=summary['mask_different_frames'],
                             mismatched_graph_frames=summary['graph_different_frames']))
                assert summary['status'] == 'PASS', 'new host output differs from preserved output'
                write(root / 'pipeline_state.json', dict(status='FORWARD_EQUIVALENCE_PASS', run_id=log.run_id,
                      research_resumed=False, whole_stage4a_complete=False))
                print('NEW_HOST_FORWARD_EQUIVALENCE PASS', eid, len(observer_rows), log.run_id, flush=True)
            except BaseException as exc:
                write(root / 'pipeline_state.json', dict(status='FAIL', error=type(exc).__name__,
                      message=str(exc), completed_observer_frames=len(observer_rows), run_id=log.run_id))
                if observer_rows:
                    write(root / 'partial_observer_rows.json', observer_rows)
                raise
            finally:
                CurrentGeometry.build = original_build


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('previous', type=Path)
    parser.add_argument('original', type=Path)
    args = parser.parse_args()
    run(args.root, args.previous, args.original)
