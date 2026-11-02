"""Crash-safe, fail-closed research coordinator. Run in task-owned tmux.

Only implemented stages can complete. Missing research never becomes NO_GO or
fabricated metric files. Source changes invalidate completed stage receipts.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

from graph.task10i_stage4a.run_pipeline import read, write, sha, STAGE3

PREVIOUS = Path('/workspace/task10i_stage4a_20261031_v5/run')
SOURCE = Path(__file__).resolve().parents[2]


def source_identity():
    files = {str(p.relative_to(SOURCE)): sha(p) for folder in
             ('graph', 'visual', 'utils', 'envs') for p in (SOURCE / folder).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.toml', '.yaml', '.json')}
    return dict(files=files, digest=hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest())


def verify_inputs(root):
    disk = shutil.disk_usage(root)
    # Budget for compact queries, masks, pair embeddings and selected media.
    budget = dict(source_copy_gb=.02, query_logits_gb=12, masks_gb=8,
                  benchmark_gb=8, media_gb=2, reserve_free_gb=20)
    write(root / 'storage_budget.json', dict(budget=budget, free_bytes=disk.free,
          minimum_free_bytes=50 * 2**30, duplicate_raw_copy=False))
    if disk.free < 50 * 2**30:
        raise RuntimeError('BLOCKED_STORAGE: need 50GiB free; no existing data deleted')
    old = read(PREVIOUS / 'protocol.json')
    reserve = old['train_reserve']
    assert len(reserve) == len(set(reserve)) == 66
    ordered = sorted(reserve, key=lambda p: hashlib.sha256(
        ('task10i-stage4a2-v1:' + p).encode()).hexdigest())
    partition = dict(development=ordered[:36], internal_holdout=ordered[36:],
                     historical_development=old['development'],
                     historical_internal=old['internal_test'],
                     historical_validation=old['independent_validation'],
                     hash_rule="sha256('task10i-stage4a2-v1:'+parent_id)",
                     internal_pixels_opened=False, source_protocol_sha256=sha(PREVIOUS / 'protocol.json'))
    assert not set(ordered) & set(sum((old[k] for k in
        ('development', 'internal_test', 'independent_validation')), []))
    freeze = root / 'dataset_partition_manifest.json'
    if freeze.exists():
        assert read(freeze) == partition
    else:
        write(freeze, partition)
    receipt = read(STAGE3 / 'stage4_raw_dataset_receipt.json')
    assert sha(STAGE3 / 'stage4_recache_input_manifest.json') == receipt['recache_manifest_sha256']
    entries = read(STAGE3 / 'stage4_recache_input_manifest.json')['unique_sources']
    checks = []
    for e in entries.values():
        for kind in ('raw', 'gt'):
            actual = sha(e[kind + '_path'])
            assert actual == e[kind + '_sha256'], e[kind + '_path']
            checks.append(dict(path=e[kind + '_path'], sha256=actual))
    import torch
    packages = {n: importlib.metadata.version(n) for n in
                ('torch', 'torchvision', 'transformers', 'tokenizers', 'numpy', 'mani_skill', 'sapien')}
    write(root / 'input_and_runtime_verification.json', dict(status='PASS', packages=packages,
          executable=sys.executable, gpu=torch.cuda.get_device_name(0),
          input_files=checks, simulator_executed=False, cuda_physx_executed=False,
          historical_roots_verified={str(p): p.exists() for p in [PREVIOUS,
              Path('/workspace/task10i_stage4a_finalization_20261101_v5/run'),
              Path('/workspace/task10i_stage4a_completion_20261101_v1/run'),
              Path('/workspace/task10i_stage4a_posthoc_20261101_v3/run')]}))
    return ['dataset_partition_manifest.json', 'storage_budget.json', 'input_and_runtime_verification.json']


def trace_baseline(root):
    """Observe actual module forwards and Python call paths without extra model calls."""
    import numpy as np
    import torch
    from torch.nn.modules.module import register_module_forward_hook
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from graph.task10i_stage4a.forward_equivalence import run as equivalence
    counts, call_counts, module_rows, memory_rows = Counter(), Counter(), [], []
    context = {}
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    original_process = OnlineVisionAdapter.process
    original_single = SAM2VideoPredictor._run_single_frame_inference

    def process(self, rgb, episode_identity, timestep):
        context.update(episode_id=episode_identity, timestep=timestep,
                       adapter_identity=id(self))
        return original_process(self, rgb, episode_identity, timestep)

    def describe(value):
        if torch.is_tensor(value):
            return dict(shape=list(value.shape), dtype=str(value.dtype))
        if isinstance(value, dict):
            return {str(k): describe(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [describe(v) for v in value]
        return type(value).__name__

    def observe(module, args, output):
        name = type(module).__name__
        if name in ('GroundingDinoForObjectDetection', 'Dinov2Model', 'SAM2VideoPredictor'):
            counts[name] += 1
            module_rows.append(dict(**context, model_class=name, output=describe(output)))

    def single(self, *args, **kwargs):
        result = original_single(self, *args, **kwargs)
        # Official internal outputs; no invented confidence or state writes.
        row = dict(**context, outputs=describe(result))
        if isinstance(result, tuple) and isinstance(result[0], dict):
            row['signals'] = {k: v.detach().float().cpu().reshape(-1).tolist()
                              for k, v in result[0].items()
                              if torch.is_tensor(v) and v.numel() <= 8}
        memory_rows.append(row)
        return result

    def profile(frame, event, arg):
        if event == 'call':
            filename = frame.f_code.co_filename
            if any(s in filename for s in ('online_vision_adapter.py', 'initialization.py',
                                           'reanchor_v1.py', 'modeling_dinov2.py')):
                key = f'{filename}:{frame.f_code.co_firstlineno}:{frame.f_code.co_name}'
                call_counts[key] += 1

    handle = register_module_forward_hook(observe)
    OnlineVisionAdapter.process = process
    SAM2VideoPredictor._run_single_frame_inference = single
    sys.setprofile(profile)
    try:
        equivalence(root / 'baseline_trace', PREVIOUS,
                    Path('/workspace/task10i_stage4a_20261031_v3/run'))
    finally:
        sys.setprofile(None)
        handle.remove()
        OnlineVisionAdapter.process = original_process
        SAM2VideoPredictor._run_single_frame_inference = original_single
        write(root / 'execution_call_graph.json', dict(module_forward_counts=dict(counts),
              python_calls=dict(call_counts), actual_module_calls=module_rows,
              dinov2_forwards=counts['Dinov2Model'],
              scope='101-action-frame historical fixture, not population evidence'))
        write(root / 'sam2_memory_state_trace.json', memory_rows)
    proof = read(root / 'baseline_trace/new_host_forward_equivalence.json')
    assert proof['status'] == 'PASS' and proof['frames'] == 101
    write(root / 'baseline_forward_equivalence.json', proof)
    write(root / 'dinov2_call_trace.json', dict(baseline_forward_count=counts['Dinov2Model'],
          scope='baseline fixture only', matching_input_count=0 if counts['Dinov2Model'] == 0 else None,
          separate_legacy_implementation='visual/scripts/reanchor_v1.py:Dino',
          baseline_guard='OnlineVisionAdapter has no DINOv2 dependency/call; found slots never re-detected',
          enabled_candidate_evaluated=False))
    return ['baseline_forward_equivalence.json', 'execution_call_graph.json',
            'dinov2_call_trace.json', 'sam2_memory_state_trace.json']


def installed_contract(root):
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter, ObservedFrames
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from transformers import AutoProcessor
    paths = read('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json')
    proc = AutoProcessor.from_pretrained(paths['dino'], local_files_only=True)
    contracts = dict(dino_processor=proc.to_dict() if hasattr(proc, 'to_dict') else str(proc),
                     image_processor=proc.image_processor.to_dict(),
                     prompts={s: proc.tokenizer.tokenize(s + '.') for s in
                              ('red and white block', 'red and white peg', 'peg', 'rectangular rod',
                               'cardboard box', 'box', 'box with a hole')},
                     sam_config='configs/sam2.1/sam2.1_hiera_s.yaml',
                     baseline_dinov2='NOT_IN_BASELINE_CALL_PATH')
    sources = {}
    for obj in (GDino, SAM2Tracker, OnlineVisionAdapter, ObservedFrames,
                SAM2VideoPredictor._run_single_frame_inference,
                SAM2VideoPredictor.add_new_points_or_box,
                SAM2VideoPredictor.reset_state):
        p = Path(inspect.getsourcefile(obj))
        lines, start = inspect.getsourcelines(obj)
        sources[obj.__qualname__] = dict(path=str(p), sha256=sha(p), first_line=start,
                                         source=''.join(lines))
    write(root / 'model_preprocessing_contract.json', contracts)
    write(root / 'installed_source_audit.json', sources)
    old = read(PREVIOUS / 'model_identity.json')
    assert sha(paths['sam']) == old['sam_sha256']
    for filename, digest in old['dino_files'].items():
        assert sha(Path(paths['dino']) / filename) == digest
    write(root / 'three_model_identity.json', dict(grounding_dino=dict(
          model='IDEA-Research/grounding-dino-tiny', files=old['dino_files']),
          sam2=dict(model='SAM2.1 Hiera Small', checkpoint=paths['sam'], sha256=old['sam_sha256']),
          dinov2=dict(model='facebook/dinov2-small', status='WEIGHTS_NOT_YET_LOCATED_OR_HASHED',
                      baseline_executed=False)))
    return ['model_preprocessing_contract.json', 'installed_source_audit.json', 'three_model_identity.json']


STAGES = [('INPUT_AND_RUNTIME_VERIFICATION', verify_inputs),
          ('INSTALLED_SOURCE_CONTRACT', installed_contract),
          ('BASELINE_FORWARD_EQUIVALENCE_AND_EXECUTION_TRACE', trace_baseline)]


def main(root):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = source_identity()
        if (root / 'source_identity.json').exists():
            assert read(root / 'source_identity.json') == identity, 'SOURCE_CHANGED: new root required'
        else:
            write(root / 'source_identity.json', identity)
        with ExperimentLogger(team='integration', model='Task10I_Stage4A2', seed=42,
                              job_type='evaluation', root=root / 'experiments',
                              config=dict(source_digest=identity['digest'], production_changes=False,
                                          stage4b_authorized=False)) as log:
            for name, function in STAGES:
                receipt = root / 'stages' / (name + '.json')
                if receipt.exists():
                    previous = read(receipt)
                    assert previous['source_digest'] == identity['digest']
                    assert all(sha(root / p) == h for p, h in previous['artifacts'].items())
                    continue
                write(root / 'pipeline_state.json', dict(status='RUNNING', stage=name,
                      pid=os.getpid(), run_id=log.run_id, whole_goal_complete=False))
                print(name, flush=True)
                try:
                    artifacts = function(root)
                    write(receipt, dict(status='PASS', source_digest=identity['digest'],
                          artifacts={p: sha(root / p) for p in artifacts}, run_id=log.run_id))
                except BaseException as exc:
                    write(root / 'pipeline_state.json', dict(status='FAILED', stage=name,
                          error=type(exc).__name__, message=str(exc), run_id=log.run_id,
                          whole_goal_complete=False, partial_outputs_preserved=True))
                    raise
            write(root / 'pipeline_state.json', dict(status='AUDIT_PHASE_COMPLETE',
                  next_stage='DINO_RAW_PROPOSAL_AUDIT_AND_DINOV2_FEATURE_AUDIT',
                  whole_goal_complete=False, run_id=log.run_id,
                  candidate_selected=False, independent_final_executed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
