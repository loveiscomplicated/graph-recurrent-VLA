"""Opt-in execution-only Vision optimizations; prompts/weights/precision unchanged.

The original prefix initializer and SAM2 propagation remain authoritative. Cache
entries are raw detections, before any prefix-dependent object assignment. The
preflight fast path is valid only while prompts/objects/conditioning rows are
unchanged and there are no temporary prompt outputs waiting for consolidation.
"""
from copy import deepcopy
import hashlib

import numpy as np


SETTINGS = dict(detector_cache=True, preflight_reuse=True, compile_image_encoder=True,
                compiler_backend='aot_eager', compiler_mode='default', dtype='float32',
                autocast=False, resize='unchanged')
REVISION_ALLOWLIST = {
    'graph/integration_baseline/vision_speed.py',
    'graph/integration_baseline/direct_dataset.py',
    'graph/integration_baseline/run_control.py',
    'graph/integration_baseline/tests/test_vision_speed.py',
    'graph/integration_baseline/README.md',
}


def revision(run):
    """Read the opt-in, checksummed execution revision without changing config."""
    from pathlib import Path
    from graph.integration_baseline.direct_runtime import read, sha
    run = Path(run)
    pointer = run / 'execution_revision.json'
    if not pointer.exists():
        return None
    link = read(pointer)
    path = run / link['path']
    if not path.resolve().is_relative_to((run / 'execution_revisions').resolve()):
        raise ValueError('execution revision path outside its audit directory')
    if sha(path) != link['sha256']:
        raise ValueError('execution revision checksum mismatch')
    receipt = read(path)
    if receipt['settings'] != SETTINGS or receipt['status'] != 'VALIDATED':
        raise ValueError('unvalidated Vision optimization settings')
    if receipt['original_sources_sha256'] != sha(run / 'source_hashes.json'):
        raise ValueError('original source audit changed')
    if receipt['config_sha256'] != sha(run / 'config.yaml'):
        raise ValueError('frozen training config changed')
    proof_path = path.parent / 'verification.json'
    if sha(proof_path) != receipt['verification_sha256']:
        raise ValueError('optimization verification checksum mismatch')
    proof = read(proof_path)
    required={'masks','packets','actions','existing_cache'}
    if (proof['status'] != 'PASS' or set(proof['exact']) != required
            or any(value is not True for value in proof['exact'].values())
            or proof.get('compiler_backend') != SETTINGS['compiler_backend']
            or proof['speedup'] <= 1 or receipt['contract_tests_returncode'] != 0):
        raise ValueError('optimization requires output equivalence, speedup and contract tests')
    return receipt


def verify_revision_sources(run, before, after):
    receipt = revision(run)
    if receipt is None:
        return False
    changed = {p for p in set(before) | set(after) if before.get(p) != after.get(p)}
    if not changed.issubset(REVISION_ALLOWLIST) or after != receipt['source_hashes']:
        raise ValueError('sources differ from validated execution revision')
    return True


class MemoizedDetector:
    def __init__(self, detector):
        self.detector = detector
        self.reset_episode()

    def reset_episode(self):
        self.cache = {}
        self.calls = self.hits = self.misses = 0

    def __call__(self, rgb, prompt):
        array = np.ascontiguousarray(rgb)
        key = (array.dtype.str, array.shape, hashlib.sha256(array.tobytes()).digest(), prompt)
        self.calls += 1
        if key not in self.cache:
            self.cache[key] = deepcopy(self.detector(rgb, prompt))
            self.misses += 1
        else:
            self.hits += 1
        return deepcopy(self.cache[key])

    def statistics(self):
        return dict(requests=self.calls, actual_inferences=self.misses, cache_hits=self.hits)


def _preflight_signature(state):
    return (tuple(state['obj_ids']), tuple(
        (idx, tuple(sorted(outputs['cond_frame_outputs'])))
        for idx, outputs in state['output_dict_per_obj'].items()))


def guard_preflight(predictor):
    """Reuse a completed preflight, invalidate on new prompts/objects/reset.

This does not bypass per-frame image encoding, tracking, memory encoding,
temporal pointer normalization, mask generation or output resizing.
"""
    original = predictor.propagate_in_video_preflight
    counters = dict(executed=0, skipped=0)

    def preflight(state):
        signature = _preflight_signature(state)
        temporary = any(rows for obj in state['temp_output_dict_per_obj'].values()
                        for rows in obj.values())
        if (not temporary and state.get('_grvla_completed_preflight') == signature
                and len(state['obj_ids']) > 0):
            counters['skipped'] += 1
            return
        original(state)
        state['_grvla_completed_preflight'] = _preflight_signature(state)
        counters['executed'] += 1

    predictor.propagate_in_video_preflight = preflight
    return counters


def enable(detector, tracker, *, compile_image_encoder=True):
    """Compile the same FP32 image encoder with AOT Eager; fail on errors.

    No BF16/FP16, TF32 toggle, resizing, VOS algorithm replacement, threshold
    change, or automatic compilation fallback is introduced here.
    """
    import torch
    wrapped = MemoizedDetector(detector)
    counters = guard_preflight(tracker.predictor)
    if compile_image_encoder:
        encoder = tracker.predictor.image_encoder
        if next(encoder.parameters()).device.type != 'cuda':
            raise ValueError('image encoder compilation requires the validated CUDA device')
        if any(p.dtype != torch.float32 for p in encoder.parameters()):
            raise ValueError('the approved image encoder precision is FP32')
        encoder.forward = torch.compile(encoder.forward, backend=SETTINGS['compiler_backend'],
                                       mode='default', fullgraph=True, dynamic=False)
    tracker.grvla_speed_statistics = dict(preflight=counters, compiler=SETTINGS['compiler_backend'] if compile_image_encoder else None,
                                         precision='float32', resize_unchanged=True)
    return wrapped, tracker
