"""Current physics/controller audits and recorded-prefix vision probes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from vla.data.atomic import atomic
from vla.data.legacy_io import save_npz
from vla.perception.online import OnlineVisionAdapter

def numpy(value):
    return value.detach().cpu().numpy().copy()

def controller_semantics(config):
    """URDF installation directory is host-specific; controller values are exact."""
    value = json.loads(json.dumps(config))
    for item in value.values():
        if 'urdf_path' in item:
            item['urdf_path'] = Path(item['urdf_path']).name
    return value

class PhysicsAudit:
    """Observe actual scene.step calls, including any accidental inference step."""
    def __init__(self, env):
        self.task = env.unwrapped
        self.count = 0
        self.scene = self.task.scene
        self.original = self.scene.step

        def step(*args, **kwargs):
            result = self.original(*args, **kwargs)
            self.count += 1
            return result

        self.scene.step = step

    def token(self):
        return self.count, int(self.task.elapsed_steps.item())

    def unchanged(self, before):
        if self.token() != before:
            raise ValueError('physics advanced during observation/inference')

    def stepped(self, before):
        if self.token() != (before[0]+5, before[1]+1):
            raise ValueError('one action must advance exactly 5 physics / 1 control steps')

    def close(self):
        self.scene.step = self.original

def controller_audit(env):
    """Record entry and actual installed preprocessing, without modifying action."""
    controller = env.unwrapped.agent.controller
    record = {}
    original = controller.set_action

    def set_action(action):
        record['controller_input'] = numpy(action)[0]
        return original(action)

    controller.set_action = set_action
    for name, child in controller.controllers.items():
        original_preprocess = child._preprocess_action

        def preprocess(action, _original=original_preprocess, _name=name):
            result = _original(action)
            record[_name+'_scaled_input'] = numpy(result)[0]
            return result

        child._preprocess_action = preprocess
    return record

def validate_controller_command(action, actual):
    """Audit the installed collection convention, including its rotation sign."""
    np.testing.assert_array_equal(actual['controller_input'], action)
    arm = action[:6].copy()
    arm[:3] = np.clip(arm[:3], -1, 1)*.1
    arm[3:] /= max(float(np.linalg.norm(arm[3:])), 1.)
    arm[3:] *= -.1
    gripper = (np.clip(action[6:], -1, 1)+1)*.025-.01
    np.testing.assert_allclose(actual['arm_scaled_input'], arm, atol=2e-8, rtol=1e-6)
    np.testing.assert_allclose(actual['gripper_scaled_input'], gripper, atol=1e-8, rtol=0)

def reset_diagnostics(env):
    """Evaluation-only reset identity; return value never enters policy inputs."""
    from envs.maniskill import get_gt_state
    gt = get_gt_state(env)
    data = np.concatenate([numpy(getattr(gt, name)).ravel() for name in
                           ('qpos', 'peg_pose', 'box_pose', 'hole_pose', 'peg_half_sizes')])
    return hashlib.sha256(data.tobytes()).hexdigest()

def compare_saved_sequence(rgb, detector, tracker, output):
    """Recorded RGB is revealed one frame at a time; separate from live smoke."""
    from graph.perception.offline import causal_masks
    online = OnlineVisionAdapter(detector, tracker)
    values = []
    try:
        online.reset('recorded_prefix')
        for t in range(len(rgb)):
            values.append(online.process(rgb[t], 'recorded_prefix', t)[0])
        initialization = online.found.copy()
    finally:
        online.close()
    offline, meta = causal_masks(rgb, detector, tracker)
    current = np.stack(values)
    differences = np.count_nonzero(current != offline, axis=(2,3))
    presence_diff = current.any(axis=(2,3)) != offline.any(axis=(2,3))
    save_npz(output/'recorded_vision_comparison.npz', online=current, offline=offline,
             differing_pixels=differences, presence_differences=presence_diff)
    result = dict(status='PASS' if not presence_diff.any() and initialization == meta['initialization'] else 'CONTRACT_FAILURE',
                  frames=len(rgb), differing_pixels=differences.tolist(), presence_differences=presence_diff.tolist(),
                  initialization_equal=initialization == meta['initialization'],
                  scope='recorded real RGB revealed incrementally; not live simulator evidence')
    # A second independent stream sees a changed suffix only after the prefix.
    changed = OnlineVisionAdapter(detector, tracker)
    try:
        changed.reset('suffix_probe')
        prefix_values = []
        for t in range(len(rgb)):
            image = rgb[t] if t < len(rgb)-1 else np.zeros_like(rgb[t])
            prefix_values.append(changed.process(image, 'suffix_probe', t)[0])
        prefix_diff = int(np.count_nonzero(np.stack(prefix_values)[:-1] != current[:-1]))
        result['changed_suffix_prefix_differing_pixels'] = prefix_diff
        if prefix_diff:
            result['status'] = 'CONTRACT_FAILURE'
    finally:
        changed.close()
    result['late_object_actual_sam2'] = late_object_probe(rgb, initialization, detector, tracker, output)
    if result['late_object_actual_sam2']['status'] == 'CONTRACT_FAILURE':
        result['status'] = 'CONTRACT_FAILURE'
    atomic(output/'recorded_vision_comparison.json', result)
    return result

def late_object_probe(rgb, initialization, detector, tracker, output):
    """Controlled prompt fixture tests real SAM2 memory, not detection quality."""
    from copy import deepcopy
    from unittest.mock import patch
    from graph.perception.offline import causal_masks
    if any(initialization.get(name,{}).get('frame') != 0 for name in ('peg','box')):
        return dict(status='NOT_EXECUTED', reason='fixture needs both RGB detections at frame zero')
    fixture = np.stack([rgb[t % len(rgb)] for t in range(7)])

    def initial(frames, detector):
        result = deepcopy(initialization)
        result['peg']['frame'] = 0
        result['box']['frame'] = 5 if len(frames) >= 6 else None
        if len(frames) < 6:
            result['box']['box'] = None
        return result

    online = OnlineVisionAdapter(detector, tracker)
    values = []
    retained = True
    try:
        online.reset('late_object_fixture')
        with patch('vla.perception.online.detect_initial', initial):
            original_state = original_memory = None
            old_keys = set()
            for t, image in enumerate(fixture):
                values.append(online.process(image, 'late_object_fixture', t)[0])
                state = online.states[(1,)]
                memory = state['output_dict_per_obj'][0]
                keys = set(memory['cond_frame_outputs']) | set(memory['non_cond_frame_outputs'])
                if t == 4:
                    original_state, original_memory, old_keys = state, memory, keys
                if t == 5:
                    retained = state is original_state and memory is original_memory and old_keys <= keys
            groups_equal = set(online.states) == {(1,),(2,)}
    finally:
        online.close()
    with patch('graph.perception.offline.detect_initial', initial):
        offline, _ = causal_masks(fixture, detector, tracker)
    values = np.stack(values)
    differences = np.count_nonzero(values != offline, axis=(2,3))
    presence_equal = np.array_equal(values.any(axis=(2,3)), offline.any(axis=(2,3)))
    success = retained and groups_equal and presence_equal and not values[:5,1].any()
    save_npz(output/'late_object_sam2_fixture.npz', online=values, offline=offline,
             differing_pixels=differences)
    return dict(status='PASS' if success else 'CONTRACT_FAILURE', previous_object_memory_retained=retained,
                object_groups_equal=groups_equal, presence_equal=presence_equal,
                differing_pixels=differences.tolist(),
                scope='REAL_SAM2_WITH_CONTROLLED_PROMPT_FIXTURE; seven repeated recorded RGB frames; '
                      'past frame-zero DINO boxes; delayed box prompt at t5; not live detection-quality evidence')
