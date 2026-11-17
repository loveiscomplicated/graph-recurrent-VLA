"""Frozen temporal branches and parameter-free Object21/State38 composition."""

import torch

from graph.gated_mpnn import assemble_state38
from graph.pose_geometry import geometric_endpoint, hole_frame_vector, unit_quaternion
from graph.temporal_gate import correct_object21
from graph.temporal.insertion import StreamingInsertionEstimator


def assemble_hybrid(mamba, gate, frozen, *, endpoint_method='geometric', hole_rotation='mamba'):
    """Combine aligned Object21 tensors in one coordinate frame, without GT.

    Method A (geometric) uses the Peg's local +X half-length. Method B
    (preserve_endpoint) retains Mamba's endpoint: r_new = r_old + h_old - h_new.
    Positions can be world or TCP-relative, but all three inputs must agree.
    """
    if mamba.shape[-1] != 21 or gate.shape != mamba.shape or frozen.shape != mamba.shape:
        raise ValueError('Expected aligned Mamba, Gate and Frozen Object21[...,21]')
    if not all(torch.isfinite(value).all() for value in (mamba, gate, frozen)):
        raise ValueError('Object21 inputs must be finite')
    if endpoint_method not in ('geometric', 'preserve_endpoint'):
        raise ValueError('endpoint_method must be geometric or preserve_endpoint')
    if hole_rotation not in ('mamba', 'gnn'):
        raise ValueError('hole_rotation must be mamba or gnn')
    rotation_source = mamba if hole_rotation == 'mamba' else frozen
    # Build without in-place slice writes: Method A must also support gradients
    # into the Peg head during relative-geometry fine-tuning.
    peg = torch.cat((mamba[..., :3], unit_quaternion(mamba[..., 3:7])), dim=-1)
    hole = torch.cat((gate[..., 10:13], unit_quaternion(rotation_source[..., 13:17])), dim=-1)
    result = torch.cat((peg, mamba[..., 7:10], hole, frozen[..., 17:21]), dim=-1)
    if endpoint_method == 'geometric':
        relative = geometric_endpoint(result)
    else:
        relative = mamba[..., 7:10] + mamba[..., 10:13] - gate[..., 10:13]
    return torch.cat((peg, relative, hole, frozen[..., 17:21]), dim=-1)


def world_object21(object21, tcp_world):
    """Translate the two pose positions to world; vector and wxyz axes stay fixed."""
    result = object21.clone()
    result[..., :3] += tcp_world
    result[..., 10:13] += tcp_world
    return result


class StreamingHybridEstimator:
    """Run each pretrained branch on its own original observations and memory.

    One observed frame per call, with explicit contiguous frame identity. IDs
    may be interleaved; reset/discard both branches together at real boundaries.
    Gate quality17 and Mamba quality25 retain their distinct trained schemas.
    """

    def __init__(self, mamba, gate):
        self.mamba = StreamingInsertionEstimator(mamba.eval().requires_grad_(False))
        self.gate = gate.eval().requires_grad_(False)
        self.episodes = {}

    def reset(self, episode_id):
        """Clear branch caches, Gate memory, TCP motion history and frame clock."""
        self.mamba.reset(episode_id)
        self.episodes[episode_id] = {'gate_world': None, 'next_frame': 0}

    def discard(self, episode_id):
        """Release all temporal state belonging to a completed episode."""
        self.mamba.discard(episode_id)
        del self.episodes[episode_id]

    @torch.inference_mode()
    def step(self, observation, *, episode_id, frame_index):
        """Consume only current prediction/proprioception, then assemble all variants."""
        allowed = {'object21', 'quality25', 'quality17', 'tcp_world', 'tcp_quaternion',
                   'gripper_qpos', 'robot15', 'freshness2'}
        required = {'object21', 'quality25', 'quality17', 'tcp_world'}
        if set(observation) - allowed or required - set(observation):
            raise ValueError('Expected current observation fields only; GT/labels are forbidden')
        if episode_id not in self.episodes:
            raise ValueError('Call reset(episode_id) before streaming')
        state = self.episodes[episode_id]
        if type(frame_index) is not int or frame_index != state['next_frame']:
            raise ValueError(f'Expected next frame {state["next_frame"]} for {episode_id}')
        frozen = observation['object21']
        tcp = observation['tcp_world']
        quality = observation['quality17']
        if frozen.shape != (1, 21) or tcp.shape != (1, 3) or quality.shape != (1, 17):
            raise ValueError('Streaming hybrid expects object21[1,21], tcp_world[1,3], quality17[1,17]')
        if observation['quality25'].shape != (1, 25):
            raise ValueError('Streaming hybrid expects quality25[1,25]')
        if not torch.equal(quality, observation['quality25'][:, 8:]):
            raise ValueError('Gate and Mamba quality must describe the same BOX/HOLE observations')
        branch_observation = {key: value for key, value in observation.items() if key != 'quality17'}
        mamba_result = self.mamba.step(branch_observation, episode_id)
        gate_world, alpha = self.gate.step(frozen[0, 10:13], tcp[0], quality[0], state['gate_world'])
        state.update(gate_world=gate_world, next_frame=frame_index + 1)
        gate = correct_object21(frozen, gate_world[None] - tcp)
        mamba = mamba_result['object21']
        predictions = {'gnn': frozen, 'learned_gate': gate, 'mamba': mamba}
        for variant, rotation in (('hybrid_h1', 'mamba'), ('hybrid_h2', 'gnn')):
            for suffix, method in (('a', 'geometric'), ('b', 'preserve_endpoint')):
                predictions[f'{variant}_{suffix}'] = assemble_hybrid(
                    mamba, gate, frozen, endpoint_method=method, hole_rotation=rotation)
        primary = predictions['hybrid_h1_a']
        world = world_object21(primary, tcp)
        result = {'predictions': predictions, 'object21': primary, 'world_object21': world,
                  'relative_hole_frame': hole_frame_vector(primary[:, 7:10], primary[:, 13:17]),
                  'gate_alpha': alpha, 'observation_age_frames': mamba_result['observation_age_frames']}
        if 'robot15' in observation:
            result['state38'] = assemble_state38(observation['robot15'], primary, observation['freshness2'])
        return result
