"""Single CPU ManiSkill subtask wrapper using uninterrupted expert prefixes."""
import copy
import json
import os
from pathlib import Path
import tomllib

import gymnasium as gym
import h5py
import numpy as np
import torch

from graph.prototype.data import reset_episode, state_at
from .conditions import ConditionContext
from .dataset import load_segments, load_audit
from .live import LiveSubtaskAccessor
from .specs import NAMES, SubtaskSpec, load_config

DEFAULT_INDEX = 'outputs/subtasks/segments/index.json'
ENV_CONFIG = Path(__file__).with_name('environment.toml')


class InvalidSubtaskReset(ValueError):
    """A selected source cannot furnish a valid independent start state."""


def tree_error(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError('state keys differ')
        return max((tree_error(actual[k], v) for k, v in expected.items()), default=0.)
    if torch.is_tensor(actual):
        actual = actual.detach().cpu().numpy()
    a, b = np.asarray(actual).reshape(-1), np.asarray(expected).reshape(-1)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite or misaligned state')
    return float(np.max(np.abs(a-b))) if a.size else 0.


class SubtaskEnv(gym.Wrapper):
    """Preserves ManiSkill batched observations/actions; returns scalar Gym flags.

    source_episode_id is local to the explicitly selected index file. The saved
    source_uid identifies that episode across datasets. A seed on a source reset
    is an assertion about its seed, never an instruction to randomize its objects.
    """
    def __init__(self, subtask='reach', *, index=DEFAULT_INDEX,
                 control_mode='pd_ee_delta_pose', obs_mode='state', max_steps=None,
                 render_mode=None, config=None):
        from envs.maniskill import make_peg_insertion_env
        if subtask not in NAMES:
            raise ValueError(subtask)
        with open(ENV_CONFIG, 'rb') as f:
            self.runtime = tomllib.load(f)
        self.index_path = str(Path(index).resolve()) if index is not None else None
        self.dataset = load_segments(index) if index is not None else None
        if self.dataset and self.dataset['control_mode'] != control_mode:
            raise ValueError('source and learner control modes must match')
        self.config = copy.deepcopy(config or (self.dataset['thresholds'] if self.dataset else load_config()))
        self.subtask = subtask
        self.subtask_spec = SubtaskSpec(subtask, self.config)
        self.align_spec = SubtaskSpec('align', self.config)
        self.budget = self.runtime['max_steps'][subtask] if max_steps is None else max_steps
        if not isinstance(self.budget, int) or self.budget < 1:
            raise ValueError('max_steps must be positive')
        self.config['timeout_steps'] = self.budget
        self.source = h5py.File(self.dataset['source'], 'r') if self.dataset else None
        self.original = h5py.File(self.dataset['initial_state_source'], 'r') if self.dataset else None
        try:
            if self.dataset and self.dataset.get('environment'):
                import mani_skill.envs  # noqa: F401
                from envs.maniskill.peg_insertion import _configure_macos_vulkan
                _configure_macos_vulkan()
                kwargs = dict(self.dataset['environment'], control_mode=control_mode,
                              obs_mode=obs_mode, render_mode=render_mode)
                if render_mode is not None:
                    kwargs['render_backend'] = 'cpu'
                original_cwd = Path.cwd()
                try:
                    wrapped = gym.make('PegInsertionSide-v1', **kwargs)
                finally:
                    os.chdir(original_cwd)
            else:
                wrapped = make_peg_insertion_env(control_mode=control_mode, obs_mode=obs_mode, render_mode=render_mode)
            # Ignore the registered full-task TimeLimit and full-task termination.
            super().__init__(wrapped.unwrapped)
        except Exception:
            self.close_sources()
            raise
        self._ready = False
        self.obs_mode = obs_mode

    def _update(self, t, contact_valid):
        self.live_state = self.accessor(self.env, contact_valid=contact_valid)
        self.metrics = self.subtask_spec.compute_metrics(self.live_state)
        self.context.update(self.metrics, t, self.config, self.align_spec.align_raw(self.metrics))

    def _info(self, base=None, *, success=False, reasons=()):
        info = dict(base or {})
        info.update(subtask=self.subtask, subtask_success=bool(success),
                    subtask_failure=bool(reasons), failure_reason=','.join(reasons) or None,
                    failure_reasons=list(reasons), subtask_step=self.learner_steps,
                    metrics=self.metrics, full_task_success=bool(self.metrics['official_success']),
                    reset_valid=True, start_state=self.start_ref,
                    initialization='expert_prefix' if self.segment is not None else 'fresh_reset',
                    prefix_steps=self.prefix_steps)
        return info

    def reset(self, *, seed=None, options=None):
        self._ready = False
        options = dict(options or {})
        if options.pop('subtask', self.subtask) != self.subtask:
            raise ValueError('create a new wrapper to select a different subtask')
        eid = options.pop('source_episode_id', None)
        if options:
            raise ValueError(f'unsupported reset options: {sorted(options)}')
        if eid is None and self.subtask != 'reach':
            raise ValueError('middle subtasks require explicit source_episode_id and index')
        self.segment = None
        self.prefix_steps = self.learner_steps = 0
        self.context = ConditionContext()
        self.max_prefix_observation_error = self.max_prefix_state_error = 0.
        if eid is None:
            obs, base_info = self.env.reset(seed=seed, options={'reconfigure': True})
            self.start_ref = dict(index=None, source_episode_id=None, seed=seed, state_index=0)
        else:
            if self.dataset is None:
                raise ValueError('source reset requires a dataset index')
            ep = next((e for e in self.dataset['episodes'] if e['episode_id'] == eid), None)
            if ep is None:
                raise ValueError(f'unknown episode {eid} in {self.index_path}')
            self.segment = next(s for s in ep['segments'] if s['subtask'] == self.subtask)
            if self.segment['status'] != 'detected':
                raise InvalidSubtaskReset(self.segment['boundary_reason'])
            ref = self.segment['initial_state_ref']
            if seed is not None and seed != ref['episode_seed']:
                raise ValueError('seed differs from selected source episode')
            replay_ref = copy.deepcopy(ref)
            if replay_ref['reset_kwargs'].get('options') is None:
                replay_ref['reset_kwargs']['options'] = {}
            obs, base_info = reset_episode(self.env, replay_ref)
            source_s0 = state_at(self.original[f"traj_{ref['original_episode_id']}/env_states"], 0)
            if ref['method'] == 'seed_reset_then_expert_prefix':
                # Newly recorded sources have correct s0. Preserve the reset's
                # native contact/solver/controller state; check it before replay.
                if tree_error(self.env.get_state_dict(),source_s0) > 2e-6:
                    raise InvalidSubtaskReset('seed reset does not reproduce recorded s0')
            elif ref['method'] == 'original_s0_then_expert_prefix':
                self.env.set_state_dict(source_s0)
                self.env.agent.controller.reset()
            else:
                raise ValueError('unknown initialization method')
            obs = self.env.get_obs()
            self.prefix_steps = self.segment['start_t']
            self.start_ref = dict(index=self.index_path, source_episode_id=eid,
                                  source_uid=self.segment.get('source_uid', f"legacy91:{eid}"),
                                  seed=ref['episode_seed'], state_index=self.prefix_steps,
                                  source_sha256=self.dataset['source_sha256'],
                                  reset_kwargs=ref['reset_kwargs'])
            self.start_ref['method'] = ref['method']
            self.group = self.source[f'traj_{eid}']
            self.audit = load_audit(ep['audit_file'])
        self.accessor = LiveSubtaskAccessor().reset(self.env)
        self._update(0, False)
        if self.segment is not None:
            for t in range(self.prefix_steps+1):
                self.max_prefix_state_error = max(self.max_prefix_state_error,
                    tree_error(self.env.get_state_dict(), state_at(self.original[f"traj_{ref['original_episode_id']}/env_states"], 0)
                               if t == 0 else state_at(self.group['env_states'], t)))
                if self.obs_mode == 'state':
                    self.max_prefix_observation_error = max(self.max_prefix_observation_error,
                        tree_error(obs, self.group['obs'][t]))
                for key in ('peg_half_sizes', 'hole_radius', 'hole_pose'):
                    if tree_error(self.live_state[key], self.audit[key][t]) > 2e-6:
                        raise InvalidSubtaskReset(f'geometry mismatch: {key} at {t}')
                if t and self.live_state['grasp'] != bool(self.audit['grasp'][t]):
                    raise InvalidSubtaskReset(f'contact mismatch at {t}')
                if t < self.prefix_steps:
                    obs, _, _, _, base_info = self.env.step(self.audit['actions'][t])
                    self._update(t+1, True)
                    if not self.metrics['workspace_valid']:
                        raise InvalidSubtaskReset(f'prefix workspace exit at {t+1}')
                    if self.context.acquired and self.context.drop_count >= self.config['drop_steps']:
                        raise InvalidSubtaskReset(f'prefix lost grasp at {t+1}')
                    if not self.context.acquired and self.metrics['peg_motion'] > self.config['peg_motion']:
                        raise InvalidSubtaskReset(f'prefix displaced peg before acquisition at {t+1}')
            if self.max_prefix_state_error > self.runtime['state_tolerance'] or self.max_prefix_observation_error > self.runtime['observation_tolerance']:
                raise InvalidSubtaskReset('prefix replay diverged from verified source')
        # Temporal evidence from actual prefix remains; only the episode clock moves.
        self.context.start_t = self.context.t
        reasons = [r for r in self.subtask_spec.failure_reasons(self.metrics, self.context) if r != 'timeout']
        if not self.subtask_spec.check_start(self.metrics, self.context):
            reasons.append('start_condition_false')
        if self.subtask_spec.check_success(self.metrics, self.context):
            reasons.append('already_successful')
        if not all(np.isfinite(np.asarray(v)).all() for k, v in self.live_state.items() if k != 'contact_valid'):
            reasons.append('nonfinite_state')
        if reasons:
            raise InvalidSubtaskReset(','.join(reasons))
        self._ready = True
        return obs, self._info(base_info)

    def step(self, action):
        if not self._ready:
            raise RuntimeError('call valid reset before step; episode may already be done')
        a = torch.as_tensor(action, dtype=torch.float32, device=self.env.device)
        if a.shape not in (self.action_space.shape, (1, *self.action_space.shape)):
            raise ValueError('action shape does not match ManiSkill action space')
        if not torch.isfinite(a).all() or not self.action_space.contains(a.detach().cpu().numpy().reshape(self.action_space.shape)):
            raise ValueError('action outside controller bounds')
        obs, _, _, _, base_info = self.env.step(a)
        self.learner_steps += 1
        self._update(self.context.t+1, True)
        success = self.subtask_spec.check_success(self.metrics, self.context)
        reasons = [r for r in self.subtask_spec.failure_reasons(self.metrics, self.context) if r != 'timeout']
        terminated = bool(success or reasons)
        truncated = bool(self.learner_steps >= self.budget and not terminated)
        info = self._info(base_info, success=success, reasons=reasons)
        info['time_limit_reached'] = truncated
        if truncated:
            info['failure_reason'] = 'timeout'
        self._ready = not (terminated or truncated)
        return obs, float(success), terminated, truncated, info

    def close_sources(self):
        for source in (self.source, self.original):
            if source is not None:
                source.close()

    def close(self):
        self.close_sources()
        self.env.close()


def make_subtask_env(subtask='reach', **kwargs):
    return SubtaskEnv(subtask, **kwargs)
