"""Independent conditions; temporal predicates use explicit per-episode context."""
from pathlib import Path
import tomllib
import numpy as np
from .metrics import compute_metrics

NAMES=('reach','grasp','transport','align','insert')
DEFAULT_CONFIG=Path(__file__).with_name('config.toml')


def load_config(path=DEFAULT_CONFIG):
    with open(path,'rb') as f: c=tomllib.load(f)
    for key in ('grasp_steps','align_steps','drop_steps','timeout_steps'):
        if not isinstance(c[key],int) or c[key]<1: raise ValueError(f'invalid {key}')
    for key in ('reach_position','transport_position','align_lateral','align_axial','grasp_relative_position'):
        if c[key]<=0: raise ValueError(f'invalid {key}')
    return c


class SubtaskSpec:
    def __init__(self,name,config=None):
        if name not in NAMES: raise ValueError(name)
        self.name=name; self.config=load_config() if config is None else config

    def compute_metrics(self,state): return compute_metrics(state,self.config)

    def _metrics(self,state):
        return state if 'position_error' in state else self.compute_metrics(state)

    def align_raw(self,m):
        c=self.config
        return bool(m['grasp'] and m['contact_valid'] and m['head_lateral']<=c['align_lateral'] and
            m['center_lateral']<=c['align_lateral'] and m['relative_rotation_error']<=np.deg2rad(c['align_rotation_deg']) and
            m['waiting_error']<=c['align_axial'] and m['surface_clearance']>=c['align_surface_margin'] and m['insertion_depth']<=0)

    def check_start(self,state,context=None):
        m=self._metrics(state)
        if self.name=='reach': return bool(m['on_table'] and m['gripper_open'] and not m['grasp'])
        if self.name=='grasp': return SubtaskSpec('reach',self.config).check_success(m,context)
        if self.name=='transport': return bool(context and context.stable_grasp and not SubtaskSpec('transport',self.config).check_success(m,context))
        if self.name=='align': return SubtaskSpec('transport',self.config).check_success(m,context)
        return SubtaskSpec('align',self.config).check_success(m,context)

    def check_success(self,state,context=None):
        m=self._metrics(state); c=self.config
        if self.name=='reach':
            return bool(m['position_error']<=c['reach_position'] and m['rotation_error']<=np.deg2rad(c['reach_rotation_deg']) and
                m['gripper_open'] and m['contact_valid'] and not m['grasp'] and not (context and context.ever_grasped) and m['peg_motion']<=c['peg_motion'])
        if self.name=='grasp': return bool(context and context.stable_grasp)
        if self.name=='transport': return bool(m['contact_valid'] and m['grasp'] and m['transport_error']<=c['transport_position'] and
            m['relative_rotation_error']<=np.deg2rad(c['transport_rotation_deg']) and m['insertion_depth']<=0)
        if self.name=='align': return bool(self.align_raw(m) and (self.config['align_steps']==1 or context and context.align_count>=c['align_steps']))
        return m['official_success']

    def failure_reasons(self,state,context=None):
        m=self._metrics(state); c=self.config; reasons=[]
        if context and context.t-context.start_t>=c['timeout_steps']: reasons.append('timeout')
        if not m['workspace_valid']: reasons.append('workspace_exit')
        if self.name in ('reach','grasp') and not (context and context.acquired) and m['peg_motion']>c['peg_motion']:
            reasons.append('peg_displaced')
        if self.name in ('grasp','transport','align','insert') and context and context.acquired and context.drop_count>=c['drop_steps']:
            reasons.append('grasp_lost')
        # Severe collision/progress failure requires evidence not defined by v0;
        # record forces as diagnostics, never silently invent a force threshold.
        return reasons

    def check_failure(self,state,context=None): return bool(self.failure_reasons(state,context))
