"""Temporal evidence resets per episode and requires consecutive state indices."""
from dataclasses import dataclass, field
import numpy as np
from .metrics import rotation_error


@dataclass
class ConditionContext:
    t: int = -1
    start_t: int = 0
    ever_grasped: bool = False
    acquired: bool = False
    grasp_count: int = 0
    drop_count: int = 0
    align_count: int = 0
    history: list = field(default_factory=list)
    stable_grasp: bool = False

    def update(self, metrics, t, config, align_raw=False):
        if t!=self.t+1:
            raise ValueError('context needs consecutive states; new episode needs new context')
        self.t=t
        live=metrics['contact_valid'] and metrics['grasp']
        self.grasp_count=self.grasp_count+1 if live else 0
        # Unknown snapshot contact is not evidence that an acquired peg dropped.
        self.drop_count=0 if live or not metrics['contact_valid'] else self.drop_count+1
        self.ever_grasped |= live
        self.history.append(metrics)
        self.history=self.history[-config['grasp_steps']:]
        stable=self.grasp_count>=config['grasp_steps']
        if stable:
            anchor=self.history[0]
            stable=all(np.linalg.norm(np.asarray(m['grasp_relative_position'])-anchor['grasp_relative_position'])<=config['grasp_relative_position'] and
                rotation_error(np.asarray(anchor['grasp_relative_rotation']).T@np.asarray(m['grasp_relative_rotation']))<=np.deg2rad(config['grasp_relative_rotation_deg'])
                for m in self.history)
        self.stable_grasp=bool(stable)
        self.acquired |= self.stable_grasp
        self.align_count=self.align_count+1 if align_raw else 0
