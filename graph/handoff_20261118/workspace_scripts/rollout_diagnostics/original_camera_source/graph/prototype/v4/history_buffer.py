"""Evaluation-only representation cache; commit commands after env.step."""
from collections import deque
import torch


class HistoryBuffer:
    def __init__(self, length=4):
        if length < 0:
            raise ValueError('negative history length')
        self.length = length
        self.reset()

    def reset(self):
        self.states = deque(maxlen=self.length)
        self.actions = deque(maxlen=self.length)
        self.first = None
        self.steps = 0

    def window(self, current, state_history=True):
        if self.first is None:
            self.first = current.detach().clone()
        missing = self.length-len(self.states)
        states = [self.first]*missing + list(self.states) + [current] if state_history else [current]
        zero = current.new_zeros(len(current), 7)
        actions = [zero]*missing + list(self.actions)
        valid = current.new_tensor([0]*missing + [1]*len(self.states)).expand(len(current), -1)
        action_tensor = torch.stack(actions, 1) if actions else current.new_empty(len(current), 0, 7)
        return torch.stack(states, 1), action_tensor, valid

    def commit(self, pre_step_state, applied_action):
        if applied_action.shape != (len(pre_step_state), 7) or not torch.isfinite(applied_action).all():
            raise ValueError('invalid applied action')
        if self.first is None:
            self.first = pre_step_state.detach().clone()
        self.states.append(pre_step_state.detach().clone())
        self.actions.append(applied_action.detach().clone())
        self.steps += 1
