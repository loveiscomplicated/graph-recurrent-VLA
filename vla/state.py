"""Controller-owned state trees, without assuming a GRU inside integration."""
import torch


def map_state(function, state):
    if state is None:
        return None
    if torch.is_tensor(state):
        return function(state)
    if isinstance(state, tuple):
        return tuple(map_state(function, value) for value in state)
    if isinstance(state, list):
        return [map_state(function, value) for value in state]
    if isinstance(state, dict):
        return {key: map_state(function, value) for key, value in state.items()}
    raise TypeError('controller state must be a tensor tree or None')


def reset_state(state, reset, *, batch_dim=1):
    def reset_tensor(value):
        if value.shape[batch_dim] != reset.numel():
            raise ValueError('controller state/reset batch mismatch')
        shape = [1] * value.ndim
        shape[batch_dim] = reset.numel()
        return value * (~reset).reshape(shape)
    return map_state(reset_tensor, state)


def select_state(previous, current, active, *, batch_dim=1):
    """Advance active streams only, including the first padded timestep."""
    if current is None:
        if previous is not None:
            raise ValueError('controller changed state structure')
        return None
    if previous is None:
        previous = map_state(torch.zeros_like, current)
    if torch.is_tensor(current):
        if not torch.is_tensor(previous) or previous.shape != current.shape:
            raise ValueError('controller changed state tensor shape')
        if current.shape[batch_dim] != active.numel():
            raise ValueError('controller state/active batch mismatch')
        shape = [1] * current.ndim
        shape[batch_dim] = active.numel()
        return torch.where(active.reshape(shape), current, previous)
    if isinstance(current, (tuple, list)):
        if type(previous) is not type(current) or len(previous) != len(current):
            raise ValueError('controller changed state structure')
        values = [select_state(a, b, active, batch_dim=batch_dim)
                  for a, b in zip(previous, current)]
        return tuple(values) if isinstance(current, tuple) else values
    if isinstance(current, dict):
        if not isinstance(previous, dict) or previous.keys() != current.keys():
            raise ValueError('controller changed state keys')
        return {key: select_state(previous[key], value, active, batch_dim=batch_dim)
                for key, value in current.items()}
    raise TypeError('controller state must be a tensor tree or None')


def detach_state(state, *, cpu=False):
    return map_state(lambda value: value.detach().cpu().clone() if cpu else value.detach(), state)


def slice_state(state, index, *, batch_dim=1):
    return map_state(lambda value: value.narrow(batch_dim, index, 1), state)


def stack_states(states, *, batch_dim=1):
    first = states[0]
    if first is None:
        if any(state is not None for state in states):
            raise ValueError('mixed controller state structures')
        return None
    if torch.is_tensor(first):
        return torch.cat(states, dim=batch_dim)
    if isinstance(first, (tuple, list)):
        values = [stack_states([state[i] for state in states], batch_dim=batch_dim)
                  for i in range(len(first))]
        return tuple(values) if isinstance(first, tuple) else values
    if isinstance(first, dict):
        return {key: stack_states([state[key] for state in states], batch_dim=batch_dim)
                for key in first}
    raise TypeError('controller state must be a tensor tree or None')
