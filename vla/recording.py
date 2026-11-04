"""Archive controller state trees without changing the baseline GRU array."""
import json
import numpy as np
from vla.state import map_state


def snapshot_state(state):
    return map_state(lambda value: value.detach().cpu().numpy().copy(), state)


def state_history_arrays(history):
    if not history:
        return {'hidden': np.empty((0,), dtype=np.float32)}
    if all(isinstance(state, np.ndarray) for state in history):
        return {'hidden': np.stack(history)}
    if all(state is None for state in history):
        return {'hidden': np.empty((len(history), 0), dtype=np.float32)}

    def flatten(state, leaves):
        if state is None:
            return {'kind': 'none'}
        if isinstance(state, np.ndarray):
            index = len(leaves)
            leaves.append(state)
            return {'kind': 'tensor', 'field': 'hidden_' + str(index)}
        if isinstance(state, (tuple, list)):
            return {'kind': type(state).__name__, 'items': [flatten(value, leaves) for value in state]}
        if isinstance(state, dict):
            if any(not isinstance(key, str) for key in state):
                raise TypeError('recorded controller state keys must be strings')
            return {'kind': 'dict', 'items': {key: flatten(value, leaves) for key, value in state.items()}}
        raise TypeError('recorded state must be an array tree or None')

    columns = None
    structure = None
    for state in history:
        leaves = []
        current = flatten(state, leaves)
        if structure is None:
            structure = current
            columns = [[] for _ in leaves]
        if current != structure:
            raise ValueError('controller state structure changed within episode')
        for column, value in zip(columns, leaves):
            column.append(value)
    return {'controller_state_structure': np.array(json.dumps(structure, sort_keys=True)),
            **{'hidden_' + str(i): np.stack(values) for i, values in enumerate(columns)}}
