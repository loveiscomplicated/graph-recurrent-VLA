"""Evaluation checks that report bad values before they reach the simulator."""
from contextlib import contextmanager

import numpy as np
import torch


def require_finite(value, component, *, active=None, batch_dim=0, positive=False):
    """Check a value/tree, optionally restricting tensors to active batch slots."""
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{component}.{key}", active=active,
                           batch_dim=batch_dim, positive=positive)
        return
    if isinstance(value, torch.Tensor):
        invalid = ~torch.isfinite(value)
        if positive:
            invalid = invalid | (value <= 0)
        if active is not None:
            invalid = invalid.movedim(batch_dim, 0).reshape(len(active), -1).any(-1) & active
        if not invalid.any().item():
            return
        slots = torch.where(invalid)[0].tolist() if active is not None else None
    else:
        array = np.asarray(value)
        invalid = ~np.isfinite(array)
        if positive:
            invalid |= array <= 0
        if not invalid.any():
            return
        slots = None
    expected = "finite positive values" if positive else "finite values"
    detail = f"; slots={slots}" if slots is not None else ""
    raise FloatingPointError(f"Invalid {component}: expected {expected}{detail}")


def require_state(state, kind, *, active=None, component="state", root=True, positive=False):
    if state is None:
        return
    if isinstance(state, torch.Tensor):
        dim = 1 if root and kind in ("gru", "lstm") else 0
        require_finite(state, component, active=active, batch_dim=dim, positive=positive)
        return
    for index, item in enumerate(state):
        require_state(item, kind, active=active, component=f"{component}[{index}]",
                      root=root and kind == "lstm",
                      positive=root and kind == "rkn" and index == 1)


@contextmanager
def numerics_context(**context):
    try:
        yield
    except FloatingPointError as exc:
        details = " ".join(f"{key}={value}" for key, value in context.items())
        raise FloatingPointError(f"{details}: {exc}") from exc
