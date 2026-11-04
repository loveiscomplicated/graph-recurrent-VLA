"""Compatibility alias; the adopted implementation is vla.workflows.rollout."""
from importlib import import_module
import sys

_implementation = import_module('vla.workflows.rollout')
if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
