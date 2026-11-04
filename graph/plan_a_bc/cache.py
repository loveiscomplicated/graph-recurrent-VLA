"""Compatibility alias; the adopted implementation is vla.data.graph_cache."""
from importlib import import_module
import sys

_implementation = import_module('vla.data.graph_cache')
if __name__ == '__main__':
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
