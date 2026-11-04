"""Compatibility alias; active implementation: graph.backends.surface_edges."""
from importlib import import_module as _import_module
import sys as _sys
_sys.modules[__name__] = _import_module('graph.backends.surface_edges')
