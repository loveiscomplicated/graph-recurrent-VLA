"""Compatibility alias; active implementation: graph.perception.geometry."""
from importlib import import_module as _import_module
import sys as _sys
_sys.modules[__name__] = _import_module('graph.perception.geometry')
