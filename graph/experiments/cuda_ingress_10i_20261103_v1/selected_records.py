"""Compatibility alias; active implementation: graph.backends.resident.selected_records."""
from importlib import import_module as _import_module
import sys as _sys
_sys.modules[__name__] = _import_module('graph.backends.resident.selected_records')
