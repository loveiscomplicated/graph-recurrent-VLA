"""Compatibility alias; active implementation: vla.action_codec."""
from importlib import import_module as _import_module
import sys as _sys
_sys.modules[__name__] = _import_module('vla.action_codec')
