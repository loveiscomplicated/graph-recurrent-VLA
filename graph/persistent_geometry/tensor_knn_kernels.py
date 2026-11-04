"""Compatibility alias; active implementation: graph.geometry.tensor_knn_kernels."""
from importlib import import_module as _import_module
import sys as _sys
_sys.modules[__name__] = _import_module('graph.geometry.tensor_knn_kernels')
