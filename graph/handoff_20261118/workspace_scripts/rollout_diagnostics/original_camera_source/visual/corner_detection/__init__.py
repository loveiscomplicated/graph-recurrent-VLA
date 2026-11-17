"""Current-frame RGB-D corner detection for peg and box observations."""

from .detector import TorchCornerDetector, TensorResult, detect, detect_tensors
from .geometry import Parameters
from .numpy_reference import detect as detect_cpu

__all__ = [
    "TorchCornerDetector",
    "TensorResult",
    "Parameters",
    "detect",
    "detect_tensors",
    "detect_cpu",
]
