"""Choose a PyTorch compute device without changing global defaults.

NumPy arrays always live on the CPU; transfer tensors to CPU before calling
``tensor.detach().cpu().numpy()``.
"""

import torch


def set_device(device: str | torch.device = "auto") -> torch.device:
    """Return an available device, preferring CUDA, then MPS, then CPU.

    Explicit requests (including indexed CUDA devices) fail when unavailable.
    Pass the returned device to tensor constructors or ``model.to(device)``.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available() or (
            selected.index is not None and selected.index >= torch.cuda.device_count()
        ):
            raise RuntimeError(f"CUDA device {selected} is not available")
    elif selected.type == "mps":
        if selected.index not in (None, 0) or not torch.backends.mps.is_available():
            raise RuntimeError(f"MPS device {selected} is not available")
    elif selected.type != "cpu" or selected.index is not None:
        raise ValueError(f"Unsupported device: {selected}")
    return selected
