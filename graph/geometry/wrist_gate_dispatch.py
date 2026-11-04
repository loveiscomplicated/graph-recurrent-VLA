"""CUDA-first wrist gate policy for current live and Task10-I geometry.

The NumPy adapter preserves the legacy prepare/correction boundary. Resident
callers use tensor_gate directly and keep their accepted masks on CUDA.
"""


def default_wrist_gate_backend():
    import torch
    return 'torch_cuda' if torch.cuda.is_available() else 'existing_cpu'


def wrist_gate_cuda_numpy(camera, raw_masks, clean_masks, cameras, clean_all, cfg):
    """Run the validated CUDA gate, explicitly exporting its accepted mask.

CUDA/import/runtime failures propagate. The caller selects CPU only when
CUDA is unavailable; this adapter never catches GPU errors to retry on CPU.
"""
    import torch
    from graph.geometry.tensor_gate import wrist_gate_tensor
    clean = {name: torch.as_tensor(clean_masks if name == camera.camera else value, device='cuda')
             for name, value in clean_all.items()}
    raw = {camera.camera: torch.as_tensor(raw_masks, device='cuda')}
    accepted, trace = wrist_gate_tensor(cameras, raw, clean, cfg, wrist=camera.camera)
    return accepted.cpu().numpy(), trace
