"""Validate installation separately from GPU runtime; no model downloads."""
import importlib
import sys

modules = (
    "torch", "torchvision", "triton", "numpy", "PIL", "cv2", "scipy",
    "matplotlib", "imageio", "imageio_ffmpeg", "trimesh", "transforms3d",
    "gymnasium", "sapien", "mani_skill.envs", "sam2.build_sam",
    "transformers.models.grounding_dino.modeling_grounding_dino",
    "transformers.models.dinov2.modeling_dinov2", "pc_gpu", "pose_v1",
)
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        print(f"IMPORT FAILED: {module}: {exc}", file=sys.stderr)
        print("Check missing shared libraries (e.g. libGL, libEGL, libvulkan), and rerun bash setup.sh.", file=sys.stderr)
        raise SystemExit(2) from exc
import torch
assert sys.version_info[:2] == (3, 12), sys.version
assert torch.__version__ == "2.8.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
print(f"Imports passed; Python {sys.version.split()[0]}, torch {torch.__version__}, CUDA runtime {torch.version.cuda}")
if not torch.cuda.is_available():
    print("Dependencies installed, but CUDA is unavailable. Check nvidia-smi, the host NVIDIA driver and container GPU passthrough (--gpus all).", file=sys.stderr)
    raise SystemExit(3)
try:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.ones(1, device="cuda").add_(1)
    torch.cuda.synchronize()
except Exception as exc:
    print(f"Dependencies installed, but CUDA execution failed: {exc}", file=sys.stderr)
    raise SystemExit(3) from exc
