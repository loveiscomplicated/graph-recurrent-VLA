"""Create the shared ManiSkill PegInsertionSide-v1 environment."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def _configure_macos_vulkan() -> None:
    """Use a local Homebrew MoltenVK ICD when one is available.

    SAPIEN initializes its renderer even for state-only observations on macOS.
    An explicit VK_ICD_FILENAMES always takes precedence.
    """
    if platform.system() != "Darwin" or os.environ.get("VK_ICD_FILENAMES"):
        return
    for prefix in ("/opt/homebrew", "/usr/local"):
        icd = Path(prefix) / "opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json"
        if icd.is_file():
            os.environ["VK_ICD_FILENAMES"] = str(icd)
            return


_configure_macos_vulkan()

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: E402,F401 - registers PegInsertionSide-v1
import torch  # noqa: E402


def make_peg_insertion_env(
    *,
    seed: int | None = None,
    device: str = "cpu",
    num_envs: int = 1,
    obs_mode: str = "state",
    control_mode: str | None = None,
    render_mode: str | None = None,
    sim_backend: str | None = None,
    observation_device: str | torch.device | None = None,
    render_backend: str | None = None,
    sensor_configs: dict | None = None,
):
    """Make the task; ``seed`` also resets its first user episode.

    ``control_mode=None`` selects ManiSkill's own default. Returned observations,
    rewards and flags are batched torch tensors, including when ``num_envs=1``.
    ``observation_device`` optionally transfers raw reset/step observations at
    the output boundary. It does not select a physics or renderer backend.
    An explicit ``render_backend="cuda"`` can expose native CUDA camera images
    while retaining ``device="cpu"`` physics on a supported Linux GPU host.
    ``sensor_configs`` forwards ManiSkill's raw camera configuration overrides;
    omitting it preserves the task's default camera settings.
    """
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    backend = sim_backend or ("physx_cpu" if device == "cpu" else "physx_cuda")
    if backend not in {"physx_cpu", "physx_cuda"}:
        raise ValueError("sim_backend must be 'physx_cpu' or 'physx_cuda'")
    if backend != ("physx_cpu" if device == "cpu" else "physx_cuda"):
        raise ValueError(f"device={device!r} conflicts with sim_backend={backend!r}")
    if device == "cpu" and num_envs != 1:
        raise ValueError("ManiSkill 3.0.1 physx_cpu requires num_envs=1")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch installation")
    target = None if observation_device is None else torch.device(observation_device)
    if target is not None:
        if target.type not in {"cpu", "cuda"}:
            raise ValueError("observation_device must be CPU or CUDA")
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA observation_device requested but CUDA is unavailable")

    kwargs = dict(
        num_envs=num_envs,
        obs_mode=obs_mode,
        render_mode=render_mode,
        sim_backend=backend,
    )
    # CPU simulation defaults to host image readback even when Torch sees CUDA.
    # Otherwise ManiSkill's default GPU renderer asks SAPIEN for cuda:0,
    # which may have no matching Vulkan physical device (e.g. lavapipe).
    # Explicit render_backend opts out; SAPIEN 3.0 cannot switch render devices
    # within a process.
    if render_backend is not None:
        kwargs["render_backend"] = render_backend
    elif device == "cpu":
        kwargs["render_backend"] = "cpu"
    if control_mode is not None:
        kwargs["control_mode"] = control_mode
    if sensor_configs is not None:
        kwargs["sensor_configs"] = sensor_configs
    original_cwd = Path.cwd()
    env = None
    try:
        env = gym.make("PegInsertionSide-v1", **kwargs)
        if target is not None:
            from .observation_device import ObservationDevice

            env = ObservationDevice(env, target)
        if seed is not None:
            env.reset(seed=seed)
        return env
    except Exception as exc:
        if env is not None:
            env.close()
        if platform.system() == "Darwin" and not os.environ.get("VK_ICD_FILENAMES"):
            raise RuntimeError(
                "ManiSkill could not start. On macOS, install Homebrew molten-vk "
                "and set VK_ICD_FILENAMES to its MoltenVK_icd.json file."
            ) from exc
        raise
    finally:
        # SAPIEN's macOS renderer may change the process working directory.
        os.chdir(original_cwd)
