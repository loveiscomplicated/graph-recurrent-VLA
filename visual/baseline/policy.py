"""Build LeRobot ACT / Diffusion policies and normalize their inputs and outputs.

LeRobot 0.7 keeps normalization in separate processor pipelines; we do it here
with dataset mean/std (state, action) and ImageNet statistics (RGB), so training
and closed-loop evaluation share exactly one code path.
"""

from __future__ import annotations

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from visual.baseline.common import ROBOT_DIM

ACTION_DIM = 7
CAMERA_NAMES = ("main", "wrist")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def image_key(camera: str) -> str:
    """LeRobot feature name for a camera, e.g. ``observation.images.main``."""
    return f"observation.images.{camera}"


# Main-camera depth fed to the policy as an extra 3-channel "image".
DEPTH_KEY = image_key("main_depth")


def build_policy(policy_config: dict, image_size: int, use_state: bool = True, use_depth: bool = False):
    """Create an untrained ACT or Diffusion policy from the YAML ``policy`` section.

    Example section: ``{type: act, chunk_size: 50, n_action_steps: 1, ...}``; all
    keys except ``type`` are passed to the LeRobot config dataclass.
    ``use_state=False`` drops the robot state from the inputs (images only; ACT only,
    because LeRobot's Diffusion policy always conditions on the state).
    """
    options = dict(policy_config)
    policy_type = options.pop("type")
    input_features = {
        image_key(camera): PolicyFeature(FeatureType.VISUAL, (3, image_size, image_size))
        for camera in CAMERA_NAMES
    }
    if use_depth:
        input_features[DEPTH_KEY] = PolicyFeature(FeatureType.VISUAL, (3, image_size, image_size))
    if use_state:
        input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (ROBOT_DIM,))
    elif policy_type != "act":
        raise ValueError("use_state: false is only supported for policy.type 'act'")
    output_features = {"action": PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))}
    features = dict(input_features=input_features, output_features=output_features, device="cuda")

    if policy_type == "act":
        return ACTPolicy(ACTConfig(**features, **options))
    if policy_type == "diffusion":
        if "crop_shape" in options:
            options["crop_shape"] = tuple(options["crop_shape"])
        return DiffusionPolicy(DiffusionConfig(**features, **options))
    raise ValueError(f"policy.type must be 'act' or 'diffusion', got {policy_type!r}")


class Normalizer:
    """Mean/std normalization of state and action, ImageNet normalization of RGB."""

    def __init__(self, stats: dict[str, torch.Tensor], device: str = "cuda"):
        self.stats = {name: value.to(device) for name, value in stats.items()}
        self.image_mean = IMAGENET_MEAN.to(device)
        self.image_std = IMAGENET_STD.to(device)

    def state(self, robot_state: torch.Tensor) -> torch.Tensor:
        return (robot_state - self.stats["state_mean"]) / self.stats["state_std"]

    def action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.stats["action_mean"]) / self.stats["action_std"]

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.stats["action_std"] + self.stats["action_mean"]

    def depth(self, depth_mm: torch.Tensor, size: int) -> torch.Tensor:
        """int16 mm depth (..., 1, h, w) → metres centred at 1 m, nearest-resized, 3 channels."""
        leading = depth_mm.shape[:-3]
        flat = depth_mm.reshape(-1, *depth_mm.shape[-3:]).float().clamp(min=0) / 1000.0
        flat = torch.nn.functional.interpolate(flat, size=(size, size), mode="nearest")
        return (flat - 1.0).repeat(1, 3, 1, 1).reshape(*leading, 3, size, size)

    def image(self, rgb_uint8: torch.Tensor) -> torch.Tensor:
        """uint8 (..., 3, H, W) → float ImageNet-normalized tensor."""
        return (rgb_uint8.float() / 255.0 - self.image_mean) / self.image_std


def compute_stats(robot_state: torch.Tensor, actions: torch.Tensor, action_mode: str) -> dict[str, torch.Tensor]:
    """Per-dimension state mean/std and action center/scale.

    ``action_mode='min_max'`` maps actions to [-1, 1] (required by Diffusion's
    ``clip_sample``); ``'mean_std'`` is the ACT default.
    """
    if action_mode == "min_max":
        low, high = actions.min(0).values, actions.max(0).values
        action_center, action_scale = (high + low) / 2, (high - low) / 2
    elif action_mode == "mean_std":
        action_center, action_scale = actions.mean(0), actions.std(0)
    else:
        raise ValueError(f"action_mode must be 'min_max' or 'mean_std', got {action_mode!r}")
    return dict(
        state_mean=robot_state.mean(0),
        state_std=robot_state.std(0).clamp_min(1e-3),
        action_mean=action_center,
        action_std=action_scale.clamp_min(1e-3),
    )
