"""SAM2 video tracking for runtime visual pipelines.

Example::

    tracker = SAM2Tracker()
    state = tracker.start(video_path, {1: [x1, y1, x2, y2]})
    for frame_idx, masks in tracker.track(state):
        ...
    tracker.reset(state)

``video_path`` is a directory of RGB frames accepted by SAM2. Boxes use pixel
coordinates in ``[x1, y1, x2, y2]`` format and integer object IDs. Returned
masks are keyed by tracker ID; each value is a boolean Torch tensor with shape
``[height, width]``. The integration prototype's
``integration.visual_graph.adapters.attach_rgbd`` joins this iterator to recorded
RGB-D, camera calibration, and TCP poses, with an explicit tracker-to-Graph ID map.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_small.pt"


class SAM2Tracker:
    """Load a SAM2 video predictor once and track prompted objects in videos.

    Args:
        device: Torch device for SAM2 inference. Defaults to ``"cuda"`` to
            match the existing Visual pipeline.
        config: SAM2 Hydra config path, relative to the SAM2 package config
            directory or an absolute path.
        checkpoint: Model checkpoint path.
        hydra_overrides: Optional SAM2 model configuration overrides.
        predictor: Existing SAM2-compatible predictor, useful when another
            pipeline owns model construction. When provided, model arguments
            are ignored.
    """

    def __init__(self, device: str = "cuda", config: str = DEFAULT_CONFIG,
                 checkpoint: str | Path = DEFAULT_CHECKPOINT,
                 hydra_overrides: Sequence[str] | None = None, predictor=None):
        self.device = device
        if predictor is None:
            from sam2.build_sam import build_sam2_video_predictor

            predictor = build_sam2_video_predictor(
                config,
                str(checkpoint),
                device=device,
                hydra_overrides_extra=list(hydra_overrides or ()),
            )
        self.predictor = predictor

    def start(self, video_path: str | Path,
              boxes: Mapping[int, Sequence[float]], frame_idx: int = 0) -> dict:
        """Create a video state and prompt each object with its initial box."""
        state = self.predictor.init_state(video_path=str(video_path))
        for object_id, box in boxes.items():
            coords = np.asarray(box, dtype=np.float32)
            if coords.shape != (4,):
                self.predictor.reset_state(state)
                raise ValueError(f"box for object {object_id} must have four coordinates")
            self.predictor.add_new_points_or_box(
                state, frame_idx=frame_idx, obj_id=int(object_id), box=coords
            )
        return state

    def track(self, state: dict, start_frame: int = 0) -> Iterator[tuple[int, dict[int, object]]]:
        """Yield ``(frame_index, {object_id: boolean_mask})`` for each frame."""
        for frame_idx, object_ids, logits in self.predictor.propagate_in_video(
            state, start_frame_idx=start_frame
        ):
            masks = {
                int(object_id): logits[i, 0] > 0
                for i, object_id in enumerate(object_ids)
            }
            yield int(frame_idx), masks

    def reset(self, state: dict) -> None:
        """Release the predictor's per-video tracking state."""
        self.predictor.reset_state(state)


# Keep the concise spelling available alongside the descriptive class name.
Tracker = SAM2Tracker
