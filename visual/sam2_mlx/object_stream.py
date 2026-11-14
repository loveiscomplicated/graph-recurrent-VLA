"""Track two prompted objects from consecutive live RGB frames on MLX."""

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import mlx.core as mx
import numpy as np

from visual.sam2_mlx.predictor import CurrentFrame, GpuMaskPredictor, preprocess_rgb


class MLXObjectStream:
    """One causal base-camera state, with frame-zero boxes and MLX GPU masks."""

    def __init__(self, checkpoint):
        if not mx.metal.is_available():
            raise RuntimeError("MLXObjectStream requires Apple Metal")
        mx.set_default_device(mx.gpu)
        self.predictor = GpuMaskPredictor(checkpoint, 512)
        self.state = None
        self.next_frame = 0

    def start(self, rgb, boxes):
        """Prompt both objects together so they share one encoding per frame."""
        if self.state is not None:
            raise RuntimeError("reset the MLX stream before supplying initial boxes")
        if set(boxes) != {1, 2}:
            raise ValueError("expected initial XYXY boxes for peg (1) and box (2)")
        with TemporaryDirectory(prefix="sam2_mlx_live_") as directory:
            path = Path(directory) / "000000.jpg"
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError("could not encode initial SAM2 frame")
            self.state = self.predictor.init_state(directory)
        for object_id in (1, 2):
            self.predictor.add_new_points_or_box(self.state, 0, object_id, box=boxes[object_id])
        return self._propagate(0)

    def track(self, rgb, timestep):
        """Consume the next observation and return the camera-size GPU masks."""
        if self.state is None or timestep != self.next_frame:
            raise ValueError("expected the next frame of an initialized MLX stream")
        size = (self.state["video_height"], self.state["video_width"])
        if rgb.dtype != np.uint8 or rgb.shape != (*size, 3):
            raise ValueError("expected aligned uint8 RGB for the current camera")
        pixels = preprocess_rgb(rgb, self.predictor.image_size)
        self.state["images"] = CurrentFrame(timestep, pixels)
        self.state["num_frames"] = timestep + 1
        return self._propagate(timestep)

    def _propagate(self, timestep):
        iterator = self.predictor.propagate_in_video(
            self.state, start_frame_idx=timestep, max_frame_num_to_track=0,
        )
        try:
            frame, ids, masks = next(iterator)
        finally:
            iterator.close()
        if frame != timestep or ids != [1, 2]:
            raise RuntimeError("MLX tracker returned unexpected frame or object IDs")
        mx.eval(masks)
        mx.synchronize()
        self.next_frame = timestep + 1
        return masks

    def reset(self):
        """Release episode state while retaining the loaded model."""
        self.state = None
        self.next_frame = 0
