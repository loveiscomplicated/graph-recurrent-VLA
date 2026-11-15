"""Track prompted objects from consecutive CPU RGB frames on CUDA."""

from contextlib import contextmanager
from numbers import Integral
from threading import RLock

import numpy as np
import torch

from .predictor import build_predictor, initialize_state
from .preprocessing import preprocess_rgb


_tf32_lock = RLock()


@contextmanager
def _tf32_inference():
    """Scope PyTorch's process-wide TF32 flags to one stream call."""
    with _tf32_lock:
        matmul = torch.backends.cuda.matmul
        cudnn = torch.backends.cudnn
        previous_matmul, previous_cudnn = matmul.allow_tf32, cudnn.allow_tf32
        try:
            matmul.allow_tf32 = True
            cudnn.allow_tf32 = True
            with torch.autocast("cuda", enabled=False):
                yield
        finally:
            matmul.allow_tf32 = previous_matmul
            cudnn.allow_tf32 = previous_cudnn


class CudaObjectStream:
    """Use SAM2.1 Small with CPU JPEG input and camera-size CUDA masks.

    Supply one uint8 NumPy RGB image in HWC order per frame. Each call returns
    a dictionary of CUDA bool tensors keyed by the input box IDs. TF32 flags
    are restored after each call. Coordinate concurrent CUDA work that needs
    different flags because PyTorch applies them process-wide.
    """

    def __init__(self, checkpoint, *, image_size=512, memory_slots=3):
        if not torch.cuda.is_available():
            raise RuntimeError("CudaObjectStream requires CUDA-enabled PyTorch and a CUDA device")
        if type(image_size) is not int or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        self.predictor = build_predictor(checkpoint, image_size, memory_slots)
        self.image_size = image_size
        self._state = None
        self._next_frame = 0

    def _prepare_frame(self, rgb):
        """Validate host RGB and upload one normalized CHW image."""
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
                or rgb.shape[-1] != 3 or min(rgb.shape[:2]) == 0):
            raise ValueError("expected nonempty uint8 NumPy RGB with shape (height, width, 3)")
        pixels = preprocess_rgb(rgb, self.image_size)[0]
        return torch.from_numpy(pixels).to(self.predictor.device)

    @torch.inference_mode()
    def start(self, rgb, boxes):
        """Initialize frame zero with one or more integer IDs and XYXY boxes."""
        if self._state is not None:
            raise RuntimeError("reset the stream before starting another sequence")
        if not boxes or any(isinstance(key, bool) or not isinstance(key, Integral) for key in boxes):
            raise ValueError("expected at least one XYXY box keyed by an integer object ID")
        pixels = self._prepare_frame(rgb)
        with _tf32_inference():
            state = initialize_state(self.predictor, pixels, rgb.shape[:2], boxes)
            try:
                masks = self._propagate(state, 0, pixels)
            except BaseException:
                self.predictor.reset_state(state)
                raise
        self._state = state
        self._next_frame = 1
        return masks

    @torch.inference_mode()
    def track(self, rgb, timestep):
        """Track the next frame using the sequence's existing SAM2 memory."""
        state = self._state
        if state is None or timestep != self._next_frame:
            raise ValueError("expected the next frame of an initialized CUDA stream")
        pixels = self._prepare_frame(rgb)
        if rgb.shape[:2] != (state["video_height"], state["video_width"]):
            raise ValueError("RGB dimensions must match the initial frame")
        with _tf32_inference():
            try:
                masks = self._propagate(state, timestep, pixels)
            except BaseException:
                self.reset()
                raise
        self._next_frame += 1
        return masks

    def _propagate(self, state, timestep, pixels):
        """Expose only the current RGB frame while retaining temporal memory."""
        state["images"] = {timestep: pixels}
        state["num_frames"] = timestep + 1
        iterator = self.predictor.propagate_in_video(state, start_frame_idx=timestep)
        try:
            frame, object_ids, logits = next(iterator)
        finally:
            iterator.close()
        if frame != timestep or list(object_ids) != state["obj_ids"]:
            raise RuntimeError("unexpected CUDA tracking frame or object IDs")
        masks = {int(object_id): logits[index, 0] > 0
                 for index, object_id in enumerate(object_ids)}
        expected_shape = (state["video_height"], state["video_width"])
        for mask in masks.values():
            if not mask.is_cuda or mask.dtype != torch.bool or mask.shape != expected_shape:
                raise RuntimeError("expected camera-size boolean masks on CUDA")
        return masks

    def reset(self):
        """Release sequence state while retaining the loaded SAM2 model."""
        if self._state is not None:
            self.predictor.reset_state(self._state)
            self._state = None
        self._next_frame = 0
