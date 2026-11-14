"""GPU mask prediction and current-frame input for the pinned MLX SAM2 port."""

from io import BytesIO

import cv2
import mlx.core as mx
from mlx_sam.preprocess import preprocess_image
from mlx_sam.video_predictor import SAM2VideoPredictor
from PIL import Image


class HalfPrecisionTrunk:
    """Run Hiera in FP16 and preserve FP32 inputs to the feature pyramid."""

    def __init__(self, trunk):
        self.trunk = trunk
        trunk.set_dtype(mx.float16)
        mx.eval(trunk.parameters())

    def __call__(self, pixels):
        features = self.trunk(pixels.astype(mx.float16))
        return [feature.astype(mx.float32) for feature in features]


def resize_logits(logits, size):
    """Apply half-pixel bilinear interpolation with replicated edge pixels."""
    for axis, length in enumerate(size, start=2):
        source_length = logits.shape[axis]
        if source_length == length:
            continue
        coordinates = (mx.arange(length, dtype=mx.float32) + 0.5) * (source_length / length) - 0.5
        lower = mx.floor(coordinates)
        first = mx.clip(lower, 0, source_length - 1).astype(mx.int32)
        second = mx.clip(lower + 1, 0, source_length - 1).astype(mx.int32)
        shape = [1] * logits.ndim
        shape[axis] = length
        weight = (coordinates - lower).reshape(shape)
        before = mx.take(logits, first, axis=axis)
        after = mx.take(logits, second, axis=axis)
        logits = before * (1 - weight) + after * weight
    return logits


class GpuMaskPredictor(SAM2VideoPredictor):
    """Return camera-size masks from SAM2.1 Small with three spatial memories."""

    def __init__(self, checkpoint, image_size, *, half_precision=True):
        super().__init__(
            checkpoint=checkpoint, model_id="facebook/sam2.1-hiera-small",
            image_size=image_size,
            memory_attention_dtype="float16" if half_precision else None,
        )
        self.encoder_calls = 0
        self.model.num_maskmem = 3
        # The pinned port uses 6 - t_pos. Keep the original prompt and the two
        # most recent temporal embeddings in the slots it actually reads.
        self.model.maskmem_tpos_enc = self.model.maskmem_tpos_enc[mx.array([0, 1, 2, 3, 0, 1, 6])]
        if half_precision:
            self.model.memory_attention.set_dtype(mx.float16)
            mx.eval(self.model.memory_attention.parameters())
            self.model.trunk = HalfPrecisionTrunk(self.model.trunk)

    def encode_pixels(self, pixels):
        """Compute all current-frame features from one normalized CHW batch."""
        return self.model.encode_image(mx.array(pixels))

    def _get_image_feature(self, state, frame_idx):
        cached = state["cached_features"].get(frame_idx)
        if cached is not None:
            return cached
        self.encoder_calls += 1
        features = self.encode_pixels(state["images"][frame_idx:frame_idx + 1])
        state["cached_features"] = {frame_idx: features}
        return features

    def _output_masks(self, state, outputs):
        size = (state["video_height"], state["video_width"])
        logits = mx.concatenate([output["pred_masks"] for output in outputs], axis=0)
        return resize_logits(logits, size)[:, 0] > 0


class CurrentFrame:
    """Expose exactly one normalized RGB frame under its original index."""

    def __init__(self, index, pixels):
        self.index = index
        self.pixels = pixels

    def __getitem__(self, selection):
        if selection != slice(self.index, self.index + 1):
            raise IndexError("only the current SAM2 frame can be read")
        return self.pixels


def preprocess_rgb(rgb, size):
    """Decode an in-memory JPEG and normalize it with the pinned MLX loader."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr)
    if not ok:
        raise OSError("could not encode the SAM2 JPEG")
    with Image.open(BytesIO(encoded.tobytes())) as image:
        return preprocess_image(image, size)
