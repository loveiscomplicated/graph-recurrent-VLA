"""Grounding DINO inference for runtime visual pipelines.

Example::

    detector = GDino()
    boxes, scores = detector(rgb, "red and white block")

``rgb`` is an H x W x 3 uint8 RGB image. ``boxes`` are pixel-coordinate
``[x1, y1, x2, y2]`` boxes ordered by descending score.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image


GDINO_ID = "IDEA-Research/grounding-dino-tiny"
CAND_THR = 0.1


class GDino:
    """Load Grounding DINO once, then detect prompted objects in RGB images.

    Args:
        device: Torch device used for inference. Defaults to ``"cuda"`` to
            match the existing Visual pipeline.
        model_id: Hugging Face model identifier or local model directory.
        threshold: Box and text threshold passed to Transformers postprocessing.
    """

    def __init__(self, device: str = "cuda", model_id: str = GDINO_ID,
                 threshold: float = CAND_THR):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._torch = torch
        self.device = device
        self.threshold = threshold
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.last_ms = {"pre": 0.0, "forward": 0.0, "post": 0.0}

    def _synchronize(self) -> None:
        if self.device.startswith("cuda") and self._torch.cuda.is_available():
            self._torch.cuda.synchronize(self.device)

    def __call__(self, rgb: np.ndarray, prompt: str) -> tuple[list[list[float]], list[float]]:
        """Return score-sorted pixel boxes and scores for one text prompt."""
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))

        t0 = time.perf_counter()
        inputs = self.proc(
            images=image,
            text=prompt.lower().strip() + ".",
            return_tensors="pt",
        ).to(self.device)
        self._synchronize()
        t1 = time.perf_counter()

        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        self._synchronize()
        t2 = time.perf_counter()

        detections = self.proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.threshold,
            text_threshold=self.threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = detections["boxes"].detach().cpu().numpy()
        scores = detections["scores"].detach().cpu().numpy()
        self.last_ms = {
            "pre": (t1 - t0) * 1000,
            "forward": (t2 - t1) * 1000,
            "post": (time.perf_counter() - t2) * 1000,
        }

        order = np.argsort(-scores)
        return boxes[order].tolist(), scores[order].tolist()


# Keep the descriptive spelling available to new callers while preserving the
# short class name used in the existing Visual scripts.
GroundingDINO = GDino

