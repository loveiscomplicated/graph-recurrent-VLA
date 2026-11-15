"""Causal image windows for online TAPNext++ on Apple Silicon."""

from functools import partial

import mlx.core as mx
import numpy as np

from visual.tapnextpp_mlx import OnlineTracker, RecurrentState
from visual.tapnextpp_window import AdaptiveState, patch_ids, remap_rows, select_window


def _compile_window_step(block, *, final):
    """Compile stable and remapped continuations with the query count in the input shape."""
    def continuation(tokens, history, hidden, image_ids, rows=None, fresh=None):
        if rows is not None:
            history = mx.where(fresh[:, None, None], 0, history[rows])
            hidden = hidden[rows]
        start = image_ids.shape[0] if final else 0
        output, state = block(tokens, RecurrentState(history, hidden), output_start=start, fresh=fresh)
        return output, state.history, state.hidden

    return mx.compile(continuation)


class AdaptiveOnlineTracker(OnlineTracker):
    """Track in an observation-driven window with patch and query state continuity."""

    def __init__(self, model, crop):
        if crop is not None:
            raise ValueError("Adaptive TAPNext++ selects its own window")
        super().__init__(model, crop)
        self.grid = self.input_resolution // self.patch_size
        self._embed_window = {scale: mx.compile(partial(self._embed_patch_region, scale=scale))
                              for scale in (1, 2)}
        self._window_steps = [_compile_window_step(block, final=i == len(self.blocks) - 1)
                              for i, block in enumerate(self.blocks)]

    def _step_window(self, image_ids, rows, fresh, index, tokens, prior):
        if prior is None:
            start = image_ids.shape[0] if index == len(self.blocks) - 1 else 0
            return self.blocks[index](tokens, output_start=start)
        output, history, hidden = self._window_steps[index](
            tokens, prior.history, prior.hidden, image_ids, rows, fresh,
        )
        return output, RecurrentState(history, hidden)

    def track_frame(self, frame_bgr, query_points_xy=None, state=None, autocast=False, *,
                    observation_mask=None):
        """Use the current optional object mask and past tracks, in source pixels."""
        if autocast:
            raise ValueError("CUDA autocast does not apply to TAPNext++ MLX")
        scale = self._frame_scale(frame_bgr)
        frame_shape = frame_bgr.shape[:2]
        window, misses = select_window(query_points_xy, state, observation_mask, frame_shape, self.grid)
        selected = patch_ids(window, self.grid).astype(np.int32)
        image_ids = mx.array(selected)
        rows = fresh = None
        prior_model = None
        if state is not None:
            prior_model = state.model_state
            if state.crop != window:
                mapping = remap_rows(state.crop, selected, prior_model[0], self.grid)
                rows = mx.array(np.maximum(mapping, 0))
                fresh = mx.array(mapping < 0)
        pixel_step = self.patch_size * scale
        left, top, right, bottom = (edge * pixel_step for edge in window)
        pixels = frame_bgr[top:bottom, left:right]
        image_tokens = self._embed_window[scale](mx.array(pixels), self.image_pos[image_ids])
        step = partial(self._step_window, image_ids, rows, fresh)
        xy, visible, model_state = self._run_frame(image_tokens, frame_shape, query_points_xy, prior_model, step)
        valid = visible & np.isfinite(xy).all(axis=1)
        next_state = AdaptiveState(window, model_state, xy[valid], misses, frame_shape)
        return xy, visible, next_state
