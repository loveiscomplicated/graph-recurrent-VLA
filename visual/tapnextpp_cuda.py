"""Observation-driven patch windows for the CUDA TAPNext++ model."""

import numpy as np
import torch
from tapnet.tapnext.tapnext_lru_modules import RecurrentBlockCache
from tapnet.tapnextpp.votsp2026 import utils
from tapnet.tapnextpp.votsp2026.model import TAPNextPP

from visual.tapnextpp_window import AdaptiveState, patch_ids, remap_rows, select_window


def _cache_rows(cache, rows):
    return RecurrentBlockCache(*(field.index_select(0, rows) for field in cache))


def _step_moved_window(block, tokens, cache, rows, count, order, restore):
    """Process new patches as initial steps and retained patches as continuations."""
    if order is None:
        return block(tokens, cache=_cache_rows(cache, rows))
    sequence = tokens[0, 0].index_select(0, order).unsqueeze(1)
    continued, kept = block.ssm_block(sequence[:count], _cache_rows(cache, rows))
    initialized, fresh = block.ssm_block(sequence[count:])
    recurrent = torch.cat((continued, initialized)).index_select(0, restore)
    merged = RecurrentBlockCache(*(
        torch.cat((old, new)).index_select(0, restore) for old, new in zip(kept, fresh)
    ))
    output = block.vit_block(recurrent.transpose(0, 1))
    return output.unsqueeze(0), merged


def _predict_tokens(model, tokens, prior, query_count, mapping=None):
    caches = []
    for block, cache in zip(model.blocks, prior):
        if mapping is None:
            tokens, cache = block(tokens, cache=cache)
        else:
            tokens, cache = _step_moved_window(block, tokens, cache, *mapping)
        caches.append(cache)
    query_output = model.encoder_norm(tokens[:, :, -query_count:])
    tracks, _, visibility = model.prediction_heads(query_output)
    return tracks, visibility, caches


class _CapturedContinuation:
    """Replay the fixed-size CUDA computation with explicit inputs and state."""

    def __init__(self, model, tokens, prior, query_count, autocast):
        self.tokens = torch.empty_like(tokens)
        self.prior = [RecurrentBlockCache(*(torch.empty_like(field) for field in cache))
                      for cache in prior]
        self._copy_inputs(tokens, prior)
        device = tokens.device
        stream = torch.cuda.Stream(device=device)
        current = torch.cuda.current_stream(device)
        stream.wait_stream(current)
        # Capture the casts too, so the graph doesn't depend on a caller's
        # autocast cache surviving across tracking sessions.
        with torch.cuda.stream(stream), torch.autocast("cuda", dtype=torch.float16,
                                                      enabled=autocast, cache_enabled=False):
            for _ in range(2):
                _predict_tokens(model, self.tokens, self.prior, query_count)
        current.wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            with torch.autocast("cuda", dtype=torch.float16, enabled=autocast, cache_enabled=False):
                self.tracks, self.visibility, self.caches = _predict_tokens(
                    model, self.tokens, self.prior, query_count,
                )

    def _copy_inputs(self, tokens, prior):
        self.tokens.copy_(tokens)
        for destination, source in zip(self.prior, prior):
            for field, other in zip(destination, source):
                field.copy_(other)

    def __call__(self, tokens, prior):
        self._copy_inputs(tokens, prior)
        self.graph.replay()
        # State returned to callers must survive later replays of this graph.
        caches = [RecurrentBlockCache(*(field.clone() for field in cache)) for cache in self.caches]
        return self.tracks, self.visibility, caches


class AdaptiveCudaTracker(TAPNextPP):
    """Use the CUDA autocast model with a causal window and source-space queries."""

    def __init__(self, inner, device, input_resolution):
        super().__init__(inner, device, input_resolution)
        if input_resolution != self.MODEL_SIZE:
            raise ValueError("Adaptive TAPNext++ requires its native model resolution")
        self.grid = self.MODEL_SIZE // inner.patch_size[0]
        self.cuda_graph_enabled = False
        self._captured_step = None
        self._capture_key = None

    def _predict(self, tokens, prior, query_count, mapping, autocast):
        use_graph = (self.cuda_graph_enabled and self.device.type == "cuda"
                     and prior[0] is not None and mapping is None
                     and tokens.shape[2] - query_count == 144)
        if not use_graph:
            return _predict_tokens(self._model, tokens, prior, query_count, mapping)
        key = (tokens.shape, tokens.dtype, autocast)
        if key != self._capture_key:
            self._captured_step = _CapturedContinuation(
                self._model, tokens, prior, query_count, autocast,
            )
            self._capture_key = key
        return self._captured_step(tokens, prior)

    def _image_tokens(self, frame_bgr, window):
        video = utils.preprocess_frame(frame_bgr, self.device, self.input_resolution)
        left, top, right, bottom = (edge * self._model.patch_size[0] for edge in window)
        crop = video[0, :, top:bottom, left:right].permute(0, 3, 1, 2)
        projected = self._model.lin_proj(crop).flatten(2).transpose(1, 2)
        x0, y0, x1, y1 = window
        positions = self._model.image_pos_emb.reshape(self.grid, self.grid, -1)
        positions = positions[y0:y1, x0:x1].reshape(1, -1, self._model.width)
        return (projected + positions).unsqueeze(1)

    def _window_mapping(self, state, window):
        query_count, _ = state.model_state
        mapping = remap_rows(state.crop, patch_ids(window, self.grid), query_count, self.grid)
        retained = np.flatnonzero(mapping >= 0)
        fresh = np.flatnonzero(mapping < 0)
        rows = torch.as_tensor(mapping[retained], device=self.device, dtype=torch.long)
        if not len(fresh):
            return rows, len(retained), None, None
        order = np.concatenate((retained, fresh))
        restore = np.argsort(order)
        return (rows, len(retained), torch.as_tensor(order, device=self.device),
                torch.as_tensor(restore, device=self.device))

    @torch.no_grad()
    def track_frame(self, frame_bgr, query_points_xy=None, state=None, autocast=True, *,
                    observation_mask=None):
        """Return tracked points and a state keyed by the observation window."""
        if state is None and (query_points_xy is None or len(query_points_xy) == 0):
            raise ValueError("The initial frame requires at least one query")
        if state is not None and query_points_xy is not None:
            raise ValueError("Adaptive TAPNext++ accepts queries only on the initial frame")
        frame_shape = frame_bgr.shape[:2]
        window, misses = select_window(query_points_xy, state, observation_mask, frame_shape, self.grid)
        model = self._model
        if state is None:
            query_count = len(query_points_xy)
            model_xy = utils.display_to_model(query_points_xy, *frame_shape, self.MODEL_SIZE)
            query_tensor = utils.make_query_tensor(model_xy, self.device)
            prior = (None,) * len(model.blocks)
        else:
            query_count, prior = state.model_state
        mapping = self._window_mapping(state, window) if state is not None and state.crop != window else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=autocast and self.device.type == "cuda"):
            image = self._image_tokens(frame_bgr, window)
            if state is None:
                queries = model.embed_queries(1, query_tensor)
            else:
                # All query timestamps precede the current frame.
                queries = model.mask_token.expand(1, 1, query_count, -1)
            tokens = torch.cat((image, queries), dim=2)
            tracks, visibility, caches = self._predict(tokens, prior, query_count, mapping, autocast)
        yx = tracks[0, 0].cpu().float().numpy()
        xy = utils.model_to_display(yx[:, ::-1], *frame_shape, self.MODEL_SIZE)
        visible = (visibility[0, 0, :, 0] > 0).cpu().numpy()
        good = visible & np.isfinite(xy).all(axis=1)
        return xy, visible, AdaptiveState(window, (query_count, caches), xy[good], misses, frame_shape)
