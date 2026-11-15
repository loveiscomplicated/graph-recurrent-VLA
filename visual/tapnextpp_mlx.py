"""Single-frame MLX inference blocks for the pinned TAPNext++ checkpoint."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _array(tensor):
    """Convert a checkpoint tensor once, outside inference."""
    array = tensor.detach().cpu().contiguous().numpy()
    return mx.array(array)


def _linear(x, weight, bias):
    return mx.addmm(bias, x, weight.T)


def _gelu(x, *, approximate):
    # Preserve the FP16 rounding at the activation boundary.
    values = x.astype(mx.float32)
    activation = nn.gelu_approx(values) if approximate else nn.gelu(values)
    return activation.astype(x.dtype)


@dataclass(frozen=True)
class RecurrentState:
    """Persistent convolution history and RG-LRU state for one block."""

    history: mx.array
    hidden: mx.array


class OnlineBlock:
    """TRecViT for a single frame of [image patches + query tokens, width]."""

    def __init__(self, block):
        self.weights = {key: _array(value) for key, value in block.state_dict().items()}
        recurrent = block.ssm_block.recurrent_block
        self.width = recurrent.width
        self.num_heads = recurrent.rg_lru.num_heads
        self.conv_width = recurrent.conv_1d.temporal_width
        self.rms_eps = block.ssm_block.temporal_pre_norm.eps
        self.channel_eps = block.ssm_block.channel_pre_norm.eps
        self.vit_eps = (block.vit_block.ln_1.eps, block.vit_block.ln_2.eps)
        self.head_scale = (self.width // self.num_heads) ** -0.5
        prefix = "ssm_block.recurrent_block.rg_lru."
        self.gate_weights = mx.concatenate(
            (self.weights[prefix + "input_gate.w"], self.weights[prefix + "a_gate.w"]), axis=-1
        )
        self.gate_biases = mx.concatenate(
            (self.weights[prefix + "input_gate.b"], self.weights[prefix + "a_gate.b"]), axis=-1
        )
        self.decay = nn.softplus(self.weights[prefix + "a_param"])
        self.temporal_gain = self.weights["ssm_block.temporal_pre_norm.scale"] + 1
        self.channel_gain = self.weights["ssm_block.channel_pre_norm.scale"] + 1
        self.mlp_up = self.weights["ssm_block.mlp_block.ffw_up.w"].transpose(1, 0, 2)
        self.mlp_up = mx.contiguous(self.mlp_up.reshape(self.width, -1))
        self.mlp_bias = self.weights["ssm_block.mlp_block.ffw_up.b"].reshape(-1)
        mx.eval(self.weights, self.gate_weights, self.gate_biases, self.decay,
                self.temporal_gain, self.channel_gain, self.mlp_up, self.mlp_bias)

    def _project(self, x, name):
        return _linear(x, self.weights[name + ".weight"], self.weights[name + ".bias"])

    def _recurrent(self, x, state):
        prefix = "ssm_block.recurrent_block."
        x = mx.fast.rms_norm(x, self.temporal_gain, self.rms_eps)
        projected = self._project(x, prefix + "linear_x")
        gate_y = _gelu(self._project(x, prefix + "linear_y"), approximate=True)
        history = state.history if state is not None else mx.zeros(
            (x.shape[0], self.conv_width - 1, self.width), dtype=x.dtype
        )
        window = mx.concatenate((history, projected[:, None]), axis=1)
        convolved = mx.sum(window * self.weights[prefix + "conv_1d.w"], axis=1)
        convolved = convolved + self.weights[prefix + "conv_1d.b"]
        grouped = convolved.reshape(x.shape[0], self.num_heads, -1).transpose(1, 0, 2)
        gates = grouped @ self.gate_weights + self.gate_biases[:, None, :]
        input_gate, decay_gate = mx.split(mx.sigmoid(gates), 2, axis=-1)
        gated = convolved * input_gate.transpose(1, 0, 2).reshape(convolved.shape)
        if state is not None:
            decay_gate = decay_gate.transpose(1, 0, 2).reshape(convolved.shape)
            log_decay = (-8.0 * decay_gate) * self.decay
            gated = mx.exp(log_decay) * state.hidden + gated * mx.sqrt(1 - mx.exp(2 * log_decay))
        output = self._project(gated * gate_y, prefix + "linear_out")
        return output, RecurrentState(window[:, 1:], gated)

    def _recurrent_mlp(self, x):
        normalized = mx.fast.rms_norm(x, self.channel_gain, self.channel_eps).astype(mx.float16)
        gate, value = mx.split(normalized @ self.mlp_up + self.mlp_bias, 2, axis=-1)
        hidden = _gelu(gate, approximate=True) * value
        return self._project(hidden, "ssm_block.mlp_block.ffw_down").astype(mx.float32)

    def _layer_norm(self, x, name, eps):
        return mx.fast.layer_norm(x, self.weights[name + ".weight"], self.weights[name + ".bias"], eps)

    def _spatial(self, x):
        x = x.astype(mx.float16)
        normalized = self._layer_norm(x, "vit_block.ln_1", self.vit_eps[0])
        prefix = "vit_block.self_attention."
        packed = _linear(normalized, self.weights[prefix + "in_proj_weight"],
                         self.weights[prefix + "in_proj_bias"])
        q, k, v = (part.reshape(1, x.shape[0], self.num_heads, -1).transpose(0, 2, 1, 3)
                   for part in mx.split(packed, 3, axis=-1))
        attention = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_scale)
        attention = attention.transpose(0, 2, 1, 3).reshape(x.shape)
        residual = x + self._project(attention, prefix + "out_proj")
        normalized = self._layer_norm(residual, "vit_block.ln_2", self.vit_eps[1])
        hidden = _gelu(self._project(normalized, "vit_block.mlp.0"), approximate=False)
        return (residual + self._project(hidden, "vit_block.mlp.3")).astype(mx.float32)

    def __call__(self, tokens, state=None):
        """Process one frame, keeping recurrence and residual inputs in FP32."""
        temporal, state = self._recurrent(tokens, state)
        residual = tokens + temporal
        return self._spatial(residual + self._recurrent_mlp(residual)), state


def compile_continuation(block):
    """Compile the fixed-shape continuation while keeping state explicit."""
    def continuation(tokens, history, hidden):
        output, state = block(tokens, RecurrentState(history, hidden))
        return output, state.history, state.hidden

    return mx.compile(continuation)


class OnlineTracker:
    """Run the native 256px TAPNext++ model with all frame state on MLX GPU."""

    def __init__(self, model, crop):
        self.device = mx.Device(mx.gpu)
        self.runtime_version = mx.__version__
        self.input_resolution = model.image_size[0]
        self.patch_size = model.patch_size[0]
        self.width = model.width
        grid = self.input_resolution // self.patch_size
        if crop is None:
            crop = (0, 0, grid, grid)
        if (not isinstance(crop, (tuple, list)) or len(crop) != 4
                or any(type(edge) is not int for edge in crop)):
            raise ValueError("Expected four integer native patch edges")
        left, top, right, bottom = crop
        if not (0 <= left < right <= grid and 0 <= top < bottom <= grid):
            raise ValueError(f"Patch crop {crop} is outside the {grid}x{grid} grid")
        self.crop = (slice(top * self.patch_size, bottom * self.patch_size),
                     slice(left * self.patch_size, right * self.patch_size))
        positions = model.image_pos_emb.detach().reshape(grid, grid, model.width)
        self.image_pos = _array(positions[top:bottom, left:right].reshape(-1, model.width))
        self.patch_weight = mx.contiguous(_array(model.lin_proj.weight).transpose(0, 2, 3, 1))
        self.patch_bias = _array(model.lin_proj.bias)
        self.query_pos = _array(model.query_pos_embed).reshape(
            self.input_resolution, self.input_resolution, model.width
        )
        self.query_token = _array(model.point_query_token).reshape(1, model.width)
        self.mask_token = _array(model.mask_token).reshape(1, model.width)
        self.norm = (_array(model.encoder_norm.weight), _array(model.encoder_norm.bias),
                     model.encoder_norm.eps)
        self.coordinate_head = self._head_weights(model.coordinate_head)
        self.visible_head = self._head_weights(model.visible_head)
        blocks = []
        for block in model.blocks:
            block.vit_block.half()
            block.ssm_block.mlp_block.half()
            blocks.append(OnlineBlock(block))
        self.blocks = blocks
        self.continuations = [compile_continuation(block) for block in blocks]
        self.bins = mx.arange(self.input_resolution)
        mx.eval(self.image_pos, self.patch_weight, self.patch_bias, self.query_pos,
                self.query_token, self.mask_token, self.norm, self.coordinate_head,
                self.visible_head, self.bins)

    @classmethod
    def from_checkpoint(cls, checkpoint, input_resolution, crop):
        """Load the official PyTorch checkpoint on CPU and convert its tensors once."""
        from tapnet.tapnextpp.votsp2026.model import TAPNextPP

        if not mx.metal.is_available():
            raise RuntimeError("TAPNext++ MLX requires an Apple Silicon Metal GPU")
        if input_resolution != TAPNextPP.MODEL_SIZE:
            raise ValueError("TAPNext++ MLX supports its native 256px resolution")
        mx.set_default_device(mx.gpu)
        reference = TAPNextPP.from_checkpoint(checkpoint, device="cpu", input_resolution=input_resolution)
        return cls(reference._model, crop)

    @staticmethod
    def _head_weights(head):
        return [(_array(layer.weight), _array(layer.bias), getattr(layer, "eps", None))
                for layer in head if hasattr(layer, "weight")]

    @staticmethod
    def _predict(x, parameters):
        for weight, bias, eps in parameters:
            if eps is not None:
                x = _gelu(mx.fast.layer_norm(x, weight, bias, eps), approximate=False)
            else:
                x = _linear(x, weight, bias)
        return x

    def _image_tokens(self, frame_bgr):
        image = mx.array(frame_bgr[..., ::-1].copy()).astype(mx.float32)
        size = self.input_resolution
        height, width, channels = image.shape
        if (height, width) == (2 * size, 2 * size):
            # Half-pixel bilinear sampling at exactly 2x gives four equal weights.
            image = image.reshape(size, 2, size, 2, channels).mean(axis=(1, 3))
        elif (height, width) != (size, size):
            raise ValueError("TAPNext++ MLX expects a 256x256 or 512x512 BGR frame")
        vertical, horizontal = self.crop
        image = image[vertical, horizontal] / 127.5 - 1
        patches = mx.conv2d(image[None], self.patch_weight, stride=self.patch_size)
        return patches.reshape(-1, self.width) + self.patch_bias + self.image_pos

    def _initial_queries(self, points_xy):
        # align_corners=False evaluates model-space pixel centres at x/y - 0.5.
        xy = mx.array(points_xy) - 0.5
        floor = mx.floor(xy)
        fraction = xy - floor
        start = floor.astype(mx.int32)
        size = self.input_resolution
        sampled = self.query_token
        for dx, x_weight in ((0, 1 - fraction[:, 0]), (1, fraction[:, 0])):
            for dy, y_weight in ((0, 1 - fraction[:, 1]), (1, fraction[:, 1])):
                x = start[:, 0] + dx
                y = start[:, 1] + dy
                valid = (0 <= x) & (x < size) & (0 <= y) & (y < size)
                embedding = self.query_pos[mx.clip(y, 0, size - 1), mx.clip(x, 0, size - 1)]
                sampled = sampled + embedding * (x_weight * y_weight * valid)[:, None]
        return sampled

    def track_frame(self, frame_bgr, query_points_xy=None, state=None, autocast=False):
        """Return source-pixel coordinates, visibility and the next MLX state."""
        from tapnet.tapnextpp.votsp2026 import utils

        if autocast:
            raise ValueError("CUDA autocast does not apply to TAPNext++ MLX")
        height, width = frame_bgr.shape[:2]
        if state is None:
            if query_points_xy is None or len(query_points_xy) == 0:
                raise ValueError("The initial frame requires at least one query")
            query_count = len(query_points_xy)
            model_xy = utils.display_to_model(query_points_xy, height, width, self.input_resolution)
            queries = self._initial_queries(model_xy)
        else:
            if query_points_xy is not None:
                raise ValueError("TAPNext++ MLX accepts queries only on the initial frame")
            query_count, caches = state
            queries = mx.broadcast_to(self.mask_token, (query_count, self.width))
        x = mx.concatenate((self._image_tokens(frame_bgr), queries))
        next_caches = []
        for index, block in enumerate(self.blocks):
            if state is None:
                x, cache = block(x)
            else:
                prior = caches[index]
                x, history, hidden = self.continuations[index](x, prior.history, prior.hidden)
                cache = RecurrentState(history, hidden)
            next_caches.append(cache)

        queries = mx.fast.layer_norm(x[-query_count:], *self.norm)
        logits = self._predict(queries, self.coordinate_head).reshape(query_count, 2, -1)
        peaks = mx.argmax(logits, axis=-1, keepdims=True)
        window = mx.abs(peaks - self.bins) <= 20
        probabilities = mx.softmax(logits * 0.5, axis=-1) * window
        weighted = mx.sum(probabilities * self.bins, axis=-1)
        yx = weighted / mx.sum(probabilities, axis=-1) + 0.5
        visible = self._predict(queries, self.visible_head)[:, 0] > 0
        mx.eval(yx, visible, [(cache.history, cache.hidden) for cache in next_caches])
        xy = utils.model_to_display(np.asarray(yx)[:, ::-1], height, width, self.input_resolution)
        return xy, np.asarray(visible), (query_count, next_caches)
