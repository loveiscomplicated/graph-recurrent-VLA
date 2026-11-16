"""Inference optimizations for the pinned TAPNext++ PyTorch model."""

from contextlib import contextmanager
from types import MethodType

import torch


def run_block_in_fp16(block):
    """Keep FP32 inputs and outputs at the boundary of an FP16 block."""
    def cast_input(module, args):
        return (args[0].half(),)

    def cast_output(module, args, output):
        return output.float()

    block.half()
    block.register_forward_pre_hook(cast_input)
    block.register_forward_hook(cast_output)


def enable_mps_fp16_vit(tracker):
    """Run spatial Transformer blocks in FP16 with FP32 recurrent inputs."""
    if tracker.device.type != "mps":
        raise ValueError("tap_mps_fp16_vit requires an MPS tracker")
    for block in tracker._model.blocks:
        run_block_in_fp16(block.vit_block)


def compile_mps_recurrent_blocks(tracker):
    """Compile the recurrent residual blocks with their existing dtypes."""
    if tracker.device.type != "mps":
        raise ValueError("tap_mps_compile_recurrent requires an MPS tracker")
    blocks = tracker._model.blocks
    # Compile the shared forward once, passing each block's weights at call time.
    recurrent_type = type(blocks[0].ssm_block)
    compiled_forward = torch.compile(recurrent_type.forward, fullgraph=True, dynamic=False)
    for block in blocks:
        recurrent = block.ssm_block
        recurrent.forward = MethodType(compiled_forward, recurrent)


@contextmanager
def use_past_query_mask_tokens(tracker):
    """Skip spatial sampling while all queries precede the current frame."""
    model = tracker._model
    embed_queries = model.embed_queries

    def embed_mask_tokens(timesteps, query_points):
        batch, queries, _ = query_points.shape
        return model.mask_token.expand(batch, timesteps, queries, -1)

    model.embed_queries = embed_mask_tokens
    try:
        yield
    finally:
        model.embed_queries = embed_queries


def crop_native_patches(tracker, crop):
    """Crop before patch projection and retain full-image position embeddings."""
    model = tracker._model
    image_height, image_width = model.image_size
    patch_height, patch_width = model.patch_size
    rows, cols = image_height // patch_height, image_width // patch_width
    if tracker.input_resolution != image_height or image_height != image_width:
        raise ValueError("tap_native_patch_crop_xyxy requires native input resolution")
    if not isinstance(crop, (list, tuple)) or len(crop) != 4 or any(
        type(edge) is not int for edge in crop
    ):
        raise ValueError("tap_native_patch_crop_xyxy must contain four integer patch edges")
    left, top, right, bottom = crop
    if not (0 <= left < right <= cols and 0 <= top < bottom <= rows):
        raise ValueError(f"Patch crop {crop} is outside the native {cols}x{rows} patch grid")

    positions = model.image_pos_emb.detach().reshape(1, rows, cols, model.width)
    positions = positions[:, top:bottom, left:right].reshape(1, -1, model.width)
    # The upstream forward uses a direct lookup when the grid and embedding
    # have the same token count. Query embeddings still use the full image.
    model.image_pos_emb = torch.nn.Parameter(positions.contiguous(), requires_grad=False)

    vertical = slice(top * patch_height, bottom * patch_height)
    horizontal = slice(left * patch_width, right * patch_width)

    def crop_projection_input(module, args):
        return (args[0][..., vertical, horizontal],)

    model.lin_proj.register_forward_pre_hook(crop_projection_input)
