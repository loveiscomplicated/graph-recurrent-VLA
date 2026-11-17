"""Execution settings for matched corner models without changing their weights."""

import torch

from graph.temporal.corner_tracking.model import differentiable_mamba_step


def configure_execution(motion, gnn, settings):
    """Compile tensor-only regions while retaining ordinary checkpoint keys.

    Python tracking decisions stay eager. CUDA graphs are excluded because many
    recurrent forwards retain their outputs until one episode-level backward.
    """
    torch.set_float32_matmul_precision(settings["float32_matmul_precision"])
    gnn.corner_graph_batch_size = settings["graph_batch_size"]
    if not settings["compile"]:
        return
    if motion.device.type != "cuda":
        raise ValueError("Corner tracking compilation requires CUDA for the verified execution path")
    # Sequential regions share Dynamo's wrapper code. Different fixed module
    # widths and train/eval modes need a bounded set of distinct specializations.
    torch._dynamo.config.recompile_limit = settings["compile_recompile_limit"]
    options = dict(dynamic=True, fullgraph=True, mode=settings["compile_mode"])
    motion.record_encoder.compile(**options)
    motion._predict_forward = torch.compile(motion._predict_values, **options)
    motion._context_forward = torch.compile(motion._context_values, **options)
    if motion.architecture == "mamba":
        motion._mamba_step = torch.compile(differentiable_mamba_step, **options)
    else:
        for block in motion.blocks:
            # The installed Inductor fails the backward stride assertion when
            # compiling streaming MultiheadAttention with changing history.
            # Keep attention eager; compile its dense normalization/FFN regions.
            for module in (block.attention_norm, block.feedforward_norm, block.feedforward):
                module.compile(**options)
    # Compile the dense message-passing regions, leaving data-dependent graph
    # selection and validation outside the compiler.
    for module in (gnn.node_encoder, *gnn.peg_layers, *gnn.hole_layers,
                   gnn.cross_layer, gnn.peg_head, gnn.hole_head, gnn.endpoint_residual):
        module.compile(**options)
