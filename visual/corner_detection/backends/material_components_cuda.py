"""CUDA connected components for the detector's two material masks."""

import torch
import triton
import triton.language as tl


@triton.jit
def _root(parents, index, active):
    parent = tl.load(parents + index, active, other=0, volatile=True)
    while tl.sum((active & (parent != index)).to(tl.int32), 0) > 0:
        index = tl.where(active, parent, index)
        parent = tl.load(parents + index, active, other=0, volatile=True)
    return index


@triton.jit
def _join(parents, source, target, active, pixels: tl.constexpr):
    while tl.sum(active.to(tl.int32), 0) > 0:
        first = _root(parents, source, active)
        second = _root(parents, target, active)
        active = active & (first != second)
        upper = tl.maximum(first, second)
        lower = tl.minimum(first, second)
        # Inactive lanes use the sentinel. CAS attaches a root only if
        # another thread has not already attached it to a smaller root.
        ptr = parents + tl.where(active, upper, pixels)
        compare = tl.where(active, upper, -1)
        previous = tl.atomic_cas(ptr, compare, lower, sem="relaxed")
        active = active & (previous != upper)


@triton.jit
def _connect(colors, parents, width: tl.constexpr, pixels: tl.constexpr, block: tl.constexpr):
    idx = tl.program_id(0) * block + tl.arange(0, block)
    color = tl.load(colors + idx, idx < pixels, other=0)
    row = idx % (width * width) // width
    col = idx % width
    for edge in tl.static_range(4):
        dy = 0 if edge == 0 else -1
        dx = -1 if edge < 2 else edge - 2
        target = idx + dy * width + dx
        inside = color & (row + dy >= 0) & (col + dx >= 0) & (col + dx < width)
        adjacent = tl.load(colors + target, inside, other=0)
        _join(parents, idx, target, inside & adjacent, pixels)


@triton.jit
def _count(colors, parents, labels, sizes, pixels: tl.constexpr, block: tl.constexpr):
    idx = tl.program_id(0) * block + tl.arange(0, block)
    color = tl.load(colors + idx, idx < pixels, other=0)
    root = _root(parents, idx, color)
    tl.store(labels + idx, tl.where(color, root, pixels), idx < pixels)
    tl.atomic_add(sizes + root, 1, color, sem="relaxed")


@triton.jit
def _mask(labels, sizes, core, recovery, image_size: tl.constexpr, block: tl.constexpr):
    idx = tl.program_id(0) * block + tl.arange(0, block)
    in_bounds = idx < 2 * image_size
    label = tl.load(labels + idx, in_bounds, other=2 * image_size)
    area = tl.load(sizes + label)
    tl.store(core + idx, area >= 35, in_bounds)
    tl.store(recovery + idx, area >= tl.where(idx < image_size, 35, 8), in_bounds)


def filter_components(colors):
    """Filter the Peg and Box masks by original eight-connected area cutoffs."""
    pixels = colors.numel()
    parents = torch.arange(pixels + 1, dtype=torch.int32, device=colors.device)
    sizes = torch.zeros_like(parents)
    labels = torch.empty_like(parents)
    core = torch.empty_like(colors)
    recovery = torch.empty_like(colors)
    block = 128
    grid = (triton.cdiv(pixels, block),)
    _connect[grid](colors, parents, colors.shape[-1], pixels, block)
    _count[grid](colors, parents, labels, sizes, pixels, block)
    _mask[grid](labels, sizes, core, recovery, pixels // 2, block)
    return core, recovery
