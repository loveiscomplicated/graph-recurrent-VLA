"""Metal connected components for the detector's two material masks."""

from functools import lru_cache

import torch


_SHADER = r"""
#include <metal_stdlib>
using namespace metal;

uint root(device atomic_uint *parents, uint index) {
    uint next = atomic_load_explicit(parents + index, memory_order_relaxed);
    while (next != index) {
        index = next;
        next = atomic_load_explicit(parents + index, memory_order_relaxed);
    }
    return index;
}

kernel void connect(
    device const bool *colors, device atomic_uint *parents,
    constant uint &width, uint idx [[thread_position_in_grid]]
) {
    if (!colors[idx]) return;
    const int row = int((idx % (width * width)) / width);
    const int col = int(idx % width);
    const int2 offsets[] = {int2(-1, 0), int2(-1, -1), int2(0, -1), int2(1, -1)};
    for (int2 delta : offsets) {
        int x = col + delta.x;
        if (x < 0 || x >= int(width) || row + delta.y < 0) continue;
        uint target = uint(int(idx) + delta.y * int(width) + delta.x);
        if (!colors[target]) continue;
        while (true) {
            uint first = root(parents, idx);
            uint second = root(parents, target);
            if (first == second) break;
            // Attach a root only if another thread has not already linked it.
            uint upper = max(first, second);
            if (atomic_compare_exchange_weak_explicit(
                parents + upper, &upper, min(first, second),
                memory_order_relaxed, memory_order_relaxed
            )) break;
        }
    }
}

kernel void count_labels(
    device const bool *colors, device atomic_uint *parents,
    device uint *labels, device atomic_uint *sizes, constant uint &pixels,
    uint idx [[thread_position_in_grid]]
) {
    if (!colors[idx]) {
        labels[idx] = pixels;
        return;
    }
    uint label = root(parents, idx);
    labels[idx] = label;
    atomic_fetch_add_explicit(sizes + label, 1, memory_order_relaxed);
}

kernel void write_masks(
    device const uint *labels, device const uint *sizes,
    device bool *core, device bool *recovery, constant uint &image_size,
    uint idx [[thread_position_in_grid]]
) {
    uint area = sizes[labels[idx]];
    core[idx] = area >= 35;
    recovery[idx] = area >= (idx < image_size ? 35 : 8);
}
"""


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SHADER)


def filter_components(colors):
    """Filter the Peg and Box masks by original eight-connected area cutoffs."""
    pixels = colors.numel()
    parents = torch.arange(pixels, dtype=torch.int32, device=colors.device)
    sizes = torch.zeros(pixels + 1, dtype=torch.int32, device=colors.device)
    labels = torch.empty_like(parents)
    core = torch.empty_like(colors)
    recovery = torch.empty_like(colors)
    kernels = _library()
    kernels.connect(colors, parents, colors.shape[-1], threads=pixels)
    kernels.count_labels(colors, parents, labels, sizes, pixels, threads=pixels)
    kernels.write_masks(labels, sizes, core, recovery, pixels // 2, threads=pixels)
    return core, recovery
