// These kernels supplement the existing score, select and weighted-fit kernels.
kernel void native_first_max(
    device const int* votes, device long* best, constant uint& count) {
    uint winner = 0;
    for (uint index = 1; index < count; ++index) {
        if (votes[index] > votes[winner]) winner = index;
    }
    best[0] = long(winner);
}

inline uint keep_outlier(device const bool* selected, uint index, uint size) {
    return uint(index < size && !selected[index]);
}

kernel void native_outlier_block_sizes(
    device const bool* selected, device uint* sizes, constant uint& size,
    uint index [[thread_position_in_grid]], uint block [[threadgroup_position_in_grid]],
    uint subgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint totals[8];
    uint local = simd_sum(keep_outlier(selected, index, size));
    if (lane == 0) totals[subgroup] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && subgroup == 0) {
        uint sum = 0;
        for (uint i = 0; i < 8; ++i) sum += totals[i];
        sizes[block] = sum;
    }
}

kernel void native_compact_outliers(
    device const float* points, device const bool* selected, device const uint* sizes,
    device float* remainder, constant uint& size,
    uint index [[thread_position_in_grid]], uint block [[threadgroup_position_in_grid]],
    uint subgroup [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint totals[8];
    threadgroup uint previous_blocks;
    uint keep = keep_outlier(selected, index, size);
    uint local_offset = simd_prefix_exclusive_sum(keep);
    uint group_count = simd_sum(keep);
    if (lane == 0) totals[subgroup] = group_count;
    if (subgroup == 0) {
        uint sum = 0;
        for (uint i = lane; i < block; i += 32) sum += sizes[i];
        sum = simd_sum(sum);
        if (lane == 0) previous_blocks = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!keep) return;
    uint target = previous_blocks + local_offset;
    for (uint i = 0; i < subgroup; ++i) target += totals[i];
    remainder[3 * target] = points[3 * index];
    remainder[3 * target + 1] = points[3 * index + 1];
    remainder[3 * target + 2] = points[3 * index + 2];
}
