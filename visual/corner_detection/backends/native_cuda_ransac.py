"""Load the native ATen loop for CUDA Hole RANSAC."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


@lru_cache(maxsize=1)
def load_cuda_hole_ransac():
    """Build once for the active PyTorch and return its plane fitter."""
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("native") / "cuda_hole_ransac.cpp"
    extension = load(
        name="corner_cuda_hole_ransac",
        sources=[str(source)],
        extra_cflags=["-O3"],
        with_cuda=False,
    )

    def fit_planes(points, parameters, seed, maximum, *, stop_when_stable):
        rng = np.random.default_rng(seed)

        def schedule(count, sample_count, trials):
            # The remaining point count determines the next draw.
            sampled = rng.choice(count, sample_count, replace=False)
            triples = rng.choice(sample_count, (trials, 3), replace=True)
            packed = np.concatenate((sampled, sampled[triples].ravel()))
            return torch.from_numpy(packed)

        return extension.fit_planes(
            points, parameters.minimum_plane_pixels, parameters.ransac_trials,
            maximum, parameters.plane_tolerance_m, stop_when_stable, schedule,
        )

    return fit_planes
