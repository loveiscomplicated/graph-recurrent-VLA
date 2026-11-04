"""Reference-sample ledgers; no model, sampler or point budget decisions."""
import numpy as np
from scipy.spatial import cKDTree


def covered(points, correct, reference, radius=.004):
    points = np.asarray(points, float).reshape(-1, 3)
    correct = np.asarray(correct, bool)
    reference = np.asarray(reference, float).reshape(-1, 3)
    if len(points) != len(correct) or not np.isfinite(points).all() or not np.isfinite(reference).all():
        raise ValueError('unaligned/nonfinite coverage geometry')
    if not correct.any():
        return np.zeros(len(reference), bool)
    return cKDTree(points[correct]).query(reference)[0] < radius


def transition(previous, current, region=None):
    previous = np.asarray(previous, bool)
    current = np.asarray(current, bool)
    if previous.shape != current.shape:
        raise ValueError('reference identity mismatch')
    region = np.ones(len(previous), bool) if region is None else np.asarray(region, bool)
    if region.shape != previous.shape:
        raise ValueError('region identity mismatch')
    n = int(region.sum())
    before = int((region & previous).sum())
    after = int((region & current).sum())
    lost = int((region & previous & ~current).sum())
    recovered = int((region & ~previous & current).sum())
    assert after-before == recovered-lost
    return dict(reference_samples=n, before=before, after=after, lost=lost, recovered=recovered,
        loss_fraction=lost/n if n else None, net_loss_fraction=(before-after)/n if n else None)


def high_resolution_gain(low_raw, high_raw):
    low_raw = np.asarray(low_raw, bool)
    high_raw = np.asarray(high_raw, bool)
    if low_raw.shape != high_raw.shape:
        raise ValueError('paired reference identity mismatch')
    return high_raw & ~low_raw


def retention(coverage, region):
    coverage, region = np.asarray(coverage, bool), np.asarray(region, bool)
    if coverage.shape != region.shape:
        raise ValueError('reference region mismatch')
    return float(coverage[region].mean()) if region.any() else None
