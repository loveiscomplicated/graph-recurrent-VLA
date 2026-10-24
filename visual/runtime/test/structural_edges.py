"""Task 02-C observation-only comparison, never called by production runtime.

Sparse raster-pair evidence is NOT graph connectivity or a semantic edge class.
Depth order only indicates potential occluding/occluded sides. No GT or RGB is
accepted. Thresholding retains overlapping causes instead of resolving them.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from visual.runtime.geometry import (
    GeometryFrame, GeometryParameters, build_organized_geometry,
    compute_geometric_features, estimate_surface_normals,
)
from visual.runtime.test.normal_estimators import finite_difference, local_plane

METHODS = ('baseline', 'fd2', 'pca5')
THRESHOLDS = (2, 5, 10, 15, 20, 25, 30, 40, 60, 90, 180)
PARAMETERS = GeometryParameters()


from visual.runtime.edge_evidence import PairEvidence, pair_evidence


def compare(frame, rotation=0):
    start = perf_counter()
    geometry = build_organized_geometry(frame)
    geometry_ms = (perf_counter()-start)*1000
    results, evidence = {}, {}
    order = METHODS[rotation % 3:]+METHODS[:rotation % 3]
    for method in order:
        start = perf_counter()
        if method == 'baseline':
            normals = {i: estimate_surface_normals(geometry, i, PARAMETERS) for i in geometry.object_masks}
        elif method == 'fd2':
            normals = {i: finite_difference(geometry, i, 2, PARAMETERS) for i in geometry.object_masks}
        else:
            normals = {i: local_plane(geometry, i, 5, PARAMETERS) for i in geometry.object_masks}
        after_normal = perf_counter()
        features = {i: compute_geometric_features(geometry, i, *normals[i], PARAMETERS) for i in normals}
        after_features = perf_counter()
        evidence[method] = {i: pair_evidence(geometry, i, *normals[i]) for i in normals}
        after_pairs = perf_counter()
        # One illustrative threshold pass; the evaluation sweep is excluded.
        for pair in evidence[method].values():
            pair.candidates(20)
        end = perf_counter()
        timing = dict(geometry=geometry_ms, normals=(after_normal-start)*1000,
                      features=(after_features-after_normal)*1000,
                      pair_evidence=(after_pairs-after_features)*1000,
                      threshold=(end-after_pairs)*1000, total=geometry_ms+(end-start)*1000)
        results[method] = GeometryFrame(frame.source_frame, frame.camera, frame.frame_idx,
                                        frame.timestamp, geometry, features, timing)
    return results, evidence
