"""Task05-E evaluation producer: all fresh EDE candidates, no pixel validator.

Candidate completeness/origins are tested offline. The Graph receiver checks
only supplied structural metadata and does not repeat the candidate algorithm.
"""
from time import perf_counter

from graph.vision_input import RAW_EDE_POLICY
from visual.runtime.extended_delaunay_edges import candidate_edges, directed_edges


def build_raw_ede(packet):
    audit = candidate_edges(packet['object_id'], packet['provenance']['source_index'],
                            packet['observed_component_id'], packet['image_shape'], extended=True)
    tick = perf_counter()
    pairs = audit['candidates']
    edge = directed_edges(pairs)
    metadata = dict(edge_policy=dict(RAW_EDE_POLICY), raw_candidate_pairs=pairs.copy(),
        raw_candidate_origins=audit['origins'].tolist(),
        raw_candidate_source_pairs=packet['provenance']['source_index'][pairs].copy(),
        raw_candidate_object_id=packet['object_id'][pairs[:, 0]].copy(),
        raw_candidate_component_id=packet['observed_component_id'][pairs[:, 0]].copy())
    audit['timing_ms']['raw_packaging_ms'] = (perf_counter()-tick)*1000
    return edge, metadata, audit
