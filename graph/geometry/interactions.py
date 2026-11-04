"""Deterministic robot/object and peg/box interaction edge helpers."""
import numpy as np

def _integer(value, minimum, name):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be integer >= {minimum}')

def nearest_robot_pairs(positions, object_id, source_index, k=4):
    _integer(k, 1, 'k')
    pairs = []
    for robot in range(3):
        for oid in (0, 1):
            rows = np.flatnonzero(object_id == oid)
            delta = positions[rows+3].astype(np.float64) - positions[robot].astype(np.float64)
            d2 = np.sum(delta*delta, axis=1)
            order = np.lexsort((rows, source_index[rows], d2))[:k]
            pairs.extend((robot, int(row)+3) for row in rows[order])
    return np.asarray(pairs, np.int64).reshape(-1, 2)

def global_peg_box_pairs(points, object_id, source_index, M=16, block_rows=64):
    """Exact all-pair block scan; O(PQ) distances, O(block_rows*Q+M) scratch.

    Total order is (squared distance, peg source, box source, peg row, box
    row). Keeping each block's first M cannot discard a global first-M pair.
    No cutoff, approximation, normal/depth gate or geometry filter is used.
    """
    _integer(M, 0, 'M')
    _integer(block_rows, 1, 'block_rows')
    peg, box = (np.flatnonzero(object_id == oid) for oid in (0, 1))
    best = np.empty((0, 2), np.int64)
    best_dist = np.empty(0, np.float64)
    if M == 0 or not len(peg) or not len(box):
        return best
    xyz = points.astype(np.float64)
    for start in range(0, len(peg), block_rows):
        rows = peg[start:start+block_rows]
        delta = xyz[rows, None] - xyz[box][None]
        d2 = np.sum(delta*delta, axis=-1).reshape(-1)
        pairs = np.column_stack((np.repeat(rows, len(box)), np.tile(box, len(rows))))
        pairs = np.concatenate((best, pairs))
        d2 = np.r_[best_dist, d2]
        order = np.lexsort((pairs[:, 1], pairs[:, 0], source_index[pairs[:, 1]],
                            source_index[pairs[:, 0]], d2))[:M]
        best, best_dist = pairs[order], d2[order]
    return best

def bidirectional(pairs):
    pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
    return np.concatenate((pairs, pairs[:, ::-1]), axis=0).T.copy()

def edge_attributes(positions, edges):
    delta = positions[edges[1]].astype(np.float64) - positions[edges[0]].astype(np.float64)
    distance = np.linalg.norm(delta, axis=1)
    direction = np.divide(delta, distance[:, None], out=np.zeros_like(delta), where=distance[:, None] > 0)
    return np.column_stack((distance, direction)).astype(np.float32)
