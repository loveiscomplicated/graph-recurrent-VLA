"""Peg/Box row ranges carried from per-object selection through packets.

Ranges describe the unchanged flat row layout; no point arrays are duplicated.
Old packets without ranges can still supply interleaved object IDs.
"""


def object_slices(peg_count, box_count):
    if any(type(n) is not int or n < 0 for n in (peg_count, box_count)):
        raise ValueError('nonnegative integer object counts required')
    return ((0, peg_count), (peg_count, peg_count+box_count))


def object_rows(packet):
    """Selectors for a validated packet: slices for grouped rows, masks for legacy.

    Both ranges must cover all rows, in Peg/Box order, including empty groups.
    Object-ID agreement is checked separately at packet validation boundaries.
    """
    if 'object_slices' not in packet:
        return tuple(packet['object_id'] == oid for oid in (0, 1))
    ranges = packet['object_slices']
    if (not isinstance(ranges, (tuple, list)) or len(ranges) != 2
            or any(not isinstance(pair, (tuple, list)) or len(pair) != 2
                   or any(type(x) is not int for x in pair) for pair in ranges)):
        raise ValueError('object_slices requires two integer start/stop pairs')
    (start, split), (next_start, stop) = ranges
    if start != 0 or split != next_start or not 0 <= split <= stop or stop != len(packet['points']):
        raise ValueError('object_slices must partition all rows in Peg/Box order')
    return slice(start, split), slice(next_start, stop)


def object_slices_match(packet):
    """Scalar NumPy/Torch predicate; Torch callers can aggregate before syncing."""
    if 'object_slices' not in packet:
        return True
    peg, box = object_rows(packet)
    return (packet['object_id'][peg] == 0).all() & (packet['object_id'][box] == 1).all()
