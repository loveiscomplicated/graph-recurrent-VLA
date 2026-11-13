"""Compatibility entrypoint for selection from checksum-verified graph caches."""

from data_collection.physical_vertex_selection import (
    initialize_cached_worker as initialize_worker,
    inspect_cached_frame as inspect_frame,
    prepare_selection,
    select_source_rows,
)


def prepare(config):
    """Select cached detector XYZ using reprojected pixels, without re-detection."""
    return prepare_selection(config, use_cache=True)
