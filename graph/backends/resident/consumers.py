"""Reuse existing exact native small-graph operations, scoped to one build.

Task-local contexts already use process-local patches. This context is likewise
not a concurrent builder API. It does not alter sampling, radii or tie rules.
"""
from contextlib import ExitStack
from unittest.mock import patch
import graph.backends.native as native
import graph.backends.surface_edges as edges


class SmallGraphConsumers(ExitStack):
    def __enter__(self):
        super().__enter__()
        import graph.geometry.core as core
        import graph.geometry.graph as gm
        original=core.deduplicate
        self.enter_context(patch.object(core,'deduplicate',lambda rows,distance=.003:
            native.deduplicate(rows,distance,fallback=original)))
        self.enter_context(patch.object(core,'farthest_rows',native.farthest_rows))
        self.enter_context(patch.object(gm,'farthest_rows',native.farthest_rows))
        self.enter_context(patch.object(gm,'surface_edges',edges.surface_edges))
        return self
