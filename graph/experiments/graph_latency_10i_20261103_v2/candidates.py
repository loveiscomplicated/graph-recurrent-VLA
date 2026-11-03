"""Opt-in successive candidates. Never edits shared Graph or simulator files."""
from fast_graph import FastGraph
import native


class Candidate(FastGraph):
    def __init__(self,stage):
        super().__init__('fast_v5_tree');self.stage=stage

    def __enter__(self):
        super().__enter__()
        import graph.plan_a_geometry.core as core
        if self.stage>=1:
            original=core.deduplicate
            self.patch(core,'deduplicate',lambda r,distance=.003:native.deduplicate(r,distance,fallback=original))
        if self.stage>=2:
            from visual.runtime.test import run_voxel_structural_sampling as voxel
            import graph.persistent_geometry.graph as gm
            self.patch(voxel,'representatives',native.representatives)
            self.patch(core,'farthest_rows',native.farthest_rows)
            self.patch(gm,'farthest_rows',native.farthest_rows)
        if self.stage>=3:
            import robot
            self.patch(core,'classify_robot',lambda *a,**kw:robot.classify(self,*a,**kw))
        if self.stage>=4:
            import dense
            import graph.integration_baseline.vision as vision
            self.patch(vision,'extract_sampling_evidence',dense.extract)
        if self.stage>=5:
            import selection
            from visual.runtime.test import run_variable_rate_shape_first as sf
            from visual.runtime.test import run_priority_fps_metric_control as prior
            import numpy as np
            def feature_rows(case,flat):
                # SF's saved_flat is strictly source-ascending; preserve the old
                # available-only join without a full Python dictionary per frame.
                saved=case['saved_flat'];rows=np.searchsorted(saved,flat)
                available=rows<len(saved)
                if len(saved):available&=saved[np.minimum(rows,len(saved)-1)]==flat
                positions=np.flatnonzero(available)
                return available,positions,{k:v[rows[available]].copy() for k,v in case['saved_features'].items()}
            self.patch(sf,'fps_indices',selection.fps)
            self.patch(prior,'feature_rows',feature_rows)
        if self.stage>=6:
            import packing
            from visual.runtime.test import run_variable_rate_shape_first as sf
            self.patch(sf,'make_packet',packing.selection_packet)
            # Capture the generic fallback before ownership-specialized SF path.
            fallback=core.deduplicate
            self.patch(core.PlanA,'represent',packing.sf_represent(core.PlanA.represent,fallback))
        if self.stage>=7:
            import packing
            import graph.persistent_geometry.graph as gm
            self.patch(gm,'global_peg_box_pairs',packing.global_pairs)
        if self.stage>=8:
            import edges
            import graph.persistent_geometry.graph as gm
            self.patch(gm,'surface_edges',edges.surface_edges)
        if self.stage>=9:
            # Same gates, two implementations. The grouped launch helps small
            # images; high-resolution clipping was no faster in paired runs.
            import robot
            def adaptive_robot(camera,surfaces,**kw):
                if camera.depth_raw_mm.size<=128*128:return robot.classify(self,camera,surfaces,**kw)
                return self.classify_robot(camera,surfaces,**kw)
            self.patch(core,'classify_robot',adaptive_robot)
        return self

    def __exit__(self,*args):
        super().__exit__(*args)
        # Nested patches store methods/closures bound to this context. Release
        # them after restoration instead of leaving context cycles for the GC.
        self.restore.clear();self.support=None
