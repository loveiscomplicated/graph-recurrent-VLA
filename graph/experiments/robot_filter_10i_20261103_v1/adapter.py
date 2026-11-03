"""Process-local opt-in scope. Never run concurrent builders in one process."""
from candidates import Candidate
from roi_filter import RobotFilter, RayCache, filtered_masks


class SparseGraph(Candidate):
    def __init__(self,variant='strict',rays=None):
        super().__init__(9);self.variant=variant;self.rays=rays or RayCache()
        self.robot_filter=None;self.raw_masks=None
    def __enter__(self):
        super().__enter__()
        import graph.plan_a_geometry.core as core
        original=core.PlanA.prepare;original_classifier=core.classify_robot;ctx=self
        def prepare(system,cameras,data,masks,surfaces):
            ctx.robot_filter=RobotFilter(ctx.variant,ctx.rays);ctx.raw_masks=masks
            try:return original(system,cameras,data,masks,surfaces)
            finally:ctx.raw_masks=None
        def classify(camera,surfaces,**kw):
            if ctx.raw_masks is None:raise RuntimeError('private ROI path requires PlanA.prepare mask scope')
            if ctx.variant=='strict' and camera.depth_raw_mm.size<=128*128:
                return original_classifier(camera,surfaces,**kw)
            return ctx.robot_filter.classify(camera,surfaces,ctx.raw_masks[camera.camera],**kw)
        self.patch(core.PlanA,'prepare',prepare)
        self.patch(core,'classify_robot',classify)
        self.patch(core,'filtered_masks',filtered_masks)
        return self
