"""Untimed diagnostic of last-bit link-owner differences in rejected candidates."""
from run import *
import graph.plan_a_geometry.core as core
geo=geometry();scene='10i_s1_0044_t080';size=1024
spec=next(s for s in C.read(FROZEN/'protocol.json')['scenes'] if s['id']==scene)
inputs=load_inputs(scene,size,geo);masks=masks_for(spec,size,0,'actual');rows=[]
with Candidate(9) as ctx:
    for c in inputs[0]:
        expected=core.classify_robot(c,inputs[2])
        for v in ('roi','hulls','packed','rays','atomic'):
            out=RobotFilter(v).classify(c,inputs[2],masks[c.camera],diagnostics=True);p=out.pixels
            bad=out.diagnostics['link_owner']!=expected['link_owner'].ravel()[p]
            ix=np.flatnonzero(bad)
            depth=out.diagnostics['predicted_robot_depth'];ref=expected['predicted_robot_depth'].ravel()[p]
            finite=np.isfinite(depth)&np.isfinite(ref)
            rows.append(dict(camera=c.camera,variant=v,owner_mismatches=len(ix),
                depth_max_abs=float(np.max(np.abs(depth[finite]-ref[finite]),initial=0)),
                confirmed_mismatches=int(np.count_nonzero(out.confirmed!=expected['confirmed_robot'].ravel()[p])),
                uncertain_mismatches=int(np.count_nonzero(out.uncertain!=expected['uncertain'].ravel()[p])),
                examples=[dict(pixel=int(p[i]),expected_owner=int(expected['link_owner'].ravel()[p[i]]),actual_owner=int(out.diagnostics['link_owner'][i]),expected_z=float(ref[i]),actual_z=float(depth[i])) for i in ix[:10]]))
C.write(Path(os.environ['TASK10I_DEBUG_OUTPUT']),dict(scene=scene,size=size,rows=rows))
print(json.dumps(rows))
