"""Compare existing full-screen CPU/GPU diagnostic contracts, without changing them."""
from run import *
import graph.plan_a_geometry.core as core
output=Path(os.environ['TASK10I_CONTRACT_OUTPUT']);output.mkdir(parents=True,exist_ok=False)
cases=C.read(HERE/'cpu_threads.json')['cases'];specs={s['id']:s for s in C.read(FROZEN/'protocol.json')['scenes']}
rows=[]
with ExperimentLogger(team='integration',model='task10i_fullscreen_contract_diagnostic',job_type='evaluate',seed=42,root=output/'experiments',config=dict(cases=cases)) as log:
    for case in cases:
        geo=geometry();inputs=load_inputs(case['scene'],case['size'],geo)
        for camera in inputs[0]:
            with Candidate(9):actual=core.classify_robot(camera,inputs[2])
            expected=cpu_filter(camera,inputs[2]);counts={}
            for field in ('confirmed_robot','uncertain','candidate','inside_known_link_volume','link_owner'):
                counts[field]=int(np.count_nonzero(actual[field]!=expected[field]))
            a,b=actual['predicted_robot_depth'],expected['predicted_robot_depth'];finite=np.isfinite(a)&np.isfinite(b)
            rows.append(dict(case=case,camera=camera.camera,mismatches=counts,finite_mismatch=int(np.count_nonzero(np.isfinite(a)!=np.isfinite(b))),
                predicted_depth_max_abs=float(np.max(np.abs(a[finite]-b[finite]),initial=0)),
                note='Diagnostic comparison of pre-existing full implementations. Stage 9 GPU remains the new-candidate reference.'))
    C.write(output/'contracts.json',dict(status='COMPLETE',rows=rows));log.link_output(output/'contracts.json')
print(json.dumps(rows))
