"""Installed simulator/FK and preserved09-C baseline regression, read-only."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from graph.integration_baseline.io import write_json
from graph.persistent_geometry.collect import FULL_PLAN
from graph.persistent_geometry.preflight import runtime,baseline,candidate_g0_identity
from graph.persistent_geometry.evaluate import load_cameras
from graph.persistent_geometry.self_filter import LinkSurface,classify_robot
from graph.persistent_geometry.fusion import make_packet,observations_from_packet,filtered_masks,consistent_views
from graph.persistent_geometry.map import current_records
from graph.persistent_geometry.graph import build_graph

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    capture=Path(json.loads(Path(__file__).with_name('protocol.json').read_text())['development_capture'])
    runtime(a.output,capture)
    plan=json.loads(FULL_PLAN.read_text());baseline(a.output,plan);candidate_g0_identity(a.output,capture,plan)
    manifest=json.loads((capture/'capture_manifest.json').read_text());checks=[]
    for ep in manifest['episodes'][:2]:
        source=capture/'collection'/f'seed{ep["seed"]}'
        with np.load(source/'robot_meshes.npz',allow_pickle=False) as ar:
            names=ar['links'].tolist();vertices=[ar[f'vertices_{i}'].copy() for i in range(len(names))]
        for t in (0,25,65):
            cameras,data=load_cameras(source/f'f{t:04d}_input.npz')
            with np.load(source/f'f{t:04d}_masks.npz',allow_pickle=False) as ar:masks={k:ar[k] for k in ar.files}
            surfaces=[LinkSurface(n,v,p) for n,v,p in zip(names,vertices,data['robot_link_world'])]
            clean={};observed={}
            for c in cameras:
                clean[c.camera],_=filtered_masks(masks[c.camera],classify_robot(c,surfaces))
                packet,_=make_packet(c,clean[c.camera]);observed[c.camera]=observations_from_packet(packet,c)
            dual,_=consistent_views(sum(observed.values(),[]),cameras,clean)
            for variant,records,cs in [('G1',current_records(observed[ep['roles']['main']],fuse_views=False),[c for c in cameras if c.camera==ep['roles']['main']]),('G3',current_records(dual),cameras)]:
                graph=build_graph(records,data['robot_world_geometry'],data['tcp_pose'],t,cameras=cs,point_cap=128)
                reference=Path('/workspace/task09c/development_geometry_runs/ba145f98f3594d28abd69ee82eb5fca6')/'geometry'/f'seed{ep["seed"]}'/f'f{t:04d}'/(variant+'_graph.npz')
                with np.load(reference,allow_pickle=False) as old:
                    for name in old.files:
                        actual=getattr(graph,name).numpy()
                        if actual.dtype.kind in 'fc':np.testing.assert_allclose(actual,old[name],atol=1e-6,rtol=1e-5)
                        else:np.testing.assert_array_equal(actual,old[name])
                checks.append(dict(seed=ep['seed'],timestep=t,variant=variant,status='PASS'))
    write_json(a.output/'G1_G3_regression.json',dict(status='PASS',checks=checks,original_artifacts_read_only=True))
if __name__=='__main__':main()
