"""Pinned real09-D graph regression on the actual target runtime, no policy."""
import argparse
from pathlib import Path
import numpy as np
from graph.persistent_geometry.evaluate import load_cameras
from .common import read,write
from .geometry import CurrentGeometry


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root;geometry=CurrentGeometry(root/'inputs',read(root/'normalizer.json')['schema']);rows=[]
    for t in [0,25,65]:
        cameras,data=load_cameras(root/'inputs'/f'09d_fk_reference_{t}.npz')
        with np.load(root/'inputs'/f'09d_masks_reference_{t}.npz',allow_pickle=False) as z:masks={k:z[k] for k in z.files}
        graph,_,_=geometry.build(cameras,data,geometry.fk.surfaces(data['robot76']),masks)
        diff={}
        with np.load(root/'inputs'/f'09d_graph_reference_{t}.npz',allow_pickle=False) as old:
            for k in old.files:
                actual=getattr(graph,k).numpy()
                if actual.dtype.kind in 'fc':np.testing.assert_allclose(actual,old[k],atol=2e-6,rtol=1e-5)
                else:np.testing.assert_array_equal(actual,old[k])
                diff[k]=float(np.max(abs(actual-old[k]))) if actual.size else 0.
        rows.append(dict(timestep=t,status='PASS',max_abs=diff))
    write(root/'09d_graph_regression_target.json',dict(status='PASS',frames=rows),immutable=True)
    print(rows,flush=True)


if __name__=='__main__':main()
