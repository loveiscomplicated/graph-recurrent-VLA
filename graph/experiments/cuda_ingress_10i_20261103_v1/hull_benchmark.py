"""Current frame preparation only; ordered parallel Qhull stays bitwise exact."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .benchmark import geometry,load
from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
from utils.experiment_logger import ExperimentLogger


def main(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    geo=geometry('triton_fused');cameras,data,surfaces,_=load('10i_s1_0924_t080',512,geo)
    filter=CudaRobotFilter();reference=filter.prepare_surfaces(surfaces,cameras)
    pools={n:ThreadPoolExecutor(n) if n else None for n in (0,2,4,8)};rows=[]
    def run(n):
        start=time.perf_counter();p=filter.prepare_surfaces(surfaces,cameras,executor=pools[n]);ms=(time.perf_counter()-start)*1000
        np.testing.assert_array_equal(p.starts,reference.starts)
        for (_,a),(_,b) in zip(p.hulls,reference.hulls):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(p.packed,reference.packed)
        return ms
    with ExperimentLogger(team='integration',model='task10i_current_hull_preparation',job_type='evaluate',seed=42,
            root=output/'experiments',config=dict(scene='10i_s1_0924_t080',size=512,repeats=15,warmup=3,
                input_identity=load.identity,boundary='current world vertices, both camera bounds, visible Qhull equations and packed frame metadata; CPU only; excludes CUDA upload and filter')) as log:
        try:
            for _ in range(3):
                for n in pools:run(n)
            for i in range(15):
                order=list(pools);order=order[i%4:]+order[:i%4]
                rows.append(dict(repeat=i,ms={n:run(n) for n in order}))
            path=output/'results.json';path.write_text(json.dumps(dict(status='PASS',rows=rows,equations_metadata_bitwise_exact=True),indent=2));log.link_output(path)
        finally:
            for pool in pools.values():
                if pool is not None:pool.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
