"""Record identical live inputs; reuse the already validated live GPU timing."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from graph.experiments.live_stage_timing_10i_20261104_v1 import run as live


COMMIT = '709798983435875e374e9a8cd6df8af9e6c7bb7f'


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--legacy-source', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index('--output')+1])
    from graph.plan_a_bc import rollout
    module = ast.parse((args.legacy_source/'graph/plan_a_bc/rollout.py').read_text())
    function, = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name=='sensors']
    namespace = dict(rollout.__dict__)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<frozen-CPU-sensors>', 'exec'), namespace)
    legacy_sensors = namespace['sensors']
    OriginalMeasurement = live.LiveMeasurement

    class CaptureMeasurement(OriginalMeasurement):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.input_rows = []
            self.host_input = None

        def install(self, stack):
            super().install(stack)
            original_sensors = rollout.sensors
            original_build = rollout.CurrentGeometry.build
            def sensors(env, obs, identity, t, schema):
                if self.mode=='natural':
                    # Before the GPU sensors/Graph timer starts. This executes
                    # the original CPU sensors function against shared native
                    # observations, including its image D2H boundary.
                    torch.cuda.synchronize()
                    start = time.perf_counter_ns()
                    host = legacy_sensors(env, obs, identity, t, schema)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter_ns()-start)/1e6
                    self.host_input = (host, ms)
                return original_sensors(env, obs, identity, t, schema)
            def build(geometry, cameras, data, surfaces, masks, **kwargs):
                result = original_build(geometry, cameras, data, surfaces, masks, **kwargs)
                # All recording and comparison below is outside Graph timing.
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                raw = {name:value.detach().cpu().numpy() for name,value in geometry.last_raw_masks.items()}
                export_ms = (time.perf_counter_ns()-start)/1e6
                hashes = {name:digest(value) for name,value in raw.items()}
                t = int(data['timestep'])
                if self.mode=='natural':
                    (host_cameras,host_data,host_surfaces), sensor_ms = self.host_input
                    payload = {'data__'+k:np.asarray(v) for k,v in host_data.items()}
                    for camera in host_cameras:
                        for field in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl','tcp_pose'):
                            payload[f'{camera.camera}__{field}'] = getattr(camera,field)
                        payload[f'{camera.camera}__mask'] = raw[camera.camera]
                    payload['surface_count'] = np.int64(len(host_surfaces))
                    for i,surface in enumerate(host_surfaces):
                        payload[f'surface{i}__link'] = np.array(surface.link)
                        payload[f'surface{i}__vertices'] = surface.local_vertices
                        payload[f'surface{i}__world'] = surface.world_from_link
                    folder=output/'inputs';folder.mkdir(exist_ok=True)
                    file=folder/f'frame{t:04d}.npz';np.savez_compressed(file,**payload)
                    self.input_rows.append(dict(timestep=t,file=file.name,sha256=live.sha(file),
                        legacy_sensors_ms=sensor_ms,mask_readback_ms=export_ms,raw_mask_sha256=hashes,
                        RGB_shape=[list(c.rgb.shape) for c in host_cameras]))
                else:
                    manifest=live.read(output/'inputs/manifest.json')
                    assert hashes==manifest['frames'][t]['raw_mask_sha256'], (self.mode,t,'raw masks changed')
                return result
            stack.enter_context(patch.object(rollout,'sensors',sensors))
            stack.enter_context(patch.object(rollout.CurrentGeometry,'build',build))

        def finish(self):
            result=super().finish()
            if self.mode=='natural':
                live.write(output/'inputs/manifest.json',dict(legacy_commit=COMMIT,
                    normalized_controls='shared512 native CUDA images, Terra and native SAM2 VOS; old CPU sensors includes image readback',
                    frames=self.input_rows))
            result['shared_live_inputs']=str(output/'inputs/manifest.json')
            result['CPU_sensor_observer_outside_GPU_timer']=self.mode=='natural'
            return result

    OriginalLogger=live.ExperimentLogger
    class Logger(OriginalLogger):
        def __init__(self,*a,**kw):
            kw['config'].update(task='10i_live_preopt_CPU_capture_20261104',legacy_commit=COMMIT,
                shared_vision=True,legacy_sensor_observer='outside sensors_to_graph/work timing; image_start_to_graph of first pass includes observer')
            super().__init__(*a,**kw)
    live.LiveMeasurement=CaptureMeasurement
    live.ExperimentLogger=Logger
    sys.argv=[sys.argv[0]]+remaining
    live.main()


if __name__=='__main__':
    main()
