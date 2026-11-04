"""Current rollout vision/graph on native recorded pairs, without GT scoring."""
import argparse
import gc
import os
from pathlib import Path

import numpy as np
import torch

from .common import CAMERAS, PARENTS, SIZES, entries, read, write, sha, sensor_frame, world_points
from utils.experiment_logger import ExperimentLogger


def main(capture, runtime, output):
    # Resolve the exact same model/adapter classes used by the current entrypoint.
    from graph.plan_a_bc.rollout import CurrentGeometry, OnlineVisionAdapter, LiveGrounding, SAM2Tracker
    from visual.runtime.live_grounding import policy_identity
    from graph.persistent_geometry.tensor_camera import tensor_camera
    from graph.persistent_geometry.tensor_graph import numpy_graph, FIELDS
    from graph.persistent_geometry.tensor_records import numpy_records
    if not torch.cuda.is_available() or not os.environ.get('OPENAI_API_KEY', '').strip():
        raise RuntimeError('CUDA and environment Terra credential required')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    index = entries(capture)
    schema = read(runtime / 'normalizer.json')['schema']
    paths = read(runtime / 'runtime_paths.json')
    detector = LiveGrounding(device='cuda', model_id=paths['dino'])
    # No fallback model load or mixed-backend paired statistic.
    def no_fallback(*args, **kwargs):
        raise RuntimeError('Terra authentication fallback: paired primary run aborted')
    import visual.runtime.live_grounding as grounding
    grounding.GDino = no_fallback
    tracker = SAM2Tracker(device='cuda', checkpoint=paths['sam'])
    grounding_calls = []
    request = detector._request
    def recorded_request(rgb):
        import hashlib
        boxes = request(rgb)
        grounding_calls.append(dict(rgb_shape=list(rgb.shape),
            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), boxes=boxes))
        write(output/'grounding_predictions.json', grounding_calls)
        return boxes
    detector._request = recorded_request
    protocol = dict(parents=PARENTS, sizes=SIZES, timesteps=list(range(21)), cameras=CAMERAS,
        runtime=str(runtime), capture=str(capture), capture_manifest_sha256=sha(capture/'capture_manifest.json'),
        checkpoint_sha256=sha(paths['sam']), grounding=policy_identity(),
        source_manifest_sha256=sha(Path(__file__).resolve().parents[3]/'source_manifest.json'),
        native_RGB_no_JPEG=True, separate_tracking_states=True, GT_arrays_used_in_inference=False,
        order='ascending sizes for even-indexed parent; descending for odd-indexed parent',
        backend=dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()),
        scope='recorded native inputs; first 21 observations; no simulator/policy execution')
    write(output/'protocol.json', protocol)
    receipts = []
    with ExperimentLogger(team='integration', model='task10i_current_graph_native128_512', seed=42,
            root=output/'experiments', job_type='evaluate', config=protocol) as log:
        try:
            for pi, parent in enumerate(PARENTS):
                for size in SIZES[::1 if pi % 2 == 0 else -1]:
                    identity = f'{parent}:native{size}'
                    geometry = CurrentGeometry(runtime/'inputs', schema)
                    adapters = {c: OnlineVisionAdapter(detector, tracker) for c in CAMERAS}
                    folder = output/parent/str(size)
                    folder.mkdir(parents=True, exist_ok=False)
                    for a in adapters.values():
                        a.reset(identity)
                    try:
                        for t in range(21):
                            host, data = sensor_frame(capture, index, parent, t, size, identity)
                            cameras = [tensor_camera(c) for c in host]
                            geometry.fk.verify_robot(data['robot76'], data['robot_world_geometry'], data['tcp_pose'])
                            surfaces = geometry.fk.surfaces(data['robot76'])
                            data.update(episode=np.array(identity), timestep=np.int64(t), timestamp=np.float64(t/20),
                                qpos=data['robot76'][:9], robot_link_world=np.stack([s.world_from_link for s in surfaces]))
                            def masks_now():
                                return {c.camera: adapters[c.camera].process(c.rgb, identity, t, mask_device='cuda')[0]
                                        for c in cameras}
                            g, records, gate = geometry.build(cameras, data, surfaces, masks_now)
                            if detector.backend != 'terra':
                                raise RuntimeError('unexpected grounding backend')
                            # Explicit archival boundary, after all current graph processing.
                            saved = numpy_graph(g)
                            rr = numpy_records(records)
                            for r in rr:
                                for e, ts, camera, pixel in r['source_observation_ids']:
                                    if e != identity or ts != t or not 0 <= pixel < size*size:
                                        raise ValueError('foreign/stale source')
                                    if r['object_id'] == 1 and camera != 'base_camera':
                                        raise ValueError('Wrist Box source')
                            raw_masks = np.stack([geometry.last_raw_masks[c].cpu().numpy() for c in CAMERAS])
                            np.savez_compressed(folder/f'raw_masks_t{t:04d}.npz', masks=raw_masks)
                            arrays = {k: getattr(saved, k).numpy() for k in FIELDS}
                            arrays.update(raw_masks=raw_masks,
                                world_xyz=world_points(rr, data['tcp_pose']),
                                object_id=np.asarray([r['object_id'] for r in rr], dtype=np.int64),
                                camera=np.asarray([CAMERAS.index(r['source_observation_ids'][0][2]) for r in rr], dtype=np.int64),
                                pixel=np.asarray([r['source_observation_ids'][0][3] for r in rr], dtype=np.int64))
                            path = folder/f't{t:04d}.npz'
                            np.savez_compressed(path, **arrays)
                            write(folder/f't{t:04d}.json', dict(gate=gate, packet=saved.packet,
                                robot_schedule=geometry.last_robot_schedule,
                                provenance=[dict(object_id=r['object_id'], sources=r['source_observation_ids']) for r in rr]))
                            if t % 5 == 0:
                                print('GRAPH', parent, size, t, len(rr), 'grounding', detector.status(), flush=True)
                        write(folder/'vision_trace.json', {c: a.records for c, a in adapters.items()})
                        receipts.append(dict(parent=parent, size=size, frames=21,
                            files={str(p.relative_to(output)): sha(p) for p in folder.iterdir() if p.is_file()}))
                        write(output/'progress.json', receipts)
                    finally:
                        geometry.close()
                        for a in adapters.values():
                            a.close()
                        del geometry, adapters
                        gc.collect()
                        torch.cuda.empty_cache()
            write(output/'inference_complete.json', dict(status='COMPLETE', run_id=log.run_id,
                protocol_sha256=sha(output/'protocol.json'), sequences=receipts, grounding=detector.status(),
                graph_frames=sum(r['frames'] for r in receipts), GT_arrays_used=False))
            log.log(dict(graph_frames=168, parent_count=4))
            log.link_output(output/'inference_complete.json')
        except BaseException as error:
            # Avoid recording provider exception objects or credential-bearing state.
            write(output/'failure.json', dict(status='FAILED', exception_type=type(error).__name__,
                completed_sequences=len(receipts), run_id=log.run_id))
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.capture, a.runtime, a.output)
