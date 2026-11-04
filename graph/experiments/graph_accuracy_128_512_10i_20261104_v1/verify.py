"""Independent depth/extrinsic reconstruction of archived graph representatives."""
import argparse
from pathlib import Path
import numpy as np
from .common import CAMERAS, PARENTS, SIZES, entries, read, write, sha
from visual.runtime.vision_packet import float32_tolerance


def verify(capture, inference, source, output):
    done = read(inference/'inference_complete.json')
    assert done['status'] == 'COMPLETE' and done['graph_frames'] == 168
    index = entries(capture)
    source_manifest = read(source/'source_manifest.json')
    changed = [p for p, h in source_manifest['files'].items() if sha(source/p) != h]
    assert not changed, changed
    maximum = 0.
    maximum_ratio = 0.
    count = 0
    backends = set()
    for parent in PARENTS:
        for size in SIZES:
            for t in range(21):
                path = inference/parent/str(size)/f't{t:04d}.npz'
                with np.load(path, allow_pickle=False) as z:
                    a = {k: z[k].copy() for k in z.files}
                assert a['raw_masks'].shape == (2, 2, size, size)
                assert a['raw_masks'].dtype == np.bool_
                assert np.isfinite(a['world_xyz']).all()
                for oid in (0, 1):
                    assert (a['object_id'] == oid).sum() == (a['node_type'] == oid+3).sum()
                meta = read(path.with_suffix('.json'))
                assert meta['robot_schedule'] == 'before_vision'
                backends.add((meta['packet']['graph_backend'], meta['packet']['wrist_gate_backend']))
                for ci, camera_name in enumerate(CAMERAS):
                    take = a['camera'] == ci
                    pixels = a['pixel'][take]
                    if not len(pixels):
                        continue
                    e = index[parent, t, size, camera_name]
                    with np.load(capture/e['sensor_file'], allow_pickle=False) as z:
                        k = z['intrinsic_cv'].astype(float)
                        extrinsic = z['extrinsic_cv'].astype(float)
                        raw = z['depth_raw_mm']
                    v, u = np.unravel_index(pixels, raw.shape)
                    assert (raw[v, u] > 0).all()
                    depth = raw[v, u].astype(float)/1000
                    # Independently use the CV extrinsic and pixel centers.
                    camera = np.stack(((u+.5-k[0,2])*depth/k[0,0],
                                       (v+.5-k[1,2])*depth/k[1,1], depth), axis=1)
                    matrix = np.eye(4)
                    matrix[:3] = extrinsic
                    inv = np.linalg.inv(matrix).astype(np.float32)
                    world = camera@inv[:3,:3].T+inv[:3,3]
                    error = np.linalg.norm(world-a['world_xyz'][take], axis=1)
                    ratio = np.abs(world-a['world_xyz'][take])/float32_tolerance(world)
                    maximum_ratio = max(maximum_ratio, float(ratio.max()))
                    maximum = max(maximum, float(error.max()))
                    count += len(error)
    assert maximum_ratio <= 1., (maximum, maximum_ratio)
    assert backends == {('torch_cuda', 'torch_cuda')}, backends
    write(output, dict(status='PASS', graph_frames=168, checked_representatives=count,
        source_pixel_reconstruction_max_error_m=maximum,
        componentwise_max_tolerance_ratio=maximum_ratio,
        tolerance='existing vision_packet.float32_tolerance: componentwise max(1e-6 m, 4 ULP)',
        source_files=len(source_manifest['files']), changed_source_files=changed,
        verification_script_sha256=sha(Path(__file__)),
        graph_and_gate_backends=sorted(backends), independent_GT_arrays_used=False,
        inference_receipt_sha256=sha(inference/'inference_complete.json')))
    print('INDEPENDENT_VERIFICATION_PASS', count, maximum, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('capture', 'inference', 'source', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    verify(a.capture, a.inference, a.source, a.output)
