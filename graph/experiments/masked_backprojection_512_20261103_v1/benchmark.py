"""Paired CPU timing of one saved 512 frame, with no robot-filter execution."""
import argparse
from functools import partial
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from unittest.mock import patch

os.environ['WANDB_MODE'] = 'disabled'
import numpy as np

from utils.experiment_logger import ExperimentLogger
from visual.runtime.pca import build_geometry
from visual.runtime.rgbd import TrackedRGBDFrame
from visual.runtime.sampling import extract_sampling_evidence, extract_candidates, SamplingConfig


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def gather(builder, frame):
    geometry = builder(frame)
    return {oid:geometry.xyz.reshape(-1, 3)[np.flatnonzero(geometry.observed_mask(oid))]
            for oid in (0, 1)}


def verify(frame, original, masked):
    a, b = original(frame), masked(frame)
    np.testing.assert_array_equal(a.valid_depth, b.valid_depth)
    np.testing.assert_array_equal(a.overlap_mask, b.overlap_mask)
    np.testing.assert_array_equal(a.xyz[..., 2], b.xyz[..., 2])
    for oid in (0, 1):
        np.testing.assert_array_equal(a.xyz[a.observed_mask(oid)], b.xyz[b.observed_mask(oid)])
    with patch('visual.runtime.sampling.build_geometry', original):
        expected = extract_sampling_evidence(frame)
    with patch('visual.runtime.sampling.build_geometry', masked):
        actual = extract_sampling_evidence(frame)
    for oid in (0, 1):
        for name in expected.frame.objects[oid].__dataclass_fields__:
            np.testing.assert_array_equal(getattr(expected.frame.objects[oid], name),
                                          getattr(actual.frame.objects[oid], name), err_msg=name)
        for name in expected.pairs[oid].__dataclass_fields__:
            np.testing.assert_array_equal(getattr(expected.pairs[oid], name),
                                          getattr(actual.pairs[oid], name), err_msg=name)
    a = extract_candidates(expected, SamplingConfig(), 'S3')
    b = extract_candidates(actual, SamplingConfig(), 'S3')
    for oid in (0, 1):
        np.testing.assert_array_equal(a[oid].flat_index, b[oid].flat_index)
        np.testing.assert_array_equal(a[oid].points, b[oid].points)
        for name in a[oid].metadata:
            np.testing.assert_array_equal(a[oid].metadata[name], b[oid].metadata[name])
    return dict(object_xyz_exact=True,source_indices_exact=True,valid_depth_exact=True,
                background_z_exact=True,cpu_pca5_features_exact=True,
                cpu_pair_evidence_exact=True,s3_candidate_metadata_exact=True,
                point_counts={str(i):len(a[i].points) for i in (0, 1)})


def main(args):
    source = Path(args.input_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Import the byte-preserved pre-edit implementation, not a reconstructed baseline.
    original_path = source/'source_before/visual/runtime/pca.py'
    spec = importlib.util.spec_from_file_location('task10i_original_pca', original_path)
    original_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_module)
    original = original_module.build_geometry
    masked = partial(build_geometry, mask_only=True)
    with np.load(source/'input.npz', allow_pickle=False) as z:
        arrays = {name:z[name].copy() for name in z.files}
    frame = TrackedRGBDFrame(80, 'hand_camera', arrays['rgb'], arrays['depth'], arrays['intrinsic'],
        np.eye(4), np.array([0., 0., 0., 1., 0., 0., 0.]), dict(enumerate(arrays['masks'])), 80, 4.)
    if frame.depth.shape != (512, 512):raise ValueError('native 512 input required')
    inputs_before = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
    checks = verify(frame, original, masked)
    functions = {
        'geometry_only':{'full':original,'mask_only':masked},
        'geometry_and_object_gather':{
            'full':partial(gather, original),'mask_only':partial(gather, masked)},
    }
    config = dict(task='10i_masked_backprojection_512_20261103',host=platform.platform(),
        cpu=platform.processor(),python=sys.version,numpy=np.__version__,
        warmups=args.warmups,repeats=args.repeats,gc_enabled=__import__('gc').isenabled(),
        input_sha256=sha(source/'input.npz'),original_source_sha256=sha(original_path),
        source_sha256={str(p):sha(p) for p in (Path('visual/runtime/pca.py'),Path(__file__))},
        boundary='CPU pca geometry construction, optionally followed by original per-object raster-index/XYZ gather; includes validation, mask copies, allocations; excludes file loading, mm conversion, normal/features, sampling, Graph and robot filter',
        mask_pixels={str(i):int(frame.masks[i].sum()) for i in (0,1)},
        mask_union_fraction=float(np.logical_or(*arrays['masks']).mean()),
        verification=checks,source_frame_identity='10i_s1_0844_t080 / hand_camera / actual mask r0',
        identity_arrays='dummy transform/TCP are unused by the measured camera-coordinate operator')
    write(output/'config.json', config)
    snapshot=output/'source';snapshot.mkdir()
    for p in (Path('visual/runtime/pca.py'),Path(__file__)):
        shutil.copy2(p,snapshot/p.name)
    results={}
    with ExperimentLogger(team='graph',model='masked_backprojection_512',job_type='evaluate',
                          seed=0,root=output/'experiments',config=config) as log:
        for boundary, variants in functions.items():
            for _ in range(args.warmups):
                for function in variants.values():
                    value=function(frame);del value
            samples={name:[] for name in variants}
            for repeat in range(args.repeats):
                order=list(variants) if repeat%2==0 else list(reversed(variants))
                for name in order:
                    start=time.perf_counter_ns()
                    value=variants[name](frame)
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    samples[name].append(elapsed)
                    del value
            stats={name:dict(median_ms=float(np.median(values)),p95_ms=float(np.quantile(values,.95)),
                              min_ms=min(values),max_ms=max(values),samples_ms=values)
                   for name,values in samples.items()}
            stats['mask_over_full_ratio']=stats['mask_only']['median_ms']/stats['full']['median_ms']
            stats['paired_delta_median_ms']=float(np.median(np.array(samples['mask_only'])-samples['full']))
            results[boundary]=stats
            log.log({boundary:{name:row['median_ms'] for name,row in stats.items() if isinstance(row,dict)}})
        assert inputs_before=={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
        write(output/'results.json',dict(status='PASS',run_id=log.run_id,
            input_unchanged=True,verification=checks,boundaries=results))
        log.link_output(output/'results.json');log.link_output(output/'config.json')
        print(json.dumps(dict(run_id=log.run_id,verification=checks,
            timings={b:{n:{k:v for k,v in row.items() if k!='samples_ms'} if isinstance(row,dict) else row
                        for n,row in rows.items()} for b,rows in results.items()}),indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--warmups',type=int,default=50)
    parser.add_argument('--repeats',type=int,default=1000)
    main(parser.parse_args())
