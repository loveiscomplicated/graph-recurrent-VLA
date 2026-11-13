"""Audit the original initial-visibility gate without filtering the fixed test."""

import json
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.visibility import CameraObservation, evaluate as measure_visibility
from utils.experiment_logger import ExperimentLogger


def evaluate(config):
    """Score every frozen test reset; retain both passing and failing cases."""
    if file_sha256(config['protocol']) != config['protocol_sha256']:
        raise ValueError('Frozen test protocol SHA256 mismatch')
    protocol = json.loads(Path(config['protocol']).read_text())
    root = Path(config['raw_root'])
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['status'] != 'COMPLETE' or manifest['protocol_sha256'] != config['protocol_sha256']:
        raise ValueError('Test dataset is incomplete or has a different protocol')
    episodes = {row['seed']: row for row in manifest['episodes'] if row['mode'] == 'expert'}
    if set(episodes) != set(protocol['seeds']):
        raise ValueError('Expert reset seeds differ from the protocol')
    with ExperimentLogger(team='graph', model='independent_test_visibility_audit', config=config,
                          seed=protocol['generation']['seed'], job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        rows = []
        for seed in protocol['seeds']:
            episode = episodes[seed]
            path = root / episode['path'] / 'frame_000000.npz'
            if file_sha256(path) != episode['frame_sha256'][0]:
                raise ValueError(f'Raw initial frame checksum mismatch: {seed}')
            with np.load(path, allow_pickle=False) as data:
                image = 'observation/sensor_data/base_camera/'
                calibration = 'observation/sensor_param/base_camera/'
                tcp = data['ground_truth/ee_pose'][0].copy()
                camera = CameraObservation(str(seed), 0, 'base_camera', 0.0,
                    data[image+'rgb'][0].copy(), data[image+'depth'][0, ..., 0].copy(),
                    data[calibration+'intrinsic_cv'][0].copy(), data[calibration+'extrinsic_cv'][0].copy(),
                    data[calibration+'cam2world_gl'][0].copy(), tcp)
                peg = data['ground_truth/peg_pose'][0].copy()
                half = data['ground_truth/peg_half_sizes'][0].copy()
            result, _ = measure_visibility(camera, peg, half, tcp)
            rows.append(dict(seed=seed, requested_pass=bool(result['requested_pass']),
                             strict_100_pass=bool(result['strict_100_pass'])))
        path = output / 'initial_visibility.json'
        summary = dict(cases=len(rows), ends_and_grasp_pass=sum(row['requested_pass'] for row in rows),
                       strict_full_surface_pass=sum(row['strict_100_pass'] for row in rows), rows=rows,
                       note='Diagnostic only; no test cases were filtered or replaced.')
        path.write_text(json.dumps(summary, indent=2))
        logger.record(cases=len(rows), ends_and_grasp_pass=summary['ends_and_grasp_pass'])
        logger.link_output(path)
        print(f'Initial visibility audit: {output}', flush=True)
