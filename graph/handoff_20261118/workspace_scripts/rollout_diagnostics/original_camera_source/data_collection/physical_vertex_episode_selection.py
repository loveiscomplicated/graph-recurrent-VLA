"""Audit physical vertex conditions for every frame of evaluated episodes."""

import json
from pathlib import Path

import numpy as np

from data_collection.episode_dataset import file_sha256
from data_collection.physical_vertex_selection import (
    initialize_cached_worker,
    inspect_cached_frame,
    iter_frame_decisions,
)
from data_collection.storage import write_json
from utils.experiment_logger import ExperimentLogger


def prepare(config):
    """Certify cached detections and compare with existing sampled decisions."""
    prediction_path = Path(config['predictions']).resolve()
    source_root = Path(config['source_root']).resolve()
    cache_root = Path(config['cache_root']).resolve()
    identity_path = cache_root / 'metadata/data_identity.json'
    inventory_path = cache_root / 'metadata/cache_inventory.json'
    identity = json.loads(identity_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    parent_map = json.loads(Path(config['parent_manifest']).read_text())['parent_by_episode']
    raw = {row['id']: row for row in identity['frames']}
    cached = {row['sample_id']: row for row in inventory['files']}
    with np.load(prediction_path, allow_pickle=False) as saved:
        sample_ids = [str(value) for value in saved['sample_ids']]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError('Prediction sample IDs must be nonempty and unique')
    rows = []
    for sample_id in sample_ids:
        episode = str(Path(sample_id).parent)
        entry = cached[sample_id]
        rows.append({'sample_id': sample_id, 'parent_id': parent_map[episode],
                     'sha256': raw[sample_id]['sha256'], 'cache_file': entry['file'],
                     'cache_sha256': entry['sha256']})
    previous_path = Path(config['previous_selection'])
    previous = {row['sample_id']: row for row in
                (json.loads(line) for line in previous_path.read_text().splitlines())}
    if len(previous) != 960 or not set(previous) <= set(sample_ids):
        raise ValueError('Original 960 validation decisions must belong to the evaluated episodes')
    worker_config = dict(config, source_root=str(source_root), cache_root=str(cache_root))
    with ExperimentLogger(team='graph', config=config, seed=0,
                          model='physical_vertex_episode_selection', job_type='data_preparation') as logger:
        output = Path(config['output']['root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        counts = {'frames': 0, 'peg_pass': 0, 'hole_pass': 0, 'both_pass': 0}
        mismatches = []
        with (output / 'frames.jsonl').open('w') as stream:
            for result in iter_frame_decisions(
                rows, worker_config, initialize_cached_worker, inspect_cached_frame
            ):
                old = previous.get(result['sample_id'])
                if old is not None:
                    differences = [name for name in ('peg_pass', 'hole_pass', 'both_pass')
                                   if result[name] != old[name]]
                    if differences:
                        mismatches.append({'sample_id': result['sample_id'], 'fields': differences,
                                           'recomputed': result, 'previous': old})
                    # Keep the original detector-pixel decision for these 960 frames.
                    result.update({key: old[key] for key in ('peg_pass', 'hole_pass', 'both_pass',
                                  'peg_single_face', 'unique_peg_ids', 'unique_rim_ids')})
                result['decision_source'] = 'original_detector_uv' if old else 'projected_cached_xyz'
                stream.write(json.dumps(result, sort_keys=True) + '\n')
                counts['frames'] += 1
                for key in ('peg_pass', 'hole_pass', 'both_pass'):
                    counts[key] += result[key]
                if counts['frames'] % 500 == 0:
                    print(f'Certified {counts["frames"]}/{len(rows)} frames', flush=True)
        summary = dict(counts, parents=len({row['parent_id'] for row in rows}),
                       original_frames=len(previous), mismatches=len(mismatches),
                       predictions_sha256=file_sha256(prediction_path),
                       previous_selection_sha256=file_sha256(previous_path),
                       cache_inventory_sha256=file_sha256(inventory_path),
                       certification=config.get('certification', {}))
        write_json(output / 'summary.json', summary)
        write_json(output / 'mismatches.json', mismatches)
        write_json(output / 'effective_config.json', config)
        for name in ('frames.jsonl', 'summary.json', 'mismatches.json', 'effective_config.json'):
            logger.link_output(output / name)
        logger.record(**summary)
        print(f'Episode physical conditions: {output}; {summary}', flush=True)
    return output
