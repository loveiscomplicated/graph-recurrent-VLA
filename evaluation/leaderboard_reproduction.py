"""Re-evaluate frozen leaderboard GNN inputs and compare recorded arrays."""

import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluation.detected_corner_gnn_comparison import evaluate_run
from utils.experiment_logger import ExperimentLogger


def difference(actual, reference):
    """Describe exact equality and magnitude of elementwise numeric differences."""
    if actual.shape != reference.shape:
        raise ValueError(f'Array shape mismatch: {actual.shape} versus {reference.shape}')
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    return dict(exact=bool(np.array_equal(actual, reference)),
                differing_elements=int(np.count_nonzero(actual != reference)),
                max_abs=float(delta.max()) if delta.size else 0.0,
                mean_abs=float(delta.mean()) if delta.size else 0.0)


def evaluate(config):
    """Run the protected evaluator on verified caches without altering its dependencies."""
    device = torch.device(config['device'])
    torch.set_num_threads(config['cpu_threads'])
    with ExperimentLogger(team='graph', model='leaderboard_reproduction', config=config,
                          seed=0, job_type='evaluation') as logger:
        output = Path(config['output_root']) / logger.run_id
        output.mkdir(parents=True, exist_ok=False)
        results = []
        for item in config['runs']:
            checkpoint = torch.load(item['checkpoint'], map_location='cpu', weights_only=False)
            if checkpoint['config']['seed'] != item['seed'] or checkpoint['config']['temporal']['architecture'] != 'transformer':
                raise ValueError('Checkpoint identity differs from requested Transformer seed')
            if file_sha256(item['cache']) != checkpoint['rollout_cache_sha256']:
                raise ValueError(f'Frozen rollout cache hash mismatch for seed {item["seed"]}')
            with np.load(item['reference'], allow_pickle=False) as data:
                reference = {key: data[key].copy() for key in data.files}
            run = dict(cache_path=Path(item['cache']), checkpoint=checkpoint)
            prediction, errors = evaluate_run(run, reference['sample_ids'], reference['target'], device)
            comparisons = {'prediction': difference(prediction, reference['prediction'])}
            for name, values in errors.items():
                comparisons[name] = difference(values, reference[name])
            result = dict(seed=item['seed'], frames=len(prediction), checkpoint_sha256=file_sha256(item['checkpoint']),
                          cache_sha256=file_sha256(item['cache']), reference_sha256=file_sha256(item['reference']),
                          recorded_hole_mean_mm=float(reference['hole_position_error_mm'].mean()),
                          rerun_hole_mean_mm=float(errors['hole_position_error_mm'].mean()),
                          recorded_peg_mean_mm=float(reference['peg_position_error_mm'].mean()),
                          rerun_peg_mean_mm=float(errors['peg_position_error_mm'].mean()), comparisons=comparisons)
            results.append(result)
            path = output / f'seed_{item["seed"]}_predictions.npz'
            np.savez_compressed(path, sample_ids=reference['sample_ids'], prediction=prediction,
                                target=reference['target'], **errors)
            logger.link_output(path)
            print(f'seed {item["seed"]}: frames {len(prediction)}, prediction max difference '
                  f'{comparisons["prediction"]["max_abs"]:.9g}', flush=True)
        path = output / 'reproduction.json'
        path.write_text(json.dumps(dict(config=config, results=results,
                                        scope='GNN-only on frozen tracked graph inputs; detector and tracker not rerun',
                                        numerical_reproduction='exact only if all elementwise comparisons match'), indent=2))
        logger.link_output(path)
        logger.record(results=results)
        print(f'Reproduction: {output}', flush=True)
