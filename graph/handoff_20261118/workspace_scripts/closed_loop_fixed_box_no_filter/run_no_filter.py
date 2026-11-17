"""Re-evaluate the same fixed-Box rollouts with the entire Recurrent filter disabled."""
import argparse
import json
from pathlib import Path
from time import monotonic

import cv2
import numpy as np
import torch

import exp2_eval as exp2
import run_rollouts as base
from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import ExperimentLogger, write

def main():
    p = argparse.ArgumentParser()
    for name in ('source', 'tracker', 'gnn', 'filtered-run', 'output'):
        p.add_argument(f'--{name}', type=Path, required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--stop', type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.start < a.stop <= base.TOTAL_SEEDS
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    device = torch.device('cuda:0')
    seeds, _, _ = base.source_seeds(a.source)
    paths = [a.source / f'recurrent/exp2/checkpoints/runs_r6_mix_filter/mamba_step_s{i}/last.pt' for i in range(5)]
    ckpts = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    assert all(c['kin'] == {'min_box': 2, 'min_peg': 5, 'robust': False} and c['rot6'] and
               c['obs_dim'] == 44 and c['chunk'] == 1 for c in ckpts)
    decoder = exp2.Ensemble([exp2.Decoder(dict(c, kin=False), 'step') for c in ckpts])
    assert decoder.kf is None and all(d.kf is None for d in decoder.decs)
    front = base.HybridFrontend(a.tracker, a.gnn, device)
    old = a.filtered_run / f'shard_{a.start}_{a.stop}'
    source_rows = old / 'rollouts.jsonl'
    baseline = {r['seed']: r['hybrid'] for r in map(json.loads, source_rows.read_text().splitlines())}
    reference_config = json.loads((old / 'summary.json').read_text())['config']
    config = dict(seeds=seeds[a.start:a.stop], camera=base.CAMERA, horizon=exp2.HORIZON,
                  filter_mode='all KinFilter outputs bypassed', hole_motion='none', hybrid_ids='GT assignment only',
                  hybrid_hashes=dict(tracker=file_sha256(a.tracker), gnn=file_sha256(a.gnn)),
                  recurrent_hashes=[file_sha256(p) for p in paths], prior_rollouts_sha256=file_sha256(source_rows),
                  script_sha256=file_sha256(__file__), base_script_sha256=file_sha256(base.__file__))
    for key in ('hybrid_hashes', 'recurrent_hashes', 'hole_motion', 'camera', 'horizon'):
        assert config[key] == reference_config[key], key
    assert config['seeds'] == list(baseline)
    a.output.mkdir(parents=True, exist_ok=False)
    path = a.output / 'rollouts.jsonl'
    start = monotonic()
    with ExperimentLogger(team='graph', model='fixed_box_mamba_no_filter', job_type='evaluate',
                          seed=base.SAMPLE_SEED, config=config, root=a.output / 'experiments') as logger:
        env = base.make_env()
        logger.record(runtime=dict(simulation=str(env.unwrapped.backend), gpu=torch.cuda.get_device_name()))
        count = 0
        try:
            with path.open('w') as stream:
                for seed in config['seeds']:
                    _, result = base.episode(env, front, decoder, seed, 'hybrid')
                    assert all(r['object21'] == r['filtered'] for r in result['trace'])
                    count += int(result['success'])
                    ref = baseline[seed]
                    np.testing.assert_allclose(result['trace'][0]['object21_gt'],
                                               ref['trace'][0]['object21_gt'], rtol=0, atol=1e-6)
                    np.testing.assert_allclose(result['trace'][0]['object21'],
                                               ref['trace'][0]['object21'], rtol=0, atol=5e-5)
                    stream.write(json.dumps(dict(seed=seed, **result), allow_nan=False) + '\n')
                    stream.flush()
                    print(seed, result['success'], result['steps'],
                          f'{monotonic()-start:.1f}s', flush=True)
        finally:
            env.close()
        summary = a.output / 'summary.json'
        write(summary, dict(run_id=logger.run_id, config=config, success=count,
                            episodes=len(config['seeds']), seconds=monotonic()-start))
        logger.log({'success': count})
        for item in (path, summary):
            logger.link_output(item)
    print('Completed matched fixed-Box no-filter rollouts', flush=True)


if __name__ == '__main__':
    main()
