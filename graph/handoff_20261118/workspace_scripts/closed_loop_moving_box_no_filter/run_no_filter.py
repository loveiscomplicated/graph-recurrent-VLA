"""Re-evaluate the same moving-Box rollouts with the entire Recurrent filter disabled."""
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

VIDEO_SEEDS = (700488, 700627, 700866, 700901)


class NoVideo:
    def add(self, *_):
        pass

    def finish(self, *_):
        return Path('unrecorded'), 0

    def close(self):
        pass


@torch.inference_mode()
def episode(env, front, dec, seed, videos):
    obs, _ = env.reset(seed=seed)
    reset = base.reset_state(obs, env)
    dec.reset()
    task = env.unwrapped
    front.reset({0: int(task.peg.per_scene_id[0]), 1: int(task.box.per_scene_id[0])})
    initial_z = float(task.peg.pose.p[0, 2])
    freq = float(task.control_freq)
    plan = base.schedule(seed, base.MOTION_START_STEP / freq)
    plan['stop_expert_insert'] = False
    motion = base.EpisodeHoleMotion(env, plan, None)
    video = base.RolloutVideo(videos, seed, 'no_filter', freq) if seed in VIDEO_SEEDS else NoVideo()
    try:
        _, result = base.moving_loop(env, front, dec, seed, 'hybrid', obs, reset, task, [], False,
                                     'horizon', initial_z, plan, motion, video, freq)
    finally:
        video.close()
    if seed not in VIDEO_SEEDS:
        result['movie'] = None
    assert all(r['object21'] == r['filtered'] for r in result['trace'])
    return result


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
                  filter_mode='all KinFilter outputs bypassed', video_seeds=VIDEO_SEEDS,
                  motion_start_step=base.MOTION_START_STEP, hybrid_ids='GT assignment only',
                  hybrid_hashes=dict(tracker=file_sha256(a.tracker), gnn=file_sha256(a.gnn)),
                  recurrent_hashes=[file_sha256(p) for p in paths], prior_rollouts_sha256=file_sha256(source_rows),
                  script_sha256=file_sha256(__file__), base_script_sha256=file_sha256(base.__file__))
    for key in ('hybrid_hashes', 'recurrent_hashes', 'motion_start_step', 'camera', 'horizon'):
        assert config[key] == reference_config[key], key
    assert config['seeds'] == list(baseline)
    a.output.mkdir(parents=True, exist_ok=False)
    path = a.output / 'rollouts.jsonl'
    videos = a.output / 'videos'
    videos.mkdir()
    start = monotonic()
    with ExperimentLogger(team='graph', model='moving_box_mamba_no_filter', job_type='evaluate',
                          seed=base.SAMPLE_SEED, config=config, root=a.output / 'experiments') as logger:
        env = base.make_env()
        logger.record(runtime=dict(simulation=str(env.unwrapped.backend), gpu=torch.cuda.get_device_name()))
        count = 0
        try:
            with path.open('w') as stream:
                for seed in config['seeds']:
                    result = episode(env, front, decoder, seed, videos)
                    count += int(result['success'])
                    ref = baseline[seed]
                    assert result['motion_plan'] == ref['motion_plan']
                    np.testing.assert_allclose(result['trace'][0]['object21_gt'],
                                               ref['trace'][0]['object21_gt'], rtol=0, atol=1e-6)
                    np.testing.assert_allclose(result['trace'][0]['object21'],
                                               ref['trace'][0]['object21'], rtol=0, atol=5e-5)
                    stream.write(json.dumps(dict(seed=seed, **result), allow_nan=False) + '\n')
                    stream.flush()
                    print(seed, result['success'], result['steps'], round(result['box_max_shift_mm'],1),
                          f'{monotonic()-start:.1f}s', flush=True)
        finally:
            env.close()
        summary = a.output / 'summary.json'
        write(summary, dict(run_id=logger.run_id, config=config, success=count,
                            episodes=len(config['seeds']), seconds=monotonic()-start))
        logger.log({'success': count})
        for item in (path, summary, *sorted(videos.glob('*.mp4'))):
            logger.link_output(item)
    print('Completed matched no-filter rollouts', flush=True)


if __name__ == '__main__':
    main()
