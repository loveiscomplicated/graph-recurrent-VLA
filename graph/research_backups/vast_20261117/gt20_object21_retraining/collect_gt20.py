"""Collect simulator-only GT20 inputs and Object21 targets on fresh expert trajectories."""
import argparse
from collections import Counter
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.scripted_policy import make_policy
from envs.maniskill import make_peg_insertion_env
from utils.experiment_logger import ExperimentLogger, write

from evaluate_full_gt import collect

FIRST_SEED = 3_100_000
SPLIT_SIZES = (120, 24, 32)
END = FIRST_SEED + sum(SPLIT_SIZES)


def collect_episode(env, seed):
    obs, _ = env.reset(seed=seed)
    policy = make_policy()
    policy.reset(env, seed)
    records = []
    for step in range(200):
        phase = policy.phase
        if phase == 'insert':
            return records, 'insert_entry'
        task = env.unwrapped
        held = bool(task.agent.is_grasping(task.peg)[0])
        gt, inputs = collect(env, obs)
        xyz, kinds, tcp = inputs['full_front20']
        records.append(dict(xyz=xyz, kinds=kinds, tcp=tcp, target=gt, seed=seed,
                            step=step, phase=phase, grasp=held))
        obs, _, term, trunc, _ = env.step(policy.act(obs))
        if bool(term[0]) or bool(trunc[0]):
            return records, 'terminated' if bool(term[0]) else 'truncated'
    return records, 'step_limit'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--stop', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not FIRST_SEED <= args.start < args.stop <= END:
        parser.error('Seed range is outside the experiment split')
    torch.set_num_threads(1)
    torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=False)
    seeds = list(range(args.start, args.stop))
    config = dict(seeds=seeds, control_mode='pd_ee_delta_pose', observation='state_dict',
                  max_steps=200, input='simulator GT20', script_sha256=file_sha256(__file__))
    started = monotonic()
    with ExperimentLogger(team='graph', model='gt20_data_collection', job_type='collect',
                          config=config, seed=FIRST_SEED, root=args.output/'experiments') as logger:
        env = make_peg_insertion_env(device='cpu', render_backend='cuda', obs_mode='state_dict',
                                     control_mode=config['control_mode'], max_episode_steps=200)
        rows, terminals = [], {}
        try:
            logger.record(runtime=str(env.unwrapped.backend))
            for idx, seed in enumerate(seeds, 1):
                records, terminal = collect_episode(env, seed)
                if not records:
                    raise ValueError(f'Empty episode {seed}')
                rows.extend(records)
                terminals[str(seed)] = terminal
                if idx == 1 or idx % 10 == 0 or idx == len(seeds):
                    print(f'{idx}/{len(seeds)} episodes; {len(rows)} frames; {monotonic()-started:.1f}s', flush=True)
            arrays = {name:np.asarray([row[name] for row in rows]) for name in rows[0]}
            for name in ('xyz', 'tcp', 'target'):
                arrays[name] = arrays[name].astype(np.float32)
                if not np.isfinite(arrays[name]).all():
                    raise ValueError(f'Nonfinite array {name}')
            path = args.output/'dataset.npz'
            np.savez_compressed(path, **arrays)
            meta = dict(run_id=logger.run_id, **config, frames=len(rows), terminals=terminals,
                        phase_counts=dict(Counter(arrays['phase'].tolist())),
                        dataset_sha256=file_sha256(path), seconds=monotonic()-started)
            write(args.output/'summary.json', meta)
            logger.log({'episodes':len(seeds),'frames':len(rows)})
            logger.link_output(path)
            logger.link_output(args.output/'summary.json')
            print('Completed',logger.run_id,flush=True)
        finally:
            env.close()


if __name__ == '__main__':
    main()
