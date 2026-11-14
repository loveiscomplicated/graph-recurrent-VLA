"""Freeze an additional test cohort using the original initial-visibility gate."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from data_collection.collect_episodes import initial_visibility
from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from envs.maniskill import make_peg_insertion_env
from utils.experiment_logger import ExperimentLogger


def fingerprint(observation, ground_truth):
    """Hash the exact initial camera/pose bytes used to make the gate decision."""
    digest = hashlib.sha256()
    arrays = (observation['sensor_data']['base_camera']['rgb'],
              observation['sensor_data']['base_camera']['depth'],
              ground_truth['peg_pose'], ground_truth['hole_pose'], ground_truth['ee_pose'])
    for value in arrays:
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def select(config):
    """Predeclare candidates, apply the GT visibility gate and freeze passing seeds."""
    if file_sha256(config['base_protocol']) != config['base_protocol_sha256']:
        raise ValueError('Base test protocol SHA256 mismatch')
    base = json.loads(Path(config['base_protocol']).read_text())
    training_path = Path(config['training_exclusions'])
    additional_path = Path(config['additional_exclusions'])
    if file_sha256(training_path) != base['exclusions']['training_inventory_sha256']:
        raise ValueError('Training exclusion inventory differs from the base protocol')
    if file_sha256(additional_path) != base['exclusions']['additional_inventory_sha256']:
        raise ValueError('Additional exclusion inventory differs from the base protocol')
    training = json.loads(training_path.read_text())
    additional = json.loads(additional_path.read_text())
    excluded = set(base['seeds']) | set(training['original_collection_seeds'])
    excluded.update(row['seed'] for row in additional['episodes'])
    start, end = config['candidate_seed_range']
    if not (0 <= start < end and 0 < config['required_passes'] <= config['candidate_count'] <= end-start):
        raise ValueError('Invalid candidate range or sample counts')
    if excluded.intersection(range(start, end)):
        raise ValueError('Candidate range overlaps a known collection or fixed test')
    rng = np.random.default_rng(config['candidate_seed_generator'])
    candidates = list(map(int, rng.choice(np.arange(start, end, dtype=np.int64),
                                           size=config['candidate_count'], replace=False)))
    root = Path(config['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    rule = dict(config=config, candidates=candidates,
                gate='original ends-and-grasp initial RGB-D/GT visibility',
                selection='first required passes in predeclared candidate order; no model metrics')
    write_json(root / 'selection_protocol.json', rule)
    sensors = {name: {'width': 512, 'height': 512} for name in ('base_camera', 'hand_camera')}
    env = make_peg_insertion_env(obs_mode='rgb+depth', control_mode=base['policy']['control_mode'],
                                 sensor_configs=sensors, max_episode_steps=base['policy']['max_frames'])
    attempts = []
    try:
        for seed in candidates:
            obs, _ = env.reset(seed=seed)
            visibility, gt = initial_visibility(env, obs)
            passed = bool(visibility['requested_pass'])
            attempts.append(dict(seed=seed, passed=passed, initial_sha256=fingerprint(obs, gt)))
            accepted = sum(row['passed'] for row in attempts)
            if len(attempts) % 10 == 0 or accepted == config['required_passes']:
                print(f'Initial visibility selection: {accepted}/{config["required_passes"]} '
                      f'passed after {len(attempts)} attempts', flush=True)
            if accepted == config['required_passes']:
                break
    finally:
        env.close()
    write_json(root / 'selection_attempts.json', attempts)
    if sum(row['passed'] for row in attempts) != config['required_passes']:
        raise RuntimeError('Candidate budget exhausted before the required gate passes')
    protocol = copy.deepcopy(base)
    protocol['seeds'] = [row['seed'] for row in attempts if row['passed']]
    protocol['role'] = 'independent_matched_visibility_test'
    protocol['policy']['visibility_filter'] = 'ends_and_grasp'
    protocol['generation'] = dict(seed=config['candidate_seed_generator'], seed_range=config['candidate_seed_range'],
                                  count=config['required_passes'], candidate_count=config['candidate_count'])
    protocol['selection'] = dict(selection_protocol_sha256=file_sha256(root/'selection_protocol.json'),
                                 attempts_sha256=file_sha256(root/'selection_attempts.json'),
                                 gate='ends_and_grasp', attempted=len(attempts),
                                 initial_fingerprints={str(row['seed']): row['initial_sha256']
                                                       for row in attempts if row['passed']},
                                 note='Follow-up cohort preregistered after the first unfiltered test; no model metrics used for its seed selection.')
    path = root / 'protocol.json'
    write_json(path, protocol)
    with ExperimentLogger(team='graph', model='matched_visibility_test_selection', config=config,
                          seed=config['candidate_seed_generator'], job_type='collection') as logger:
        logger.record(protocol_sha256=file_sha256(path), attempted=len(attempts), accepted=len(protocol['seeds']))
        logger.link_output(path)
        logger.link_output(root/'selection_attempts.json')
    print(f'Frozen matched test: {path} sha256={file_sha256(path)}', flush=True)
