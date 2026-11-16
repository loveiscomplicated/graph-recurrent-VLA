"""Collect and verify a moving-Box counterpart to the fixed-Hole trajectories."""

import json
from pathlib import Path

import numpy as np

from data_collection import mixed_trajectory_collection, mixed_trajectory_recovery
from data_collection.collection_batches import collection_defaults
from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json


def verify(root, reference_root, reference, start, stop, settings, *, initial_atol=1e-7):
    """Check paired initial states and realized GT translation in all four modes."""
    expected_modes = (*mixed_trajectory_collection.MODES, mixed_trajectory_recovery.MODE)
    fixed = {row['seed']: row for row in reference['episodes'] if row['mode'] == 'expert'}
    protocol = json.loads((root / 'pilot' / 'protocol.json').read_text())
    seeds = [row['seed'] for row in protocol['records'] if row['split'] == 'pilot'][start:stop]
    expected = {(seed, mode) for seed in seeds for mode in expected_modes}
    expected_offset = [*settings['translation_m'], 0, 0, 0, 0]
    offsets = []
    cases = set()
    initial_fields = ('qpos', 'peg_pose', 'box_pose', 'hole_pose', 'peg_half_sizes', 'hole_radius')
    for cohort in ('pilot', 'recovery'):
        manifest = json.loads((root / cohort / 'manifest.json').read_text())
        if manifest['status'] != 'COMPLETE':
            raise ValueError(f'Expected a completed {cohort} manifest')
        for row in manifest['episodes']:
            seed = row['seed']
            case = (seed, row['mode'])
            if case in cases or case not in expected:
                raise ValueError(f'Duplicate or unexpected trajectory: {case}')
            fixed_row = fixed[seed]
            exact_initial_match = row['initial_fingerprint'] == fixed_row['initial_fingerprint']
            initial_max_difference = 0.0
            cases.add(case)
            directory = root / cohort / row['path']
            first = directory / 'frame_000000.npz'
            last = directory / f'frame_{row["steps"]:06d}.npz'
            with np.load(first, allow_pickle=False) as initial, np.load(last, allow_pickle=False) as final:
                if not exact_initial_match:
                    # Platform rounding in the quaternion can change a float32 fingerprint.
                    fixed_first = reference_root / fixed_row['path'] / first.name
                    if file_sha256(fixed_first) != fixed_row['frame_sha256'][0]:
                        raise ValueError(f'Fixed-Hole reset-frame SHA-256 mismatch: {fixed_first}')
                    with np.load(fixed_first, allow_pickle=False) as fixed_initial:
                        initial_max_difference = max(
                            float(np.max(np.abs(initial[f'ground_truth/{key}']
                                                - fixed_initial[f'ground_truth/{key}'])))
                            for key in initial_fields
                        )
                    if not initial_max_difference <= initial_atol:
                        raise ValueError(f'Initial state differs from the fixed-Hole reference: {case}')
                offset = final['ground_truth/box_pose'] - initial['ground_truth/box_pose']
                hole_offset = final['ground_truth/hole_pose'] - initial['ground_truth/hole_pose']
            reached_target = all(np.allclose(value, expected_offset, rtol=0, atol=1e-6)
                                 for value in (offset, hole_offset))
            offsets.append(dict(seed=case[0], mode=case[1], path=f'{cohort}/{row["path"]}',
                                box_offset_m=offset[0, :3].tolist(), reached_target=reached_target,
                                exact_initial_match=exact_initial_match,
                                initial_max_difference=initial_max_difference))
    if cases != expected:
        raise ValueError('Expected one trajectory per selected seed and mode')
    count = sum(row['reached_target'] for row in offsets)
    write_json(root / 'motion_audit.json', dict(episodes=len(cases), full_translation=count,
                                               reference_seeds=seeds, initial_atol=initial_atol, records=offsets))
    if count != len(cases):
        raise RuntimeError(f'{count}/{len(cases)} episodes reached the configured translation')


def collect(config):
    """Collect three moving pilot modes, then branch the fourth from moving frames."""
    config = collection_defaults(config)
    source = Path(config['reference_pilot'])
    manifest_path = source / 'manifest.json'
    if file_sha256(manifest_path) != config['reference_manifest_sha256']:
        raise ValueError(f'Fixed-Hole reference manifest SHA-256 mismatch: {manifest_path}')
    reference = json.loads(manifest_path.read_text())
    if reference['status'] != 'COMPLETE':
        raise ValueError('Expected a complete fixed-Hole reference pilot')
    start, stop = config['record_range']
    root = Path(config['output_root'])
    pilot = root / 'pilot'
    settings = config['hole_motion']
    if settings.get('stop_expert_insert', False):
        raise ValueError('Use a Box motion schedule independent of expert phase')
    pilot_config = dict(
        reference['effective_config'], output_root=str(pilot), collect_splits=['pilot'],
        source_protocol=str(source / 'protocol.json'), source_protocol_sha256=reference['protocol_sha256'],
        record_range=[start, stop], hole_motion=settings, minimum_free_gib=config['minimum_free_gib'],
        workers=config['workers'], batch_episodes=config['batch_episodes'], render_backend=config.get('render_backend'),
    )
    if 'ram_root' in config:
        pilot_config['ram_root'] = config['ram_root']
    if 'compress_frames' in config:
        pilot_config['compress_frames'] = config['compress_frames']
    physics_device = config.get('physics_device', 'cpu')
    if physics_device not in {'cpu', 'cuda'}:
        raise ValueError('physics_device must be cpu or cuda')
    if physics_device == 'cuda':
        config = dict(config, reset_reference_root=str(source))
        pilot_config.update(physics_device='cuda', batch_size=config.get('batch_size', config['batch_episodes']),
                            writers=config.get('writers', config['workers']), reset_reference_root=str(source))
    pilot_manifest_path = pilot / 'manifest.json'
    if pilot.exists():
        completed = json.loads(pilot_manifest_path.read_text())
        if completed['status'] != 'COMPLETE' or collection_defaults(completed['effective_config']) != pilot_config:
            raise ValueError(f'Expected a completed pilot with the same settings: {pilot}')
    else:
        mixed_trajectory_collection.collect(pilot_config)

    protocol = json.loads((pilot / 'protocol.json').read_text())
    records = [row for row in protocol['records'] if row['split'] == 'pilot'][start:stop]
    transitions = {row['seed']: row['transition_phase'] for row in records}
    fixed_switch_steps = {}
    for row in reference['episodes']:
        seed = row['seed']
        if seed not in transitions or row['mode'] != 'expert_random':
            continue
        step = row['switch_step']
        if step is None:
            step = mixed_trajectory_recovery.last_phase_action(source, row, transitions[seed])
        if step is not None:
            fixed_switch_steps[seed] = max(1, step)

    mixed_trajectory_recovery.collect(dict(
        config, pilot_root=str(pilot), pilot_manifest_sha256=file_sha256(pilot_manifest_path),
        output_root=str(root / 'recovery'), missing_phase_switch_steps=fixed_switch_steps,
    ))
    initial_atol = 1e-6 if physics_device == 'cuda' else 1e-7
    verify(root, source, reference, start, stop, settings, initial_atol=initial_atol)
