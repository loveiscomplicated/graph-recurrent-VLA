"""Local-only identities, immutable files and the preregistered protocol."""
import csv
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

from vla.data.files import sha,now,write,read,append_csv,npz,tree_identity,tensor_identity,checkpoint_load,digest




















def protocol(task08_plan, dataset_manifest, split_manifest, source_hash):
    """Fixed BEFORE policy outputs. No values derived from Plan B results."""
    config = read(task08_plan)['config']
    splits = split_manifest['splits']; episodes = dataset_manifest['episodes']
    if {k:len(v) for k,v in splits.items()} != dict(train=96, validation=12, test=12):
        raise ValueError('original 96/12/12 split required')
    if 30003 not in splits['train']: raise ValueError('registered train seed absent')
    dev = list(range(42000,42005))
    if set(dev) & {e['seed'] for e in episodes}: raise ValueError('dev leakage')
    return dict(version='plan_a_cached_graph_bc_v1', created_at=now(), seed=42,
        source_sha256=source_hash, source_reference='09-D c7c1934765394dc4866e6564a6dbc7ec frozen v2',
        geometry=dict(condition='C3', representation='SF', point_cap_per_object=128, coordinate_frame='ee',
            persistent=False, temporal_extra_features=0, node_dimension=21, edge_dimension=4,
            geometry_configuration='exact frozen09D protocol.json, unmodified'),
        model=dict(d=64, L=2, gru_hidden=144, gru_layers=1, enc_hidden=128),
        training={**config['training'], 'patience':None, 'updates':11487,
            'loss':'Task08 Method B; (6*motion_z_mse + continuous_gripper_mse)/7',
            'validation_best':'minimum validation Method B optimization loss, first tie',
            'scheduler_clock':'original cosine T_max50, advance at completed epoch, max11487 updates'},
        splits=splits, episode_lengths={str(e['episode_id']):e['actions'] for e in episodes},
        evaluation=dict(train_seed=30003, dev_seeds=dev, smoke_seed=44001,
            schedule_steps=[500]+list(range(1000,11001,1000))+[11487],
            expanded_steps=list(range(2000,10001,2000))+[11487],
            normal_dev_seeds=dev[:2], expanded_dev_seeds=dev,
            horizon=250, smoke_horizon=3, control_freq=20, physics_freq=100,
            stop_the_clock=True, early_failure_termination=False,
            test_access='after BOTH E0/E1 finish and validation selection frozen',
            final_models=['last','validation_best'],
            reach='TCP distance to current peg actor origin <=30mm (diagnostic proximity)',
            reach_distance_m=.03, grasp='installed agent.is_grasping(peg)',
            lift='is_grasping and peg origin >initial_z+5mm for >=3 consecutive post-action observations',
            lift_diagnostic_height_m=.005, lift_consecutive_steps=3,
            alignment='official evaluate peg_head_pos_at_hole: transverse |y|,|z|<=hole_radius and peg/hole x-axis <=10deg',
            alignment_angle_deg=10., insertion='official env info.success; independently crosscheck evaluate.success'),
        no_publish=True, no_hyperparameter_adaptation=True,
        baseline_task08_config_sha256=digest(config),
        parameter_count_caveat='E0/E1 differ in parameter count; not an isolated message-passing causal effect')
