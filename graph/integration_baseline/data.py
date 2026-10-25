"""Hash/identity/alignment gates and episode-local ragged graph sequences."""
from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from graph.vision_input import RAW_EDE_POLICY, validate_packet
from graph.integration_baseline.graph import build_graph, pack_graphs
from graph.integration_baseline.io import read_packet, sha256, write_json
from graph.integration_baseline.robot import geometry_in_ee, RobotNormalizer


@dataclass
class AlignedFrame:
    episode_id: int
    timestep: int
    packet: dict
    robot_state: np.ndarray
    robot_positions_ee: np.ndarray
    action: np.ndarray
    terminated: bool
    truncated: bool


class ReplayDataset:
    """Only exact source/episode/timestep joins; no nearest-time or stale fill."""
    def __init__(self, replay, vision):
        self.replay, self.vision = Path(replay).resolve(), Path(vision).resolve()
        audit=self.vision/'quality_audit.json'
        if audit.exists() and json.loads(audit.read_text()).get('training_eligible') is False:
            raise ValueError('Vision dataset was explicitly rejected by a quality audit')
        self.manifest = json.loads((self.replay/'manifest.json').read_text())
        self.graph_manifest = json.loads((self.vision/'manifest.json').read_text())
        m, g = self.manifest, self.graph_manifest
        if (g['replay_manifest_sha256'] != sha256(self.replay/'manifest.json')
                or g['source_sha256'] != m['source_sha256'] or g['splits'] != m['splits']
                or g['robot_state_schema'] != m['robot_state_schema']):
            raise ValueError('Vision/replay manifest identity or schema mismatch')
        self.schema, self.splits = m['robot_state_schema'], m['splits']
        frozen = json.loads(Path(m['frozen_split_manifest']).read_text())
        if sha256(m['frozen_split_manifest']) != m['frozen_split_sha256'] or frozen['splits'] != self.splits:
            raise ValueError('frozen split changed')
        flattened = [i for ids in self.splits.values() for i in ids]
        expected = [e['episode_id'] for e in frozen['episodes']]
        if len(flattened) != len(set(flattened)) or set(flattened) != set(expected):
            raise ValueError('episode splits are overlapping or incomplete')
        groups = {}
        for split, ids in self.splits.items():
            for e in frozen['episodes']:
                if e['episode_id'] in ids:
                    seed = e['episode_seed']
                    if seed in groups and groups[seed] != split:
                        raise ValueError('reset group crosses splits')
                    groups[seed] = split
        if sha256(m['source']) != m['source_sha256']:
            raise ValueError('expert source hash changed')
        if 'source_json_sha256' in m:
            metadata_path=Path(m['source']).with_suffix('.json')
            if sha256(metadata_path)!=m['source_json_sha256'] or sha256(m['initial_source'])!=m['initial_source_sha256']:
                raise ValueError('source metadata/original initial state changed')
            source_episodes={e['episode_id']:e for e in json.loads(metadata_path.read_text())['episodes']}
            for episode in m['episodes']:
                original=source_episodes[episode['episode_id']]
                if (episode['reset_identity']!=original['reset_kwargs'] or episode['reset_seed']!=original['episode_seed']
                        or episode['control_mode']!=original['control_mode']):
                    raise ValueError('source/reset identity mismatch')
        self.raw = {}
        self.episodes = {e['episode_id']: e for e in m['episodes']}
        for eid, e in self.episodes.items():
            path = self.replay/e['path']
            if sha256(path) != e['sha256']:
                raise ValueError('raw replay episode hash mismatch')
            with np.load(path, allow_pickle=False) as a:
                self.raw[eid] = {k: a[k].copy() for k in a.files}
        self.packet_rows = {}
        for e in g['episodes']:
            eid = e['episode_id']
            if e['replay_sha256'] != self.episodes[eid]['sha256']:
                raise ValueError('graph episode uses different replay')
            provenance = e['mask_provenance']
            if (not provenance or provenance.get('gt_segmentation_used') is not False
                    or provenance.get('future_observations_used') is not False
                    or not provenance.get('sam2_horizon','').startswith('num_frames=t+1;')):
                raise ValueError('unverified mask provenance')
            if sha256(self.vision/e['mask_path']) != e['mask_sha256']:
                raise ValueError('mask artifact hash changed')
            for row in e['packets']:
                key = (eid, row['timestep'])
                if key in self.packet_rows:
                    raise ValueError('duplicate episode/timestep')
                self.packet_rows[key] = row
        self.validate_raw()

    def validate_raw(self):
        m = self.manifest
        with h5py.File(m['source'], 'r') as source:
            for eid, a in self.raw.items():
                ep = self.episodes[eid]
                n = len(a['actions'])
                if (n != ep['aligned_targets'] or len(a['robot_state']) != n+1
                        or not np.array_equal(a['timestep'], np.arange(n+1))
                        or not np.all(a['episode_id'] == eid)
                        or str(a['source_sha256']) != m['source_sha256']):
                    raise ValueError('raw identity/continuity/length mismatch')
                ref = source[f'traj_{eid}']
                for name in ('actions', 'terminated', 'truncated'):
                    if not np.array_equal(a[name], ref[name][:n]):
                        raise ValueError('expert target/flag alignment changed: ' + name)
                if (a['robot_state'].shape != (n+1, self.schema['dimension'])
                        or not np.isfinite(a['robot_state']).all()
                        or not np.isfinite(a['robot_world_geometry']).all()):
                    raise ValueError('full robot schema invalid')
                np.testing.assert_allclose(a['timestamp'], np.arange(n+1)/ep['control_freq'], rtol=0, atol=0)
                # Only robot prefix25 enters this comparison. The remaining
                # source GT columns are audit data, not inputs to the model.
                specs = {f['name']: f for f in self.schema['fields']}
                columns = []
                for name in ('qpos', 'qvel', 'tcp_pose'):
                    spec = specs[name]
                    columns.append(a['robot_state'][:, spec['offset']:spec['offset']+spec['dimension']])
                robot = np.concatenate(columns, -1)
                if np.max(np.abs(robot-ref['obs'][:n+1, :25])) > m['pilot_gate']['replay_atol']:
                    raise ValueError('raw robot/source replay alignment error')
                np.testing.assert_array_equal(columns[-1], a['tcp_pose'])
                for t in range(n+1):
                    geometry_in_ee(a['robot_world_geometry'][t], a['tcp_pose'][t])
                # Verify the SAME-time finger link poses reconstruct precisely
                # the existing rubber-pad collision centres (no object state).
                from envs.maniskill.gt_state import _LEFT_TIP_LOCAL, _RIGHT_TIP_LOCAL
                from visual.runtime.vision_packet import pose_matrix
                for index,name,offset in [(1,'left_finger_pose',_LEFT_TIP_LOCAL),(2,'right_finger_pose',_RIGHT_TIP_LOCAL)]:
                    spec=specs[name]
                    for t in range(n+1):
                        matrix=pose_matrix(a['robot_state'][t,spec['offset']:spec['offset']+spec['dimension']])
                        point=matrix[:3,3]+matrix[:3,:3]@np.asarray(offset)
                        np.testing.assert_allclose(point,a['robot_world_geometry'][t,index],rtol=0,atol=1e-6)

    def frame(self, eid, t):
        row = self.packet_rows[(eid, t)]
        path = self.vision/row['path']
        if sha256(path) != row['sha256']:
            raise ValueError('packet hash changed')
        p, a = read_packet(path), self.raw[eid]
        validate_packet(p)
        if t >= len(a['actions']) or not row['has_action_target']:
            raise ValueError('final observation has no expert next-action target')
        identity = self.manifest['source_sha256'] + f':traj_{eid}'
        provenance = p['observation_provenance']
        if (p['episode'] != identity or p['source_frame'] != t or p['frame_idx'] != t
                or p['timestamp'] != float(a['timestamp'][t]) or provenance['episode_id'] != eid
                or provenance['timestep'] != t or provenance['source_sha256'] != self.manifest['source_sha256']
                or provenance['replay_sha256'] != self.episodes[eid]['sha256']
                or p['edge_policy'] != RAW_EDE_POLICY):
            raise ValueError('packet identity/timing/policy mismatch')
        np.testing.assert_array_equal(p['observation_ee_pose'], a['tcp_pose'][t].astype(np.float64))
        camera = self.graph_manifest['camera']
        np.testing.assert_array_equal(p['camera_to_world'], a[camera+'/cam2world'][t].astype(np.float64))
        if 'rgb_sha256' in provenance:
            import hashlib
            if provenance['rgb_sha256']!=hashlib.sha256(a[camera+'/rgb'][t].tobytes()).hexdigest():
                raise ValueError('packet refers to a different RGB observation')
        return AlignedFrame(eid, t, p, a['robot_state'][t].copy(),
                            geometry_in_ee(a['robot_world_geometry'][t], a['tcp_pose'][t]),
                            a['actions'][t].copy(), bool(a['terminated'][t]), bool(a['truncated'][t]))

    def sequence(self, eid, start, length):
        if length < 1:
            raise ValueError('positive sequence length required')
        result = [self.frame(eid, t) for t in range(start, start+length)]
        if any(r.terminated or r.truncated for r in result[:-1]):
            raise ValueError('sequence crosses terminated/truncated boundary')
        return result

    def fit_normalizer(self):
        ids = [eid for eid in self.raw if eid in self.splits['train']]
        rows = np.concatenate([self.raw[e]['robot_state'][:-1] for e in ids])
        return RobotNormalizer.fit(rows, ids, self.splits, self.schema)

    def validation_report(self):
        valid = []
        for (eid, t), row in sorted(self.packet_rows.items()):
            if row['has_action_target']:
                self.frame(eid, t)
                valid.append((eid, t))
        return dict(status='REAL_ALIGNED' if valid else 'BLOCKED_DATA', validated_targets=len(valid),
                    scope='independently verified pre-failure observation prefixes only',
                    full_source_episodes=self.manifest['episodes_expected'],
                    full_source_actions=self.manifest['actions_expected'],
                    uncaptured_or_unlabelled_source_actions=self.manifest['actions_expected']-len(valid),
                    full_replay_pilot_gate=self.manifest['pilot_gate']['status'],
                    complete_source_episodes=self.manifest['coverage']['complete_episodes'],
                    captured_source_episodes=self.manifest['coverage']['episodes'],
                    observation_action_alignment='PASS', robot_geometry_same_time='PASS',
                    packet_policy='raw_extended_delaunay/none', splits=self.splits,
                    missing_full_replay_episodes=self.manifest['missing_episode_ids'],
                    source_replay_failure=self.manifest['failures'],
                    frames_by_episode={str(e):sum(i==e for i,t in valid) for e in self.raw},
                    gt_task_inputs=False, action_or_future_input=False)


def collate_sequences(sequences, *, k=4, M=16, block_rows=64):
    graphs, mapping, states = [], [], []
    max_length = max(map(len, sequences))
    targets = torch.zeros((len(sequences), max_length, 7))
    mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
    for b, seq in enumerate(sequences):
        if not seq or any(r.episode_id != seq[0].episode_id or r.timestep != seq[0].timestep+t for t,r in enumerate(seq)):
            raise ValueError('episode-local contiguous frames required')
        if any(r.terminated or r.truncated for r in seq[:-1]):
            raise ValueError('terminal crossing')
        for t, row in enumerate(seq):
            graphs.append(build_graph(row.packet, row.robot_positions_ee, k=k, M=M, block_rows=block_rows))
            mapping.append((b, t))
            states.append(row.robot_state)
            targets[b, t] = torch.from_numpy(row.action)
            mask[b, t] = True
    return pack_graphs(graphs, mapping), torch.from_numpy(np.stack(states)), targets, mask


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--vision', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset = ReplayDataset(args.replay, args.vision)
    write_json(args.output, dataset.validation_report())


if __name__ == '__main__':
    main()
