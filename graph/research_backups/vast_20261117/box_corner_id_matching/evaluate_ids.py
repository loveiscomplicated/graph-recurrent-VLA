"""Select a Box matcher distance on validation and audit GT-free test assignments."""
import argparse
from pathlib import Path
from time import monotonic

import numpy as np

from box_matcher import BoxCornerMatcher
from data_collection.episode_dataset import file_sha256
from train_tracker import read_run
from utils.experiment_logger import ExperimentLogger, write

GATES_MM = (4, 6, 8, 10, 12, 15, 18)
MATCHABLE_MM = 5
STRONG_PRECISION = .99
RANGES = [(3100120, 3100132), (3100132, 3100144)] + [(s, s + 8) for s in range(3100144, 3100176, 8)]


def infer(raw):
    """Input is only flattened detector xyz, object class, and frame offsets."""
    xyz, kinds = raw['raw_xyz'], raw['raw_kind']
    end = np.cumsum(raw['raw_count'])
    offsets = np.r_[0, end]
    index = np.full((len(end), 8), -1, dtype=int)
    residual = np.full(index.shape, np.inf)
    initialized = np.zeros(len(end), bool)
    for i, step in enumerate(raw['step']):
        if step == 0:
            matcher = BoxCornerMatcher(assignment_m=max(GATES_MM) / 1000)
        start, stop = offsets[i:i+2]
        boxes = np.flatnonzero(kinds[start:stop] == 1) + start
        assignment = matcher.step(xyz[boxes])
        match = assignment >= 0
        selected = boxes[assignment[match]]
        assert len(np.unique(selected)) == len(selected)
        index[i, match] = selected
        if matcher.pose is not None:
            initialized[i] = True
            residual[i, match] = np.linalg.norm(xyz[selected] - matcher.pose.corners()[match], axis=-1)
    return index, residual, initialized


def score(xyz_raw, gt, index, oracle, visibility, frame_mask):
    safe = index.clip(min=0)
    accepted = index >= 0
    coords = xyz_raw[safe]
    dist = np.linalg.norm(coords[:, :, None] - gt[:, None], axis=-1)
    nearest = dist.argmin(axis=-1)
    close = dist.min(axis=-1) <= MATCHABLE_MM / 1000
    same = nearest == np.arange(8)
    tp = accepted & close & same
    wrong = accepted & close & ~same
    outside = accepted & ~close
    valid = frame_mask[:, None]
    num = int((accepted & valid).sum())
    right = int((tp & valid).sum())
    swaps = int((wrong & valid).sum())
    covered = int((tp & oracle & valid).sum())
    denom = int((oracle & valid).sum())
    visible = int((visibility & valid).sum())
    right_visible = int((tp & visibility & valid).sum())
    assert right + swaps + int((outside & valid).sum()) == num
    return dict(frames=int(frame_mask.sum()), correct=right, assigned=num, wrong_id=swaps,
                outside_gt_5mm=int((outside & valid).sum()),
                precision_5mm=right / num if num else 0.,
                identity_accuracy_given_gt_match=right / (right + swaps) if right else 0.,
                could_match=denom, correct_of_could_match=covered,
                matchable_recall=covered / denom if denom else 0.,
                matchable_coverage=denom / visible if visible else 0.,
                visible=visible, visible_recall=right_visible / visible if visible else 0.,
                correct_per_frame=right / int(frame_mask.sum()))


def load(root, gold_root):
    pieces, ids = [], []
    for start, stop in RANGES:
        folder = root / f'shard_{start}_{stop}'
        data, link = read_run(folder, 'detections.npz', 'sha256')
        assert np.unique(data['seed']).tolist() == list(range(start, stop))
        pieces.append(data)
        ids.append(link)
    raw = {key: np.concatenate([part[key] for part in pieces]) for key in pieces[0]}
    gt_parts = []
    for name in ('data_3100088_3100132','data_3100132_3100176'):
        data, link = read_run(gold_root / name, 'dataset.npz', 'dataset_sha256')
        gt_parts.append({key: arr[data['seed'] >= RANGES[0][0]] for key, arr in data.items()})
        ids.append(link)
    gt = {key: np.concatenate([part[key] for part in gt_parts]) for key in gt_parts[0]}
    for key in ('seed', 'step'):
        np.testing.assert_array_equal(raw[key], gt[key])
    assert len(raw['seed']) == 6058
    assert len(raw['raw_xyz']) == raw['raw_count'].sum()
    return raw, gt, ids


def main():
    parser = argparse.ArgumentParser()
    for name in ('capture','reference','output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    a = parser.parse_args()
    start = monotonic()
    raw, gt, sources = load(a.capture, a.reference)
    landmarks = gt['xyz'][:, 8:16]
    val = raw['seed'] < 3100144
    test = ~val
    index, residual, initialized = infer(raw)
    oracle = raw['observed'][:, 8:16]
    visible = raw['visible'][:, 8:16]
    metrics = {gate: score(raw['raw_xyz'], landmarks, np.where(residual <= gate / 1000, index, -1),
                           oracle, visible, val) for gate in GATES_MM}
    possible = [gate for gate, row in metrics.items() if row['precision_5mm'] >= STRONG_PRECISION]
    if possible:
        chosen = max(possible, key=lambda gate: (metrics[gate]['matchable_recall'], metrics[gate]['precision_5mm']))
    else:
        def f1(gate):
            p, r = (metrics[gate][key] for key in ('precision_5mm', 'matchable_recall'))
            return 2 * p * r / max(p + r, 1e-9)
        chosen = max(GATES_MM, key=f1)
    final = np.where(residual <= chosen / 1000, index, -1)
    late = np.zeros(len(test), bool)
    for seed in np.unique(raw['seed'][test]):
        ix = np.flatnonzero((raw['seed'] == seed) & (gt['phase'] == 'align') & gt['grasp'])
        assert len(ix) >= 5
        late[ix[-5:]] = True
    periods = dict(all=test, motion=test & gt['moving_now'], last5=late, first=test & (raw['step'] == 0))
    result = {name: score(raw['raw_xyz'], landmarks, final, oracle, visible, mask)
              for name, mask in periods.items()}
    first = [int(raw['step'][(raw['seed'] == s) & initialized].min())
             for s in np.unique(raw['seed'][test]) if ((raw['seed'] == s) & initialized).any()]
    votes = (final >= 0).sum(axis=1)
    config = dict(sources=sources, gates_mm=GATES_MM, gt_scoring_m=.005, validation_target=STRONG_PRECISION,
                  chosen_mm=chosen, matcher_sha256=file_sha256(Path(__file__).with_name('box_matcher.py')),
                  runner_sha256=file_sha256(__file__))
    a.output.mkdir(parents=True, exist_ok=False)
    with ExperimentLogger(team='graph', model='box_corner_id_matching', job_type='evaluate',
                          seed=0, config=config, root=a.output / 'experiments') as logger:
        data, out = a.output / 'assignments.npz', a.output / 'summary.json'
        logger.record(runtime=dict(evaluation='CPU NumPy', detector='CUDA RANSAC'))
        write(out, dict(run_id=logger.run_id, config=config, validation=metrics, test=result,
                        first_init_histogram=dict(zip(*[arr.tolist() for arr in np.unique(first, return_counts=True)])),
                        first_init_episodes=len(first), elapsed_seconds=monotonic() - start,
                        test_frames_with_no_assignment=int(((votes == 0) & test).sum())))
        np.savez_compressed(data, seed=raw['seed'], step=raw['step'], assigned_candidate_index=final,
                            assigned=final >= 0,
                            assigned_xyz=np.where((final >= 0)[..., None], raw['raw_xyz'][final.clip(min=0)], 0.),
                            oracle_available=oracle, gt_xyz=landmarks, visible=visible, raw_count=raw['raw_count'],
                            raw_xyz=raw['raw_xyz'], raw_oracle_id=raw['raw_oracle_id'], moving=gt['moving_now'],
                            test=test, last5=late, initialized=initialized)
        logger.log(result)
        logger.link_output(data)
        logger.link_output(out)
    print('Completed GT-free Box ID evaluation', flush=True)


if __name__ == '__main__':
    main()
