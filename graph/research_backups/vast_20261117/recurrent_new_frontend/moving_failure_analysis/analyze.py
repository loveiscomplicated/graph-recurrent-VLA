"""Summarize moving Box failure traces, using the installed task's geometry definitions."""
import json
from pathlib import Path

import numpy as np


OUTPUT = Path(__file__).resolve().parent
ARTIFACTS = OUTPUT.parents[1]
BASELINE = ARTIFACTS / 'closed_loop_moving_box_no_filter/source_runs'
RETRAINED = OUTPUT.parent / 'source_runs/eval'


def load(root, retrained):
    rows = {}
    for path in root.glob('shard_*/rollouts.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            seed = row['seed']
            assert seed not in rows
            rows[seed] = row['moving'] if retrained else row
    assert len(rows) == 32
    return rows


def summarize_rollout(episode):
    trace = episode['trace']
    predicted = np.array([frame['object21'] for frame in trace])
    truth = np.array([frame['object21_gt'] for frame in trace])
    tip = np.array([frame['actual_relative_mm'] for frame in trace])
    grasp = np.array([frame['grasp'] for frame in trace], dtype=bool)
    moving = np.array([frame['box_moving'] for frame in trace], dtype=bool)
    counts = np.array([frame['observed'] for frame in trace])
    hole_error = np.linalg.norm(predicted[:, 10:13] - truth[:, 10:13], axis=1) * 1000
    lateral_error = np.array([frame['raw_lateral_mm'] for frame in trace])
    assert all(frame['filtered'] == frame['object21'] for frame in trace)
    assert episode['steps'] == len(trace) and moving.any()
    assert episode['success'] or (len(trace) == 250 and episode['end_reason'] == 'horizon')
    assert np.allclose(truth[:, 17], truth[0, 17])
    assert np.isclose(truth[0, 20] - truth[0, 18], .003, atol=1e-6)
    lateral = np.linalg.norm(tip[:, 1:3], axis=1)
    half_length = truth[0, 17] * 1000
    # Hole frame's x=0 is inside the box. The entry plane is x=-half_length.
    from_entry = tip[:, 0] + half_length
    after = np.arange(len(trace)) > np.flatnonzero(moving)[-1]
    tail = slice(-20, None)

    def p90(x):
        return float(np.percentile(x, 90))

    return dict(success=episode['success'], steps=len(trace), shift=episode['box_max_shift_mm'],
                last_motion=int(np.flatnonzero(moving)[-1]), grasp_end=bool(grasp[-1]),
                grasp_dropped_after_motion=bool((~grasp[after]).any()),
                hole_p90_moving=p90(hole_error[moving]), tip_error_p90_moving=p90(lateral_error[moving]),
                hole_p90_post=p90(hole_error[after]), tip_error_p90_post=p90(lateral_error[after]),
                hole_error_end=float(hole_error[-1]), tip_error_end=float(lateral_error[-1]),
                peg_counts_last20_median=float(np.median(counts[tail, 0])),
                lateral_end=float(lateral[-1]), lateral_last20_median=float(np.median(lateral[tail])),
                depth_at_end=float(from_entry[-1]), depth_last20_median=float(np.median(from_entry[tail])),
                depth_last20_range=float(np.ptp(from_entry[tail])),
                deepest_from_entry_post=float(from_entry[after].max()),
                success_depth_gap_end=float(-15-tip[-1, 0]),
                closest_success_depth_gap_post=float(-15-tip[after, 0].max()),
                lateral_at_end_yz=tip[-1, 1:3].tolist())


def describe(rows):
    successes = sum(row['success'] for row in rows)
    return dict(n=len(rows), successes=successes,
                grasp_end=sum(row['grasp_end'] for row in rows),
                grasp_dropped_after_motion=sum(row['grasp_dropped_after_motion'] for row in rows),
                within_10mm_of_entry_at_tail=sum(abs(row['depth_last20_median']) <= 10 for row in rows),
                advanced_40mm_from_entry_post=sum(row['deepest_from_entry_post'] >= 40 for row in rows),
                medians={field: round(float(np.median([row[field] for row in rows])), 3) for field in
                         ('hole_p90_moving','hole_p90_post','tip_error_p90_moving','tip_error_p90_post',
                          'lateral_last20_median','depth_last20_median','depth_last20_range',
                          'peg_counts_last20_median','closest_success_depth_gap_post')})


def main():
    old = load(BASELINE, retrained=False)
    new = load(RETRAINED, retrained=True)
    assert old.keys() == new.keys()
    paired = {}
    for seed in sorted(new):
        assert old[seed]['motion_plan'] == new[seed]['motion_plan']
        a, b = [summarize_rollout(row) for row in (old[seed], new[seed])]
        paired[seed] = dict(old=a, new=b)
    categories = {}
    for policy in ('old', 'new'):
        for success in (False, True):
            cohort = [pair[policy] for pair in paired.values() if pair[policy]['success'] == success]
            categories[f'{policy}_{success}'] = describe(cohort)
    lost = [seed for seed, pair in paired.items() if pair['old']['success'] and not pair['new']['success']]
    gained = [seed for seed, pair in paired.items() if pair['new']['success'] and not pair['old']['success']]
    assert len(lost) == 6 and len(gained) == 3
    result = dict(categories=categories, lost=lost, gained=gained, paired=paired)
    (OUTPUT / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(categories, indent=2))
    for seed, pair in paired.items():
        if not pair['new']['success']:
            row = pair['new']
            print(seed, 'lost' if seed in lost else 'both failed', *[round(row[field], 2) for field in
                  ('depth_last20_median', 'depth_last20_range', 'lateral_last20_median',
                   'hole_p90_post','tip_error_p90_post','closest_success_depth_gap_post')])


if __name__ == '__main__':
    main()
