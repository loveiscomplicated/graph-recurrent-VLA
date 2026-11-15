"""Validate paired targeted Graph replays and compare each expert seed."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
FOLDERS = ('hole_front_object21_ablation/full', 'hole_front_object21_latest_detector/full',
           'hole_detector_cuda_regression/ransac_graph')
METRICS = ('hole_center_mm', 'axis_deg', 'tip_lateral_mm')
SEEDS = (350, 800, 950, 1175)


def collect(folder):
    result = defaultdict(list)
    path = ROOT.parent / folder / 'frames.jsonl'
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['seed'] in SEEDS:
            result[row['seed']].append(row)
    return result


def compare():
    sources = [collect(folder) for folder in FOLDERS]
    settings = [json.loads((ROOT.parent / folder / 'summary.json').read_text()) for folder in FOLDERS]
    for setting in settings:
        for key in ('camera', 'device', 'checkpoint_sha256', 'tracker_gate_m'):
            assert setting['config'][key] == settings[0]['config'][key]
    outputs = {}
    for seed in SEEDS:
        stages, tails = [], []
        for rows in (s[seed] for s in sources):
            stages.append([(r['step'], r['stages'], r['gt21']) for r in rows])
            tails.append([r['detected']['errors'] for r in rows if 'held_align' in r['stages']][-5:])
        assert stages[0] == stages[1] == stages[2], seed
        assert all(len(t) == 5 for t in tails)
        outputs[seed] = {
            label: {metric: float(np.median([r[metric] for r in tail])) for metric in METRICS}
            for label, tail in zip(('previous', 'grid', 'ransac'), tails, strict=True)}
    (ROOT / 'graph_comparison.json').write_text(json.dumps(outputs, indent=2)+'\n')


if __name__ == '__main__':
    compare()
