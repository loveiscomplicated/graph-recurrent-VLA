"""Render completed audit summaries without rerunning perception or geometry."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root):
    rows = json.loads((root / 'remote/information_loss_v1/summary.json').read_text())['stages']
    lookup = {(r['size'], r['object_id'], r['stage']): r for r in rows}
    gains = json.loads((root / 'remote/information_loss_v1/gain_summary.json').read_text())
    gain = {(r['object_id'], r['stage']): r['retained_fraction_mean'] for r in gains
            if r['size'] == 512 and r['region'] == 'all'}
    stages = ['raw', 'gate', 'shape_first', 'cross_view', 'dedup', 'fps128']
    labels = ['Raw', 'Gate', 'Shape-first', 'Cross-view', 'Dedup', 'FPS128']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {128: '#667085', 512: '#1769aa'}
    for oid, ax in enumerate(axes[0]):
        for size in (128, 512):
            ax.plot(labels, [100 * lookup[size, oid, s]['coverage_all_mean'] for s in stages],
                    marker='o', color=colors[size], label=f'Native {size}')
        ax.set(title=f'{("Peg", "Box")[oid]}: common visible-surface coverage', ylabel='Coverage (%)', ylim=(0, 105))
        ax.grid(alpha=.2)
        ax.legend()
    ax = axes[1, 0]
    for oid, color in ((0, '#1769aa'), (1, '#d97706')):
        ax.plot(labels, [100 * gain[oid, s] for s in stages], marker='o', color=color,
                label=('Peg', 'Box')[oid])
    ax.set(title='Retention of surface samples observed only at 512', ylabel='Retained gain (%)', ylim=(0, 105))
    ax.grid(alpha=.2)
    ax.legend()
    ax = axes[1, 1]
    region_labels = ['Peg end -X', 'Peg end +X', 'Box approach rim']
    regions = [(0, 'end_minus'), (0, 'end_plus'), (1, 'observed_rim')]
    for offset, stage, label, color in ((-.27, 'gate', '512 Gate', '#b5d5ec'),
            (0, 'dedup', '512 pre-FPS', '#4292c6'), (.27, 'fps128', '512 final', '#084594')):
        values = [100 * lookup[512, oid, stage]['coverage_' + region + '_mean'] for oid, region in regions]
        ax.bar([i + offset for i in range(3)], values, width=.25, label=label, color=color)
    ax.set(xticks=range(3), xticklabels=region_labels, title='Structural regions lost at final compression',
           ylabel='Region coverage (%)', ylim=(0, 105))
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=.2)
    fig.suptitle('Frozen 128/512 inputs: where observed geometry is lost', fontsize=16)
    fig.savefig(root / 'information_loss_localization.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
