"""Standalone, exportable figures for measured vertex-GNN diagnostics."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluation.gnn.vertex_diagnostics import MODELS, group_masks
from evaluation.gnn.vertex_metrics import load_aligned_predictions


LABELS = ('Baseline', 'A', 'B', 'AB')
COLORS = ('#586172', '#3476ae', '#cf8031', '#38856b')


def plot_probe_comparison(summary_path, output):
    """Show paired uncertainty and training-seed variation on identical features."""
    summary = json.loads(Path(summary_path).read_text())
    pairs = (
        ('multiseed', 'Multi-seed − Attention'),
        ('attention_capacity', 'Capacity − Attention'),
        ('multiseed_vs_attention_capacity', 'Multi-seed − Capacity'),
    )
    metrics = (
        ('peg_half_size_x_error_mm', 'Size X'),
        ('relative_3d_error_mm', 'Relative 3D'),
        ('lateral_error_mm', 'Lateral'),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for axis, (metric, title) in zip(axes, metrics):
        for row, (name, label) in enumerate(pairs):
            measured = summary['paired_comparisons'][name]['averaged'][metric]
            interval = measured['parent_ci95_conditional_on_seeds']
            mean = measured['seed_mean_delta']
            axis.errorbar(mean, row, xerr=[[mean - interval['low']], [interval['high'] - mean]],
                          fmt='o', color='black', capsize=4, markersize=6, zorder=4)
            for seed, value in enumerate(measured['seed_deltas']):
                axis.scatter(value, row + (seed - 1) * .10, s=28,
                             color=COLORS[seed + 1], alpha=.8, zorder=3)
        axis.axvline(0, color='#777777', ls='--', lw=1)
        axis.set_title(title)
        axis.set_xlabel('Paired error difference (mm); + is worse')
        axis.set_yticks(range(len(pairs)), [label for _, label in pairs])
        axis.grid(axis='x', alpha=.15)
    axes[0].invert_yaxis()
    figure.suptitle('Fixed features: black = mean / parent 95% CI; blue/orange/green = training seeds 0/1/2')
    figure.tight_layout()
    figure.savefig(output/'frozen_readout_comparison.png', dpi=180)
    plt.close(figure)


def render(config):
    """Render recorded distributions and interventions without refitting models."""
    predictions, _ = load_aligned_predictions(config['predictions'])
    output = Path(config['output']['root']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    baseline = predictions['baseline']
    gt = baseline['target'][:,17]*1000
    high = group_masks(baseline['counts'])['high_support_proxy']
    figure, axes = plt.subplots(2,4,figsize=(14,7),sharex=True,sharey=True)
    for column, (name,label,color) in enumerate(zip(MODELS,LABELS,COLORS)):
        for row, mask in enumerate((high,~high)):
            axis = axes[row,column]
            axis.hexbin(gt[mask],predictions[name]['object21'][mask,17]*1000,
                        gridsize=35,mincnt=1,bins='log',cmap='viridis',extent=(80,130,60,155))
            axis.plot([80,130],[80,130],color='tomato',lw=1)
            axis.set_title(f'{label}: {"high" if row==0 else "low"} support')
            if row == 1:
                axis.set_xlabel('GT peg half-size X (mm)')
            if column == 0:
                axis.set_ylabel('Predicted half-size X (mm)')
    figure.suptitle('Size tracking on identical validation frames; counts do not imply full visibility')
    figure.tight_layout()
    figure.savefig(output/'size_tracking.png',dpi=180)
    plt.close(figure)

    relative = json.loads(Path(config['relative_summary']).read_text())
    conditions = ('original','gt_peg_position','gt_peg_rotation','gt_peg_size','gt_hole_position','gt_peg_endpoint','no_residual')
    labels = ('Original','GT peg pos','GT peg rot','GT peg size','GT hole pos','GT peg tip','Residual = 0')
    figure, axes = plt.subplots(1,2,figsize=(13,5))
    x = np.arange(len(conditions))
    for axis,metric,title in zip(axes,('relative_3d_mm','lateral_mm'),('Relative 3D','Lateral')):
        for offset,(name,label,color) in enumerate(zip(MODELS,LABELS,COLORS)):
            values=[relative[name]['oracle_fixed_residual'][condition][metric]['mean'] for condition in conditions]
            axis.bar(x+(offset-1.5)*.18,values,width=.18,label=label,color=color)
        axis.set_xticks(x,labels,rotation=35,ha='right')
        axis.set_ylabel('Mean error (mm)')
        axis.set_title(title)
    figure.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc='upper center',
                  bbox_to_anchor=(.5, .94), frameon=False)
    figure.suptitle('Oracle output substitutions; original learned residual held fixed except Residual = 0')
    figure.tight_layout(rect=(0, 0, 1, .87))
    figure.savefig(output/'oracle_components.png',dpi=180)
    plt.close(figure)

    if config.get('train_geometry'):
        with np.load(config['train_geometry']) as archive:
            train = archive['target'][:, 17] * 1000
        figure,axes=plt.subplots(1,2,figsize=(11,4))
        bins=np.linspace(80,130,26)
        axes[0].hist(train,bins=bins,density=True,alpha=.55,label=f'Train n={len(train):,}')
        axes[0].hist(gt,bins=bins,density=True,alpha=.55,label=f'Validation n={len(gt):,}')
        axes[0].set_xlabel('GT peg half-size X (mm)')
        axes[0].set_ylabel('Frame-weighted density')
        axes[0].legend()
        for name,label,color in zip(MODELS,LABELS,COLORS):
            error=predictions[name]['peg_half_size_x_error_mm']
            ordered=np.sort(error)
            axes[1].plot(ordered,np.arange(1,len(error)+1)/len(error),label=label,color=color)
        axes[1].set_xlim(0,30)
        axes[1].set_xlabel('Size X absolute error (mm)')
        axes[1].set_ylabel('Empirical CDF')
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(output/'size_distribution.png',dpi=180)
        plt.close(figure)
    if config.get('cases'):
        cases = json.loads(Path(config['cases']).read_text())
        selected = [case for direction in ('worse', 'better')
                    for case in [row for row in cases if row['direction'] == direction][:2]]
        figure, axes = plt.subplots(2, 2, figsize=(12, 7))
        for axis, case in zip(axes.flat, selected):
            points = np.asarray(case['peg_xyz_gt_local_m']) * 1000
            half_size = baseline['target'][case['index'], 17:20] * 1000
            axis.axvspan(-half_size[0], half_size[0], color='#dddddd', alpha=.35)
            axis.axvline(-half_size[0], color='black', ls='--', lw=1)
            axis.axvline(half_size[0], color='black', ls='--', lw=1)
            axis.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='coolwarm',
                         s=65, vmin=-30, vmax=30, edgecolors='black', linewidths=.5)
            predicted = case['predicted_size_x_mm']
            axis.set_title(f"AB {case['direction']}: {case['sample_id'].split('/')[0]} / "
                           f"{case['sample_id'].split('/')[1]}\n"
                           f"GT={case['gt_size_x_mm']:.1f}, Base={predicted['baseline']:.1f}, "
                           f"AB={predicted['combined']:.1f} mm")
            axis.set_xlabel('Observed peg X in GT local frame (mm)')
            axis.set_ylabel('Observed peg Y (mm)')
            axis.set_xlim(-135, 135)
            axis.set_ylim(-35, 35)
        figure.suptitle('Coordinates span both ends even in the largest regressions; color denotes local Z')
        figure.tight_layout()
        figure.savefig(output/'observed_peg_cases.png', dpi=180)
        plt.close(figure)
    if config.get('probe_summary'):
        plot_probe_comparison(config['probe_summary'], output)
    return output
