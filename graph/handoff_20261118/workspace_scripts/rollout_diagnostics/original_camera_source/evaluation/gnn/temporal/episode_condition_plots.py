"""Plot 21D prediction errors alongside GT-audited per-frame vertex coverage."""

import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from data_collection.storage import write_json
from evaluation.gnn.vertex_metrics import per_frame_errors, statistics
from graph.pose_geometry import geometric_endpoint
from utils.experiment_logger import ExperimentLogger


PANELS = (
    ('peg_position_error_mm', 'Peg position', 'mm'),
    ('hole_position_error_mm', 'Hole position', 'mm'),
    ('relative_3d_error_mm', 'Peg tip minus hole', 'mm'),
    ('peg_rotation_error_deg', 'Peg rotation', 'degrees'),
    ('hole_rotation_error_deg', 'Hole rotation', 'degrees'),
    ('peg_half_size_x_error_mm', 'Peg half-size X', 'mm'),
    ('peg_half_size_y_error_mm', 'Peg half-size Y', 'mm'),
    ('peg_half_size_z_error_mm', 'Peg half-size Z', 'mm'),
    ('hole_radius_error_mm', 'Hole radius', 'mm'),
)
COLORS = ('#9ca3af', '#f59e0b', '#8b5cf6', '#059669')
LABELS = ('Neither', 'Peg only', 'Hole only', 'Both')


def score(prediction, target):
    """Use the same physical metrics as the original frame-level M0 evaluation."""
    pred = torch.from_numpy(prediction.copy())
    truth = torch.from_numpy(target.copy())
    output = {'object21': pred, 'endpoint_geom': geometric_endpoint(pred),
              'endpoint_residual': torch.zeros_like(pred[:, :3])}
    return per_frame_errors(output, truth)


def plot(config):
    """Create per-episode plots, an index and a multipage PDF from saved inference."""
    inputs = config['predictions']
    prediction_paths = [Path(value).resolve() for value in (inputs if isinstance(inputs, list) else [inputs])]
    if not prediction_paths:
        raise ValueError('At least one predictions file is required')
    if 'seeds' in config and len(config['seeds']) != len(prediction_paths):
        raise ValueError('The number of seed labels must match the predictions files')
    prediction_path = prediction_paths[0]
    reference_path = Path(config.get('reference_predictions', prediction_path)).resolve()
    decisions_path = Path(config['decisions']).resolve()
    decision_summary = json.loads(Path(config['decision_summary']).read_text())
    if decision_summary['predictions_sha256'] != file_sha256(reference_path):
        raise ValueError('Physical decisions were produced for different predictions')
    rows = [json.loads(line) for line in decisions_path.read_text().splitlines()]
    lookup = {row['sample_id']: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate physical decision sample IDs')
    alignment_keys = ('sample_ids', 'episode', 'frame', 'target', 'tcp_world', 'pose_quality', 'quality')
    with np.load(reference_path, allow_pickle=False) as saved:
        reference = {key: saved[key].copy() for key in alignment_keys}
    predictions = []
    for path in prediction_paths:
        with np.load(path, allow_pickle=False) as saved:
            for key in alignment_keys:
                if not np.array_equal(saved[key], reference[key]):
                    raise ValueError(f'Prediction {path} differs from the physical audit in {key}')
            predictions.append({key: saved[key].copy() for key in ('gnn_object21', 'primary_object21')})
    data = reference
    ids = [str(value) for value in data['sample_ids']]
    if len(ids) != len(set(ids)) or set(ids) != set(lookup):
        raise ValueError('Prediction and physical decision frame IDs must match exactly')
    decisions = [lookup[value] for value in ids]
    peg = np.array([row['peg_pass'] for row in decisions], dtype=bool)
    hole = np.array([row['hole_pass'] for row in decisions], dtype=bool)
    status = peg.astype(int) + 2 * hole.astype(int)
    sampled = np.array([row['decision_source'] == 'original_detector_uv' for row in decisions])
    seed_scores = {
        'M0': [score(row['gnn_object21'], data['target']) for row in predictions],
        'Mamba': [score(row['primary_object21'], data['target']) for row in predictions],
    }
    scores = {name: {key: np.stack([row[key] for row in values]).mean(axis=0)
                     for key in values[0]} for name, values in seed_scores.items()}
    seed_label = 'seed 0' if len(predictions) == 1 else f'{len(predictions)} seeds: mean of frame errors'
    with ExperimentLogger(team='graph', config=config, seed=config.get('seed', 0),
                          model='episode_condition_plots', job_type='evaluation') as logger:
        output = Path(config['output']['root']) / logger.run_id
        images = output / 'episodes'
        images.mkdir(parents=True, exist_ok=False)
        index_rows = []
        cmap = ListedColormap(COLORS)
        norm = BoundaryNorm(np.arange(-0.5, 4.5), cmap.N)
        handles = [Patch(facecolor=color, label=label) for color, label in zip(COLORS, LABELS)]
        with PdfPages(output / 'episodes_21d.pdf') as pdf:
            for number, episode in enumerate(np.unique(data['episode']), start=1):
                mask = data['episode'] == episode
                frames = data['frame'][mask]
                if not np.array_equal(np.diff(frames), np.ones(max(len(frames) - 1, 0))):
                    raise ValueError(f'Non-contiguous frames in episode {episode}')
                fig = plt.figure(figsize=(16, 12), constrained_layout=True)
                grid = fig.add_gridspec(4, 3, height_ratios=(0.27, 1, 1, 1))
                ribbon = fig.add_subplot(grid[0, :])
                ribbon.imshow(status[mask][None], aspect='auto', interpolation='nearest', cmap=cmap,
                              norm=norm, extent=(frames[0] - .5, frames[-1] + .5, 0, 1))
                ribbon.scatter(frames[sampled[mask]], np.full(sampled[mask].sum(), .5),
                               s=5, c='black', label='Original 960 frames', zorder=5)
                ribbon.set_yticks([])
                ribbon.set_title(f'{episode}  |  physical condition per frame  |  black dots: original validation frames')
                ribbon.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1.65),
                              ncol=4, frameon=False, fontsize=9)
                for position, (key, title, unit) in enumerate(PANELS):
                    ax = fig.add_subplot(grid[1 + position // 3, position % 3])
                    ax.plot(frames, scores['M0'][key][mask], color='#64748b', alpha=.8,
                            linewidth=1, label='M0 mean' if len(predictions) > 1 else 'M0')
                    if len(predictions) > 1 and position < 5:
                        for seed_index, seed_score in enumerate(seed_scores['Mamba']):
                            ax.plot(frames, seed_score[key][mask], color='#93c5fd', alpha=.55,
                                    linewidth=.6, label='Mamba individual seeds' if seed_index == 0 else None)
                    ax.plot(frames, scores['Mamba'][key][mask], color='#2563eb',
                            linewidth=1.3, label='Mamba mean' if len(predictions) > 1 else 'M0 + Mamba')
                    ax.set_title(title, fontsize=10)
                    ax.set_ylabel(unit)
                    ax.set_xlabel('Frame')
                    ax.grid(alpha=.2)
                    if position == 0:
                        ax.legend(frameon=False, fontsize=8)
                fig.suptitle(f'Object21 errors | {seed_label} | GT-assisted coverage certificates', fontsize=13)
                name = f'episode_{number:02d}.png'
                fig.savefig(images / name, dpi=110)
                pdf.savefig(fig)
                plt.close(fig)
                index_rows.append({'episode': str(episode), 'image': f'episodes/{name}',
                                   'frames': int(mask.sum()), 'both': int((status[mask] == 3).sum())})
                if number % 10 == 0:
                    print(f'Plotted {number} episodes', flush=True)
        summary = {'seeds': config.get('seeds', [config.get('seed', 0)]),
                   'aggregation': 'mean of per-frame errors across seeds',
                   'frames': len(ids), 'episodes': len(index_rows),
                   'original_sampled_frames': int(sampled.sum()),
                   'category_counts': {label: int((status == i).sum()) for i, label in enumerate(LABELS)},
                   'prediction_sha256': [file_sha256(path) for path in prediction_paths],
                   'reference_prediction_sha256': file_sha256(reference_path),
                   'decisions_sha256': file_sha256(decisions_path), 'metrics': {}}
        for name, values in scores.items():
            summary['metrics'][name] = {}
            for label, group in [('all_frames', np.ones(len(ids), dtype=bool)),
                                 ('original_960', sampled)] + [
                                     (LABELS[i], status == i) for i in range(4)]:
                summary['metrics'][name][label] = {key: statistics(values[key][group])
                                                    for key, _, _ in PANELS}
        write_json(output / 'summary.json', summary)
        write_json(output / 'episodes.json', index_rows)
        write_json(output / 'effective_config.json', config)
        links = ''.join(f'<li><a href="#{i}">{html.escape(row["episode"])}</a> '
                        f'({row["frames"]} frames, both {row["both"]})</li>'
                        for i, row in enumerate(index_rows))
        figures = ''.join(f'<section id="{i}"><h2>{html.escape(row["episode"])}</h2>'
                          f'<img loading="lazy" src="{row["image"]}" alt="{html.escape(row["episode"])} error plots">'
                          f'<p><a href="#top">Back to episode list</a></p></section>'
                          for i, row in enumerate(index_rows))
        page = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>에피소드별 21D 오차</title>'
                '<style>body{font:16px system-ui;max-width:1300px;margin:auto;padding:24px;line-height:1.5}'
                'img{width:100%;height:auto;border:1px solid #ddd}li{margin:4px}section{margin:3em 0}</style>'
                f'<h1 id="top">에피소드별 21D 오차 — {html.escape(seed_label)}</h1>'
                '<p>회색: 두 조건 모두 미충족, 주황: Peg만 충족, 보라: Hole만 충족, 초록: 둘 다 충족. '
                '검은 점은 기존 검증 프레임 960개입니다. 파란 선은 Mamba, 회색 선은 M0입니다. '
                '여러 시드의 경우 진한 선은 각 프레임 오차의 평균이며, 연한 파란 선은 개별 시드입니다. '
                '조건은 GT와 RGB-D를 이용한 보수적인 인증이며, 960개 이외는 캐시된 3D 검출점을 투영해 판정했습니다.</p>'
                '<p><a href="episodes_21d.pdf">전체 에피소드 PDF</a></p><ol>' + links + '</ol>' + figures + '</html>')
        (output / 'index.html').write_text(page, encoding='utf-8')
        for name in ('index.html', 'episodes_21d.pdf', 'episodes.json', 'summary.json', 'effective_config.json'):
            logger.link_output(output / name)
        logger.record(frames=len(ids), episodes=len(index_rows), counts=summary['category_counts'])
        print(f'Episode condition plots: {output}', flush=True)
    return output
