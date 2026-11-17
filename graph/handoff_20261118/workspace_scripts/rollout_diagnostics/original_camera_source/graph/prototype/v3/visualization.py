"""Collision union, region points and exact graph edges with reproducible filters."""
import argparse
from itertools import product
from pathlib import Path
import numpy as np
import torch
from ..geometry import GeometryState, transform
from ..surface_sampling import task_box_bounds
from ..visualization import plt
from .geometry_audit import REGIONS, regions
from .edge_diagnostics import edge_rows

COLORS = ['black', 'silver', 'green', 'teal', 'goldenrod']
NAMES = ['kinematic', 'local surface', 'finger–peg', 'peg–hole', 'global goal']


def mesh_faces(bounds, pose):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    corners = torch.tensor(list(product((0, 1), repeat=3)), dtype=bounds.dtype)
    vertices = bounds[0]+corners*(bounds[1]-bounds[0])
    vertices = transform(vertices[None], pose)[0].numpy()
    faces = []
    for axis in range(3):
        for side in (0, 1):
            ids = torch.where(corners[:, axis] == side)[0].tolist()
            faces.append(vertices[[ids[i] for i in (0, 1, 3, 2)]])
    return Poly3DCollection(faces, facecolor='lightsteelblue', edgecolor='slategray', alpha=.12, linewidth=.3)


def draw(g, graph, output, title='', selected=None, zoom=False, goals=False, collision=True):
    points = torch.cat((g.ee_pose[:, None, :3], torch.stack((g.left_fingertip, g.right_fingertip), 1),
                        g.peg_points, g.hole_points), 1)[0].numpy()
    labels = regions(g)[0].numpy()
    rows = edge_rows(graph, selected)
    fig = plt.figure(figsize=(13, 6))
    axes = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')]
    for ax, angle in zip(axes, (-60, 25)):
        if collision:
            bounds = task_box_bounds(g.box_half_size, g.hole_radius, g.hole_offset)[0]
            for bound in bounds:
                ax.add_collection3d(mesh_faces(bound, g.box_pose))
            ax.add_collection3d(mesh_faces(torch.stack((-g.peg_half_size[0], g.peg_half_size[0])), g.peg_pose))
        for region in np.unique(labels):
            if region == 0:
                for i, name, marker in ((0, 'TCP', 'x'), (1, 'left pad centre', '^'), (2, 'right pad centre', 'v')):
                    ax.scatter(*points[i], s=30, marker=marker, label=name)
            else:
                ax.scatter(*points[labels == region].T, s=13, label=REGIONS[region])
        for a, b, kind in rows:
            # Two arrows are represented by one segment. Counts remain directed.
            if a < b:
                ax.plot(*points[[a, b]].T, c=COLORS[kind], alpha=.23, lw=.65)
        if goals:
            local = torch.zeros(1, 1, 3); local[0, 0, 0] = g.peg_half_size[0, 0]
            head = transform(local, g.peg_pose)[0, 0].numpy()
            from ..v2.goals import goal_geometry
            goal = goal_geometry(g)
            # Dashed: diagnostic relation, explicitly NOT added to edge_index.
            ax.plot(*np.stack((head, g.hole_pose[0, :3].numpy())).T, 'k--', lw=2, label='head→hole (overlay only)')
            ax.plot(*np.stack((g.ee_pose[0,:3].numpy(), goal['grasp_position'][0].numpy())).T, 'r--', label='TCP→grasp (overlay only)')
            ax.quiver(*g.hole_pose[0, :3].numpy(), *goal['insertion_axis'][0].numpy(), length=.07, color='black')
        view = points
        if zoom and rows:
            ids = sorted({v for a,b,_ in rows for v in (a,b)})
            view = points[ids]
        centre = (view.min(0)+view.max(0))/2
        span = max(float((view.max(0)-view.min(0)).max())*1.12, .06)
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
            setter(c-span/2, c+span/2)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(25, angle)
        ax.set(xlabel='world x (m)', ylabel='world y (m)', zlabel='world z (m)')
    counts = ', '.join(f'{NAMES[t]}={sum(k==t for _,_,k in rows)}' for t in sorted({k for _,_,k in rows}))
    axes[0].legend(fontsize=7)
    fig.suptitle(f'{title}\nActual directed edges: {counts or "none"}; solid segments collapse reverses')
    fig.tight_layout(); Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140); plt.close(fig)
    # Machine-checkable exact directed edges underlying every figure.
    import json
    Path(output).with_suffix('.edges.json').write_text(json.dumps(rows))


def histogram(graphs, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (stage, graph) in zip(axes.flat, graphs.items()):
        for t in graph.edge_type.unique().tolist():
            values = graph.edge_attr[graph.edge_type == t, 3].numpy()
            ax.hist(values, bins=np.linspace(0, .65, 45), histtype='step', label=f'{NAMES[t]} ({len(values)})', color=COLORS[t])
        ax.set(title=stage, xlabel='edge length (m)', ylabel='directed edge count'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output, dpi=140); plt.close(fig)


def aperture_view(original, sampled, output):
    """Orthographic box-local YZ view; white aperture is genuinely empty."""
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, g, name in zip(axes, (original, sampled), ('original', 'sampling')):
        h = g.hole_local[0]; labels = regions(g)[0, 3+g.peg_local.shape[1]:]
        half, offset, radius = g.box_half_size[0], g.hole_offset[0], g.hole_radius[0, 0]
        ax.add_patch(Rectangle((-float(half[1]), -float(half[2])), float(2*half[1]), float(2*half[2]),
                               facecolor='lightsteelblue', edgecolor='slategray', alpha=.6))
        ax.add_patch(Rectangle((float(offset[1]-radius), float(offset[2]-radius)), float(2*radius), float(2*radius),
                               facecolor='white', edgecolor='black'))
        for region, label, marker in ((4, 'Entry face', 'o'), (5, 'Inner wall projection', 'x')):
            points = h[labels == region]
            ax.scatter(points[:, 1], points[:, 2], s=25, marker=marker, label=label)
        ax.set(xlabel='box-local y (m)', ylabel='box-local z (m)', title=f'{name} / negative-X opening')
        ax.set_aspect('equal'); ax.legend(fontsize=8)
    fig.suptitle('Actual collision opening and surface samples / episode 29')
    fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', choices=['initial', 'acquired', 'transport', 'alignment'], default='initial')
    p.add_argument('--variant', choices=['original', 'interaction', 'sampling'], default='original')
    p.add_argument('--edge-types', type=int, nargs='*', help='0 kinematic, 1 local, 2 finger, 3 peg–hole, 4 global; empty means no edges')
    p.add_argument('--zoom', action='store_true'); p.add_argument('--goal-overlay', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from ..graph_builder import GraphBuilder
    from .interaction_graph import InteractionGraph
    from .geometry_audit import resample
    g = GeometryState(**torch.load(f'outputs/graph/v3/audit/{args.state}.pt', weights_only=True)['geometry'])
    if args.variant == 'sampling': g = resample(g, entry_band=.02)
    graph = GraphBuilder()(g) if args.variant == 'original' else InteractionGraph()(g)
    draw(g, graph, args.output, f'episode 29 / {args.state} / {args.variant}', args.edge_types, args.zoom, args.goal_overlay)


if __name__ == '__main__': main()
