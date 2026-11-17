"""PNG geometry diagnostics in world coordinates, including optional typed edges."""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/graph/mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np


def plot_geometry(g, output, graph=None, title="GT collision surface"):
    fig = plt.figure(figsize=(12, 6))
    axes = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]
    pts = np.concatenate((g.ee_pose[0, :3].cpu().numpy()[None],
        g.left_fingertip[0].cpu().numpy()[None], g.right_fingertip[0].cpu().numpy()[None],
        g.peg_points[0].cpu().numpy(), g.hole_points[0].cpu().numpy()))
    colors = ["crimson", "limegreen", "royalblue", "darkorange", "purple"]
    names = ["TCP", "left pad centre", "right pad centre", "peg", "box/hole"]
    bounds = [(0, 1), (1, 2), (2, 3), (3, 3+g.peg_points.shape[1]), (3+g.peg_points.shape[1], len(pts))]
    for ax in axes:
        for (a, b), color, name in zip(bounds, colors, names):
            ax.scatter(*pts[a:b].T, c=color, label=name, s=22 if a < 3 else 10)
        for points, normals in ((g.peg_points, g.peg_normals), (g.hole_points, g.hole_normals)):
            p, n = points[0].cpu().numpy(), normals[0].cpu().numpy()
            ax.quiver(*p[::4].T, *n[::4].T, length=.012, alpha=.4, color="gray")
        if graph is not None:
            ei = graph.edge_index.cpu().numpy()
            types = graph.edge_type.cpu().numpy()
            edge_colors = ["black", "silver", "green", "teal"]
            for i in range(ei.shape[1]):
                a, b = ei[:, i]
                if a >= b:
                    continue
                ax.plot(*pts[[a, b]].T, c=edge_colors[types[i]], alpha=.2, lw=.5)
        ax.set(xlabel="world x (m)", ylabel="world y (m)", zlabel="world z (m)")
        ax.set_box_aspect((1, 1, 1))
    axes[0].view_init(25, -60)
    axes[1].view_init(10, 15)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_curves(history, path):
    fig, ax = plt.subplots()
    ax.plot([h["epoch"] for h in history], [h["train_mse"] for h in history], label="train")
    ax.plot([h["epoch"] for h in history], [h["val_mse"] for h in history], label="validation")
    ax.set(xlabel="epoch", ylabel="normalized action MSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
