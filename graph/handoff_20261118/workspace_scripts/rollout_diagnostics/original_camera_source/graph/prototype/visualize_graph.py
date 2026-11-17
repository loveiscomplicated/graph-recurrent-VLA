import argparse
from pathlib import Path

import torch

from .common import config, DEFAULT_CONFIG, write_json
from .geometry import GeometryState
from .graph_builder import GraphBuilder
from .maniskill_gt import GTAdapter
from .visualization import plot_geometry


def main():
    p = argparse.ArgumentParser(description="Save typed surface graph diagnostics")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--geometry", type=Path, help="Saved geometry dictionary instead of simulator")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("outputs/graph/graph.png"))
    args = p.parse_args()
    cfg = config(args.config)
    if args.geometry:
        g = GeometryState(**torch.load(args.geometry, weights_only=True))
    else:
        from envs.maniskill import make_peg_insertion_env
        env = make_peg_insertion_env(seed=args.seed, control_mode=cfg["control_mode"])
        try:
            g = GTAdapter(cfg["peg_points"], cfg["hole_points"], cfg["sampling_seed"])(env)
        finally:
            env.close()
    graph = GraphBuilder(cfg["surface_k"], cfg["interaction_k"])(g)
    plot_geometry(g, args.output, graph, title="Edges: black=kinematic; gray=surface; green=finger-peg; teal=peg-hole")
    write_json(args.output.with_suffix(".json"), graph.stats)
    print(graph.stats)


if __name__ == "__main__":
    main()
