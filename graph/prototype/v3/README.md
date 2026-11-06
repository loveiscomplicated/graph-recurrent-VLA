# Surface interaction audit v3

This Graph-team experiment preserves v1/v2 and all shared/Visual/Recurrent code.
The original graph is audited before candidate construction; the initial evidence
is retained under `outputs/graph/v3/audit/phase_a_evidence.json` and
`phase_b_evidence.json`. See `docs/graph_prototype_v3.md` for interpretation.

```bash
.venv/bin/python -m graph.prototype.v3.graph_audit --figures
.venv/bin/python -m graph.prototype.v3.train --output outputs/graph/v3/experiments/interaction_new
.venv/bin/python -m graph.prototype.v3.compare --checkpoint outputs/graph/v3/experiments/interaction_new/best.pt --rollouts --output outputs/graph/v3/comparison/new.json
```

A is the unchanged Surface v2. B keeps the same samples and 131 nodes, separates
global proxy relations from region-eligible radius interactions, and adds one
edge-type column. C additionally resamples the hole: outer/inner/entry/exit
weights 1:3:3:1 and a 2cm entry band, retaining broad coverage. C is a hypothesis,
not a replacement default for B. Explicit normals/clearance-based interactions
and relation-specific aggregation are not implemented.

Global relations use existing surface representatives, not new object-pose
nodes. Exact pose goals remain in the unchanged v2 common input and context.
The representative edge is not an exact grasp or insertion waypoint.

Sampling and Flat control are independently runnable. Their short smoke runs
check execution only and do not support a performance claim:

```bash
.venv/bin/python -m graph.prototype.v3.train --variant sampling --output outputs/graph/v3/experiments/sampling_new
.venv/bin/python -m graph.prototype.v3.train --variant flat_sampling --output outputs/graph/v3/experiments/flat_sampling_new
```

The default config is `graph/prototype/v2/config.toml`; a separate config can be
selected with `--config` (do not edit v2 for a v3 run). V3 additionally exposes
`--local-k`, `--cross-k`, `--radius`, `--distance-threshold`,
`--sampling-weights` (outer, inner, entry, exit), and `--entry-band`.
Audit point counts are configurable with `--peg-points` and `--hole-points`.
Full training deliberately validates the existing 64+64 dataset convention.
Alternate counts need a newly extracted dataset and a matching training config;
they are not silently substituted into the original paired comparison.

Figures always use actual tensors. Both directions are exported into the
adjacent `.edges.json`; a pair is drawn once as a solid segment. Dashed goal
overlays and the insertion direction arrow are diagnostic annotations.
`--edge-types` toggles types; an empty list hides all edges; `--zoom` fits the
selected relation's endpoints; two camera views use equal physical axis scales.

```bash
.venv/bin/python -m graph.prototype.v3.visualization --state acquired --variant interaction --edge-types 2 --zoom --output outputs/graph/v3/visualizations/finger_zoom.png
.venv/bin/python -m graph.prototype.v3.visualization --state initial --variant original --edge-types 3 --output outputs/graph/v3/visualizations/original_cross.png
.venv/bin/python -m graph.prototype.v3.visualization --state alignment --variant interaction --edge-types 4 --goal-overlay --output outputs/graph/v3/visualizations/global.png
.venv/bin/python -m graph.prototype.v3.visualization --state initial --edge-types --output outputs/graph/v3/visualizations/geometry.png
```

```bash
.venv/bin/python -m unittest discover -v
.venv/bin/python -m unittest graph.prototype.tests.test_geometry graph.prototype.tests.test_data graph.prototype.tests.test_models graph.prototype.v2.tests.test_v2 graph.prototype.v3.tests.test_v3 -v
```

Only CPU PhysX simulation and CPU Torch are executed. CUDA and MPS inference or
training are not validated. Outputs are ignored by Git; the report contains the
key measurements and reproduction commands.
