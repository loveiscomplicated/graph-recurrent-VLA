# Task09-C independent geometry diagnostic

Opt-in feasibility implementation for RGB-D geometry, object-local surfels,
multi-view association, surface graphs and diagnostic geometry encoders. It does
not change the shared simulator, existing vision defaults, BC datasets, policy
checkpoints or the production Graph/GNN interface. No GRU or action head is
constructed by these runners.

The practical `pipeline.GeometrySystem` accepts synchronized camera observations,
predicted masks, current robot observations and known robot link geometry. Its
input whitelist rejects actor poses and segmentation sidecars. `oracle.py`
separately implements evaluation-only GT-motion variants. Masks and poses from
the simulator are stored in separate capture archives and used only by metrics,
diagnostic targets and explicitly labelled oracle conditions.

## Representations

| Variant | Robot filter | Memory | Camera input |
|---|---|---|---|
| G0 | none | none | original Main Shape-First/F5/Raw EDE |
| G1 | per-link collision projection/interior | none | Main |
| G2 | same | object-local surfels | Main |
| G3 | same | none | Main + Wrist |
| G4 | same | object-local surfels | Main + Wrist |
| O1 | same | estimated registration | both, GT visible masks |
| O2 | same | GT relative rigid motion | both, predicted masks |
| O3 | same | GT relative rigid motion | both, GT visible masks |

Oracle motion transports only previously observed surfaces. It never creates
full-mesh nodes. Known robot collision shapes are handled individually; the
filter does not remove a large radius or a convex hull around the whole robot.

The original G0 packet and 14-feature graph are saved unchanged. The separately
labelled common schema has 21 features: original XYZ/F5/availability/validity14
plus status3, confidence, age, positional uncertainty and pose confidence.
Historical depth-jump and normal-variation fields are unavailable and invalid.
Current measured normals and positional uncertainty remain distinct from
historical pose-dependent estimates.

Persistent edges use a versioned 3D radius/k-neighbour rule with signed normals,
tangency, surface witnesses and camera free-space checks. They are never called
Raw EDE. Robot–object k4 and Peg–Box M16 reuse the existing helpers. Conservative
edge rejection can fragment the graph; `geometry_audit` reports this cost.

The primary GNN input has a maximum of128 nodes per object. It is a common cap,
not a claim that every condition has the same number of supported nodes.
Native variable-N maps and an additional geometry-only matched-count subset are
saved separately. Matched counts use predicted G0–G4 counts, never GT. An object
missing from any representation can make its matched count zero.

## Runners

Run from the repository root with existing dependencies and offline perception
checkpoints. Each runner creates its own `ExperimentLogger` UUID. Output roots
are parents of UUID directories; finalized inputs are required by downstream
runners.

```bash
python -m graph.persistent_geometry.collect --pilot --output /workspace/task09c/pilot
python -m graph.persistent_geometry.preflight --capture CAPTURE_UUID_DIR --output PREFLIGHT_ROOT
python -m graph.persistent_geometry.evaluate --capture CAPTURE_UUID_DIR --workers 4 --output GEOMETRY_ROOT
python -m graph.persistent_geometry.branch_eval --capture CAPTURE_UUID_DIR --geometry GEOMETRY_UUID_DIR --output BRANCH_ROOT
python -m graph.persistent_geometry.geometry_audit --geometry BRANCH_UUID_DIR --output AUDIT_ROOT
python -m graph.persistent_geometry.train --geometry BRANCH_UUID_DIR --updates 2000 --output TRAIN_ROOT
python -m graph.persistent_geometry.probes --training TRAIN_UUID_DIR --output PROBE_ROOT
```

`collect --branch-smoke` captures one development episode with same-pose repeats
and actual actor-pose interventions. All branches share the parent robot,
calibration and causal history through t−1. Invalid collision conditions remain
in the artifacts and are excluded from fitting/primary paired results. These
are live kinematic render interventions without a physics step, not dynamic
contact-force tests.

`collect --development` uses the fixed8 Train/4 Validation episodes and defers
reserved final seeds62091/62092. Only after `probes` writes
`frozen_selection.json`, run `collect --evaluation-only`, evaluate that capture
and its branches, then apply the frozen models:

```bash
python -m graph.persistent_geometry.final_apply --training TRAIN_UUID_DIR --probes PROBE_UUID_DIR --geometry FINAL_BRANCH_UUID_DIR --output FINAL_ROOT
python -m graph.persistent_geometry.cases --capture CAPTURE_UUID_DIR --geometry BRANCH_UUID_DIR --output NEGATIVE_ROOT
python -m graph.persistent_geometry.verify --runs UUID_DIR_1 UUID_DIR_2 --output READBACK_ROOT
python -m unittest graph.persistent_geometry.tests.test_contracts -v
```

The final runner has no optimizer or selection call. It checks freeze timestamps,
reserved seeds, actual reset-group separation and all weight hashes, and loads
development-only normalizers. A causal perturbation family is a parent physical
episode/timestep plus all its branches; those rows stay in the parent's split.

E0/E1/E2 use the same21-feature input, seed42, AdamW budget and validation rule.
E1 removes message passing; E2 adds a deterministic30-value summary of the same
input XYZ. Their diagnostic decoder predicts relative actor-position6 and
current visible-surface-summary30. No grasp/hole goal, phase, reward, action or
success target enters the loss. These are newly trained diagnostic encoders,
not rankings based on applying old BC checkpoints to a changed schema.

## Executed environment and limits

The research execution uses Vast instance57640710, RTX3090, existing CUDA
perception/GNN dependencies, CPU PhysX and SAPIEN rendering. CUDA PhysX was not
executed. Local macOS tests are synthetic/array contracts, not a simulator/GPU
execution claim. No dependencies are installed or artifacts published by these
runners.

Partial surfaces leave rigid pose directions ambiguous. Registration explicitly
reports partial observability, failures and increasing uncertainty. A wrong
predicted object mask can still contaminate a map, and persistence can worsen
geometry or create ghosts. Current observations, historical estimates and GT
evaluation overlays must remain visibly distinct. The completed benchmark
report and its artifacts, rather than node count or probe error alone, determine
which representations have supporting evidence.
