# Agent guidance

Read `README.md`, `envs/maniskill/README.md`, and `docs/PROJECT_CONTEXT.md`
before changing code. On `develop`, also read `data_collection/README.md` before
changing collection. The approved pivot retains the main simulator foundation,
GPU observation output and initial peg visibility collection.

Keep selection and storage in `data_collection/`, outside the raw simulator.
Preserve the documented ends-plus-grasp versus full-surface distinction and raw
RGB/depth dtype, shape and device behavior. Avoid legacy experiment imports and
task-number names. Do not delete historical artifacts during extraction.

Keep shared code limited to the working ManiSkill simulator, its raw GT accessor,
examples, and smoke tests. Do not prescribe Visual, Graph, or Recurrent models,
features, graph construction, sequence design, datasets, training loops, configs,
or final interfaces. The team directories are team-owned and may use any
internal structure. Make architecture decisions after prototype evidence.

Verify simulator API claims against the installed package and runnable examples.
State clearly when a backend or device was not executed on the current host.
