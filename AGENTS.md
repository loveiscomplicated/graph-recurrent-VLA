# Agent guidance

Read `README.md`, `envs/maniskill/README.md`, and `docs/project_context.md`
before changing code. On `develop`, also read `docs/data_collection.md` before
changing collection. The approved pivot retains the main simulator foundation,
GPU observation output and initial peg visibility collection.

Keep selection and storage in `data_collection/`, outside the raw simulator.
Preserve the documented ends-plus-grasp versus full-surface distinction and raw
RGB/depth dtype, shape and device behavior. Avoid legacy experiment imports and
task-number names. Do not delete historical artifacts during extraction.

## Naming and documentation

- Name directories, modules, scripts and documents after their function, using
  descriptive lowercase `snake_case` names. Examples: `data_collection`,
  `visibility.py`, `gpu_observations.md`.
- Do not use legacy experiment/task identifiers (such as `10i`, `task10g`,
  `stage3`, or `plan_a`) as names or prefixes for new code, documentation or
  output directories. Do not recreate the old `visual-graph-integration` report
  hierarchy on `develop`.
- Keep feature guides directly under `docs/`, such as `data_collection.md` and
  `gpu_observations.md`. Component-local READMEs may link to these guides.
  Update the relevant existing feature document instead of creating a report
  for every task, chat, stage or date. Use Git history for change chronology.
- Record dates, source experiment IDs and commit hashes inside a document when
  needed for evidence or reproducibility; do not put them in new document names.
  Preserve the actual names of referenced historical artifacts and external APIs.
- When moving documentation, update all links and agent instructions, including
  `CLAUDE.md`, and preserve relative links to raw verification artifacts.

Keep shared code limited to the working ManiSkill simulator, its raw GT accessor,
examples, and smoke tests. Do not prescribe Visual, Graph, or Recurrent models,
features, graph construction, sequence design, datasets, training loops, configs,
or final interfaces. The team directories are team-owned and may use any
internal structure. Make architecture decisions after prototype evidence.

Verify simulator API claims against the installed package and runnable examples.
State clearly when a backend or device was not executed on the current host.

## Training and evaluation workflows

- For new training/evaluation workflows, keep experiment
  settings in a team-owned config file, preferably YAML under a `configs/` directory,
  and accept a config path at execution. Include relevant data, model, training,
  evaluation, and output settings as applicable; document a short runnable command
  and prerequisites near the beginning of the team's README. Record the effective
  config and any CLI overrides for reproducibility. Teams choose their own schema
  and internal structure. Do not impose a shared config schema.
- For new training/evaluation workflows, use the repository-root `main.py` as
  the common entrypoint (for example, `python main.py --config path/to/config.yaml`).
  It may dispatch to team-owned implementations. Document the runnable command,
  config path, and prerequisites in the relevant team README. These rules apply
  going forward; existing implementations do not need to be changed to comply.
