# Agent guidance

Read `README.md`, `envs/maniskill/README.md`, and `docs/PROJECT_CONTEXT.md`
before changing code. This is the independent feasibility prototype stage.

Keep shared simulator code limited to the working ManiSkill simulator, its raw GT
accessor, examples, and smoke tests. Existing experiment tracking and artifact
publishing utilities are shared operational infrastructure. Do not prescribe
Visual, Graph, or Recurrent models, features, graph construction, sequence design,
datasets, training loops, configs, or final interfaces. The team directories are
team-owned and may use any internal structure. Make architecture decisions after
prototype evidence.

Verify simulator API claims against the installed package and runnable examples.
State clearly when a backend or device was not executed on the current host.

## Naming and documentation

- Name directories, modules, scripts and documents after their function, using
  descriptive lowercase `snake_case` names. Examples: `corner_detection`,
  `geometry.py`, `corner_detection.md`.
- Do not use legacy experiment/task identifiers (such as `10i`, `task10g`,
  `stage3`, or `plan_a`) as names or prefixes for new code, documentation or
  output directories. Do not recreate the old `visual-graph-integration` report
  hierarchy on this branch.
- Keep feature guides directly under `docs/`, such as `corner_detection.md`.
  Component-local READMEs may link to these guides.
  Update the relevant existing feature document instead of creating a report
  for every task, chat, stage or date. Use Git history for change chronology.
- Record dates, source experiment IDs and commit hashes inside a document when
  needed for evidence or reproducibility; do not put them in new document names.
  Preserve the actual names of referenced historical artifacts and external APIs.
- When moving documentation, update all links and agent instructions, including
  `CLAUDE.md`, and preserve relative links to raw verification artifacts.

For corner-detection changes, also read `docs/corner_detection.md`.

## Development and experiment infrastructure

- Follow [engineering rules](docs/engineering_rules.md) for the work sequence,
  experiment identity, verification, and completion report.
- For training/evaluation, reuse `utils.experiment_logger.ExperimentLogger`;
  follow [experiment tracking](docs/experiment_tracking.md). Do not add a
  team-specific W&B logger or change the existing run ID/project/group rules.
- For artifacts, reuse `utils.experiment_artifacts` through
  `tools.publish_experiment`; follow [HF publishing](docs/huggingface_artifacts.md).
  Publish only on explicit request or in an explicitly defined publishing workflow;
  do not automatically upload all checkpoints/videos from training.
- Keep credentials out of code, logs, configs, manifests, and committed files.
  Preserve local/offline logging and separate publishing failures from training.
- Check the current branch's implementation before using an API. If an interface
  is missing, report the gap and consider a shared extension before adding a
  parallel implementation. Update the corresponding docs when interfaces change.
