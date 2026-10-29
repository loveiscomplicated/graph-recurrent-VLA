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

## Research report location

- Collect final experiment, analysis, and diagnostic Markdown reports in
  `docs/visual-graph-integration/`, using the task identifier in the filename.
  Save the full report there, not only a summary or a link to `outputs/`.
- Raw artifacts, CSVs, figures, static HTML, and execution logs may remain in
  their experiment output directory. If a report is generated in `outputs/`,
  also copy its full content into `docs/visual-graph-integration/` and adjust
  relative artifact links so they resolve from the documentation directory.
- Link to the report in `docs/visual-graph-integration/` in completion messages.
