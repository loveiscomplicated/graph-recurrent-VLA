# Claude Code entrypoint

@AGENTS.md

Use the imported repository guidance as the common development rules. Follow its
links to engineering, experiment tracking, and Hugging Face publishing details.
If file imports are unavailable in your client, read `AGENTS.md` directly.

For new training/evaluation workflows, follow the
experiment config and README guidance in `AGENTS.md`: prefer team-owned YAML
configs, document a short command, and record effective settings and overrides.
Existing formats remain supported; do not impose a shared config schema.
Use the repository-root `main.py` as the common entrypoint for new training/evaluation
workflows, as described in `AGENTS.md`. Existing implementations do not need to be
changed to comply.
For new code, keep team models and their internal modules in `visual/`, `graph/`,
and `recurrent/`; put training and evaluation code in `training/` and `evaluation/`.
Follow the independent team implementation and existing-code guidance in `AGENTS.md`.
`training/` loads or constructs models, prepares training data, runs training,
and saves checkpoints and metrics. `evaluation/` loads models and checkpoints,
runs evaluation, and records results. The root `main.py` reads the config and
dispatches to these workflows; team directories contain model definitions.
