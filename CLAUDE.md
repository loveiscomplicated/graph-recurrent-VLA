# Project guidance

@AGENTS.md

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
