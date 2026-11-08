# Claude Code entrypoint

@AGENTS.md

Use the imported repository guidance as the common development rules. Follow its
links to engineering, experiment tracking, and Hugging Face publishing details.
If file imports are unavailable in your client, read `AGENTS.md` directly.

Graph corner workflows have separate guides for
[fixed-ID prediction](docs/corner_prediction.md),
[lifecycle tracking](docs/corner_tracking.md), and
[detector-only association and GNN](docs/detected_corner_gnn.md).
Follow the detector-only guide's [exact evaluation preservation criteria](docs/detected_corner_gnn.md#evaluation-preservation)
before changing dependencies of its six protected leaderboard results.

Follow the **Code readability** rules in `AGENTS.md`: use descriptive names,
give functions clear responsibilities, and separate logical steps with readable
control flow and spacing. Avoid compressed one-liners and semicolon-separated
statements. Explain non-obvious intent and input formats with concise comments
or examples, and use actionable validation errors. Review readability before
finishing; keep abstractions proportional to the task.

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
