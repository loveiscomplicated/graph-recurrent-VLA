# Task 10-I Terra reasoning-effort comparison

One user-requested four-effort comparison on the existing 60 known native512
images. New Standard requests are `none`, `low`, `medium`;
the previously frozen `high` responses, SAM2 masks and GT metrics are reused.
One attempt per image and effort, sequential, no warmup, retries, replacement,
prompt changes, schema forcing, coordinate repair, or additional paid instance.

The existing OpenAI HTTP adapter, parsers, SAM2 segmentation, scoring and PIL
independent checker are reused in an isolated source/output directory. The new
common model mapping represents effort levels; the request includes explicit
`service_tier=default`, and effective tier is checked in every completed response.
The earlier high request omitted this field but returned `default`.

Stages: `run.py prepare --dry-run`, `run.py prepare`, `run.py infer --dry-run`,
`run.py infer`, `run.py segment`, `run.py evaluate --mode strict`, then fence,
`run.py verify` for both modes, `run.py report`, `run.py complete`. Predictions
are frozen before SAM2, all new masks before GT. ExperimentLogger is local with
W&B disabled. GPU work uses only existing Vast 58576054 and keys stay local.
Previously frozen outputs, production, PAUSED_SCOPE_REVIEW and Stop Stub stay
unchanged. No simulation, tracking, training, graph, commits, or publishing.
