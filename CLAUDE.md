# Project guidance

Start with `README.md`, `envs/maniskill/README.md`, and
`docs/PROJECT_CONTEXT.md`. The current goal is three independent feasibility
prototypes on one shared ManiSkill `PegInsertionSide-v1` environment.

Shared code owns simulator setup, raw ground truth access, examples, and smoke
checks. Visual, Graph, and Recurrent teams own their own representations,
models, data, training, evaluation, configs, and directory layouts. Do not add
cross-team input contracts or choose the final architecture before prototype
results. Check installed ManiSkill behavior directly before documenting APIs.
