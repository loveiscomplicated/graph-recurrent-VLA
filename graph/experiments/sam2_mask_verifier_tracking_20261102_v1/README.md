# Task10I — SAM2 mask verifier and identity tracking

Independent experiment for a frozen existing cohort: 15 parents, 60 native512
images, 986 saved DINO proposals. Compare frozen baseline A, common binding
prompt B, and same prompt with SAM2 mask-rendered inputs C. Candidate SAM states
are independent. All static/tracking predictions freeze before GT scoring.

Hash-fixed eight parents supply sixteen native512 sequences at t0–t80, restored
from recorded states without physics steps. Each method tracks only its t0
predicted masks; no redetection, GT prompt, or future conditioning.

Run `test_contracts.py` before deployment; `bash run.sh` on existing instance
58576054 performs the frozen bounded experiment. No automatic retry. Source,
datasets, model files, environment and paused production runner are protected.
Reuse ExperimentLogger with W&B disabled. No publishing or git commit.

Outputs are under the independent `/workspace/task10i_stage4a2_` root declared in
`common.py`; the final full report is copied to docs/visual-graph-integration.
