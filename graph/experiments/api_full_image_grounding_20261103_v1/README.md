# Task10I API full-image grounding (20261103_v1)

Independent, $20-bounded comparison of gpt-6-astra, glm-5.3-flash, and
Ollama Cloud gemma4:31b. Frozen existing 15-parent/60-native512-image cohort.
One image, one fresh request per provider. No DINO/GT/masks/history in API input.

From the graph-recurrent-VLA repository root:

```sh
.venv/bin/python graph/experiments/api_full_image_grounding_20261103_v1/test_contracts.py
.venv/bin/python graph/experiments/api_full_image_grounding_20261103_v1/run.py prepare --dry-run
.venv/bin/python graph/experiments/api_full_image_grounding_20261103_v1/run.py prepare
.venv/bin/python graph/experiments/api_full_image_grounding_20261103_v1/run.py infer --dry-run
.venv/bin/python graph/experiments/api_full_image_grounding_20261103_v1/run.py infer
.venv/bin/python graph/experiments/api_full_image_grounding_20261103_v1/run.py segment
.venv/bin/python graph/experiments/api_full_image_grounding_20261103_v1/run.py evaluate
.venv/bin/python graph/experiments/api_full_image_grounding_20261103_v1/verify.py
.venv/bin/python graph/experiments/api_full_image_grounding_20261103_v1/run.py report
```

Only infer reads the root .env (process environment takes precedence).
Credentials are never logged/transferred. requests performs one POST per durable
request ID without retries or redirects. A RESERVED intent after interruption
is not resent. No model fallback or prompt adjustment. Missing usage retains
its worst-case reservation; recorded estimates are not invoices.

SAM2 uses existing instance58576054/environment/checkpoint, no new paid instance.
All API outputs freeze before segmentation; masks freeze before GT scoring.
Semantic GT fields are isolated to evaluation_manifest/evaluator. Opaque GT
hash checks in prepare do not load actor arrays. Existing artifacts/source are
protected. ExperimentLogger is local/W&B-disabled. No production changes,
tracking, simulation, training, commit, or publishing.

## Follow-up timing and reviewed results

The user authorized one additional repeat of the first image for each provider.
`repeat_first.py` has its own durable ledger and never promotes repeats into
primary accuracy. `review.py` adds the timing and diagnostic-coordinate analysis
after the primary report; `audit_remote.py` checks installed SAM2 preservation.
Offline state-machine and synthetic evaluation tests are in
`test_inference_state.py` and `test_evaluation_pipeline.py`.

This measured run encountered host sleep. All six transport errors overlapped
macOS sleep transitions. Provider availability and whole-run wall time therefore
have a HOST_SLEEP_CONTAMINATED limitation. The first/repeated-request pairs did
not overlap sleep. Do not rerun the experiment automatically to replace failures.

After this execution, `run.py` was hardened to hold a macOS caffeinate idle-sleep
assertion for the command lifetime (never for dry-run). Lid-close/manual sleep
is not guaranteed to be prevented. This change has not been used to repeat the
paid evaluation. A nonfinite-coordinate serialization edge case was also fixed;
all saved response parses are checked unchanged. Exact original inference source
is preserved in the output's `frozen_inference_source/`; source amendments are
recorded with before/after hashes.
