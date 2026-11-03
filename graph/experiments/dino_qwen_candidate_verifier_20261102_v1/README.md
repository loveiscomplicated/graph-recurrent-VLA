# Task 10-I Stage4A-2 DINO → Qwen candidate verifier

Frozen 512 native captures, four previously evaluated train parents, 16 camera-
frames. All 251 saved DINO proposals at box/text threshold0.1 are classified once.
No DINO rerun, new simulator, SAM2, training, download, publishing or production
change. Original PAUSED_SCOPE_REVIEW remains.

One full-BF16 Qwen3.5-9B load; greedy, batch1, thinking off, max_new_tokens16.
Use the unchanged successful functional-role prompt and original image-order
preamble. Native original RGB + neutral marked copy + 10% margin crop, each
processed at fixed pixel budget512². Strict one-word label only.

Primary: per-query semantic gate then original DINO score rank. Compare against
raw Top1 on identical raw candidate pools. Also report saved legacy, rejection-only
legacy gate and predeclared union-role selection. All policies and parsing are
frozen before model inference; GT is opened only after all labels are SHA-frozen.
Candidate role accuracy and geometric IoU are evaluated separately.

Run prepare.py -> infer.py -> evaluate.py -> finalize.py. No inference retry.
The shared ExperimentLogger writes W&B-disabled local runs. Final report belongs
in docs/visual-graph-integration/10i_stage4a2_dino_qwen_candidate_verifier.md.
