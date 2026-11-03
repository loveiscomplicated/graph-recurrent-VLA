# Task 10-I Stage4A-2 — DINO native-resolution paired pilot

User-authorized detector evaluation on the frozen 128/512 native capture pairs
from the preceding Qwen pilot (4 train parents, t0/t80, Main/Wrist = 16 pairs).
No new capture, simulator, Qwen, SAM2, training, model download, or publishing.

Reuse the existing GDino class and candidates/assign consumer from the frozen
Stage4A-2 source. Only native image resolution changes. Current prompts remain
`red and white block` and `cardboard box`, threshold .1, area .25, top-5, null .3,
exclusion IoU .3. Task10-J optimized prompts are not substituted.

One existing DINO Tiny FP32 load and 64 forwards (32 images x 2 phrases), no
warmup/retry calls. The processor stays at its default shortest800/longest1333;
square images both reach 800x800. Native-original pixel postprocessing is retained.

Execution: prepare.py -> infer.py -> evaluate.py -> finalize.py. Freeze all
proposals and selections before GT access. Report actual selected accuracy,
raw top1 and oracle any-proposal coverage separately. Same per-frame consumer is
applied at t80 counterfactually; no temporal scan/commit/tracking is claimed.

Four-parent macro/paired bootstrap is exploratory. Existing Qwen outputs may be
compared only on these exact images after the DINO freeze; Qwen diagnostic-only
coordinates remain separate from accepted outputs. No existing source or paused
runner is edited. Shared ExperimentLogger uses W&B disabled local records.
