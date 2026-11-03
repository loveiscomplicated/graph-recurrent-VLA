# Task10-J Grounding DINO prompt feasibility

A bounded experiment on stored failure/normal RGB frames. Reuses
`visual.runtime.grounding.GDino` and the unchanged candidate/assignment/scan
functions in `visual.runtime.initialization`. No SAM2 or simulator is instantiated.

## Run

Use an environment containing the existing model dependencies and an existing
local DINO Tiny snapshot. Package installation and model downloads are not part
of this runner. From the repository/source root:

```bash
python -m visual.experiments.dino_prompt.prepare --workspace /workspace --out /path/to/new/inputs
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m visual.experiments.dino_prompt.run --inputs /path/to/new/inputs \
  --out /path/to/new/run --model /workspace/full_pipeline_models/dino --device cuda
python -m unittest visual.experiments.dino_prompt.test_run -v
```

Output directories must be new. Preparation reads existing Task10-G and
Task10-I source artifacts, hashes their contents, checks reference receipts,
and writes an input NPZ plus manifest. Missing GT is explicit and excluded
from numerical detection metrics. Missing required RGB stops inference.

## Protocol

The two prompt rounds contain 8+4 phrases per object, registered in `run.py`.
There are 10 search images and 4 confirmation images. Each call processes one
image and one phrase, with an overall maximum of 300 forward calls including
warmup/repeat checks. Predictions are cached by RGB hash and phrase.

First the existing baseline is checked against three historical score
fingerprints and the 0355 incorrect t30 commitment. Any mismatch stops search.
After round 1, its best combination and the rationale for the fixed complementary
round 2 phrases are written. All 144 combinations are evaluated from cached
outputs; this does not execute an ensemble. Successful baseline normal controls
have priority, followed by correct selections, fewer object confusions, higher
mean selected IoU, shorter text, and registration order. Selection is frozen on
disk before confirmation forwards. Confirmation does not retune the selection.

For 0355, the seven stored scan frames are passed through the actual
`detect_initial` function. Only t0/5/10/15/20/25/30 can be queried. Other images
receive a per-frame counterfactual joint-initialization evaluation, not a claim
that the existing online system would reinitialize at that frame.

GT enters metric evaluation only. The cache adapter and existing consumer see
only RGB and text. IoU >= .5 is a hit; fewer than 8 GT pixels is excluded from
main hit denominators. Missing GT differs from invisible/small targets.
Success rates are descriptive counts for this hand-picked historical set;
seven adjacent frames of one parent are correlated.

## Outputs

`calls.json` stores every forward including warmups and repeats. `protocol.json`
stores input/model/source identities. `prompt_details.csv` gives phrase-level
proposal and selected-box metrics with the other object's baseline phrase;
`round*_combinations.csv` covers every joint phrase pair. `baseline_selected.csv`
and `transitions.csv` expose both improvements and regressions. The summary
includes the actual temporal commitments, call budget, missing inputs and latency.
`comparison.png` shows GT, baseline and selected boxes for each input.

`ExperimentLogger` supplies the existing UUID/project/group conventions. Logging
is local with W&B disabled; no artifacts are published. The full measured report
belongs in `docs/visual-graph-integration/10j_dino_prompt_feasibility.md`.
