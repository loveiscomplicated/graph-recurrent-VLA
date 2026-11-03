# Initial RGB reference diagnostic

Two conditions on all 149 preserved later-frame DINO proposals (four parents,
eight native512 camera-frames), 298 requests total, one resident BF16 Qwen load.
`current_only` gets the original three current images; `with_initial_RGB` adds
an unmarked initial RGB from the same parent and camera as image four.
Both conditions get exactly the same text, including a frozen optional-reference
description. The semantic role prompt is unchanged. The new control is also
compared with historical labels to expose the effect of the shared description.

Run prepare, infer, evaluate, finalize in that order in the existing remote
Qwen environment. Sources live under the new output's `.execution/` directory.
No prior labels, GT object correspondence, new DINO/SAM2 forwards, simulator,
training, prompt sweep, publishing, or production changes. No runtime retries.
All labels freeze before posthoc GT evaluation. This diagnoses known development
scenes; it is not an independent generalization benchmark.
