# Sol and Kimi full-image grounding comparison

Independent extension of the completed API grounding experiment. Uses the exact
same 60 native512 RGB images and frozen appearance prompt. New calls: at most
60 GPT-6.1 Sol + 60 Kimi K3. No retries, no additional first-image repeats.
Prior conservatively accounted API cost is included in the $20 aggregate cap.

`run.py prepare --dry-run`, `run.py prepare`, `run.py infer --dry-run`,
`run.py infer`, `run.py segment`, `evaluate.py --mode strict`,
`evaluate.py --mode fence`, then independent verification and report.

Two predeclared output modes: strict JSON and strict JSON with only a whole-answer
code fence removed. Never extract boxes from reasoning or explanatory prose.
The same saved final response feeds both; no response resampling or correction.
SAM2 generates each accepted object mask once; strict scoring uses it only if
that object's strict response passed. Prior Gemma responses may be measured
under this newly declared fence rule with no API calls; prior artifacts stay intact.

Scoped macOS caffeinate prevents idle sleep during each non-dry-run stage.
Keys remain local; only RGB, predicted boxes, and isolated source reach the
existing Vast.ai 58576054 RTX3090. No rendering, training, tracking, production
changes, publishing, or commits. Reports belong in docs/visual-graph-integration/.
