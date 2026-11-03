# Terra and Luna native512 grounding

Authorized extension while the isolated Sol run continues. Models are
`gpt-6.1-terra` and `gpt-6-luna`, each up to 60 requests on the frozen known
Train cohort and exact same appearance prompt. Both use Responses API,
detail=original, reasoning=high, max_output_tokens=5120, no tools, no schema
forcing, no conversation history, and no automatic retries.

Strict JSON and the predeclared whole-answer code-fence-only parser are scored
separately from the same responses. SAM2 receives valid fence-mode boxes;
strict scoring gates those masks by strict validity. No coordinate repair.

External experiments reserve an intentionally conservative $16.32923086
ceiling, which includes even the unused original Kimi request allocation.
The additional Terra/Luna request ceiling fits under the aggregate $20 cap.
This external reservation is not an actual charge estimate. Final reporting
reads the completed separate ledgers to compute actual usage estimates and
remaining unknown-charge reservations without double counting.

CLI: `run.py prepare --dry-run`, `run.py prepare`, `run.py infer --dry-run`,
`run.py infer`, `run.py segment`, `evaluate.py --mode strict|fence`,
`verify.py --mode strict|fence`. CUDA SAM2 runs on existing instance58576054;
no API key is transferred. Serialize GPU segmentation behind the Sol pipeline.
All predictions and masks are frozen before this experiment opens GT arrays.
No rendering, training, tracking, production changes, commits, or publishing.
