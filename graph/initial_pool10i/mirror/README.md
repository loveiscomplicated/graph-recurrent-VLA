# Task10-I Stage1.5 mirror recovery

Byte-preserving delivery only. No simulator, visibility gate, model or selection
is executed. `verify.py` independently opens the full remote tree and all 300
eligible candidates; `runner.py` coordinates bounded SSH/rsync retries from
local tmux. Audit files and rsync partials live outside the candidate tree.

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest graph.initial_pool10i.mirror.test_verify -v
.venv/bin/python -m graph.initial_pool10i.mirror.runner prepare outputs/task10i/stage1_5/mirror_recovery
tmux new-session -d -s task10i-stage1-5-mirror '.venv/bin/python -u -m graph.initial_pool10i.mirror.runner sync outputs/task10i/stage1_5/mirror_recovery >> outputs/task10i/stage1_5/mirror_recovery/runner.log 2>&1'
.venv/bin/python -m graph.initial_pool10i.mirror.runner status outputs/task10i/stage1_5/mirror_recovery
```

After interruption use `resume AUDIT` in local tmux with the same frozen code.
Completed step proofs are reused; source changes block resume. `verify AUDIT`
finishes verification/promotion/readback after transfer; a fresh independent
audit is needed to certify later modifications. `report AUDIT` reconstructs
the handoff from this run's immutable verification proofs. `status` never
changes the source or destination. Canonical source mismatches are hard failures.

Promotion archives the untouched prior partial, journals intent and performs
same-filesystem renames. A crash between renames can resume using that intent.
No `--delete`, recompression, JSON editing or hardlinks are used. Every Candidate
path resolves relative to the pool root; historical absolute provenance fields
remain byte-preserved. The raw channel keys are `rgb` uint8 and `depth_raw_mm`
int16 mm, never the annotated gallery image. Snapshot pickle is not executed.

The completed recovery audit is
`outputs/task10i/stage1_5/mirror_recovery_v2/`, run
`7934732624c148ad86a8f4d518bf7c18`. The first `mirror_recovery/` remains BLOCKED
after SSH transport failures; its logs and original config are preserved.
The second run uses compression, bounded Direct/Proxy route fallback and SSH
connection reuse. To reproduce from scratch, choose a new, nonexistent audit
directory. Do not prepare over either existing audit.

`finalize.py AUDIT` independently validates the local/remote inventory proof and
consumer receipts, creates the immutable audit index, sends the handoff outside
the candidate tree, and reads back its SHA256 on the server. It requires the
coordinator's COMPLETED proof and does not run a simulator or selector.
