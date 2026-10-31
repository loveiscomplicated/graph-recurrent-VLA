# Task 10-H phase handoff diagnosis

Evaluation-only runner for the frozen Task10G B step9000. Run against the verified
Task10G source via PYTHONPATH. It reads the old run and writes a distinct output
root, uses the shared ExperimentLogger, and never trains or publishes.

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/workspace/task10g_f1_abc_20261031/source \
/venv/main/bin/python -u runner.py \
 --old /workspace/task10g_f1_abc_20261031/full \
 --out /workspace/task10h_phase_handoff_20261031
```

Keep analysis10h.py beside runner.py. A process lock prevents duplicate execution.
A completed episode is resumed only after its trace and video hashes match.
Failed partial episode directories are preserved. A run resumes with a new
ExperimentLogger UUID; previous execution identities remain in experiments/.

The final lift/align command is applied before switching at the next observation.
Every observation is processed exactly once, including the full expert prefix.
E/H1/H2 prefix input, physical-state and hidden hashes are compared, including
the handoff observation, without comparing their later divergent trajectories.
No simulator snapshot restore optimization or episode parallelism is used.

Pilot parents108180–108181 gate final parents108200–108211. Anchor failure does
not trigger seed replacement, threshold relaxation or checkpoint selection.
All conditions use the existing official termination and total250-action limit.
