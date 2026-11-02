"""Stage-ordered recovery and audits. Never labels partial work completed."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .run_pipeline import read, sha, write


def main(root):
    root.mkdir(parents=True, exist_ok=False)
    with (root / 'coordinator.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = Path('/workspace/task10i_stage4a_20261031_v5/run')
        repaired = Path('/workspace/task10i_stage4a_20261031_v6/run')
        proof = Path('/workspace/task10i_stage4a_forward_equivalence_20261101_v1/run')
        assert read(proof / 'pipeline_state.json')['status'] == 'FORWARD_EQUIVALENCE_PASS'
        assert shutil.disk_usage(root).free > 20 * 1024**3
        source = Path(__file__).resolve().parents[2]
        frozen = read(previous / 'source_freeze.json')['files']
        changes = [p for p, h in frozen.items() if not p.startswith('graph/task10i_stage4a/') and sha(source / p) != h]
        assert not changes, changes
        write(root / 'source_freeze.json', dict(files={str(p.relative_to(source)): sha(p)
              for p in source.rglob('*.py') if '__pycache__' not in p.parts},
              unchanged_baseline_dependencies=True))
        recovery, corrected, extended, filters = [root / n for n in ('baseline', 'corrected', 'extended', 'filters')]
        stages = [
            ('CONTRACT_TESTS', ['-m', 'unittest', 'graph.task10i_stage4a.test_contracts',
                'graph.task10i_stage4a.test_storage_contract', 'graph.task10i_stage4a.test_aggregation',
                'graph.task10i_stage4a.test_temporal_detector', 'graph.task10i_stage4a.test_reanchor', '-v']),
            ('BASELINE_RECOVERY', ['-m', 'graph.task10i_stage4a.resume_baseline', str(recovery), str(previous), str(proof)]),
            ('CORRECTED_REFERENCE_EVALUATION', ['-m', 'graph.task10i_stage4a.reevaluate_references', str(corrected), str(recovery), str(repaired)]),
            ('DETECTOR_TEMPORAL_AUDIT', ['-m', 'graph.task10i_stage4a.extended_audit_runner', str(extended), str(recovery), str(corrected)]),
            ('FILTER_GATE_GRAPH_AUDIT', ['-m', 'graph.task10i_stage4a.filter_graph_runner', str(filters), str(recovery), str(corrected)]),
        ]
        try:
            for stage, args in stages:
                write(root / 'coordinator_state.json', dict(status='RUNNING', stage=stage, pid=os.getpid()))
                with (root / (stage.lower() + '.log')).open('x') as output:
                    subprocess.run([sys.executable, '-B', '-u', *args], stdout=output,
                                   stderr=subprocess.STDOUT, check=True)
                print(stage, 'FINISHED', flush=True)
            write(root / 'coordinator_state.json', dict(status='AUDITS_FINISHED',
                  remaining='oracle, development candidate comparison, freeze, internal test, independent validation, handoff',
                  whole_goal_complete=False))
        except BaseException as exc:
            write(root / 'coordinator_state.json', dict(status='FAILED', stage=stage,
                  error=type(exc).__name__, message=str(exc), whole_goal_complete=False))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
