"""Render Task08-B report once the protected A/C evaluations are complete."""
import argparse
import json
from pathlib import Path
import time
import traceback

from graph.integration_baseline.direct_runtime import atomic, read


def main(plan_path):
    root = Path(plan_path).resolve().parent
    while True:
        path = root/'task08b_status.json'
        if path.exists():
            stage = read(path)['stage']
            if stage == 'COMPLETE':
                break
            if stage in ('TRAINING_FAILED', 'EVALUATION_FAILED', 'PROTECTION_FAILED'):
                atomic(root/'task08b_report_status.json',dict(status='NOT_RUN',reason=stage))
                return 1
        time.sleep(60)
    try:
        from graph.integration_baseline.bc08b_report import render
        result = render(plan_path)
        atomic(root/'task08b_report_status.json',dict(status='COMPLETE',**result))
        print(json.dumps(result),flush=True)
        return 0
    except BaseException as exc:
        atomic(root/'task08b_report_status.json',dict(status='FAILED',error_type=type(exc).__name__,
            traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(main(args.plan))
