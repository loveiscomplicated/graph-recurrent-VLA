"""Preserve the failed engineering attempt and seed a separate corrected Graph attempt."""
from common import *
import argparse
import shutil


def initialize():
    old=ROOT;new=old/'repaired_graph_v2'
    assert not new.exists(),'A corrected attempt already exists; never overwrite it'
    for name,h in read(old/'actual_masks_freeze.json')['files'].items():assert sha(old/name)==h
    new.mkdir()
    def copy(source,dest):
        if Path(source).suffix in ('.npz','.png','.jpg'):os.link(source,dest)
        else:shutil.copy2(source,dest)
    for name in ('sensors','GT','jpeg','reference'):shutil.copytree(old/name,new/name,copy_function=copy)
    (new/'masks').mkdir();shutil.copytree(old/'masks/actual',new/'masks/actual',copy_function=copy)
    for name in ('capture_manifest.json','smoke_capture_manifest.json','grounding_prompt.txt',
                 'api_predictions.json','api_freeze.json','sam_static_actual.json','sam_tracking_actual.json',
                 'actual_masks_freeze.json','runtime_before_main.json','geometry_identity_verification.json',
                 'seed98202_snapshot_verification.json'):
        if (old/name).exists():shutil.copy2(old/name,new/name)
    (new/'protocol_history').mkdir();shutil.copy2(old/'protocol.json',new/'protocol_history/source_attempt_protocol.json')
    (new/'code').mkdir()
    reuse={}
    for base in ('sensors','GT','jpeg','masks/actual'):
        for f in (new/base).rglob('*'):
            if f.is_file():reuse[str(f.relative_to(new))]=dict(source=str(old/f.relative_to(new)),sha256=sha(f))
    write(new/'reused_inputs.json',dict(source_root=str(old),files=reuse,API_requests_reused=840,new_API_requests=0))
    write(new/'engineering_recovery.json',dict(
        reason='Diagnostic timing keys violated the original sensor-only data guard. Move timings to a separate tuple member; add real PlanA regression test and fail fast on schema assertions. The shared builder and guard are unchanged.',
        failed_attempt_root=str(old),original_graph_successes=0,original_graph_failures=2300,
        API_requests_total=840,new_API_requests=0,actual_masks_reused_immutable=True,
        old_GT_controls_reused=False,performance_results_used_for_fix=False))
    write(new/'pipeline_status.json',dict(completed=['sam_static_actual','sam_tracking_actual'],
          started_utc=now(),reused_from=str(old),status='READY'))
    print('CORRECTED_ROOT_INPUT_FILES',len(reuse),flush=True)


def finalize():
    new=ROOT/'repaired_graph_v2';p=read(new/'protocol_history/source_attempt_protocol.json');old=p['source_sha256']
    p.update(attempt='graph_input_fixed_v2',source_sha256={x.name:sha(x) for x in (new/'code').glob('*.py')},
        new_API_requests_in_this_attempt=0,API_requests_total_across_attempts=840,
        GT_barrier='Immutable actual API/SAM outputs reused; corrected actual Graph and estimates freeze before fresh oracle construction/scoring. Failed attempt retained unchanged; no GT controls reused.',
        engineering_recovery=read(new/'engineering_recovery.json'))
    write(new/'protocol.json',p)
    write(new/'protocol_history/correction_004.json',dict(before=old,after=p['source_sha256'],
        reason=p['engineering_recovery']['reason'],model_prompt_graph_policy_changed=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['initialize','finalize']);a=ap.parse_args()
    (initialize if a.stage=='initialize' else finalize)()
