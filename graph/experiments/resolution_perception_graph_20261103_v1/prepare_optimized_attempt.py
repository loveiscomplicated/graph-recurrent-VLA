"""Create one isolated optimized Task10-I attempt from frozen existing inputs.

The previous attempt is never resumed or overwritten. No API/model inference is
performed here. All Graphs and controls start empty in the new attempt.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

PARENT=Path('/workspace/task10i_resolution_perception_graph_20261103_v1')
OLD=PARENT/'repaired_graph_v2'
NEW=PARENT/'optimized_graph_v3'
OPT=PARENT/'graph_optimization_v1'


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main():
    old=read(OLD/'protocol.json')
    assert not NEW.exists(),'never overwrite an existing attempt'
    for name,h in old['source_sha256'].items():assert sha(OLD/'code'/name)==h,name
    assert sha(OLD/'api_predictions.json')==read(OLD/'api_freeze.json')['predictions_sha256']
    assert read(OLD/'api_freeze.json')['requested']==840
    validation=read(OPT/'pre_adoption_validation.json')
    assert validation['status']=='PASS' and len(validation['cases'])>=8
    assert validation['GPU_source_sha256']==sha(OPT/'code/gpu_graph.py')
    for size in [128,512,768,1024]:
        before=read(OPT/f'original_10i_s1_0686_t000_{size}.json')
        after=read(OPT/f'cuda_fused_v5_10i_s1_0686_t000_{size}.json')
        assert before['status']==after['status']=='COMPLETE'
        assert len(before['rows'])==len(after['rows'])==4
        assert all(x['status']=='COMPLETE' for x in before['rows']+after['rows'])
        assert all(x['hashes']==before['rows'][0]['hashes'] for x in after['rows'])
    assert read(OLD/'actual_masks_freeze.json')['files']
    for name in ['sensors','GT','jpeg','reference','masks/actual','code']:
        assert (OLD/name).exists(),name
    for name in ['gpu_graph.py','radius_components.py']:
        assert (OPT/'code'/name).exists(),name

    NEW.mkdir()
    for name in ['sensors','GT','jpeg','reference']:
        shutil.copytree(OLD/name,NEW/name,copy_function=os.link)
    (NEW/'masks').mkdir();shutil.copytree(OLD/'masks/actual',NEW/'masks/actual',copy_function=os.link)
    for name in ['capture_manifest.json','grounding_prompt.txt','api_predictions.json','api_freeze.json',
                 'sam_static_actual.json','sam_tracking_actual.json','actual_masks_freeze.json',
                 'runtime_before_main.json','geometry_identity_verification.json','seed98202_snapshot_verification.json',
                 'smoke_capture_manifest.json','smoke_verification.json','reused_inputs.json']:
        if (OLD/name).exists():os.link(OLD/name,NEW/name)
    shutil.copytree(OLD/'code',NEW/'code')
    for name in ['gpu_graph.py','radius_components.py']:shutil.copy2(OPT/'code'/name,NEW/'code'/name)
    pipeline=NEW/'code/pipeline.py';text=pipeline.read_text()
    start=text.index('def graphs(arm):');end=text.index('\ndef evaluate():',start)
    text=text[:start]+"def graphs(arm):\n    run('graphs_'+arm,'gpu_graph.py','graph_worker.py','quality','--arm',arm)\n\n"+text[end:]
    old_call="run('timing_graph','timing.py','graph')"
    assert text.count(old_call)==1
    text=text.replace(old_call,"run('timing_graph','gpu_graph.py','timing.py','graph')")
    pipeline.write_text(text)
    verify=NEW/'code/verify.py';text=verify.read_text()
    old_assert='summary_rebuilt_from_raw=True';assert text.count(old_assert)==1
    verify.write_text(text.replace(old_assert,"summary_verification='pending report generation and independent aggregate readback'"))
    (NEW/'protocol_history').mkdir();shutil.copy2(OLD/'protocol.json',NEW/'protocol_history/cpu_before_optimization.json')
    source={p.name:sha(p) for p in (NEW/'code').glob('*.py')}
    protocol=dict(old,attempt='gpu_optimized_graph_v3',source_sha256=source,
        graph_execution_backend='cuda_fused_v5',optimization_reference_root=str(OLD),
        quality_workers=1,quality_threads_per_worker=4,quality_execution_deadline_seconds=180,
        quality_sensor_cache='Within each build only; original depth and inverse matrices, no temporal cache',
        graph_execution='CUDA robot ray clipping, radius components, Shape-First FPS and final edge gates; CPU contour, PCA5 and graph packing retained',
        GPU_timing_boundary='CPU-loaded RGB-D/masks to CPU Graph tensors, includes internal transfers and sync',
        GT_barrier='Reused immutable actual API/SAM; all actual Graphs and estimates rebuild/freeze before fresh oracle construction and scoring',
        optimization_changed_model_prompt_thresholds_or_graph_policy=False,
        new_API_requests_in_this_attempt=0,API_requests_total_across_attempts=840)
    (NEW/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    status=dict(completed=['sam_static_actual','sam_tracking_actual'],started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        reused_from=str(OLD),status='READY',active=None)
    (NEW/'pipeline_status.json').write_text(json.dumps(status,indent=2)+'\n')
    evidence=dict(previous_root=str(OLD),previous_protocol_sha256=sha(OLD/'protocol.json'),
        initial_protocol_sha256=sha(NEW/'protocol.json'),source_sha256=source,
        API_predictions_sha256=sha(NEW/'api_predictions.json'),actual_masks_freeze_sha256=sha(NEW/'actual_masks_freeze.json'),
        input_hashes='capture_manifest, actual_masks_freeze and reused_inputs.json; inputs hardlinked read-only by convention',
        actual_graphs_reused=False,GT_controls_reused=False,new_API_requests=0,
        before_after='graph_optimization_v1 original and cuda_fused_v5 benchmark JSONs',
        pre_adoption_validation_sha256=sha(OPT/'pre_adoption_validation.json'),
        source_comparison='Each optimized Graph with matching old completed actual input is compared against all five original Graph tensor hashes. An assertion fails the stage immediately.')
    (NEW/'optimization_adoption.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('OPTIMIZED_ATTEMPT_READY',NEW,flush=True)


if __name__=='__main__':main()
