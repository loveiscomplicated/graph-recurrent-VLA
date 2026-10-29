"""Reuse existing event Detector/deduplication with Task10D progress records.

This service writes only ERROR/STALL/COMPLETION events. It never restarts jobs,
changes experiment conditions, or claims a verified Codex callback.
"""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import time
import json
from graph.plan_a_bc.common import read,write,sha,now


def snapshot(root):
    state=read(root/'pipeline_state.json') if (root/'pipeline_state.json').exists() else dict(status='RUNNING',phase='data')
    phase=state['phase'];progress=read(root/'graph_cache_progress.json') if (root/'graph_cache_progress.json').exists() else {}
    signature=dict(phase=phase,cached_episodes=progress.get('completed',0),cached_frames=progress.get('graphs',0))
    for c in ['A','B']:
        marker=root/(c+'_run.json')
        if marker.exists():
            folder=Path(read(marker)['path']);p=folder/'progress.json'
            if p.exists():signature[c+'_step']=read(p)['step']
    if phase=='data':signature['captured']=len(list((root/'raw').glob('*/receipt.json')))
    if phase.startswith('final') or phase.startswith('train'):
        signature['evaluated']=len(list((root/'evaluations').glob('**/episode.json')))
    service='task10d_data' if phase=='data' else 'task10d_experiment'
    st=subprocess.run(['supervisorctl','status',service],capture_output=True,text=True,timeout=10).stdout
    # A completed data producer can be waiting for the experiment consumer.
    running='RUNNING' in st or phase=='data' and (root/'data_pipeline_complete.json').exists()
    errors=[p for p in root.glob('*failure.json') if p.name!='data_pipeline_failure.json' or not (root/'data_pipeline_complete.json').exists()]
    if errors and state.get('status')!='COMPLETE':state=dict(state,status='FAILED',exception=errors[-1].name)
    return dict(pipeline=state,phase=phase,signature=signature,supervisor_running=running,
        supervisor=st.strip(),substage='cache' if phase=='data' else 'training' if phase.startswith('train') else 'rollout',
        recent_log='',active_log=str(root/(phase+'.log')),
        completion_checks={'missing':[] if (root/'verification.json').exists() else ['verification.json']} if state.get('status')=='COMPLETE' else {})


def run(root):
    ops=Path('/workspace/plan_a_ops_3788798a719f4ec4a66c224ca42f5d63/event_ops.py')
    spec=importlib.util.spec_from_file_location('task10d_inherited_event_ops',ops);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    folder=root/'monitor';folder.mkdir(exist_ok=True)
    cfg=dict(comparison_uuid=read(root/'run_identity.json')['run_id'],remote_root=str(root),
        timeouts_seconds=dict(cache=1200,training=1800,rollout=1800,preparation=900,finalization=3600))
    write(folder/'monitor_config.json',dict(**cfg,inherited_detector_sha256=sha(ops),
        event_classes=['ERROR','STALL','COMPLETION'],polling_agent_required=False,
        automatic_codex_callback='NOT_TESTED; not connected',discord='local relay only if verified'))
    detector=module.Detector(folder,cfg)
    while True:
        for event in detector.tick(snapshot(root)):
            with (folder/'event_journal.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
            print(json.dumps(event),flush=True)
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
