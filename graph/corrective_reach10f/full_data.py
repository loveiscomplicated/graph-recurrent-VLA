"""Guarded continuation from successful pilot through fixed raw collection."""
import argparse,subprocess,sys,os
from pathlib import Path
from graph.plan_a_bc.common import read,write,sha
from utils.experiment_logger import ExperimentLogger

def run(root,base):
    assert os.getenv('CONTAINER_ID')=='57640710';assert read(root/'pilot_completion_gate.json')['status']=='PASS'
    assert read(root/'pilot_teacher_alignment_gate.json')['checks']['ordered_reach_phases']=='PASS'
    assert read(root/'history_smoke02/verification.json')['weights_discarded']
    plan=root/'full_experiment_plan.json';cfg=read(plan)
    with ExperimentLogger(team='graph',model='Task10F_full_corrective_data_continuation',config=dict(plan=cfg,plan_sha256=sha(plan)),root=root/'experiments',job_type='prepare') as log:
        write(root/'full_data_identity.json',dict(run_id=log.run_id,parent_pilot_uuid=cfg['parent_pilot_uuid']),immutable=True)
        stages=[('additional10mm','train_additional_collection_manifest.json',40),('development','development_collection_manifest.json',32)]
        for name,manifest,count in stages:
            if not (root/name/'verification.json').exists():
                subprocess.run([sys.executable,'-u','-B','-m','graph.corrective_reach10f.recovery_collect','--root',str(root/name),'--base',str(base),'--manifest',str(root/manifest)],check=True)
            status=read(root/name/'verification.json');receipts=read(root/name/'recovery_pilot_manifest.json')
            assert status['status']!='FAIL' and len(receipts)==count and all(r['approved'] for r in receipts),(name,status)
            log.record(**{name:dict(status='PASS',branches=count)})
        if not (root/'nominal/nominal_capture_manifest.json').exists():
            subprocess.run([sys.executable,'-u','-B','-m','graph.corrective_reach10f.nominal_continue','--root',str(root/'nominal'),'--base',str(base),'--plan',str(plan)],check=True)
        nominal=read(root/'nominal/nominal_capture_manifest.json');assert len(nominal)==100 and all(r['status']=='ACCEPTED' for r in nominal)
        write(root/'full_data_complete.json',dict(status='PASS',normal_train=96,normal_development=4,recovery_train=80,recovery_development=32,plan_sha256=sha(plan)),immutable=True)
        log.link_output(root/'full_data_complete.json',kind='metrics')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,default=Path('/workspace/task10f_corrective_reach'));a=p.parse_args();run(a.root,a.base)
