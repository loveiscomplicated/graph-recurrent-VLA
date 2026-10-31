"""Current-runtime normal Reach via unchanged Task10-D collector, with safety audit."""
import argparse,os
from pathlib import Path
import torch
from graph.plan_a_bc.common import read,write
from graph.geometry_reach10d.runtime import environment
from graph.geometry_reach10d import collect as original
from graph.corrective_reach10f.recovery_audit import Audit
from graph.corrective_reach10f.safety_v2 import PhysicalSafety
from graph.corrective_reach10f.recovery_collect import SAFETY
from graph.integration_baseline.geometry09a import physical_state
from utils.experiment_logger import ExperimentLogger

def run(root,base,plan):
    assert os.getenv('CONTAINER_ID')=='57640710';torch.set_num_threads(1);root.mkdir(parents=True,exist_ok=False)
    for n in ['inputs','config.json','normalizer.json','action_codec.json','renderer_gpu_gate.json']:(root/n).symlink_to(base/n)
    p=read(plan);specs=[dict(episode_id=5100000+i,seed=s,variant='P0',offset_mm=[0,0,0],split='train',parent_id=s) for i,s in enumerate(p['train_parent96'])]
    specs += [dict(episode_id=5200000+i,seed=s,variant='P0',offset_mm=[0,0,0],split='validation',parent_id=s) for i,s in enumerate(p['development_nominal_seeds'])]
    write(root/'collection_manifest.json',specs,immutable=True);env=environment(root);receipts=[];safety_rows=[]
    install=original.install_audit
    try:
        with ExperimentLogger(team='graph',model='Task10F_nominal_current_runtime',config=dict(plan=str(plan),episodes=len(specs)),root=root/'experiments',job_type='prepare') as log:
            for spec in specs:
                hooks={}
                def instrument(env,schema):
                    result=install(env,schema);observer=Audit(env);physical=PhysicalSafety(env,SAFETY);initial=physical_state(env);step=env.step
                    checks=[]
                    def observed(action):
                        observer.substeps=[];obs=step(action);s=physical.evaluate(initial,observer.substeps);checks.append(s)
                        if not s['valid']:raise RuntimeError('nominal physical safety failed')
                        return obs
                    env.step=observed;hooks.update(observer=observer,step=step,checks=checks);return result
                original.install_audit=instrument
                try:r=original.collect_episode(root,spec,env)
                finally:
                    if hooks:
                        env.step=hooks['step'];env.unwrapped.scene.step=hooks['observer'].original_step
                assert r['status']=='ACCEPTED';receipts.append(r);safety_rows.append(dict(seed=spec['seed'],checks=hooks['checks']))
                write(root/'nominal_capture_manifest.json',receipts);write(root/'nominal_physical_safety.json',safety_rows)
                log.log(dict(completed=len(receipts)),step=len(receipts))
            log.link_output(root/'nominal_capture_manifest.json',kind='metrics')
    finally:original.install_audit=install;env.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,default=Path('/workspace/task10f_corrective_reach'));p.add_argument('--plan',type=Path,required=True);a=p.parse_args();run(a.root,a.base,a.plan)
