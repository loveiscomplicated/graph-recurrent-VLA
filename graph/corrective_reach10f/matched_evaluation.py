"""Evaluate only the missing R0 update3000, preserving the frozen evaluator."""
import argparse
import importlib.metadata
import os
import time
from pathlib import Path

import numpy as np
import torch

from graph.corrective_reach10f import matched_diagnosis as diagnosis
from graph.corrective_reach10f.recovery_eval import episode
from graph.plan_a_bc.common import read, write, sha, checkpoint_load, tensor_identity, npz, append_csv, digest
from graph.plan_a_bc.model import normalizer, CachedGraphPolicy
from graph.plan_a_bc.cache import make_vision
from graph.plan_a_bc.train import assert_nested_equal
from graph.integration_baseline.bc08_codec import ActionCodec
from utils.experiment_logger import ExperimentLogger


def inventory(base, output):
    manifest=read(base/'checkpoint_manifest.json'); identities={}; reference=None
    for condition in ['R0','R1']:
        for step in [0,3000]:
            receipt=next(r for r in manifest[condition] if r['step']==step); path=Path(receipt['path'])
            if not path.exists(): raise FileNotFoundError('BLOCKED_CHECKPOINT_NOT_AVAILABLE: '+str(path))
            assert sha(path)==receipt['sha256']; cp=checkpoint_load(path)
            assert cp['progress']['step']==step and cp['variant']=='E0'
            assert tensor_identity(cp['model'])==receipt['model_sha256']
            assert cp['config_sha256']==sha(base/'full_cache/config.json')
            assert cp['cache_manifest_sha256']==sha(base/'full_cache/cache_manifest.json')
            if reference is None: reference=cp
            for name in ['codec','normalizer','config']:assert_nested_equal(reference[name],cp[name])
            identities[f'{condition}_step{step}']=dict(**receipt,exists=True,codec_sha256=digest(cp['codec']),
                normalizer_sha256=digest(cp['normalizer']),config_payload_sha256=digest(cp['config']),strict_readback='PASS')
    assert identities['R0_step0']['model_sha256']==identities['R1_step0']['model_sha256']
    frozen=read(base/'final_evaluation/evaluation_protocol.json')
    for path,expected in frozen['source_sha256'].items():assert sha(path)==expected,path
    dependencies=['graph/corrective_reach10f/recovery_eval.py','graph/corrective_reach10f/recovery_audit.py',
        'graph/corrective_reach10f/recovery_collect.py','graph/corrective_reach10f/safety_v2.py',
        'graph/corrective_reach10f/pilot.py','graph/geometry_reach10d/runtime.py',
        'graph/plan_a_bc/geometry.py','graph/plan_a_bc/model.py','graph/plan_a_bc/rollout.py',
        'graph/plan_a_bc/cache.py','graph/integration_baseline/online_vision_adapter.py',
        'graph/integration_baseline/geometry09a.py','graph/integration_baseline/bc08_codec.py',
        'graph/integration_baseline/robot.py','recurrent/expert.py']
    source_identity={p:sha(p) for p in dependencies}
    versions={k:importlib.metadata.version(k) for k in ['torch','mani_skill','sapien','numpy','scipy','transformers']}
    assert versions['mani_skill']=='3.0.1' and versions['sapien']=='3.0.3'
    assert torch.__version__=='2.8.0+cu128' and torch.version.cuda=='12.8'
    receipt=dict(status='PASS',checkpoints=identities,same_codec_normalizer_config=True,
        common_initial_model_identical=True,source_sha256=source_identity,versions=versions,
        container_id=os.environ['CONTAINER_ID'],gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,
        cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        environment='CPU PhysX; GPU renderer; control20Hz physics100Hz; unchanged frozen runtime',
        frozen_protocol_sha256=sha(base/'final_evaluation/evaluation_protocol.json'),
        r1_reuse_checkpoint_sha256=read(base/'final_evaluation/evaluation_complete.json')['checkpoint_sha256']['R1'])
    assert receipt['r1_reuse_checkpoint_sha256']==identities['R1_step3000']['sha256']
    write(output/'checkpoint_identity.json',receipt,immutable=True)
    write(output/'evaluation_protocol.json',frozen,immutable=True)
    return receipt


def compare_immutable_prefix(folder, old, kind):
    a=read(folder/'trace.json');b=read(old/'trace.json')
    count=0
    if kind=='handoff':
        for x,y in zip(a,b):
            if x['segment']=='student_recovery' or y['segment']=='student_recovery':break
            for key in ['timestep','segment','executed_action7','tcp_pose','target_position_error_m',
                        'target_orientation_error_rad','approach_position_error_m','physical_safety']:
                assert x[key]==y[key],(folder,key,x['timestep'])
            count+=1
        assert count>=25
    graphs=[]
    for name in (['graph_t0000.npz','graph_t0025.npz'] if kind=='handoff' else ['graph_t0000.npz']):
        with np.load(folder/name) as x,np.load(old/name) as y:
            assert x.files==y.files
            for k in x.files:assert np.array_equal(x[k],y[k]),(folder,name,k)
        graphs.append(name)
    return dict(exact_prefix_and_disturbance_poststates=count,exact_initial_live_graphs=graphs)


def run(base,output,dino,sam):
    assert os.environ.get('CONTAINER_ID')=='57640710'
    base=base.resolve();output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    assert not (output/'identity.json').exists()
    torch.set_num_threads(1)
    with ExperimentLogger(team='graph',model='Task10F_matched3000_handoff_diagnosis',seed=42,
        root=output/'experiments',job_type='evaluate',config=dict(parent_comparison_uuid='d7c26fee195e4f32a7ddb99a94877b72',
            prior_evaluation_uuid='2bfe867c20e340aab2157c5e3ff33e8e',new_training=False,new_expert_data=False,
            new_rollouts='R0_step3000 only15',reused_rollouts='R1_step3000 15; R0_step0 15; R1_step0 identical model alias of R0_step0',
            final_seeds_previously_disclosed=True,base=str(base))) as log:
        write(output/'identity.json',dict(run_id=log.run_id,parent_comparison_uuid='d7c26fee195e4f32a7ddb99a94877b72'),immutable=True)
        identity=inventory(base,output)
        log.record(parent_comparison_uuid='d7c26fee195e4f32a7ddb99a94877b72',checkpoint_identity=identity,
            prior_evaluation_uuid='2bfe867c20e340aab2157c5e3ff33e8e')
        diagnosis.run(base,output/'stage1_existing')
        print('STAGE1_TRACE_ANALYSIS_COMPLETE',flush=True)
        checkpoint=identity['checkpoints']['R0_step3000']['path'];cp=checkpoint_load(checkpoint)
        log.link_output(checkpoint,checkpoint=True)
        norm=normalizer(base/'full_cache');config=read(base/'full_cache/config.json');codec=ActionCodec(cp['codec'])
        model=CachedGraphPolicy(norm.schema,'E0',**config['model']).cuda();model.load_state_dict(cp['model'],strict=True);model.eval()
        assert tensor_identity(model.state_dict())==identity['checkpoints']['R0_step3000']['model_sha256']
        detector,tracker=make_vision('cuda',dino,sam)
        cases=read(base/'full_experiment_plan.json')['final_cases'];protocol=read(output/'evaluation_protocol.json')
        contracts=[];results=[];started=time.perf_counter()
        for kind in ['handoff','autonomous','nominal']:
            specs=cases if kind!='nominal' else [dict(seed=s,anchor=25) for s in sorted({c['seed'] for c in cases})]
            for spec in specs:
                tag=f'R0_{kind}_{spec["seed"]}_{spec.get("offset_mm",[0,0,0])[0]}_{spec.get("offset_mm",[0,0,0])[1]}'
                folder=output/'rollouts'/tag;old=base/'final_evaluation/rollouts'/tag
                result,trace,actions,robots=episode(base,folder,model,norm,codec,detector,tracker,'R0',kind,spec,horizon=protocol['horizon'])
                result['video_sha256']=sha(folder/'rollout.mp4')
                write(folder/'trace.json',trace,immutable=True);npz(folder/'trajectory.npz',dict(proposed_action7=actions,robot76=robots))
                write(folder/'episode.json',result,immutable=True)
                assert result['reset_sha256']==read(old/'episode.json')['reset_sha256']
                check=compare_immutable_prefix(folder,old,kind)
                contracts.append(dict(episode=tag,reset_exact=True,**check))
                results.append(result);append_csv(output/({'handoff':'expert_handoff_rollouts.csv','autonomous':'autonomous_recovery_rollouts.csv','nominal':'nominal_reach_rollouts.csv'}[kind]),result)
                write(output/'progress.json',dict(complete=len(results),total=15,last_episode=tag,wall_seconds=time.perf_counter()-started))
                print('MATCHED_EVAL',len(results),tag,result['corrective_reach_success'],result['position_error_m'],result['physical_safety'],flush=True)
        assert len(results)==15
        for p,h in identity['source_sha256'].items():assert sha(p)==h
        for c,r in identity['checkpoints'].items():assert sha(r['path'])==r['sha256']
        write(output/'evaluation_contract_verification.json',dict(status='PASS',same_original_evaluator_and_threshold=True,
            new_episodes=15,reused_R1_episodes=15,paired_reset_prefix_graph_checks=contracts,
            live_graph_causal_gru='unchanged episode() implementation: hidden=None and tracker reset per episode; all prefix and disturbance obs forwarded',
            post_action_alignment='action_t=t; observation and robot76 pre-action; trace metrics post_state_t=t+1',
            GT_input_leakage='unchanged Robot76 and current RGB-D graph only; GT targets used only by measure()',
            no_training_or_expert_demonstration_collection=True,source_checkpoint_unchanged_after=True,
            blind_test=False),immutable=True)
        write(output/'evaluation_complete.json',dict(status='PASS',episodes=15,checkpoint_sha256=sha(checkpoint),wall_seconds=time.perf_counter()-started),immutable=True)
        log.link_output(output/'evaluation_contract_verification.json',kind='metrics')
        log.link_output(output/'evaluation_complete.json',kind='metrics')
        diagnosis.run(base,output/'combined_diagnosis',output)
        print('MATCHED_COMPLETE',log.run_id,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dino',type=Path,required=True);p.add_argument('--sam',type=Path,required=True)
    a=p.parse_args();run(a.base,a.output,a.dino,a.sam)
