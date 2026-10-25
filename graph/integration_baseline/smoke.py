"""Task06 contract evaluation and exactly ONE optimizer step, locally logged.

Collection and offline graph generation are separate reproducible commands.
Run this after their manifests finish; no rollout/training loop is provided.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
from time import perf_counter, process_time
import tomllib

import numpy as np
import torch

from utils.experiment_logger import ExperimentLogger
from graph.integration_baseline.io import read_packet, save_npz, write_json, sha256, packet_digest
from graph.integration_baseline.graph import build_graph, pack_graphs
from graph.integration_baseline.model import GraphEncoder, IntegrationPolicy, EpisodeStream, bc_loss, structured_readout
from graph.integration_baseline.data import ReplayDataset, collate_sequences

REPO = Path(__file__).resolve().parents[2]
OLD = REPO/'experiments/0f107550c9b64f1a84677a18e988cb51/packets'
REPORT = REPO/'docs/visual-graph-integration/06_graph_gnn_gru_baseline.md'


def stats(values):
    a=np.asarray(values,float)
    return dict(count=len(a),mean=float(a.mean()),median=float(np.median(a)),
                p95=float(np.percentile(a,95)),max=float(a.max()))


def rss_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def run_tests(out):
    modules=['graph.integration_baseline.tests.test_baseline','graph.integration_baseline.tests.test_data',
             'graph.integration_baseline.tests.test_causality',
             'visual.runtime.test.test_raw_ede_comparison','visual.runtime.test.test_extended_delaunay_edges',
             'visual.runtime.test.test_variable_rate_graph_contract','visual.runtime.test.test_variable_rate_shape_first',
             'visual.runtime.test.test_vision_features', 'graph.prototype.tests.test_models.GraphTests']
    command=[sys.executable,'-m','unittest',*modules,'-v']
    with (out/'tests.log').open('w') as stream:
        stream.write('COMMAND '+' '.join(command)+'\n');stream.flush()
        result=subprocess.run(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise AssertionError('contract tests failed; inspect tests.log')
    return dict(status='PASS',command=command,log='tests.log')


def stored_packets(out,cfg=None):
    cfg=cfg or tomllib.loads(Path(__file__).with_name('config.toml').read_text())['provisional_smoke_config']
    encoder=GraphEncoder(d=cfg['d'],L=cfg['L']).eval()
    robot=np.array([[0,0,0],[0,.02,0],[0,-.02,0]],np.float32)
    rows=[];arrays={};inputs={}
    with torch.inference_mode():
        for path in sorted(OLD.glob('*_raw.npz')):
            p=read_packet(path);digest=packet_digest(p)
            graph=build_graph(p,robot,k=cfg['k'],M=cfg['M'],block_rows=cfg['pair_block_rows'])
            packed=pack_graphs([graph]);trace={}
            z,presence=encoder(packed,trace)
            if not torch.isfinite(z).all() or digest!=packet_digest(graph.packet) or digest!=packet_digest(p):
                raise AssertionError('stored packet immutable/finite contract')
            key=path.stem
            rows.append(dict(condition=key,status='REAL_PACKET_WITH_SYNTHETIC_ROBOT',
                nodes=len(graph.node_type),edges=graph.edge_type.numel(),
                edge_counts=torch.bincount(graph.edge_type,minlength=3).tolist(),
                point_count=len(p['points']),object_present=presence[0].tolist(),
                unavailable_fields=int((~p['feature_field_available']).sum()),invalid_fields=int((~p['feature_field_valid']).sum()),
                zero_degree_nodes=int((torch.bincount(graph.edge_index[1],minlength=len(graph.node_type))==0).sum()),
                trace=trace))
            for name,value in dict(node_input=packed.node_input,node_type=packed.node_type,edge_index=packed.edge_index,
                    edge_attr=packed.edge_attr,edge_type=packed.edge_type,readout=z,presence=presence).items():
                arrays[key+'/'+name]=value.numpy()
            arrays[key+'/source_index']=p['provenance']['source_index']
            arrays[key+'/original_F5']=p['geometric_features']
            inputs[str(path.relative_to(REPO))]=sha256(path)
    if len(rows)!=16:
        raise ValueError('expected 8 basic + 8 stress Raw EDE packets')
    save_npz(out/'stored_packet_tensors.npz',**arrays)
    write_json(out/'stored_packet_forward.json',dict(status='REAL_PACKET_WITH_SYNTHETIC_ROBOT',conditions=rows))
    return rows,inputs


def real_smoke(out,dataset,cfg):
    normalizer=dataset.fit_normalizer()
    available=[e for e in dataset.raw if e in dataset.splits['train']]
    b,t=cfg['sequence_batch'],cfg['sequence_length']
    if len(available)<b:
        raise ValueError('not enough verified train episode prefixes for configured batch')
    connectivity=dict(k=cfg['k'],M=cfg['M'],block_rows=cfg['pair_block_rows'])
    sequences=[dataset.sequence(e,0,t) for e in available[:b]]
    graphs,states,targets,mask=collate_sequences(sequences,**connectivity)
    model=IntegrationPolicy(dataset.schema,d=cfg['d'],L=cfg['L'],gru_hidden=cfg['gru_hidden'],
                             gru_layers=cfg['gru_layers'],enc_hidden=cfg['enc_hidden']).eval()
    trace={}
    fused=model.fuse(graphs,states,normalizer,trace)
    with torch.no_grad():
        initial=torch.randn(cfg['gru_layers'],b,cfg['gru_hidden'])
        full,hn=model(graphs,states,normalizer,initial)
        outputs=[];hidden=initial.clone()
        for step_id in range(t):
            g,s,_,_=collate_sequences([[seq[step_id]] for seq in sequences],**connectivity)
            action,hidden=model.step(g,s,normalizer,hidden)
            outputs.append(action)
        step=torch.stack(outputs,1)
        torch.testing.assert_close(full,step,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(hn,hidden,rtol=1e-5,atol=1e-6)
        # Lifecycle tests with actual frames, preserving hidden between real t.
        stream=EpisodeStream(model,normalizer)
        first=None
        for t,row in enumerate(sequences[0]):
            g,s,_,_=collate_sequences([[row]],**connectivity)
            action=stream.step(g,s,(dataset.manifest['source_sha256'],row.episode_id),row.timestep)
            if t==0:first=action
        g,s,_,_=collate_sequences([[sequences[0][0]]],**connectivity)
        again=stream.step(g,s,'new-reset-identity',0)
        torch.testing.assert_close(first,again,rtol=1e-5,atol=1e-6)
        for flag in ('previous_terminated','previous_truncated'):
            action=stream.step(g,s,'new-reset-identity',0,**{flag:True})
            torch.testing.assert_close(first,action,rtol=1e-5,atol=1e-6)
        # The actual recorded TimeLimit boundary is checked separately from
        # synthetic reset signals applied to real input tensors above.
        eid=available[0]
        flags=np.flatnonzero(dataset.raw[eid]['truncated'])
        if not len(flags):
            raise ValueError('expected captured real truncation boundary')
        boundary=int(flags[0])
        previous=dataset.frame(eid,boundary);current=dataset.frame(eid,boundary+1)
        stream.reset()
        g0,s0,_,_=collate_sequences([[previous]],**connectivity)
        stream.step(g0,s0,'real-timeout',previous.timestep)
        g1,s1,_,_=collate_sequences([[current]],**connectivity)
        reset_action=stream.step(g1,s1,'real-timeout',current.timestep,previous_truncated=previous.truncated)
        independent,_=model.step(g1,s1,normalizer,None)
        torch.testing.assert_close(reset_action,independent,rtol=1e-5,atol=1e-6)
    # Exactly one backward and one optimizer.step in this evaluation.
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    raw,_=model(graphs,states,normalizer)
    loss=bc_loss(raw[mask],targets[mask],gripper_mode='regression')
    if not torch.isfinite(loss):
        raise AssertionError('nonfinite BC loss')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    groups=['graph_encoder.node_encoder','graph_encoder.node_types','graph_encoder.edge_types',
            *[f'graph_encoder.layers.{i}.{kind}' for i in range(cfg['L']) for kind in ('message','update')],
            'policy.encoder','policy.rnn','policy.head']
    gradients={}
    for group in groups:
        params=[p for name,p in model.named_parameters() if name.startswith(group)]
        if not params or any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise AssertionError('missing/nonfinite gradient '+group)
        norm=float(torch.stack([p.grad.square().sum() for p in params]).sum().sqrt())
        if norm<=0:
            raise AssertionError('zero module gradient '+group)
        gradients[group]=dict(l2_norm=norm)
    optimizer.step()
    updates={}
    for group in groups:
        change=sum(float((p.detach()-before[name]).square().sum()) for name,p in model.named_parameters() if name.startswith(group))**.5
        if change<=0:
            raise AssertionError('no actual parameter update '+group)
        updates[group]=change
    arrays=dict(actions_sequence=full.detach().numpy(),actions_step=step.detach().numpy(),target=targets.numpy(),
                target_mask=mask.numpy(),normalizer_mean=normalizer.stats.mean.numpy(),normalizer_std=normalizer.stats.std.numpy(),
                frame_map=graphs.frame_map.numpy(),ptr=graphs.ptr.numpy(),edge_index=graphs.edge_index.numpy(),
                node_input=graphs.node_input.numpy(),node_type=graphs.node_type.numpy(),edge_type=graphs.edge_type.numpy(),
                edge_attr=graphs.edge_attr.numpy(),robot_state=states.numpy(),fused=fused.detach().numpy(),
                initial_hidden=initial.numpy(),final_hidden_sequence=hn.numpy(),final_hidden_step=hidden.numpy())
    for name,p in model.named_parameters():
        arrays['before/'+name]=before[name].numpy()
        arrays['after/'+name]=p.detach().numpy()
        arrays['gradient/'+name]=p.grad.numpy()
    save_npz(out/'real_forward_backward.npz',**arrays)
    result=dict(status='REAL_ALIGNED',scope='two verified train prefixes, not complete expert replays',
        episode_ids=available[:b],timesteps=list(range(cfg['sequence_length'])),sequence_batch=b,sequence_length=cfg['sequence_length'],
        nodes=graphs.node_type.numel(),edges=graphs.edge_type.numel(),
        node_counts=(graphs.ptr[1:]-graphs.ptr[:-1]).tolist(),frame_map=graphs.frame_map.tolist(),
        action_shape=list(raw.shape),hidden_shape=list(hn.shape),schema_dimension=dataset.schema['dimension'],
        fused_dimension=model.fused_dim,trace=trace,
        sequence_step_max_abs_error=float((full-step).abs().max()),hidden_resets='PASS',
        actual_truncated_reset=dict(episode_id=eid,previous_timestep=boundary,current_timestep=boundary+1,
            source_flag=previous.truncated,max_action_error=float((reset_action-independent).abs().max())),
        terminated_reset_scope='synthetic lifecycle signal on actual input; no actual terminated event in verified prefix',
        bc_loss=float(loss.detach()),loss='existing graph.prototype.v2.models.bc_loss regression7 MSE',
        optimizer_steps=1,gradients=gradients,parameter_updates_l2=updates,
        normalizer=dict(fit_split='train',episode_ids=normalizer.fitted_episode_ids,
            rows=sum(len(dataset.raw[e]['actions']) for e in normalizer.fitted_episode_ids),std_floor=.001,
            reused='graph.prototype.v2.state.StateNormalizer statistics/storage; schema-sized adapter'),
        validation_test_normalizer='FROZEN; validation/test data unavailable in captured prefixes',
        variable_n=len(set((graphs.ptr[1:]-graphs.ptr[:-1]).tolist()))>1)
    write_json(out/'real_forward_backward.json',result)
    return result,model.eval(),normalizer,sequences


def benchmark(out,model,normalizer,sequences,cfg):
    warmup,repeats=3,10
    frame=sequences[0][0]
    samples=[];base_rss=rss_bytes()
    with torch.inference_mode():
        for i in range(warmup+repeats):
            wall,cpu=perf_counter(),process_time()
            graph=build_graph(frame.packet,frame.robot_positions_ee,k=cfg['k'],M=cfg['M'],block_rows=cfg['pair_block_rows'])
            a=perf_counter();packed=pack_graphs([graph],[(0,0)]);pack_ms=(perf_counter()-a)*1000
            trace={};z,presence=model.graph_encoder(packed,trace)
            a=perf_counter();state=normalizer.apply(torch.from_numpy(frame.robot_state)[None])
            fused=torch.cat((z,presence,state),-1)[:,None];fusion_ms=(perf_counter()-a)*1000
            a=perf_counter();encoded=model.policy.encoder(fused);enc_ms=(perf_counter()-a)*1000
            a=perf_counter();y,_=model.policy.rnn(encoded,None);gru_ms=(perf_counter()-a)*1000
            a=perf_counter();model.policy.head(y);head_ms=(perf_counter()-a)*1000
            if i>=warmup:
                samples.append(dict(**graph.timings_ms,disjoint_union_pack_ms=pack_ms,
                    **{k:v for k,v in trace.items() if k.endswith('_ms')},fusion_ms=fusion_ms,
                    gru_input_encoder_ms=enc_ms,gru_ms=gru_ms,action_head_ms=head_ms,
                    total_wall_ms=(perf_counter()-wall)*1000,total_cpu_ms=(process_time()-cpu)*1000))
    logical=sum(t.numel()*t.element_size() for t in vars(packed).values())
    result=dict(status='REAL_ALIGNED',device='cpu',dtype='float32',torch_threads=torch.get_num_threads(),
        warmup=warmup,repeats=repeats,batch=1,sequence_length=1,reset_hidden_each_sample=True,
        nodes=packed.node_type.numel(),edges=packed.edge_type.numel(),mode='eval/inference_mode',
        boundary='in-memory packet -> receiver/interactions/packing/GNN/readout/fusion/encoder/GRU/head; excludes upstream Vision, IO and reports',
        samples=samples,summary={k:stats([s[k] for s in samples]) for k in samples[0]},
        memory=dict(process_prior_high_water_bytes=base_rss,process_peak_rss_bytes=rss_bytes(),
                    packed_tensor_logical_bytes=logical,model_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                    caveat='same-process ru_maxrss high-water; includes imports and prior backward; not an isolated forward allocation peak'),
        gpu=dict(CUDA='NOT_EXECUTED',MPS='NOT_EXECUTED'))
    write_json(out/'latency_memory.json',result)
    return result


def snapshot(out,start):
    files=list((REPO/'graph/integration_baseline').rglob('*.py'))+[
        REPO/'graph/integration_baseline/config.toml', REPO/'graph/integration_baseline/README.md']
    imports=['graph/vision_input.py','graph/prototype/models.py','graph/prototype/v2/state.py','graph/prototype/v2/models.py',
        'graph/prototype/data.py','graph/prototype/prepare_dataset.py','recurrent/models.py','utils/experiment_logger.py',
        'utils/experiment_artifacts.py','visual/runtime/sampling.py','visual/runtime/pca.py','visual/runtime/vision_packet.py',
        'visual/runtime/intra_object_edges.py','visual/runtime/extended_delaunay_edges.py','visual/runtime/tracking.py',
        'visual/runtime/initialization.py','visual/runtime/grounding.py','visual/runtime/rgbd.py',
        'visual/runtime/test/run_variable_rate_shape_first.py','visual/runtime/test/raw_ede_comparison_adapter.py',
        'visual/runtime/test/run_vision_feature_edge.py','envs/maniskill/gt_state.py','envs/maniskill/peg_insertion.py']
    files += [REPO/p for p in imports]
    hashes={}
    for p in files:
        target=out/'source_snapshot'/p.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,target);hashes[str(p.relative_to(REPO))]=sha256(p)
    write_json(out/'source_hashes.json',hashes)
    shutil.copytree(start,out/'before_snapshot')
    initial=json.loads((start/'protected.json').read_text())
    changed=[p for p,digest in initial.items() if not (REPO/p).exists() or sha256(REPO/p)!=digest]
    if changed:
        raise AssertionError('preexisting sources/artifacts changed: '+str(changed))
    write_json(out/'source_protection.json',dict(status='PASS',files=len(initial),changed=changed))
    diff=subprocess.check_output(['git','diff','--binary'],cwd=REPO)
    parts=[b'CURRENT TRACKED WORKTREE DIFF (includes preexisting changes):\n',diff,
           b'\nTASK06 ADDED FILES:\n']
    for p in sorted((REPO/'graph/integration_baseline').rglob('*')):
        if p.is_file() and p.suffix in ('.py','.toml','.md'):
            parts.append(subprocess.run(['git','diff','--no-index','--','/dev/null',str(p)],stdout=subprocess.PIPE).stdout)
    (out/'implementation.diff').write_bytes(b''.join(parts))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,default=Path('outputs/graph/integration_baseline/replay_v2'))
    parser.add_argument('--vision',type=Path,default=Path('outputs/graph/integration_baseline/vision_v2'))
    parser.add_argument('--start-snapshot',type=Path,default=Path('/tmp/task06_start'))
    parser.add_argument('--wait-dataset',action='store_true',help='wait for the already running offline builder, never start another replay')
    args=parser.parse_args()
    if os.environ.get('WANDB_MODE')!='disabled':
        raise ValueError('WANDB_MODE=disabled required; no external upload authorized')
    torch.set_num_threads(1);torch.manual_seed(0);np.random.seed(0)
    config=tomllib.loads(Path(__file__).with_name('config.toml').read_text())
    config.update(replay=str(args.replay),vision=str(args.vision),platform=platform.platform())
    cfg=config['provisional_smoke_config']
    with ExperimentLogger(team='graph',model='vision_graph_gru_baseline',job_type='evaluate',config=config,seed=0) as logger:
        out=logger.directory
        print('RUN_DIRECTORY',out,flush=True)
        tests=run_tests(out)
        legacy=Path('/tmp/task06_preflight/tests.log')
        if legacy.exists():shutil.copyfile(legacy,out/'preflight_legacy_tests.log')
        stored,inputs=stored_packets(out,cfg)
        if args.wait_dataset:
            while True:
                manifest_path=args.vision/'manifest.json'
                state=json.loads(manifest_path.read_text())
                if state['status']!='RUNNING':break
                print('WAITING_FOR_OFFLINE_BUILDER',sum(len(e['packets']) for e in state['episodes']),flush=True)
                time.sleep(30)
        dataset=ReplayDataset(args.replay,args.vision)
        validation=dataset.validation_report();write_json(out/'dataset_validation.json',validation)
        actual,model,normalizer,sequences=real_smoke(out,dataset,cfg)
        timing=benchmark(out,model,normalizer,sequences,cfg)
        write_json(out/'graph_tensor_schema.json',dict(
            node_input=['position_ee_m[3]','finite_F5[5]','field_available[3]','field_valid[3]'],
            node_type=['TCP','Left','Right','Peg','Box'],
            edge_attr=['distance_m','direction_x','direction_y','direction_z'],
            direction='destination - source, normalized; zero distance -> zero direction',
            edge_type=['Raw intra-object','Robot-Object','Peg-Box'],
            graph_readout=['h_TCP','h_Left','h_Right','mean_h_Peg','mean_h_Box'],
            readout_dimension=5*cfg['d'],fused_dimension=model.fused_dim,
            robot_state_schema=dataset.schema,action_schema='pd_ee_delta_pose_regression7',
            raw_values='original packet separately preserved, no NaN/dtype/source-row mutation',
            batching='disjoint union; graph ptr offsets; frame_map[G,2] -> B,T; no padded fake graph'))
        from huggingface_hub import try_to_load_from_cache
        from visual.runtime.grounding import GDINO_ID
        from visual.runtime.tracking import DEFAULT_CHECKPOINT
        foundation=dict(sam2=dict(path=str(DEFAULT_CHECKPOINT),sha256=sha256(DEFAULT_CHECKPOINT)),grounding_dino={})
        for name in ('model.safetensors','config.json','preprocessor_config.json','tokenizer_config.json','tokenizer.json','vocab.txt'):
            cached=try_to_load_from_cache(GDINO_ID,name)
            if isinstance(cached,str) and Path(cached).is_file():
                foundation['grounding_dino'][name]=dict(path=cached,sha256=sha256(cached))
        if not foundation['grounding_dino'].get('model.safetensors'):
            raise ValueError('could not resolve installed cached Grounding DINO weight identity')
        write_json(out/'foundation_model_inputs.json',foundation)
        for name,path in [('replay_alignment_coverage.json',dataset.replay/'manifest.json'),
                          ('dataset_manifest.json',dataset.vision/'manifest.json'),('robot_state_schema.json',dataset.replay/'schema.json')]:
            shutil.copyfile(path,out/name)
            inputs[str(path.relative_to(REPO))]=sha256(path)
        for name in ('source','initial_source','frozen_split_manifest'):
            path=Path(dataset.manifest[name]);inputs[str(path.relative_to(REPO))]=sha256(path)
        write_json(out/'input_manifest.json',dict(sha256=inputs,
            raw_episode_and_packet_hashes='replay_alignment_coverage.json / dataset_manifest.json'))
        for name in ('/tmp/task06_replay.log','/tmp/task06_replay_v2.log','/tmp/task06_diagnostic.log','/tmp/task06_vision.log',
                     '/tmp/task06_vision_v2.log','/tmp/task06_actual_causality.log','/tmp/task06_actual_causality_v2.log',
                     '/tmp/task06_tests.log'):
            path=Path(name)
            if path.exists():shutil.copyfile(path,out/path.name.removeprefix('task06_'))
        for path in args.start_snapshot.glob('diagnostic_ep*.npz'):
            shutil.copyfile(path,out/path.name)
        diagnostic=args.start_snapshot/'replay_diagnostic.json'
        if diagnostic.exists():shutil.copyfile(diagnostic,out/diagnostic.name)
        rejected=REPO/'outputs/graph/integration_baseline/vision_v1/quality_audit.json'
        if rejected.exists():shutil.copyfile(rejected,out/'rejected_vision_v1_audit.json')
        summary=dict(run_id=logger.run_id,tests=tests,stored_packet_conditions=stored,
            dataset_paths=dict(replay=str(dataset.replay),vision=str(dataset.vision)),
            real_forward_backward=actual,replay_coverage=dataset.manifest['coverage'],
            preflight_legacy_regression=dict(status='PASS',scope='existing synthetic GT prototype unit test',
                optimizer_steps=240,detail='3 existing model fixtures ×80 steps; not the new integration smoke or expert training',
                log='preflight_legacy_tests.log'),
            full_replay_gate=dataset.manifest['pilot_gate']['status'],vision_frames=dataset.graph_manifest['frames_built'],
            schema=dataset.schema,latency_memory=timing,device='cpu',
            status='MINIMAL_REAL_PREFIX_PATH_VERIFIED_FULL_REPLAY_BLOCKED',
            not_executed=['remaining 88 expert replays','complete 91-episode dataset','CUDA','MPS',
                'long BC training','HPO','closed-loop policy rollout','success-rate evaluation','publishing'],
            limitations=['all three full pilot replays exceed existing 1e-3 bound in late qvel',
                'only independently verified observation prefixes enter actual smoke',
                'source TimeLimit truncates at t=99 although saved expert episodes continue; no sequence crosses a flag',
                'SAM2 optional C extension absent; installed predictor skips hole-filling postprocess',
                'first Vision collection rejected for SAM2 future total-length dependence; regenerated using current horizon only',
                'mask physical correctness and learned control success are not established'])
        write_json(out/'summary.json',summary)
        snapshot(out,args.start_snapshot)
        from graph.integration_baseline.report import generate,verify
        generate(out,summary,dataset,sequences)
        logger.record(task='06',environment=dict(env_id='PegInsertionSide-v1',backend=dataset.manifest['backend']),
                      dataset_hash=dataset.manifest['source_sha256'],coverage=dataset.manifest['coverage'],
                      full_replay_gate=dataset.manifest['pilot_gate']['status'])
        logger.log(dict(bc_smoke_loss=actual['bc_loss'],optimizer_steps=1,aligned_targets=validation['validated_targets']))
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name not in ('metadata.json','manifest.json','metrics.json','config.yaml') and not path.name.endswith('.tracking.json'):
                logger.link_output(path,kind='metrics' if path.suffix=='.json' else 'trajectory' if path.suffix=='.npz' else 'reports')
        logger.link_output(REPORT,kind='reports')
        write_json(out/'verification.json',dict(status='PENDING_READBACK'))
        logger.link_output(out/'verification.json',kind='metrics')
        logger.finish()
        verify(out,REPORT)
        print('COMPLETED',out,flush=True)


if __name__=='__main__':
    main()
