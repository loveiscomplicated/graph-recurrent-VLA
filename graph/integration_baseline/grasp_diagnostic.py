"""Fixed-checkpoint, saved-input grasp diagnostic; no training or simulator steps.

The reference is the actual direct_train.evaluate() with hooks that only copy
outputs. Single-layer GRU output exposes every reference hidden state. Compare
the rollout wrapper both with that identical past hidden and with its own carry.
"""
import argparse
import ast
import csv
import hashlib
import html
import inspect
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np
import torch

from graph.integration_baseline import direct_train
from graph.integration_baseline.direct_dataset import DirectDataset
from graph.integration_baseline.direct_runtime import atomic, read, sha
from graph.integration_baseline.graph import build_graph, pack_graphs
from graph.integration_baseline.io import save_npz, write_json
from graph.integration_baseline.model import EpisodeStream
from graph.integration_baseline.rollout_checkpoint import load_bundle
from graph.integration_baseline.rollout_stop_clock import infer

PHASES = ('reach', 'descend', 'close')
AXES = ('dx', 'dy', 'dz', 'rotation_X', 'rotation_Y', 'rotation_Z', 'gripper')


def array(value):
    return value.detach().cpu().numpy().copy()


def digest(*values):
    h = hashlib.sha256()
    for value in values:
        value = np.asarray(value)
        h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def audit_training_source(training_run):
    """The loader's inference class must also match the training snapshot."""
    expected=read(training_run/'source_hashes.json')
    repo=Path(inspect.getfile(direct_train)).resolve().parents[2]
    exact={}
    for relative in ['graph/integration_baseline/direct_train.py','graph/integration_baseline/data.py',
                     'graph/integration_baseline/model.py','graph/integration_baseline/graph.py',
                     'graph/integration_baseline/robot.py','recurrent/models.py']:
        exact[relative]=expected.get(relative)==sha(repo/relative)
    if not all(exact.values()):raise ValueError('training evaluator/model source differs')
    relative='graph/integration_baseline/direct_dataset.py'
    original=training_run/'source_snapshot'/relative
    if sha(original)!=expected[relative]:raise ValueError('original loader source snapshot corrupt')
    def nodes(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
    before,after=nodes(original),nodes(repo/relative)
    matches={name:before[name]==after[name] for name in ['DirectDataset','load_episode']}
    if not all(matches.values()):raise ValueError('training data loading logic changed')
    return dict(exact_sources=exact,dataset_file_exact=expected[relative]==sha(repo/relative),
                dataset_loading_ast_exact=matches,
                note='whole Dataset file differs only outside loading logic; no Vision processing is invoked')


def select_episodes(dataset, config):
    selected = []
    vision = {e['episode_id']:e for e in dataset.graph_manifest['episodes']}
    for split in config['splits']:
        ids = sorted(dataset.splits[split])
        first = ids[0]
        eligible = []
        for eid in ids:
            close = [i for i,p in enumerate(dataset.episodes[eid]['phases']) if p['before']=='close']
            rows = {p['timestep']:p for p in vision[eid]['packets']}
            if close and all(all(rows[t]['object_counts']) for t in close):
                eligible.append(eid)
        second = next(e for e in eligible if e != first)
        for eid, reason in [(first,'lowest episode ID'), (second,'lowest other ID with both objects on all close frames')]:
            phases = dataset.episodes[eid]['phases']
            close = [i for i,p in enumerate(phases) if p['before']=='close']
            stop = min(len(phases),close[-1]+1+config['post_close_context_steps'])
            selected.append(dict(split=split,episode_id=eid,reason=reason,window_start=0,
                window_stop_exclusive=stop,close_start=close[0],close_stop_exclusive=close[-1]+1))
    return selected


def reference_episode(dataset, model, norm, eid, train_config):
    """Call the unchanged evaluator; copy its real packed inputs and GRU outputs."""
    rnn = model.policy.rnn
    if rnn.num_layers != 1 or rnn.bidirectional:
        raise ValueError('this diagnostic requires the checkpoint single-layer forward GRU')
    calls, recurrent = [], []

    def before(module, args):
        graphs,state,normalizer,hidden = args
        if normalizer is not norm:
            raise ValueError('reference normalizer changed')
        calls.append(dict(graphs=graphs.to('cpu'), state=state.detach().cpu().clone(),
                          hidden=None if hidden is None else array(hidden)))

    def after_rnn(module, args, output):
        recurrent.append(array(output[0]))

    def after(module, args, output):
        calls[-1].update(action=array(output[0]), final_hidden=array(output[1]))

    handles=[model.register_forward_pre_hook(before),model.policy.rnn.register_forward_hook(after_rnn),
             model.register_forward_hook(after)]
    try:
        metrics=direct_train.evaluate(dataset,model,norm,[eid],train_config)
    finally:
        for handle in handles:handle.remove()
    if len(calls)!=len(recurrent):
        raise ValueError('reference GRU capture count mismatch')
    actions, hidden = [], []
    for call,y in zip(calls,recurrent):
        np.testing.assert_array_equal(y[:,-1][None],call['final_hidden'])
        actions.append(call['action'][0]);hidden.append(y[0])
    actions=np.concatenate(actions);post=np.concatenate(hidden)[:,None,None,:]
    pre=np.concatenate((np.zeros_like(post[:1]),post[:-1]))
    cursor=0
    for call in calls:
        if cursor:
            np.testing.assert_array_equal(call['hidden'],pre[cursor])
        elif call['hidden'] is not None:
            raise ValueError('reference did not start with zero hidden')
        cursor+=call['action'].shape[1]
    return dict(actions=actions,pre_hidden=pre,post_hidden=post,calls=calls,metrics=metrics)


def assert_same_inputs(reference, graphs, states):
    """Compare every original packed graph, edge order and Robot76 byte-exactly."""
    flat=0
    for call in reference['calls']:
        packed=call['graphs']
        for local in range(packed.count):
            if flat >= len(graphs):
                return
            g=graphs[flat];start,end=map(int,packed.ptr[local:local+2])
            chosen=packed.node_graph[packed.edge_index[0]]==local
            for key in ('node_input','node_type'):
                if not torch.equal(getattr(packed,key)[start:end],getattr(g,key)):
                    raise ValueError('reference/step graph input mismatch: '+key)
            if not torch.equal(packed.edge_index[:,chosen]-start,g.edge_index):
                raise ValueError('reference/step edge order mismatch')
            for key in ('edge_attr','edge_type'):
                if not torch.equal(getattr(packed,key)[chosen],getattr(g,key)):
                    raise ValueError('reference/step edge input mismatch: '+key)
            np.testing.assert_array_equal(array(call['state'][local]),states[flat])
            flat+=1
    if flat != len(graphs):
        raise ValueError('reference coverage incomplete')


def compare_streams(model,norm,graphs,states,reference,identity,*,stream_type=EpisodeStream):
    device=next(model.parameters()).device
    same,one,carry,h_same,h_carry=[],[],[],[],[]
    carried=stream_type(model,norm)
    for t,graph in enumerate(graphs):
        packed=pack_graphs([graph]).to(device);state=torch.from_numpy(states[t:t+1]).to(device)
        past=torch.from_numpy(reference['pre_hidden'][t].copy()).to(device)
        with torch.no_grad():
            action,_=model(packed,state,norm,past.clone())
        one.append(array(action)[0,0])
        controlled=stream_type(model,norm)
        controlled.identity=identity
        controlled.timestep=None if t==0 else t-1
        controlled.hidden=past.clone()
        action,_=infer(controlled,packed,state,identity,t)
        same.append(array(action)[0]);h_same.append(array(controlled.hidden))
        action,_=infer(carried,packed,state,identity,t)
        carry.append(array(action)[0]);h_carry.append(array(carried.hidden))
    return dict(step_same_hidden=np.stack(same),forward_one_step=np.stack(one),
                step_carried=np.stack(carry),hidden_same=np.stack(h_same),hidden_carried=np.stack(h_carry))


def difference(a,b,config):
    error=np.abs(a-b)
    failed=error > config['atol']+config['rtol']*np.abs(b)
    return dict(pass_tolerance=not bool(failed.any()),max_abs=float(error.max()),
                max_abs_per_component=error.reshape(-1,error.shape[-1]).max(0).tolist(),
                failed_values=int(failed.sum()),exact=bool(np.array_equal(a,b)))


def path_checks(result,ref,config):
    n=len(result['step_same_hidden'])
    return dict(same_hidden_forward_vs_rollout=difference(result['step_same_hidden'],result['forward_one_step'],config),
        chunk_reference_vs_same_hidden_step=difference(result['step_same_hidden'],ref['actions'][:n],config),
        chunk_reference_vs_carried_step=difference(result['step_carried'],ref['actions'][:n],config),
        hidden_reference_vs_same_hidden_step=difference(result['hidden_same'],ref['post_hidden'][:n],config),
        hidden_reference_vs_carried_step=difference(result['hidden_carried'],ref['post_hidden'][:n],config))


def controller_commands(actions):
    """Original installed native controller preprocessing, for intent comparison."""
    actions=np.asarray(actions,np.float32)
    result=actions.copy()
    result[:,:3]=np.clip(actions[:,:3],-1,1)*.1
    result[:,3:6]=actions[:,3:6]/np.maximum(np.linalg.norm(actions[:,3:6],axis=1,keepdims=True),1)*-.1
    result[:,6:]=(np.clip(actions[:,6:],-1,1)+1)*.025-.01
    return result


def imitation_stats(pred,target):
    if not len(pred):return dict(frames=0)
    delta=pred.astype(np.float64)-target.astype(np.float64)
    physical=controller_commands(pred).astype(np.float64)-controller_commands(target).astype(np.float64)
    return dict(frames=len(pred),mse=float(np.mean(delta**2)),mae_per_component=np.mean(np.abs(delta),0).tolist(),
        translation_command_error_l2_mm_mean=float(np.linalg.norm(physical[:,:3],axis=1).mean()*1000),
        translation_command_error_l2_mm_max=float(np.linalg.norm(physical[:,:3],axis=1).max()*1000),
        rotation_component_error_l2_rad_mean=float(np.linalg.norm(physical[:,3:6],axis=1).mean()),
        gripper_target_error_per_finger_mm_mean=float(np.abs(physical[:,6]).mean()*1000),
        gripper_opposite_sign_frames=int(((pred[:,6]<0)!=(target[:,6]<0)).sum()))


def write_episode(folder,spec,dataset,model,norm,train_config,config):
    folder.mkdir()
    eid=spec['episode_id'];n=spec['window_stop_exclusive']
    reference=reference_episode(dataset,model,norm,eid,train_config)
    frames=[dataset.frame(eid,t) for t in range(n)]
    graphs=[build_graph(f.packet,f.robot_positions_ee,**train_config['graph']) for f in frames]
    states=np.stack([f.robot_state for f in frames]);target=np.stack([f.action for f in frames])
    assert_same_inputs(reference,graphs,states)
    result=compare_streams(model,norm,graphs,states,reference,f'expert:{eid}')
    checks=path_checks(result,reference,config)
    phases=np.array([p['before'] for p in dataset.episodes[eid]['phases'][:n]])
    presence=np.array([[bool((f.packet['object_id']==o).any()) for o in (0,1)] for f in frames])
    point_counts=np.array([[int((f.packet['object_id']==o).sum()) for o in (0,1)] for f in frames])
    report=dict(**spec,path_checks=checks,graph_robot_inputs_exact=True,
                reference_full_episode_metrics=reference['metrics'],phase_metrics={})
    pred=result['step_carried']
    for phase in PHASES:
        selected=phases==phase
        both=selected & presence.all(1)
        report['phase_metrics'][phase]=dict(all_frames=imitation_stats(pred[selected],target[selected]),
            both_objects_present=imitation_stats(pred[both],target[both]),
            both_objects_present_frames=int(both.sum()))
    def first_negative(values):
        found=np.flatnonzero(values[:,6]<0)
        return int(found[0]) if len(found) else None
    report['first_negative_gripper']=dict(expert=first_negative(target),predicted=first_negative(pred))
    close=phases=='close';cmd=controller_commands(pred)
    report['close_prediction_commands']=dict(translation_l2_mm_mean=float(np.linalg.norm(cmd[close,:3],axis=1).mean()*1000),
        rotation_component_l2_rad_mean=float(np.linalg.norm(cmd[close,3:6],axis=1).mean()),
        gripper_raw_min=float(pred[close,6].min()),gripper_raw_max=float(pred[close,6].max()))
    # Original recorded controller audit is an independent check of scaling.
    with h5py.File(dataset.run/dataset.episodes[eid]['path'],'r') as raw:
        expected=np.column_stack((raw['arm_scaled_input'][:n],raw['gripper_scaled_input'][:n]))
        np.testing.assert_allclose(controller_commands(target),expected,atol=2e-8,rtol=1e-6)
        rgb=raw['base_camera/rgb'][:n]
        for t,frame in enumerate(frames):
            if hashlib.sha256(rgb[t].tobytes()).hexdigest()!=frame.packet['observation_provenance']['rgb_sha256']:
                raise ValueError('stored graph/RGB source mismatch')
            np.testing.assert_array_equal(frame.packet['camera_intrinsic'],raw['base_camera/intrinsic'][t].astype(np.float64))
            np.testing.assert_array_equal(frame.packet['camera_to_world'],raw['base_camera/cam2world'][t].astype(np.float64))
    save_npz(folder/'predictions.npz',expert_action=target,reference_action=reference['actions'][:n],
        **result,reference_pre_hidden=reference['pre_hidden'][:n],reference_post_hidden=reference['post_hidden'][:n],
        robot_state=states,robot_normalized=array(norm.apply(torch.from_numpy(states))),
        presence=presence,point_counts=point_counts,timestep=np.arange(n),phase=phases,
        expert_controller_command=expected,predicted_controller_command=cmd,episode_id=np.int64(eid))
    records=[]
    for t in range(n):
        row=dict(episode_id=eid,split=spec['split'],timestep=t,simulation_time=t/20,phase=phases[t],
                 peg_present=bool(presence[t,0]),box_present=bool(presence[t,1]),
                 peg_points=int(point_counts[t,0]),box_points=int(point_counts[t,1]),
                 reference_hidden_norm=float(np.linalg.norm(reference['pre_hidden'][t])))
        for j,axis in enumerate(AXES):
            row.update({f'expert_{axis}':float(target[t,j]),f'reference_{axis}':float(reference['actions'][t,j]),
                        f'step_{axis}':float(pred[t,j]),f'error_{axis}':float(pred[t,j]-target[t,j])})
        records.append(row)
    write_json(folder/'frames.json',records)
    with (folder/'frames.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write_json(folder/'summary.json',report)
    plots(folder,spec,target,reference['actions'][:n],pred,phases,rgb)
    return report


def plots(folder,spec,expert,reference,pred,phases,rgb):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(7,1,figsize=(11,12),sharex=True,layout='constrained')
    t=np.arange(len(pred))
    for j,axis in enumerate(axes):
        axis.plot(t,expert[:,j],color='black',label='Expert',linewidth=1.6)
        axis.plot(t,reference[:,j],color='#d97706',label='Evaluation T32',linestyle='--')
        axis.plot(t,pred[:,j],color='#0284c7',label='Rollout step / carried hidden',linestyle=':')
        for phase,color in [('descend','#ffedd5'),('close','#fee2e2')]:
            ids=np.flatnonzero(phases==phase)
            if len(ids):axis.axvspan(ids[0]-.5,ids[-1]+.5,color=color,zorder=-10)
        axis.set_ylabel(AXES[j]);axis.grid(alpha=.2)
    axes[0].legend(loc='upper right',fontsize=8)
    axes[-1].set_xlabel('Expert episode timestep (20Hz); shaded: descend / close')
    fig.suptitle(f"{spec['split']} episode {spec['episode_id']}: predictions on stored expert states")
    fig.savefig(folder/'actions.png',dpi=150);plt.close(fig)
    chosen=sorted(set(max(0,min(len(rgb)-1,spec['close_start']+offset)) for offset in [-2,0,3,7]))
    fig,axes=plt.subplots(1,len(chosen),figsize=(12,3.4),layout='constrained')
    for axis,timestep in zip(np.atleast_1d(axes),chosen):
        axis.imshow(rgb[timestep]);axis.axis('off')
        axis.set_title(f"t={timestep}, {phases[timestep]}\ngrip expert={expert[timestep,6]:+.2f}, pred={pred[timestep,6]:+.2f}",fontsize=10)
    fig.suptitle('Stored expert RGB; predicted actions were not applied to a simulator')
    fig.savefig(folder/'close_frames.png',dpi=150);plt.close(fig)


def replay_rollout(folder,episode,model,norm,graph_config,config):
    """Replay only recorded live inputs; no simulator/vision invocation."""
    from graph.integration_baseline.io import read_packet
    folder.mkdir()
    with np.load(episode/'observations.npz',allow_pickle=False) as data:
        states=data['robot_state'].copy();geometry=data['robot_ee_geometry'].copy()
    with np.load(episode/'actions.npz',allow_pickle=False) as data:original=data['actions'].copy()
    with np.load(episode/'hidden.npz',allow_pickle=False) as data:original_hidden=data['hidden'].copy()
    n=min(len(original),config['rollout_context_steps']);device=next(model.parameters()).device
    stream=EpisodeStream(model,norm);actions=[];hidden=[]
    for t in range(n):
        packet=read_packet(episode/f'packet_{t:04d}.npz');graph=build_graph(packet,geometry[t],**graph_config)
        with np.load(episode/f'graph_{t:04d}.npz',allow_pickle=False) as saved:
            for name in ('node_input','node_type','edge_index','edge_attr','edge_type'):
                np.testing.assert_array_equal(array(getattr(graph,name)),saved[name])
            np.testing.assert_array_equal(packet['provenance']['source_index'],saved['source_index'])
        output,_=infer(stream,pack_graphs([graph]).to(device),torch.from_numpy(states[t:t+1]).to(device),'stored:41005',t)
        actions.append(array(output)[0]);hidden.append(array(stream.hidden))
    actions=np.stack(actions);hidden=np.stack(hidden)
    result=dict(frames=n,scope='recorded live inputs only; no environment rerun',graph_input_exact=True,
                action=difference(actions,original[:n],config),hidden=difference(hidden,original_hidden[:n],config))
    save_npz(folder/'replay.npz',recorded_action=original[:n],replayed_action=actions,
             recorded_hidden=original_hidden[:n],replayed_hidden=hidden,timestep=np.arange(n))
    write_json(folder/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--checkpoint-manifest',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--rollout-episode',type=Path)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    os.environ.update(WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    config=read(args.config);torch.set_num_threads(1);torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if args.device=='cuda':
        if torch.cuda.mem_get_info()[0] < config['minimum_cuda_free_bytes']:
            raise RuntimeError('BLOCKED_RESOURCE: diagnostic does not interrupt/free other GPU jobs')
        torch.cuda.set_per_process_memory_fraction(config['cuda_memory_fraction'])
    train_config=read(args.training_run/'config.yaml');train_config['device']=args.device
    model,norm,provenance=load_bundle(args.checkpoint_manifest,device=args.device)
    if provenance['manifest']['files']['checkpoint']['sha256'] != config['checkpoint_sha256']:
        raise ValueError('the fixed checkpoint changed')
    source_audit=audit_training_source(args.training_run)
    for name,key in [('config.yaml','config_sha256'),('dataset_manifest.json','dataset_sha256'),
                     ('split_manifest.json','split_sha256'),('vision_manifest.json','vision_sha256')]:
        if sha(args.training_run/name)!=provenance['identity'][key]:
            raise ValueError('diagnostic input/checkpoint identity mismatch: '+name)
    dataset=DirectDataset(args.training_run)
    if dataset.schema != norm.schema:
        raise ValueError('dataset/checkpoint normalizer schema mismatch')
    selection=select_episodes(dataset,config)
    from utils.experiment_logger import ExperimentLogger
    with ExperimentLogger(team='integration',model='fixed_checkpoint_grasp_diagnostic',seed=config['seed'],
        config=dict(config,device=args.device,selection=selection,training_run=str(args.training_run),
                    checkpoint_manifest=str(args.checkpoint_manifest)),job_type='evaluate',root=args.output_root) as logger:
        run=logger.directory;print('RUN_DIR='+str(run),flush=True)
        atomic(run/'selection.json',dict(rule=config['selection'],selected_before_predictions=True,episodes=selection),immutable=True)
        atomic(run/'training_source_audit.json',source_audit,immutable=True)
        atomic(run/'provenance.json',dict(checkpoint=provenance['manifest'],identity=provenance['identity'],
             scope='stored expert input diagnostic, not new task success evaluation',numerics=dict(dtype='float32',
             deterministic=True,matmul_tf32=False,cudnn_tf32=False,device=args.device)),immutable=True)
        shutil.copy2(__file__,run/'diagnostic_source.py')
        source={}
        modules=[direct_train,inspect.getmodule(DirectDataset),inspect.getmodule(model.__class__),
                 inspect.getmodule(build_graph),inspect.getmodule(pack_graphs),inspect.getmodule(infer)]
        for module in modules:
            path=Path(inspect.getfile(module));target=run/'source_snapshot'/path.name;target.parent.mkdir(exist_ok=True)
            shutil.copy2(path,target);source[str(path)]=sha(path)
        atomic(run/'source_hashes.json',source,immutable=True)
        save_npz(run/'normalizer.npz',mean=array(norm.stats.mean),std=array(norm.stats.std),
                 fitted_episode_ids=np.asarray(norm.fitted_episode_ids,np.int64))
        weights_before=digest(*(array(p) for p in model.parameters()))
        stats_before=digest(array(norm.stats.mean),array(norm.stats.std))
        results=[]
        for spec in selection:
            print('EXPERT',spec,flush=True)
            report=write_episode(run/f"{spec['split']}_{spec['episode_id']}",spec,dataset,model,norm,train_config,config)
            results.append(report)
        rollout=replay_rollout(run/'rollout_41005',args.rollout_episode,model,norm,train_config['graph'],config) if args.rollout_episode else None
        passed=all(c['pass_tolerance'] for r in results for c in r['path_checks'].values())
        if rollout:passed=passed and rollout['action']['pass_tolerance'] and rollout['hidden']['pass_tolerance']
        assert weights_before==digest(*(array(p) for p in model.parameters()))
        assert stats_before==digest(array(norm.stats.mean),array(norm.stats.std))
        summary=dict(status='PATHS_MATCH_WITHIN_FIXED_FP32_TOLERANCE' if passed else 'PATH_MISMATCH_INVESTIGATE_CONNECTION_FIRST',
            run_id=logger.run_id,checkpoint_sha256=config['checkpoint_sha256'],atol=config['atol'],rtol=config['rtol'],
            selection=selection,episodes=results,recorded_rollout_replay=rollout,weights_and_normalizer_unchanged=True,
            limits=['four preselected train/validation episodes, not population-wide accuracy',
                    'no predicted action was applied to an environment; grasp success is not inferred',
                    'upstream live Vision accuracy is not validated by a stored-input comparison'])
        write_json(run/'summary.json',summary)
        sections=[]
        for spec in selection:
            name=f"{spec['split']}_{spec['episode_id']}"
            sections.append(f'<h2>{name}</h2><a href="{name}/frames.csv">frames.csv</a> · '
                f'<a href="{name}/predictions.npz">predictions.npz</a><br><img width="900" src="{name}/actions.png">'
                f'<img width="900" src="{name}/close_frames.png">')
        (run/'grasp_diagnostic.html').write_text('<!doctype html><meta charset="utf-8"><title>Fixed checkpoint grasp diagnostic</title>'
            '<style>body{font:15px system-ui;max-width:1000px;margin:35px auto}pre{white-space:pre-wrap}img{max-width:100%}</style>'
            '<h1>Stored Expert Grasp Diagnostic</h1><p>No training, no new Vision, no simulator steps. Fixed checkpoint.</p>'
            '<a href="summary.json">summary.json</a> · <a href="selection.json">selection.json</a><pre>'+
            html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(sections))
        for r in results:
            for phase in PHASES:
                logger.log(r['phase_metrics'][phase]['all_frames'],prefix=f"{r['split']}/{r['episode_id']}/{phase}")
        logger.record(source_run_id=provenance['manifest']['source_run_id'],diagnostic_status=summary['status'],
                      checkpoint_sha256=config['checkpoint_sha256'])
        for path in sorted(run.rglob('*')):
            if path.is_file() and 'source_snapshot' not in path.parts:
                logger.link_output(path,kind='trajectory' if path.suffix=='.npz' else 'reports')
        print('RESULT='+summary['status'],flush=True)


if __name__=='__main__':
    main()
