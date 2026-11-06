"""Replay each original Reach segment; trim at the first live predicate success."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import torch

from graph.prototype.common import config, seed_all, versions, write_json
from graph.prototype.data import load_dataset, sha256, state_at, split_indices
from graph.prototype.geometry import concatenate
from graph.prototype.maniskill_gt import GTAdapter
from graph.subtasks.dataset import load_segments
from graph.subtasks.env import make_subtask_env, tree_error
from .policy import ROOT, fit_inputs, inputs, raw_features, V2_CONFIG

SOURCE_ROOT = Path('outputs/subtasks/generated_50')


def prepare(source_root=SOURCE_ROOT, output=ROOT):
    if platform.system() != 'Linux':
        raise RuntimeError('benchmark replay requires the verified Linux stack')
    source_root, output = Path(source_root), Path(output)
    target = output/'data/reach_first_success.pt'
    if target.exists():
        raise FileExistsError(target)
    seed_all(0, 4)
    index_path = source_root/'segments/index.json'
    index = load_segments(index_path)
    original_path = source_root/'bc/reach.pt'
    old, geometry = load_dataset(original_path)
    if old['manifest']['splits'] != index['splits']:
        raise ValueError('export/index splits differ')
    cfg = config(V2_CONFIG)
    for key in ('peg_points','hole_points','sampling_seed','control_mode'):
        if cfg[key] != old['manifest']['config'][key]:
            raise ValueError(f'configuration differs: {key}')
    seeds = [e['segments'][0]['initial_state_ref']['episode_seed'] for e in index['episodes']]
    uids = list(old['manifest']['source_uids'].values())
    if len(set(seeds)) != len(seeds) or len(set(uids)) != len(uids):
        raise ValueError('duplicate reproduction/seed/source')
    hashes = {str(p): sha256(p) for p in (original_path, index_path, Path(index['source']), Path(index['initial_state_source']))}
    env = make_subtask_env('reach', index=index_path)
    adapter = GTAdapter(cfg['peg_points'],cfg['hole_points'],cfg['sampling_seed'])
    kept, qvels, live_geometries, episodes, records = [], [], [], [], []
    all_raw_error = all_geometry_error = all_obs_error = all_state_error = 0.
    try:
        for ep in index['episodes']:
            eid = ep['episode_id']
            seg = next(s for s in ep['segments'] if s['subtask']=='reach')
            i, j = seg['start_t'], seg['end_t']
            rows = torch.where(old['episode_id']==eid)[0]
            if not torch.equal(old['t'][rows],torch.arange(i,j)):
                raise ValueError('original export state/action indices differ')
            if not np.array_equal(old['actions'][rows].numpy(), np.asarray(env.source[f'traj_{eid}/actions'][i:j])):
                raise ValueError('converted action mismatch')
            obs, _ = env.reset(options={'source_episode_id':eid})
            first, flags, states, gs, qs, observations, metrics, motions = None, [], [], [], [], [], [], []
            for t in range(i,j+1):
                g = adapter(env)
                q = env.unwrapped.agent.robot.get_qvel().detach().cpu().clone()
                gs.append(g); qs.append(q); observations.append(obs.detach().cpu().clone())
                states.append({k:{kk:torch.as_tensor(vv).cpu().clone() for kk,vv in v.items()} for k,v in env.unwrapped.get_state_dict().items()})
                metrics.append(dict(env.metrics))
                all_obs_error = max(all_obs_error,tree_error(obs,env.group['obs'][t]))
                all_state_error = max(all_state_error,tree_error(env.unwrapped.get_state_dict(),state_at(env.group['env_states'],t)))
                if t < j:
                    offline = geometry.select(rows[t-i:t-i+1])
                    err = max(float((g.as_dict()[k]-offline.as_dict()[k]).abs().max()) for k in g.as_dict())
                    all_geometry_error = max(all_geometry_error,err)
                    np.testing.assert_allclose(q.numpy()[0],env.group['obs'][t,9:18],atol=2e-6,rtol=0)
                    np.testing.assert_allclose(g.qpos.numpy()[0],env.group['obs'][t,:9],atol=2e-6,rtol=0)
                    live_raw, off_raw = raw_features(g,q), raw_features(offline,q)
                    all_raw_error = max(all_raw_error,max(float((a-b).abs().max()) for a,b in zip(live_raw,off_raw)))
                success = env.subtask_spec.check_success(env.metrics,env.context)
                flags.append(success)
                if success and first is None:
                    first = t
                if t < j:
                    before = g.ee_pose.clone()
                    if first is None:
                        obs, _, terminated, truncated, info = env.step(env.audit['actions'][t])
                        if (terminated or truncated) and not info['subtask_success']:
                            raise ValueError(f'expert failed before success: {eid} {info["failure_reason"]}')
                    else:
                        # Diagnostic tail only, outside the independent wrapper clock.
                        obs, *_ = env.unwrapped.step(env.audit['actions'][t])
                        env._update(t+1,True)
                    motions.append(float((adapter(env).ee_pose[:,:3]-before[:,:3]).norm()))
            if first is None or not i < first <= j:
                raise ValueError(f'no valid first-success index in episode {eid}')
            n = first-i
            kept.extend(rows[:n].tolist()); qvels.extend(qs[:n]); live_geometries.extend(gs[:n])
            episodes.append(dict(episode_id=eid,seed=seeds[eid],split=ep['split'],start_t=i,end_t=first,
                                 states=states[:n+1],observations=torch.cat(observations[:n+1]),
                                 geometry=concatenate(gs[:n+1]).as_dict(),qvel=torch.cat(qs[:n+1]),
                                 actions=old['actions'][rows[:n]].clone()))
            tail = old['actions'][rows[n:]]
            pre = old['actions'][rows[:n]]
            def stats(a):
                return dict(count=len(a),mean=a.mean(0).tolist(),mean_ee_command_norm=float(a[:,:6].norm(dim=-1).mean()),
                            gripper_min=float(a[:,6].min()),gripper_max=float(a[:,6].max()),
                            non_open_commands=int((a[:,6]<.99).sum())) if len(a) else dict(count=0)
            records.append(dict(episode_id=eid,seed=seeds[eid],split=ep['split'],original_actions=j-i,
                                first_success_t=first,retained_actions=n,removed_actions=j-first,
                                success_exits_in_saved_segment=sum(a and not b for a,b in zip(flags,flags[1:])),
                                success_states_in_saved_segment=sum(flags),endpoint_success=flags[-1],
                                before_success=stats(pre),after_success=stats(tail),
                                mean_realized_translation_before_m=float(np.mean(motions[:n])),
                                mean_realized_translation_after_m=float(np.mean(motions[n:])) if motions[n:] else None,
                                metrics_at_first_success=metrics[n],metrics_at_end=metrics[-1]))
            print(f'audit ep={eid} segment={j-i} first={n} removed={j-first}',flush=True)
        backend = str(env.unwrapped.backend)
    finally:
        env.close()
    if max(all_obs_error,all_state_error) > .001 or max(all_raw_error,all_geometry_error) > 2e-6:
        raise ValueError('replay/live features disagree; do not train')
    ids = torch.tensor(kept)
    data = dict(geometry=geometry.select(ids).as_dict(),actions=old['actions'][ids].clone(),
                episode_id=old['episode_id'][ids].clone(),t=old['t'][ids].clone(),
                qvel=torch.cat(qvels),episodes=episodes,
                manifest=dict(old['manifest'],source_dataset_sha256=sha256(original_path),
                              source_index=str(index_path.resolve()),source=index['source'],
                              trimming='actions [start_t, first live Reach success), states include terminal success',
                              architecture='unchanged graph.prototype.v2.models.CommonPolicy flat/goal/regression',
                              input_exclusions=['expert action','planner phase','future waypoint','success','contact','reward']))
    norm, sn = fit_inputs(data,geometry.select(ids))
    live = concatenate(live_geometries)
    a, b = inputs(live,data['qvel'],norm,sn), inputs(geometry.select(ids),data['qvel'],norm,sn)
    normalized_error = max(float((x-y).abs().max()) for x,y in zip(a,b))
    if normalized_error > .002:
        raise ValueError('normalized live/offline feature mismatch')
    for p,h in hashes.items():
        if sha256(p) != h:
            raise ValueError('original source modified')
    target.parent.mkdir(parents=True,exist_ok=True)
    torch.save(data,target)
    report = dict(records=records,source_hashes=hashes,source_preserved=True,split_preserved=True,
                  episodes=len(episodes),original_samples=len(old['actions']),retained_samples=len(ids),removed_samples=len(old['actions'])-len(ids),
                  split_samples={k:len(split_indices(data,k)) for k in index['splits']},
                  maximum_geometry_error=all_geometry_error,maximum_raw_feature_error=all_raw_error,
                  maximum_normalized_feature_error=normalized_error,maximum_observation_error=all_obs_error,
                  maximum_simulator_state_error=all_state_error,flat_dimension=a[0].shape[1],shared_dimension=a[1].shape[1],
                  normalization_train_episode_ids=index['splits']['train'],versions=versions(),backend=backend,
                  dataset_sha256=sha256(target),mean_original_actions=float(np.mean([r['original_actions'] for r in records])),
                  mean_retained_actions=float(np.mean([r['retained_actions'] for r in records])))
    write_json(output/'reports/trimming_audit.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=SOURCE_ROOT)
    p.add_argument('--output',type=Path,default=ROOT)
    a=p.parse_args();r=prepare(a.source_root,a.output)
    print({k:v for k,v in r.items() if k not in ('records','source_hashes')})
