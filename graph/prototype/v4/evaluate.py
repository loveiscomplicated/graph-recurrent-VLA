"""Offline metrics, cache timings and live paired v2-compatible rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from ..common import seed_all, write_json, versions
from ..data import load_dataset, split_indices, sha256
from ..normalization import FeatureNormalizer
from ..graph_builder import GraphBuilder
from ..metrics import action_metrics
from ..geometry import rotation
from ..v2.state import StateNormalizer
from ..v2.goals import angle
from ..v2.diagnostics import ReplaySession, snapshot, stage_summary, qvel_dataset, THRESHOLDS
from .history_models import HistoryPolicy
from .history_dataset import HistoryDataset
from .history_buffer import HistoryBuffer


class Runtime:
    def __init__(self, model, norm, sn, builder, ablate=False):
        if model.training:
            raise ValueError('cache requires a fixed evaluation encoder')
        self.model, self.norm, self.sn, self.builder = model, norm, sn, builder
        self.buffer = HistoryBuffer(model.length)
        self.ablate = ablate
        self.timing = []

    def encode(self, g, qvel):
        start = perf_counter()
        inputs = self.norm.apply(self.model.name, g, self.builder)
        shared = self.sn.apply(g, qvel, 'goal')
        construction = (perf_counter()-start)*1000
        start = perf_counter()
        h = self.model.encode(inputs, shared)
        return h, construction, (perf_counter()-start)*1000

    def predict(self, g, qvel):
        h, construction, encoder = self.encode(g, qvel)
        start = perf_counter()
        states, actions, valid = self.buffer.window(h, self.model.use_state_history)
        if self.ablate:
            states = h[:, None].expand_as(states)
            actions = torch.zeros_like(actions)
            # Keep validity: replace past content, not the padding condition.
        processing = (perf_counter()-start)*1000
        start = perf_counter()
        raw = self.model.from_history(states, actions, valid)
        head = (perf_counter()-start)*1000
        self.timing.append([construction, encoder, processing, head])
        return raw, h, actions, valid


class HistoryReplaySession(ReplaySession):
    def __init__(self, data, runtime, horizon=250):
        super().__init__(data, horizon=horizon)
        self.runtime = runtime

    def prepare(self, ep, prefix=0):
        from mani_skill.utils.registration import TimeLimitWrapper
        super().prepare(ep, 0)
        self.runtime.buffer.reset()
        self.env = TimeLimitWrapper(self.env.unwrapped, self.horizon+prefix)
        error = 0.
        for t, action in enumerate(self.group(ep['episode_id'])['actions'][:prefix] if prefix else []):
            g = self.adapter(self.env)
            h, *_ = self.runtime.encode(g, self.env.unwrapped.agent.robot.get_qvel())
            raw = torch.tensor(action)[None]
            applied = raw.clamp(torch.tensor(self.env.action_space.low), torch.tensor(self.env.action_space.high))
            obs, *_ = self.env.step(applied)
            self.runtime.buffer.commit(h, applied)
            error = max(error, float(np.max(np.abs(obs.numpy()[0]-self.group(ep['episode_id'])['obs'][t+1]))))
        if error > .001:
            raise ValueError(f'prefix diverged {error}')
        g = self.adapter(self.env)
        fingerprint = hashlib.sha256(b''.join(v.numpy().tobytes() for v in g.as_dict().values())).hexdigest()
        return error, fingerprint


def run_episode(session, ep, prefix=0):
    error, fingerprint = session.prepare(ep, prefix)
    env, runtime = session.env, session.runtime
    initial = snapshot(env, session.adapter(env))
    trace = []
    for t in range(session.horizon):
        g = session.adapter(env)
        before = snapshot(env, g)
        raw, h, actions, valid = runtime.predict(g, torch.tensor(before['qvel'])[None])
        if raw.shape != (1, 7) or not torch.isfinite(raw).all():
            raise ValueError('invalid prediction')
        applied = raw.clamp(torch.tensor(env.action_space.low), torch.tensor(env.action_space.high))
        _, _, terminated, truncated, info = env.step(applied)
        # The only action entering the next learner window is this exact command.
        runtime.buffer.commit(h, applied)
        after_g = session.adapter(env)
        after = snapshot(env, after_g)
        trace.append(dict(t=t, absolute_t=prefix+t, before=before, after=after,
            prediction=raw[0].tolist(), applied_action=applied[0].tolist(),
            history_actions=actions[0].tolist(), history_valid=valid[0].tolist(),
            history_state_l2=[float(v.norm()) for v in runtime.buffer.states],
            realized_ee_translation_m=(after_g.ee_pose[0,:3]-g.ee_pose[0,:3]).tolist(),
            realized_ee_rotation_rad=float(angle(rotation(g.ee_pose[:,3:]).transpose(-1,-2) @ rotation(after_g.ee_pose[:,3:]))[0]),
            success=bool(info['success'].item()), terminated=bool(terminated.item()),
            truncated=bool(truncated.item()), elapsed_steps=int(env.unwrapped.elapsed_steps.item())))
        if trace[-1]['success'] or trace[-1]['terminated'] or trace[-1]['truncated']:
            break
    commands = torch.tensor([r['applied_action'] for r in trace])
    delta = commands[1:]-commands[:-1]
    return dict(episode_id=ep['episode_id'], seed=ep['episode_seed'], prefix=prefix,
        initial_geometry_sha256=fingerprint, prefix_observation_max_error=error,
        prefix_history_steps=prefix, initial=initial, trace=trace, steps=len(trace),
        final=trace[-1]['after'], action_delta_rms=float(delta.square().mean().sqrt()) if len(delta) else 0.,
        gripper_switches=int(((commands[1:,6]>0)!=(commands[:-1,6]>0)).sum()),
        **stage_summary(trace, initial))


def load(path, data, dataset):
    ck = torch.load(path, weights_only=True)
    if ck['dataset_sha256'] != sha256(dataset) or ck['splits'] != data['manifest']['splits']:
        raise ValueError('checkpoint dataset/split mismatch')
    model = HistoryPolicy(ck['model_name'], ck['config'], ck['history_mode'], ck['history_length'])
    model.load_state_dict(ck['model_state']); model.eval()
    return ck, model, FeatureNormalizer(**ck['normalizer']), StateNormalizer(**ck['state_normalizer']), GraphBuilder(ck['config']['surface_k'], ck['config']['interaction_k'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=Path('outputs/graph/dataset.pt'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rollouts', action='store_true')
    p.add_argument('--stages', nargs='+', choices=['full','grasp','transport','alignment'], default=['full','grasp','transport','alignment'])
    p.add_argument('--new-seeds', nargs='+', type=int)
    p.add_argument('--episode-ids', nargs='+', type=int)
    p.add_argument('--horizon', type=int, default=250)
    p.add_argument('--latency-repeats', type=int, default=50)
    p.add_argument('--ablate-history', action='store_true')
    args = p.parse_args()
    if args.horizon < 1 or args.latency_repeats < 1: p.error('positive budgets required')
    if args.output.exists(): p.error('choose a fresh output file')
    data, geometry = load_dataset(args.dataset)
    ck, model, norm, sn, builder = load(args.checkpoint, data, args.dataset)
    seed_all(ck['config']['seed'], ck['config']['threads'])
    qvel = qvel_dataset(data); windows = HistoryDataset(data, model.length)
    test = split_indices(data, 'test')
    report = dict(checkpoint=str(args.checkpoint), config=ck['config'], history=model.history,
        history_length=model.length, model=model.name, parameters=ck['parameters'], epoch=ck['epoch'],
        dataset_sha256=ck['dataset_sha256'], splits=ck['splits'], versions=versions(),
        ablate_history=args.ablate_history, settings={}, horizon=args.horizon, thresholds=THRESHOLDS,
        torch_device='cpu', physics_backend='physx_cpu' if args.rollouts else None)
    with torch.inference_mode():
        predictions = torch.cat([windows.forward(model, ids, geometry, qvel, norm, sn, builder) for ids in test.split(ck['config']['batch_size'])])
        report['offline'] = action_metrics(predictions, data['actions'][test])
        if model.history != 'current':
            ablated = []
            for ids in test.split(ck['config']['batch_size']):
                g = geometry.select(ids)
                h = model.encode(norm.apply(model.name, g, builder), sn.apply(g, qvel[ids], 'goal'))
                _, actions, valid = windows.window(ids, model.use_state_history)
                ablated.append(model.from_history(h[:,None].expand(-1,model.state_count,-1), torch.zeros_like(actions), valid))
            report['offline_repeated_current_zero_action'] = action_metrics(torch.cat(ablated), data['actions'][test])
        runtime = Runtime(model, norm, sn, builder, args.ablate_history)
        for i in range(args.latency_repeats+10):
            ids = test[i%len(test):i%len(test)+1]; g = geometry.select(ids)
            _, h, *_ = runtime.predict(g, qvel[ids]); runtime.buffer.commit(h, data['actions'][ids])
        means = np.mean(runtime.timing[10:], axis=0)
        report['latency'] = dict(zip(['feature_construction_ms','encoder_ms','history_processing_ms','action_head_ms'], means.tolist()))
        report['latency'].update(total_ms=float(means.sum()), cache='one current encoding per step, fixed eval encoder', threads=ck['config']['threads'], repeats=args.latency_repeats, excludes='GT extraction and simulator stepping')
        write_json(args.output, report)
        if args.rollouts:
            episodes = [e for e in data['manifest']['episodes'] if e['episode_id'] in ck['splits']['test']]
            if args.episode_ids:
                if not set(args.episode_ids).issubset(ck['splits']['test']): p.error('held-out IDs required')
                episodes = [e for e in episodes if e['episode_id'] in args.episode_ids]
            if args.new_seeds:
                used = {e['episode_seed'] for e in data['manifest']['episodes']}
                if args.stages != ['full'] or len(set(args.new_seeds)) != len(args.new_seeds) or used & set(args.new_seeds) or min(args.new_seeds)<0:
                    p.error('fresh seeds require full stage, unique nonnegative seeds absent from dataset')
                episodes = [dict(episode_id=-s, episode_seed=s, reset_kwargs={'seed':s}, new_configuration=True) for s in args.new_seeds]
            starts = json.loads(Path('outputs/graph/v2/diagnostics/expert.json').read_text())['starts']
            session = HistoryReplaySession(data, runtime, args.horizon)
            try:
                for stage in args.stages:
                    records, skipped = [], []
                    for ep in episodes:
                        eid = ep['episode_id']; prefix = 0
                        if stage == 'alignment':
                            rows = torch.where((data['episode_id']==eid)&(data['phase']==3))[0]
                            if not len(rows): rows = torch.where((data['episode_id']==eid)&(data['phase']==4))[0]
                            if not len(rows): skipped.append(eid); continue
                            prefix = int(data['timestep'][rows[0]])
                        elif stage != 'full':
                            prefix = starts[str(eid)].get(stage)
                            if prefix is None: skipped.append(eid); continue
                        r = run_episode(session, ep, prefix); records.append(r)
                        print(f'{model.name}/{model.history} {stage} ep{eid}: success={r["success"]} failure={r["failure_stage"]}', flush=True)
                    report['settings'][stage] = dict(episodes=records, denominator=len(records), skipped_episode_ids=skipped,
                        successes=sum(r['success'] for r in records), stage_counts={k:sum(r[k] for r in records) for k in ['approach','acquisition','transport','alignment','insertion']},
                        failure_counts={k:sum(r['failure_stage']==k for r in records) for k in ['acquisition','retention','transport','alignment','insertion']})
                    write_json(args.output, report)
            finally: session.close()
    print(json.dumps({k:report[k] for k in ['model','history','offline','latency']}, indent=2))


if __name__ == '__main__': main()
