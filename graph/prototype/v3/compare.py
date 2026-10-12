"""Held-out offline and closed-loop comparison; no test-set selection."""
import argparse
from pathlib import Path
import torch
from ..common import seed_all, write_json, versions
from ..data import load_dataset, split_indices, sha256
from ..metrics import action_metrics
from ..normalization import FeatureNormalizer
from ..v2.state import StateNormalizer
from ..v2.diagnostics import load_predictor, qvel_dataset, ReplaySession, run_episode
from .interaction_graph import InteractionGraph, policy
from .geometry_audit import resample
from ..v2.models import CommonPolicy


def load_candidate(path, data, dataset_path):
    ck = torch.load(path, weights_only=True)
    if ck['schema_version'] != 3 or ck['splits'] != data['manifest']['splits'] or ck['dataset_sha256'] != sha256(dataset_path):
        raise ValueError('checkpoint/dataset/split mismatch')
    model = CommonPolicy('flat', ck['config']) if ck['model_name']=='flat' else policy(ck['config'])
    model.load_state_dict(ck['model_state']); model.eval()
    norm = FeatureNormalizer(**ck['normalizer']); sn = StateNormalizer(**ck['state_normalizer'])
    builder = InteractionGraph(**ck['graph_config'])
    def predict(g, qvel):
        if ck.get('variant', 'interaction') != 'interaction':
            g = resample(g, count=ck['config']['hole_points'], seed=ck['config']['sampling_seed'], weights=ck['sampling_weights'], entry_band=ck['entry_band'])
        return model(norm.apply(ck['model_name'], g, builder), sn.apply(g, qvel, 'goal'))
    return predict, ck


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', type=Path, default=Path('outputs/graph/dataset.pt'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rollouts', action='store_true')
    args = p.parse_args(); seed_all(0, 4)
    data, g = load_dataset(args.dataset); qvel = qvel_dataset(data)
    candidate, ck = load_candidate(args.checkpoint, data, args.dataset)
    baseline_path = f'outputs/graph/v2/training/{ck["model_name"]}_goal/best.pt'
    original, old_ck, *_ = load_predictor(baseline_path, data, args.dataset)
    if ck.get('smoke_max_batches') is not None or ck['config']['epochs'] != old_ck['config']['epochs']:
        p.error('comparison requires a full equal-budget checkpoint; smoke checkpoints are execution checks only')
    report = dict(versions=versions(), backend='physx_cpu' if args.rollouts else None, torch_device='cpu',
                  dataset_sha256=sha256(args.dataset), splits=data['manifest']['splits'], baseline_checkpoint=baseline_path,
                  candidate_checkpoint=str(args.checkpoint), models={})
    ids = split_indices(data, 'test')
    episodes = [e for e in data['manifest']['episodes'] if e['episode_id'] in data['manifest']['splits']['test']]
    with torch.inference_mode():
        for name, predict, checkpoint in [('original', original, old_ck), (ck.get('variant', 'interaction'), candidate, ck)]:
            predictions = torch.cat([predict(g.select(batch), qvel[batch]) for batch in ids.split(32)])
            result = dict(offline=action_metrics(predictions, data['actions'][ids]), parameters=checkpoint['parameters'],
                          checkpoint_epoch=checkpoint['epoch'], settings={}, training_budget=checkpoint['config']['epochs'])
            report['models'][name] = result
            if args.rollouts:
                session = ReplaySession(data, horizon=250)
                try:
                    for stage in ('full', 'alignment'):
                        records = []
                        for ep in episodes:
                            prefix = 0
                            if stage == 'alignment':
                                rows = torch.where((data['episode_id'] == ep['episode_id']) & (data['phase'] == 3))[0]
                                if not len(rows): rows = torch.where((data['episode_id'] == ep['episode_id']) & (data['phase'] == 4))[0]
                                prefix = int(data['timestep'][rows[0]])
                            record = run_episode(session, ep, predict, prefix)
                            records.append(record)
                            print(f'{name} {stage} ep{ep["episode_id"]}: success={record["success"]}', flush=True)
                        result['settings'][stage] = dict(successes=sum(r['success'] for r in records), denominator=len(records), episodes=records)
                        write_json(args.output, report)
                finally: session.close()
            write_json(args.output, report)


if __name__ == '__main__': main()
