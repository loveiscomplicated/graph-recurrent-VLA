"""Reuse v2's split, normalization, BC, optimizer, schedule and budget for B."""
from utils.experiment_logger import ExperimentLogger
import argparse
from pathlib import Path
from time import perf_counter
import torch
from ..common import config, seed_all, versions, write_json
from ..data import load_dataset, split_indices, sha256
from ..normalization import FeatureNormalizer
from ..models import parameter_count
from ..metrics import action_metrics
from ..v2.diagnostics import qvel_dataset
from ..v2.state import StateNormalizer
from ..v2.models import bc_loss, CommonPolicy
from .interaction_graph import InteractionGraph, policy
from .geometry_audit import resample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('outputs/graph/dataset.pt'))
    p.add_argument('--config', type=Path, default=Path(__file__).parents[1]/'v2/config.toml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--max-batches', type=int)
    p.add_argument('--radius', type=float, default=.05)
    p.add_argument('--cross-k', type=int, default=4)
    p.add_argument('--local-k', type=int, default=8)
    p.add_argument('--distance-threshold', type=float)
    p.add_argument('--variant', choices=['interaction', 'sampling', 'flat_sampling'], default='interaction')
    p.add_argument('--sampling-weights', type=float, nargs=4, default=(1, 3, 3, 1))
    p.add_argument('--entry-band', type=float, default=.02)
    args = p.parse_args()
    if args.epochs < 1 or (args.max_batches is not None and args.max_batches < 1): p.error('positive budget required')
    if (args.output/'best.pt').exists(): p.error('checkpoint exists; choose a fresh output')
    cfg = config(args.config)
    cfg['epochs'] = args.epochs
    with ExperimentLogger(team='graph', config=dict(parameters=cfg, command=vars(args)), seed=cfg['seed'], model=getattr(args, 'model', getattr(args, 'variant', None))) as experiment:
        experiment.record(dataset=str(Path(args.dataset).resolve()))
        seed_all(cfg['seed'], cfg['threads'])
        data, g = load_dataset(args.dataset); qvel = qvel_dataset(data)
        if args.variant != 'interaction': g = resample(g, count=cfg['hole_points'], weights=args.sampling_weights, seed=cfg['sampling_seed'], entry_band=args.entry_band)
        model_name = 'flat' if args.variant == 'flat_sampling' else 'surface'
        for key in ('peg_points', 'hole_points', 'sampling_seed', 'control_mode'):
            if cfg[key] != data['manifest']['config'][key]: raise ValueError(f'dataset mismatch: {key}')
        train, val = split_indices(data, 'train'), split_indices(data, 'validation')
        norm = FeatureNormalizer.fit(g.select(train)); sn = StateNormalizer.fit(g.select(train), qvel[train], 'goal')
        model = CommonPolicy('flat', cfg) if model_name == 'flat' else policy(cfg)
        experiment.record(model_class=type(model).__module__ + '.' + type(model).__qualname__)
        graph_config = dict(surface_k=args.local_k, cross_k=args.cross_k, radius=args.radius, distance_threshold=args.distance_threshold)
        builder = InteractionGraph(**graph_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
        generator = torch.Generator().manual_seed(cfg['seed'])
        args.output.mkdir(parents=True, exist_ok=True); history = []; best = float('inf'); steps = 0; start = perf_counter()
        dataset_hash = sha256(args.dataset)
        experiment.record(dataset_sha256=dataset_hash, environment=data['manifest']['config'])
        for epoch in range(1, cfg['epochs']+1):
            model.train(); predictions = []; targets = []; losses = []; es = perf_counter()
            order = train[torch.randperm(len(train), generator=generator)]
            for bi, ids in enumerate(order.split(cfg['batch_size'])):
                if args.max_batches is not None and bi >= args.max_batches: break
                inputs = norm.apply(model_name, g.select(ids), builder)
                shared = sn.apply(g.select(ids), qvel[ids], 'goal'); target = data['actions'][ids]
                optimizer.zero_grad(set_to_none=True); raw = model.raw_action(inputs, shared)
                loss = bc_loss(raw, target, 'regression')
                if not torch.isfinite(loss): raise FloatingPointError('BC loss')
                loss.backward(); optimizer.step(); steps += 1
                predictions.append(raw.detach()); targets.append(target); losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():
                vp = torch.cat([model(norm.apply(model_name, g.select(ids), builder), sn.apply(g.select(ids), qvel[ids], 'goal')) for ids in val.split(cfg['batch_size'])])
                vm = action_metrics(vp, data['actions'][val]); tm = action_metrics(torch.cat(predictions), torch.cat(targets))
            row = dict(epoch=epoch, train_mse=tm['action_mse'], val_mse=vm['action_mse'], train_metrics=tm,
                       validation_metrics=vm, optimization_loss=sum(losses)/len(losses), seconds=perf_counter()-es,
                       learning_rate=optimizer.param_groups[0]['lr'])
            history.append(row); scheduler.step()
            if row['val_mse'] < best:
                best = row['val_mse']
                torch.save(dict(schema_version=3, model_name=model_name, variant=args.variant, sampling_weights=args.sampling_weights,
                                entry_band=args.entry_band, config=cfg, graph_config=graph_config,
                                state_mode='goal', gripper_mode='regression', model_state=model.state_dict(),
                                normalizer=norm.as_dict(), state_normalizer=sn.as_dict(), epoch=epoch,
                                validation_mse=best, dataset_sha256=dataset_hash, splits=data['manifest']['splits'],
                                parameters=parameter_count(model), versions=versions(), smoke_max_batches=args.max_batches,
                                optimizer_state=optimizer.state_dict(),
                                scheduler_state=scheduler.state_dict()), args.output/'best.pt')
                experiment.link_output(args.output / 'best.pt')
            write_json(args.output/'history.json', history)
            experiment.log_epoch(row)
            print(f'{args.variant} epoch {epoch}: train={row["train_mse"]:.6f} val={best:.6f} {row["seconds"]:.1f}s', flush=True)
        write_json(args.output/'training.json', dict(config=cfg, variant=args.variant, sampling_weights=args.sampling_weights,
                   entry_band=args.entry_band, graph_config=graph_config, training_seconds=perf_counter()-start,
                   optimizer_steps=steps, parameters=parameter_count(model), dataset_sha256=dataset_hash,
                   splits=data['manifest']['splits'], best_validation_mse=best, smoke_max_batches=args.max_batches, versions=versions(), history=history))
        from ..visualization import plot_curves
        plot_curves(history, args.output/'loss.png')


if __name__ == '__main__': main()
