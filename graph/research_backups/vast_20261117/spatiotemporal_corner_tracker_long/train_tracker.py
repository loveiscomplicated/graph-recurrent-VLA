"""Train paired temporal and spatial-temporal trackers on ID-matched detections."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from torch.nn import functional as F

from data_collection.episode_dataset import file_sha256
from utils.experiment_logger import ExperimentLogger, write
from tracker import BODY, CornerTransformer

FIRST, VAL, TEST, END = 3100000, 3100120, 3100144, 3100176
MODES = {'temporal': (False, 0.), 'spatial_temporal': (True, 0.), 'rigid_loss': (True, .25)}


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def read_run(path, kind, checksum):
    summary = json.loads((path / 'summary.json').read_text())
    metadata = json.loads((path / 'experiments' / summary['run_id'] / 'metadata.json').read_text())
    assert metadata['status'] == 'finished' and metadata['git_commit'].startswith('b2e8e4c')
    assert not metadata['git_dirty']
    file = path / kind
    digest = file_sha256(file)
    assert summary[checksum] == digest
    return read_npz(file), {'run_id': summary['run_id'], 'sha256': digest}


def concat(parts, fields):
    return {key: np.concatenate([part[key] for part in parts]) for key in fields}


def read_data(reference_root, capture_root, test_root):
    refs, captures, tests, links = [], [], [], []
    for start in range(FIRST, END, 44):
        data, source = read_run(reference_root / f'data_{start}_{start + 44}', 'dataset.npz', 'dataset_sha256')
        refs.append(data)
        links.append(source)
    for start in range(FIRST, TEST, 24):
        data, source = read_run(capture_root / f'shard_{start}_{start + 24}', 'detections.npz', 'sha256')
        captures.append(data)
        links.append(source)
    for start in range(TEST, END, 8):
        data, source = read_run(test_root / f'shard_{start}_{start + 8}', 'matches.npz', 'matches_sha256')
        np.testing.assert_array_equal(data['xyz'].shape[1:], (20, 3))
        tests.append(dict(seed=data['seed'], step=data['step'], observed=data['detected'],
                          detected_xyz=np.where(data['detected'][..., None], data['tracked_xyz'], 0.),
                          visible=data['visible']))
        links.append(source)
    reference = concat(refs, ('seed', 'step', 'xyz', 'tcp', 'target', 'moving_now', 'phase', 'grasp'))
    capture = concat(captures + tests, ('seed', 'step', 'observed', 'detected_xyz', 'visible'))
    for field in ('seed', 'step'):
        np.testing.assert_array_equal(reference[field], capture[field])
    assert np.unique(reference['seed']).tolist() == list(range(FIRST, END))
    observed = capture['observed']
    assert np.all(~observed | capture['visible'])
    assert np.all(np.linalg.norm(reference['xyz'][observed] - capture['detected_xyz'][observed], axis=1) < .005002)
    reference.update(capture)
    return reference, links


def prepare(data, fallback, ids, device):
    n = len(ids)
    groups = [np.flatnonzero(data['seed'] == seed) for seed in ids]
    length = max(map(len, groups))
    features = np.zeros((n, length, 20, 10), dtype=np.float32)
    anchor = np.zeros((n, length, 20, 3), dtype=np.float32)
    detector, target = np.zeros_like(anchor), np.zeros_like(anchor)
    observed, ever, valid = (np.zeros((n, length, 20), bool), np.zeros((n, length, 20), bool),
                             np.zeros((n, length), bool))
    for i, ix in enumerate(groups):
        positions = data['detected_xyz'][ix]
        mask = data['observed'][ix]
        frame = np.arange(len(ix))[:, None]
        last = np.maximum.accumulate(np.where(mask, frame, -1), axis=0)
        seen = last >= 0
        last_xyz = positions[np.maximum(last, 0), np.arange(20)]
        prior = np.where(seen[..., None], last_xyz, fallback[None])
        end = len(ix)
        center = (prior - fallback[None]) / .1
        relative_tcp = (data['tcp'][ix, None, :] - prior) / .1
        features[i, :end] = np.concatenate((center, relative_tcp, mask[..., None], seen[..., None],
                                            (np.minimum(frame - last, 40) / 40)[..., None],
                                            np.broadcast_to(frame[..., None] / 200, (end, 20, 1))), axis=-1)
        anchor[i, :end], detector[i, :end], target[i, :end] = prior, positions, data['xyz'][ix]
        observed[i, :end], ever[i, :end], valid[i, :end] = mask, seen, True
    return {key: torch.as_tensor(value, device=device) for key, value in
            dict(features=features, anchor=anchor, detected=detector, gt=target,
                 observed=observed, ever=ever, valid=valid).items()}


def model_args(batch):
    return (batch[k] for k in ('features', 'anchor', 'detected', 'observed', 'valid'))


def batches(data, indices, size):
    for take in indices.split(size):
        length = int(data['valid'][take].sum(dim=1).max())
        yield {key: value[take, :length] for key, value in data.items()}


def rigid_loss(pred, batch, edges, missing):
    u, v = edges
    mask = (missing[..., u] | missing[..., v]) & batch['valid'][..., None]
    pd = torch.linalg.vector_norm(pred[..., u, :] - pred[..., v, :], dim=-1)
    gt = torch.linalg.vector_norm(batch['gt'][..., u, :] - batch['gt'][..., v, :], dim=-1)
    return F.smooth_l1_loss(pd[mask] * 100, gt[mask] * 100)


def evaluate(model, data, size):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for batch in batches(data, torch.arange(len(data['gt']), device=data['gt'].device), size):
            pred = model(*model_args(batch))
            assert torch.isfinite(pred).all()
            torch.testing.assert_close(pred[batch['observed']], batch['detected'][batch['observed']],
                                       rtol=0, atol=0)
            outputs.append(pred[batch['valid']].cpu().numpy())
    return np.concatenate(outputs)


def test_causality(model, val):
    batch = next(batches(val, torch.tensor([0], device=val['features'].device), 1))
    pivot = 12
    changed = {key: value.clone() for key, value in batch.items()}
    changed['features'][:, pivot:] += 1
    changed['anchor'][:, pivot:] += .2
    changed['detected'][:, pivot:] -= .2
    changed['observed'][:, pivot:] = ~changed['observed'][:, pivot:]
    old_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        with torch.inference_mode():
            full = model(*model_args(batch))
            perturbed = model(*model_args(changed))
            truncated = {key: tensor[:, :pivot] for key, tensor in batch.items()}
            prefix = model(*model_args(truncated))
        torch.testing.assert_close(full[:, :pivot], perturbed[:, :pivot], rtol=0, atol=1e-7)
        torch.testing.assert_close(full[:, :pivot], prefix, rtol=0, atol=3e-6)
    finally:
        torch.set_float32_matmul_precision(old_precision)


def train(mode, train, val, val_data, fallback, edges, args, config, device):
    spatial, rigidity = MODES[mode]
    torch.manual_seed(args.seed)
    model = CornerTransformer(spatial).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    folder = args.output / mode
    folder.mkdir()
    cfg = dict(config, spatial=spatial, rigidity_weight=rigidity,
               parameters=sum(p.numel() for p in model.parameters()))
    start = monotonic()
    with ExperimentLogger(team='graph', model=f'fixed_id_corner_{mode}', job_type='train',
                          config=cfg, seed=args.seed, root=folder / 'experiments') as logger:
        logger.record(runtime=dict(device=str(device), gpu=torch.cuda.get_device_name(), torch=str(torch.__version__)))
        best, best_epoch = float('inf'), 0
        path = folder / 'best.pt'
        history = []
        missing_val = ~val_data['observed']
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for batch in batches(train, torch.randperm(len(train['gt']), device=device, generator=rng), 6):
                missing = batch['valid'][..., None] & ~batch['observed']
                prediction = model(*model_args(batch))
                loss = F.smooth_l1_loss(prediction[missing] * 100, batch['gt'][missing] * 100)
                if rigidity:
                    loss = loss + rigidity * rigid_loss(prediction, batch, edges, missing)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
                losses.append(loss.detach())
            prediction = evaluate(model, val, 6)
            distances = np.linalg.norm(prediction - val_data['xyz'], axis=-1)[missing_val] * 1000
            mean = float(distances.mean())
            metrics = dict(train_loss=float(torch.stack(losses).mean()), val_missing_mean_mm=mean,
                           val_missing_p90_mm=float(np.percentile(distances, 90)))
            history.append(dict(epoch=epoch, **metrics))
            if mean < best:
                best, best_epoch = mean, epoch
                torch.save(dict(model=model.state_dict(), epoch=epoch, config=cfg, run_id=logger.run_id,
                                fallback=torch.from_numpy(fallback)), path)
            logger.log(metrics, step=epoch)
            print(mode, epoch, f'{mean:.2f} mm', f'best={best_epoch}', f'{monotonic()-start:.1f}s', flush=True)
            if epoch - best_epoch >= args.patience:
                break
        resume = folder / 'resume.pt'
        temporary = folder / 'resume.pt.tmp'
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), rng=rng.get_state(),
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
                        epoch=history[-1]['epoch'], best_epoch=best_epoch, best_val_missing_mean_mm=best,
                        history=history, config=cfg, run_id=logger.run_id,
                        fallback=torch.from_numpy(fallback)), temporary)
        temporary.replace(resume)
        model.load_state_dict(torch.load(path, weights_only=True, map_location=device)['model'])
        test_causality(model, val)
        write(folder / 'summary.json', dict(best_epoch=best_epoch, val_missing_mean_mm=best,
                                            history=history, seconds=monotonic() - start, run_id=logger.run_id,
                                            checkpoint_sha256=file_sha256(path), resume_sha256=file_sha256(resume),
                                            config=cfg))
        logger.link_output(path, checkpoint=True)
        logger.link_output(resume, checkpoint=True)
        logger.link_output(folder / 'summary.json')
    return model


def main():
    p = argparse.ArgumentParser()
    for field in ('reference', 'capture', 'test', 'output'):
        p.add_argument(f'--{field}', type=Path, required=True)
    p.add_argument('--mode', choices=tuple(MODES), action='append')
    p.add_argument('--epochs', type=int, default=36)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda:0')
    data, sources = read_data(args.reference, args.capture, args.test)
    ids = [range(FIRST, VAL), range(VAL, TEST), range(TEST, END)]
    train_frames = data['seed'] < VAL
    val_frames = (VAL <= data['seed']) & (data['seed'] < TEST)
    test_frames = data['seed'] >= TEST
    fallback = data['xyz'][train_frames].mean(axis=0)
    train_data, val_data, test_data = (prepare(data, fallback, split, device) for split in ids)
    slots = torch.as_tensor(BODY, device=device)
    edges = tuple((slots[:, None] == slots[None]).triu(1).nonzero(as_tuple=True))
    config = dict(sources=sources, splits=[list(split) for split in ids], epochs=args.epochs,
                  seed=args.seed, patience=args.patience, width=48, heads=4, batch_size=6, lr=3e-4,
                  tracker_sha256=file_sha256(Path(__file__).with_name('tracker.py')),
                  train_script_sha256=file_sha256(__file__))
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for mode in (args.mode or MODES):
        model = train(mode, train_data, val_data, {k: v[val_frames] for k, v in data.items()},
                      fallback, edges, args, config, device)
        outputs[mode] = evaluate(model, test_data, 6)
    arrays = {key: value[test_frames] for key, value in data.items()}
    arrays.update({key: test_data[key][test_data['valid']].cpu().numpy() for key in ('anchor', 'ever')})
    np.savez_compressed(args.output / 'test_predictions.npz', **arrays, **outputs)
    print('Completed all tracker models', flush=True)


if __name__ == '__main__':
    main()
