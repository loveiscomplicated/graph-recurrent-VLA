"""Fine-tune only the existing Object21 regressor until validation loss flattens."""
import argparse
import json
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from evaluate_tracker import get_masks
from graph.gated_mpnn import VertexEdgeGatedMPNN
from train_gt20 import describe, error_metrics, predict
from train_tracker import VAL, TEST, read_npz
from training.detected_corner_gnn import run_epoch
from utils.experiment_logger import ExperimentLogger, write

LR = (.0003, .0001, .00003)
MIN_RELATIVE_GAIN = .005
PATIENCE = 12
MAX_EPOCHS = 240


def save(path, payload):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def prepare_tensors(data, device):
    arrays = {key: torch.as_tensor(data[key], device=device) for key in ('xyz', 'observed', 'target', 'tcp')}
    arrays['kinds'] = torch.tensor([0] * 8 + [1] * 12, device=device).expand(len(data['xyz']), 20)
    arrays['valid'] = torch.ones_like(arrays['observed'])
    return arrays


def train(a, data, arrays, initial, config, logger):
    device = arrays['xyz'].device
    model = VertexEdgeGatedMPNN(**config['gnn']).to(device)
    model.load_state_dict(initial['model'], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR[0],
                                  weight_decay=config['training']['weight_decay'])
    shuffle = np.random.default_rng(0)
    train_idx = torch.as_tensor(np.flatnonzero(data['seed'] < VAL), device=device)
    val_idx = torch.as_tensor(np.flatnonzero((VAL <= data['seed']) & (data['seed'] < TEST)), device=device)
    rollout = a.output
    best_path = rollout / 'best.pt'
    last_path = rollout / 'resume.pt'
    initial_val = run_epoch(model, arrays, val_idx, config, None, 0, logger, rollout)
    best_loss = threshold_loss = initial_val['loss']
    best_epoch = last_gain = lr_index = 0
    history = [dict(epoch=0, lr=LR[0], validation=initial_val)]
    lr_events = []
    stop = 'epoch_limit'

    def weights(epoch):
        return dict(model=model.state_dict(), epoch=epoch, config=config, run_id=logger.run_id)

    def persist(epoch):
        save(last_path, dict(**weights(epoch), optimizer=optimizer.state_dict(),
                             shuffle_state=shuffle.bit_generator.state, cuda_rng=torch.cuda.get_rng_state(),
                             torch_rng=torch.get_rng_state(), history=history, lr_events=lr_events,
                             lr_index=lr_index, last_gain=last_gain, threshold_loss=threshold_loss,
                             best_loss=best_loss, best_epoch=best_epoch))

    save(best_path, weights(0))
    persist(0)
    print(f"Baseline validation loss {best_loss:.8f}", flush=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        started = monotonic()
        order = torch.as_tensor(shuffle.permutation(len(train_idx)), device=device)
        train_metrics = run_epoch(model, arrays, train_idx[order], config, optimizer, epoch, logger, rollout)
        validation = run_epoch(model, arrays, val_idx, config, None, epoch, logger, rollout)
        loss = validation['loss']
        history.append(dict(epoch=epoch, lr=LR[lr_index], train=train_metrics, validation=validation,
                            seconds=monotonic() - started))
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            save(best_path, weights(epoch))
        print(f"epoch {epoch} train {train_metrics['loss']:.8f} val {loss:.8f} "
              f"best {best_epoch}/{best_loss:.8f} lr {LR[lr_index]:.5g} "
              f"{history[-1]['seconds']:.1f}s", flush=True)
        if loss <= threshold_loss * (1 - MIN_RELATIVE_GAIN):
            threshold_loss, last_gain = loss, epoch
        if epoch - last_gain >= PATIENCE:
            if lr_index + 1 == len(LR):
                stop = 'plateau_at_lowest_lr'
                break
            lr_index += 1
            for group in optimizer.param_groups:
                group['lr'] = LR[lr_index]
            lr_events.append(dict(epoch=epoch, next_lr=LR[lr_index], best_val_loss=best_loss))
            print(f"Learning rate from epoch {epoch + 1}: {LR[lr_index]:.5g}", flush=True)
            threshold_loss, last_gain = best_loss, epoch
            persist(epoch)
        elif epoch % 5 == 0:
            persist(epoch)
    persist(epoch)
    write(rollout / 'history.json', history)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)['model'], strict=True)
    return model, dict(best_epoch=best_epoch, final_epoch=epoch, best_validation_loss=best_loss,
                       initial_validation_loss=initial_val['loss'], stop_reason=stop, lr_events=lr_events)


def heldout(a, data, arrays, trained, initial, config):
    keep = data['seed'] >= TEST
    index = torch.as_tensor(np.flatnonzero(keep), device=arrays['xyz'].device)
    split = {key: value[keep] for key, value in data.items()}
    prior = read_npz(a.prior_hybrid)
    for key in ('seed', 'step', 'target'):
        np.testing.assert_array_equal(split[key], prior[key])
    device = arrays['xyz'].device
    baseline = VertexEdgeGatedMPNN(**config['gnn']).to(device).eval().requires_grad_(False)
    baseline.load_state_dict(initial['model'], strict=True)
    trained.eval().requires_grad_(False)
    models = dict(gt_trained=baseline, hybrid_finetuned=trained)
    predictions = {k: predict(model, arrays, index, config['training']['batch_size'])
                   for k, model in models.items()}
    np.testing.assert_allclose(predictions['gt_trained'].numpy(), prior['hybrid'], atol=4e-5, rtol=0)
    groups = get_masks(split)
    gt = torch.as_tensor(split['target'])
    scores = {}
    for key, pred in predictions.items():
        error = error_metrics(pred, gt)
        scores[key] = {group: describe(error, selected) for group, selected in groups.items()}
    path = a.output / 'test_predictions.npz'
    np.savez_compressed(path, **{k: split[k] for k in ('seed', 'step', 'target')},
                        **{k: p.numpy() for k, p in predictions.items()})
    return scores


def main():
    parser = argparse.ArgumentParser()
    for name in ('cache', 'initial', 'prior-hybrid', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    a = parser.parse_args()
    start = monotonic()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    cache = a.cache / 'input.npz'
    source = json.loads((a.cache / 'summary.json').read_text())
    assert source['inputs_sha256'] == file_sha256(cache)
    data = read_npz(cache)
    assert len(data['seed']) == 19506 and data['xyz'].shape == (19506, 20, 3)
    initial = torch.load(a.initial, map_location='cpu', weights_only=True)
    assert file_sha256(a.initial).startswith('e2650277aafa189')
    config = {key: initial['config'][key] for key in ('gnn','loss')}
    config['training'] = dict(initial['config']['training'], epochs=MAX_EPOCHS)
    config.update(inputs_sha256=source['inputs_sha256'], cache_run_id=source['run_id'],
                  initial_sha256=file_sha256(a.initial), lr_schedule=LR, patience=PATIENCE,
                  min_relative_val_gain=MIN_RELATIVE_GAIN, script_sha256=file_sha256(__file__))
    arrays = prepare_tensors(data, device)
    a.output.mkdir(parents=True, exist_ok=False)
    with ExperimentLogger(team='graph', model='hybrid_input_gnn_finetune', job_type='train',
                          seed=0, config=config, root=a.output / 'experiments') as logger:
        logger.record(runtime=dict(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__)))
        model, stopping = train(a, data, arrays, initial, config, logger)
        result = heldout(a, data, arrays, model, initial, config)
        report = a.output / 'summary.json'
        write(report, dict(config=config, run_id=logger.run_id, **stopping, seconds=monotonic() - start,
                           best_sha256=file_sha256(a.output / 'best.pt'),
                           resume_sha256=file_sha256(a.output / 'resume.pt'), test=result))
        for key in ('best.pt', 'resume.pt', 'summary.json', 'history.json', 'test_predictions.npz'):
            logger.link_output(a.output / key, checkpoint=key.endswith('.pt'))
    print('Completed hybrid-input GNN fine-tuning', stopping, flush=True)


if __name__ == '__main__':
    main()
