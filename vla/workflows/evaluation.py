"""Standalone offline evaluation through the existing cache and action codec."""
import argparse
from pathlib import Path
import torch

from utils.experiment_logger import ExperimentLogger
from vla.data.files import read, write, sha, checkpoint_load
from vla.data.graph_cache import CachedDataset
from vla.integration.live_policy import normalizer
from vla.workflows.components import components_from_config
from vla.workflows.training import offline
from vla.action_codec import ActionCodec


def evaluate(root, checkpoint, *, split='validation', device='cuda'):
    root = Path(root).resolve()
    config = read(root/'config.json')
    if split not in ('train', 'validation', 'test'):
        raise ValueError('unknown evaluation split')
    if split == 'test' and not (root/'model_selection_frozen.json').exists():
        raise ValueError('test gate locked')
    cp = checkpoint_load(checkpoint)
    if cp['config_sha256'] != sha(root/'config.json'):
        raise ValueError('checkpoint/config identity mismatch')
    if cp['normalizer'] != read(root/'normalizer.json') or cp['codec'] != read(root/'action_codec.json'):
        raise ValueError('checkpoint normalization identity mismatch')
    components = components_from_config(config)
    components.validate_checkpoint(cp)
    norm = normalizer(root)
    model = components.model(norm.schema, cp['variant'], config['model']).to(device)
    model.load_state_dict(cp['model'], strict=True)
    dataset = CachedDataset(root, allow_test=split == 'test')
    if cp['cache_manifest_sha256'] != sha(root/'cache_manifest.json'):
        raise ValueError('checkpoint/cache identity mismatch')
    with ExperimentLogger(team='integration', model='modular_live_offline_'+cp['variant'],
            seed=config['seed'], job_type='evaluate', root=root/'evaluation_runs',
            config={'split': split, 'checkpoint_sha256': sha(checkpoint),
                    'components': components.identity()}) as logger:
        logger.link_output(checkpoint, checkpoint=True)
        folder = logger.directory/'predictions'
        folder.mkdir()
        result = offline(dataset, model, norm, ActionCodec(cp['codec']),
                         config['splits'][split], config, device, output=folder)
        logger.log(result)
        path = logger.directory/'offline_summary.json'
        write(path, {'split': split, 'checkpoint_sha256': sha(checkpoint), **result})
        logger.link_output(path, kind='metrics')
        for output in list(folder.iterdir()):
            logger.link_output(output)
        return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='validation')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(evaluate(args.root, args.checkpoint, split=args.split, device=args.device))


if __name__ == '__main__':
    main()
