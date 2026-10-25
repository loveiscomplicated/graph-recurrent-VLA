"""Read-only, checksummed transfer bundle for direct_bc_v1 evaluation.

No optimizer/RNG is restored and no unrestricted pickle load is permitted.
NumPy's exact legacy RNG types are the only additions to weights_only loading.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from uuid import UUID

import numpy as np
import torch

from graph.integration_baseline.direct_runtime import atomic, read, sha
from graph.integration_baseline.model import IntegrationPolicy
from graph.integration_baseline.robot import RobotNormalizer
from graph.prototype.v2.state import StateNormalizer

ROOT = Path(__file__).resolve().parents[2]
TRANSFER_FILES = dict(checkpoint=None, checkpoint_receipt=None, receipt='training/best.json', config='config.yaml',
                      robot_schema='robot_state_schema.json', action_schema='expert_action_schema.json',
                      dataset='dataset_manifest.json', split='split_manifest.json',
                      vision='vision_manifest.json', source='source_hashes.json',
                      vision_runtime='vision_runtime.json', conditions='executed_conditions.json')
SOURCE_FILES = (
    'graph/integration_baseline/model.py', 'graph/integration_baseline/graph.py',
    'graph/integration_baseline/robot.py', 'graph/integration_baseline/vision.py',
    'graph/vision_input.py', 'graph/prototype/models.py', 'graph/prototype/v2/state.py',
    'recurrent/models.py',
)


def required_sources():
    return sorted(set(SOURCE_FILES) | {str(p.relative_to(ROOT)) for p in
        (ROOT/'visual/runtime').rglob('*.py')})


def vision_runtime_identity():
    """Local cached weights and installed SAM2 source, without a network request."""
    import sam2
    from huggingface_hub import snapshot_download
    from visual.runtime.tracking import DEFAULT_CHECKPOINT
    dino = Path(snapshot_download('IDEA-Research/grounding-dino-tiny', local_files_only=True))
    names = ('added_tokens.json', 'config.json', 'model.safetensors', 'preprocessor_config.json',
             'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt')
    files = {name: sha(dino/name) for name in names}
    sam_root = Path(sam2.__file__).parent
    return dict(dino_revision=dino.name, dino_files=files, sam2_checkpoint_sha256=sha(DEFAULT_CHECKPOINT),
                sam2_source={str(p.relative_to(sam_root)): sha(p) for p in sorted(sam_root.rglob('*.py'))})


def _file(root, relative):
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('missing/unsafe transfer file: '+relative)
    if any('.partial' in part for part in path.parts):
        raise ValueError('partial checkpoint/bundle is not valid')
    return path


def make_manifest(bundle, run_id, output):
    """Run on an already received immutable bundle, never an active run pointer."""
    if UUID(run_id).hex != run_id:
        raise ValueError('original run UUID hex required')
    bundle = Path(bundle).resolve()
    paths = dict(TRANSFER_FILES)
    paths['checkpoint'] = read(_file(bundle, paths['receipt']))['path']
    paths['checkpoint_receipt'] = str(Path(paths['checkpoint']).with_suffix('.json'))
    result = dict(version='stop_clock_checkpoint_v1', source_run_id=run_id,
                  files={key: dict(path=str(_file(bundle, value).relative_to(Path(output).resolve().parent)),
                                   sha256=sha(_file(bundle, value))) for key, value in paths.items()})
    atomic(output, result, immutable=True)
    return result


def load_bundle(manifest_path, *, device='cpu', eval_config=None, check_runtime=True):
    manifest_path = Path(manifest_path).resolve()
    manifest = read(manifest_path)
    if manifest['version'] != 'stop_clock_checkpoint_v1' or UUID(manifest['source_run_id']).hex != manifest['source_run_id']:
        raise ValueError('checkpoint manifest version/run UUID mismatch')
    if set(manifest['files']) != set(TRANSFER_FILES):
        raise ValueError('incomplete checkpoint transfer metadata')
    # Load the verified bytes, not the file again after a possible pointer change.
    content = {}
    for name, item in manifest['files'].items():
        data = _file(manifest_path.parent, item['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('transfer SHA mismatch: '+name)
        content[name] = data
    meta = {k: json.loads(v) for k, v in content.items() if k != 'checkpoint'}
    receipt = meta['receipt']
    if receipt != meta['checkpoint_receipt']:
        raise ValueError('Best pointer/immutable checkpoint receipt mismatch')
    if receipt['sha256'] != manifest['files']['checkpoint']['sha256']:
        raise ValueError('Best receipt/checkpoint mismatch')
    if Path(receipt['path']).name != Path(manifest['files']['checkpoint']['path']).name:
        raise ValueError('Best receipt path mismatch')
    # direct_bc_v1 stores NumPy MT19937 RNG. Restricted Torch unpickling rejects
    # unknown globals; do not retry weights_only=False or allowlist payload names.
    with torch.serialization.safe_globals([np.core.multiarray._reconstruct, np.ndarray,
                                           np.dtype, type(np.dtype('uint32'))]):
        cp = torch.load(io.BytesIO(content['checkpoint']), map_location='cpu', weights_only=True)
    if cp['version'] != 'direct_bc_v1':
        raise ValueError('unsupported checkpoint version')
    ident, config, schema = cp['identity'], meta['config'], meta['robot_schema']
    for field, key in [('dataset_sha256', 'dataset'), ('split_sha256', 'split'),
                       ('vision_sha256', 'vision'), ('config_sha256', 'config')]:
        if ident[field] != manifest['files'][key]['sha256']:
            raise ValueError('checkpoint identity mismatch: '+field)
    for field, expected in [('robot_schema', schema), ('action_schema', meta['action_schema']),
                            ('model_schema', config['model']), ('graph_config', config['graph']),
                            ('training_config', config['training'])]:
        if ident[field] != expected:
            raise ValueError('checkpoint schema/config mismatch: '+field)
    if schema['dimension'] != 76 or meta['action_schema']['id'] != 'pd_ee_delta_pose_regression7' or meta['action_schema']['dimension'] != 7:
        raise ValueError('Robot76/Action7 contract mismatch')
    if meta['split']['dataset_sha256'] != ident['dataset_sha256']:
        raise ValueError('split/dataset identity mismatch')
    if meta['dataset']['dataset_id'] != meta['split']['dataset_id'] or meta['vision']['dataset_id'] != meta['dataset']['dataset_id']:
        raise ValueError('dataset/split/vision UUID mismatch')
    if meta['vision']['dataset_sha256'] != ident['dataset_sha256']:
        raise ValueError('vision/dataset hash mismatch')
    splits = meta['split']['splits']
    all_ids = [i for group in splits.values() for i in group]
    all_groups = [i for group in meta['split']['reset_groups'].values() for i in group]
    if len(all_ids) != len(set(all_ids)) or len(all_groups) != len(set(all_groups)):
        raise ValueError('frozen split overlap')
    n = cp['normalizer']
    if n['schema'] != schema or n['fitted_episode_ids'] != sorted(splits['train']):
        raise ValueError('normalizer must match schema and exactly the train split')
    for key in ('mean', 'std'):
        if not torch.is_tensor(n[key]) or n[key].dtype != torch.float32 or n[key].shape != (76,) or not torch.isfinite(n[key]).all():
            raise ValueError('invalid normalizer statistics')
    if (n['std'] < .001).any():
        raise ValueError('normalizer std floor violated')
    if receipt['global_step'] != cp['progress']['global_step'] or receipt['epoch'] != cp['progress'].get('completed_epoch', cp['progress']['epoch']):
        raise ValueError('receipt progress mismatch')
    for path in required_sources():
        if meta['source'].get(path) != sha(ROOT/path):
            raise ValueError('training source missing/changed: '+path)
    if check_runtime and meta['vision_runtime'] != vision_runtime_identity():
        raise ValueError('Vision weight/SAM2 source identity mismatch')
    if eval_config is not None:
        for key in ('model', 'graph', 'vision'):
            if eval_config[key] != config[key]:
                raise ValueError('evaluation/training config mismatch: '+key)
        for key in ('horizon', 'control_mode', 'sim_backend', 'control_freq', 'physics_freq'):
            if eval_config['environment'][key] != config['collection'][key]:
                raise ValueError('evaluation/collection condition mismatch: '+key)
    model = IntegrationPolicy(schema, **ident['model_schema'])
    expected_weights = model.state_dict()
    if set(cp['model']) != set(expected_weights):
        raise RuntimeError('strict model weight keys mismatch')
    for name, value in cp['model'].items():
        if not torch.is_tensor(value) or value.dtype != expected_weights[name].dtype or value.shape != expected_weights[name].shape:
            raise ValueError('model weight dtype/shape mismatch: '+name)
    model.load_state_dict(cp['model'], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite model weights')
    model.to(device).eval()
    normalizer = RobotNormalizer(schema, StateNormalizer(n['mean'], n['std']), n['fitted_episode_ids'])
    return model, normalizer, dict(manifest=manifest, identity=ident, progress=cp['progress'],
                                   split=meta['split'], dataset=meta['dataset'],
                                   vision_runtime=meta['vision_runtime'], conditions=meta['conditions'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--source-run-id')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--write-runtime-identity', action='store_true')
    args = parser.parse_args()
    if args.write_runtime_identity:
        atomic(args.output, vision_runtime_identity(), immutable=True)
    elif args.bundle and args.source_run_id:
        make_manifest(args.bundle, args.source_run_id, args.output)
    else:
        parser.error('--bundle and --source-run-id required')


if __name__ == '__main__':
    main()
