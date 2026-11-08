"""Restore pinned temporal-training artifacts from private Hugging Face repositories."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

from huggingface_hub import HfApi, hf_hub_download
import torch
import yaml

ROOT = Path('/workspace/temporal_continuation_120')
DATASET = 'Graph-Recurrent-VLA/graph-recurrent-rollouts'
MODELS = 'Graph-Recurrent-VLA/graph-recurrent-models'
RAW_REVISION = '94cb85e8009f20b5b47063e8295f6c69040549c1'
CACHE_REVISION = '2d042d285c61d0c9bb622db471b8d30bcee3bfb0'
RUNTIME_REVISION = '556a14bfd5376d60ac13c629f1facf29ff7de70d'
MODEL_REVISION = 'b9477c44b3f8b55b767cb1c00b23d9696dd3cc6e'
CACHE_PATH = ('graph/ffe33e7fa80c428fb3d9e50dc43a14ff/artifacts/trajectory/'
              '00de2d6cc124806cbfc270d0e71abc77fd41e3d806c30f3b3812a8149f51dabc/graph_cache.tar')
RUNTIME_PATH = ('graph/62b0c436d82642b890cc52db7a01451a/artifacts/trajectory/'
                '6b636a22a0bc752620571eec24ca1d4862045b2e0cbf27ae04162bf1c7e3c2c8/'
                'temporal_runtime_inputs.tar.gz')
RUNS = {
    'mamba': ['157c3a43624f447895193301d82a6bc0', '31d5bd0dfcec45e09ef8953ae9cf93a5',
              '06a79063394f43049f8274f3af0eb1b6'],
    'transformer': ['9dd99e65c3d54eeca895a73099574aa8', '124c6d5f7c364a289c5bdfa10f514ce2',
                    '0ca635be76f34bb7ade6bf862e1ac124'],
}


def sha256(path):
    """Hash local bytes without loading large archives into memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(repo, kind, path, revision):
    """Download one immutable published object and verify its content-addressed name."""
    filename = Path(hf_hub_download(repo_id=repo, repo_type=kind, filename=path,
                                    revision=revision, local_dir=ROOT / 'downloads', token=True))
    expected = PurePosixPath(path).parent.name
    if len(expected) != 64 or sha256(filename) != expected:
        raise ValueError(f'Published artifact checksum mismatch: {PurePosixPath(path).name}')
    return filename


def extract_archive(path, destination):
    """Extract trusted, checksum-verified archives with Python's safe data filter."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'r:*') as archive:
        archive.extractall(destination, filter='data')


def restore_shard(item):
    """Restore only selected training-pool frames and check each raw-frame hash."""
    path, expected_frames = item
    archive_path = download(DATASET, 'dataset', path, RAW_REVISION)
    restored = set()
    episodes = ROOT / 'data/episodes'
    with tarfile.open(archive_path, 'r:') as archive:
        for member in archive:
            sample_id = member.name.removeprefix('./')
            if sample_id not in expected_frames:
                continue
            if not member.isfile() or sample_id in restored:
                raise ValueError('Expected unique regular frame members')
            target = episodes / sample_id
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and sha256(target) == expected_frames[sample_id]:
                restored.add(sample_id)
                continue
            temporary = target.with_suffix('.npz.partial')
            digest = hashlib.sha256()
            with archive.extractfile(member) as src, temporary.open('wb') as dst:
                for block in iter(lambda: src.read(4 * 1024 * 1024), b''):
                    digest.update(block)
                    dst.write(block)
            if digest.hexdigest() != expected_frames[sample_id]:
                temporary.unlink(missing_ok=True)
                raise ValueError(f'Raw frame checksum mismatch: {sample_id}')
            os.replace(temporary, target)
            restored.add(sample_id)
    if restored != set(expected_frames):
        raise ValueError(f'Archive missing {len(set(expected_frames) - restored)} selected frames')
    archive_path.unlink()
    print(f'Restored {PurePosixPath(path).name}: {len(restored)} verified frames', flush=True)
    return len(restored)


def prepare():
    """Restore checkpoints, runtime and contiguous train/validation episode data."""
    ROOT.mkdir(parents=True, exist_ok=True)
    runtime = download(DATASET, 'dataset', RUNTIME_PATH, RUNTIME_REVISION)
    extract_archive(runtime, ROOT)
    runtime.unlink()
    cache = download(DATASET, 'dataset', CACHE_PATH, CACHE_REVISION)
    cache_root = ROOT / 'data/gnn_cache'
    extract_archive(cache, cache_root)
    cache.unlink()
    identity = json.loads((cache_root / 'metadata/data_identity.json').read_text())
    if identity['dataset_sha256'] != '41fe23fed067f116b6bfc39694e19a5f98714381207802a618ac6ea9545e2dcb':
        raise ValueError('Published cache identity differs from the original experiment')
    train_episodes = set(identity['splits']['train'])
    frames = {row['id']: row['sha256'] for row in identity['frames']
              if str(PurePosixPath(row['id']).parent) in train_episodes}
    if len(train_episodes) != 630 or len(frames) != 64702:
        raise ValueError('Expected the original 630-parent, 64,702-frame train pool')
    api = HfApi()
    published = list(api.list_repo_files(DATASET, repo_type='dataset', revision=RAW_REVISION, token=True))
    raw_prefix = 'integration/c555bed7462d4fc09e55ae3bb04421a1/artifacts/trajectory/'
    shards = {PurePosixPath(path).stem: path for path in published
              if path.startswith(raw_prefix) and PurePosixPath(path).name.startswith('shard_') and path.endswith('.tar')}
    by_shard = {}
    for sample_id, digest in frames.items():
        shard = PurePosixPath(sample_id).parts[0]
        by_shard.setdefault(shard, {})[sample_id] = digest
    if len(shards) != 30 or not set(by_shard) <= set(shards):
        raise ValueError('Published raw episode archive inventory is incomplete')
    model_files = list(api.list_repo_files(MODELS, repo_type='model', revision=MODEL_REVISION, token=True))
    for method, runs in RUNS.items():
        for seed, run in enumerate(runs):
            matching = [path for path in model_files if path.startswith(f'graph/{run}/artifacts/checkpoint/')
                        and path.endswith('/last.pt')]
            if len(matching) != 1:
                raise ValueError(f'Expected one published {method} seed-{seed} last.pt')
            checkpoint = download(MODELS, 'model', matching[0], MODEL_REVISION)
            target = ROOT / f'inputs/checkpoints/{method}/seed_{seed}.pt'
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(checkpoint, target)
            payload = torch.load(target, map_location='cpu', weights_only=True)
            if payload['epoch'] != 90 or payload['kind'] != 'temporal_insertion_state' or not payload['optimizer']['state']:
                raise ValueError(f'Invalid {method} seed-{seed} resume checkpoint')
            original = payload['config']
            matrix = list(Path(f'/workspace/selected_vertex_continuation_matched/training_seed_{seed}').glob('*/matrix.json'))
            if len(matrix) != 1:
                raise ValueError('Original M0 matrix is missing or ambiguous')
            m0_path = Path(json.loads(matrix[0].read_text())['runs'][0]['final_checkpoint'])
            if sha256(m0_path) != original['source_checkpoint_sha256']:
                raise ValueError('Frozen M0 weights differ from temporal training source')
            split = ROOT / 'inputs/splits.json'
            if sha256(split) != original['split_manifest_sha256']:
                raise ValueError('Temporal parent split differs from training source')
            config = dict(original)
            config.update(trainer='training.gnn.temporal.selected_estimator:train_and_evaluate',
                          checkpoint=str(m0_path), split_manifest=str(split),
                          continuation_checkpoint=str(target),
                          comparison_output=str(ROOT / f'evaluation/{method}/seed_{seed}'),
                          evaluation_selection=str(ROOT / 'inputs/validation_frames.jsonl'))
            config['training'] = dict(original['training'], epochs=30, cuda_memory_fraction=0.10)
            config['data'] = dict(original['data'], cache_dir=str(cache_root),
                                  raw_root=str(ROOT / 'data/episodes'))
            config['selected_m0'] = {'dataset': '/workspace/selected_vertex_m0/data',
                                     'cache_source_checkpoint': str(ROOT / 'inputs/cache_source_best.pt')}
            config['output'] = {'root': str(ROOT / f'training/{method}/seed_{seed}')}
            (ROOT / f'{method}_seed_{seed}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
            print(f'Checkpoint verified: {method} seed {seed}', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(restore_shard, (shards[shard], subset)) for shard, subset in sorted(by_shard.items())]
        total = sum(future.result() for future in as_completed(futures))
    if total != len(frames):
        raise ValueError('Restored frame count does not match the published cache')
    (ROOT / 'data_ready.json').write_text(json.dumps({'frames': total, 'episodes': len(train_episodes),
        'cache_identity': identity['dataset_sha256'], 'raw_revision': RAW_REVISION,
        'cache_revision': CACHE_REVISION, 'runtime_revision': RUNTIME_REVISION,
        'model_revision': MODEL_REVISION}, indent=2) + '\n')
    print(f'Data restored: {total} verified frames', flush=True)


if __name__ == '__main__':
    try:
        prepare()
    except Exception as error:
        print(f'Restoration failed: {type(error).__name__}', flush=True)
        (ROOT / 'RESTORE_FAILED').touch()
        raise SystemExit(1)
