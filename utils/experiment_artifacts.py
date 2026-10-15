"""Local artifact inventory and explicit Hugging Face publishing.

No network calls occur in the experiment logger. Team formats remain opaque.
"""
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import time

from .experiment_logger import REPO, clean, load_env_file, write

ORGANIZATION = 'Graph-Recurrent-VLA'
DEFAULT_HF_ENV_FILE = Path('/root/.secrets/hf.env')
TEAMS = ('visual', 'graph', 'recurrent', 'integration')
CATEGORIES = {'checkpoint': 'checkpoint_paths', 'rollouts': 'rollout_video_paths',
              'episode_metadata': 'episode_metadata_paths', 'trajectory': 'trajectory_paths',
              'metrics': 'additional_metrics_paths'}


@dataclass(frozen=True)
class Repository:
    repo_id: str
    repo_type: str


DEFAULT_REPOSITORIES = {
    'checkpoint': Repository(f'{ORGANIZATION}/graph-recurrent-models', 'model'),
    'rollouts': Repository(f'{ORGANIZATION}/graph-recurrent-rollouts', 'dataset'),
}


def validate_repositories(repositories):
    if set(repositories) != set(DEFAULT_REPOSITORIES):
        raise ValueError('only checkpoint and rollouts repositories are supported')
    for key, repo in repositories.items():
        if not re.fullmatch(r'Graph-Recurrent-VLA/[A-Za-z0-9][A-Za-z0-9._-]*', repo.repo_id):
            raise ValueError('repositories must belong to Graph-Recurrent-VLA')
        if repo.repo_type != DEFAULT_REPOSITORIES[key].repo_type:
            raise ValueError('repository type does not match artifact kind')


def infer_kind(path):
    suffix = Path(path).suffix.lower()
    if suffix in ('.pt', '.pth', '.ckpt', '.safetensors', '.bin'):
        return 'checkpoint'
    if suffix in ('.mp4', '.webm', '.mov', '.avi', '.mkv'):
        return 'rollouts'
    if suffix in ('.h5', '.hdf5', '.npz', '.parquet'):
        return 'trajectory'
    if suffix == '.json':
        return 'metrics'
    return None


def save_manifest(directory, metadata=None):
    """Cheap inventory of registered outputs, without hashing or scanning folders.

    References are relative to the experiment directory, including external files.
    Missing registered files remain visible and are validated only when selected.
    """
    directory = Path(directory).resolve()
    metadata = metadata or json.loads((directory / 'metadata.json').read_text())
    manifest = dict(schema_version=1, run_id=metadata['run_id'], team=metadata['team'],
                    model_name=metadata.get('model'), task=metadata.get('task'),
                    environment=metadata.get('environment'), git_commit=metadata.get('git_commit'),
                    git_dirty=metadata.get('git_dirty'), random_seed=metadata.get('seed'),
                    wandb_run_id=metadata['run_id'], config_path='config.yaml',
                    metrics_path='metrics.json', **{field: [] for field in CATEGORIES.values()})
    for value in metadata.get('outputs', []):
        path = Path(value)
        kind = metadata.get('artifact_kinds', {}).get(str(path)) or infer_kind(path)
        if kind:
            manifest[CATEGORIES[kind]].append(os.path.relpath(path, directory))
    write(directory / 'manifest.json', manifest)
    return manifest


def encoded(value):
    return (json.dumps(clean(value), sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def digests(source):
    sha256, git = hashlib.sha256(), hashlib.sha1()
    size = len(source) if isinstance(source, bytes) else source.stat().st_size
    git.update(f'blob {size}\0'.encode())
    if isinstance(source, bytes):
        sha256.update(source); git.update(source)
    else:
        with source.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                sha256.update(block); git.update(block)
    return size, sha256.hexdigest(), git.hexdigest()


@dataclass
class UploadFile:
    path: str
    source: object
    size: int
    sha256: str
    git_sha1: str


def upload_file(prefix, category, name, source):
    size, sha256, git_sha1 = digests(source)
    return UploadFile(f'{prefix}/artifacts/{category}/{sha256}/{name}', source, size, sha256, git_sha1)


def plan_upload(directory, include, repositories=None):
    """Validate only explicit selections; metadata/config accompany each target."""
    directory = Path(directory).resolve()
    repositories = repositories or DEFAULT_REPOSITORIES
    validate_repositories(repositories)
    include = set(include)
    if not include or include - {'checkpoint', 'rollouts', 'metrics'}:
        raise ValueError('select checkpoint, rollouts and/or metrics explicitly')
    metadata = json.loads((directory / 'metadata.json').read_text())
    run_id = metadata['run_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', run_id) or metadata['team'] not in TEAMS:
        raise ValueError('invalid run_id or team')
    if directory.name != run_id:
        raise ValueError('experiment directory and run_id do not match')
    if metadata.get('status') == 'running':
        raise ValueError('finish the experiment before publishing')
    manifest = save_manifest(directory, metadata)
    prefix = f'{metadata["team"]}/{run_id}'
    config = clean(json.loads((directory / 'config.yaml').read_text()))
    plans = {}
    targets = include & {'checkpoint', 'rollouts'}
    if not targets:  # metrics-only uses the current model repository, not a future evaluation repo.
        targets = {'checkpoint'}
    for target in sorted(targets):
        remote = {k: v for k, v in manifest.items() if k not in CATEGORIES.values()}
        remote.pop('config_path'); remote.pop('metrics_path')
        remote['selection'] = sorted(include)
        remote['artifacts'] = []
        files = []

        def add(path, category, *, snapshot=False):
            path = Path(path)
            if not path.is_file():
                raise ValueError(f'missing selected {category} file: {path.name}')
            if path.is_symlink():
                raise ValueError('selected files must not be symlinks')
            sidecar = path.with_name(path.name + '.tracking.json')
            if sidecar.exists():
                owner = json.loads(sidecar.read_text()).get('run_id')
                if owner not in (run_id, metadata.get('source_run_id')):
                    raise ValueError('artifact belongs to another run; refusing stale output link')
            # JSON is sanitized using the existing logger policy; never load checkpoint code.
            source = encoded(json.loads(path.read_text())) if snapshot else path
            item = upload_file(prefix, category, path.name, source)
            if not item.size:
                raise ValueError(f'empty selected {category} file: {path.name}')
            if item.path not in {f.path for f in files}:
                files.append(item)
            remote['artifacts'].append(dict(kind=category, path=item.path,
                                            size=item.size, sha256=item.sha256))
            return item.path

        # Use the already sanitized snapshot, without persisting a second local config.
        cfg = upload_file(prefix, 'config', 'config.yaml', encoded(config))
        files.append(cfg); remote['config_path'] = cfg.path
        if 'metrics' in include:
            remote['metrics_path'] = add(directory / manifest['metrics_path'], 'metrics', snapshot=True)
            for value in manifest['additional_metrics_paths']:
                add(directory / value, 'metrics', snapshot=True)
        if target in include:
            field = CATEGORIES[target]
            if not manifest[field]:
                raise ValueError(f'no registered {target} files; use link_output(kind=...)')
            remote[field] = [add(directory / value, target) for value in manifest[field]]
        if target == 'rollouts':
            remote['episode_metadata_paths'] = [add(directory / value, 'episode_metadata', snapshot=True)
                                                for value in manifest['episode_metadata_paths']]
        # Immutable per-selection manifest: separate invocations cannot overwrite inventory.
        body = encoded(remote)
        digest = hashlib.sha256(body).hexdigest()
        size, sha256, git_sha1 = digests(body)
        files.append(UploadFile(f'{prefix}/manifests/{digest}/manifest.json', body, size, sha256, git_sha1))
        plans[target] = dict(repository=repositories[target], prefix=prefix, files=files,
                             manifest_path=files[-1].path)
    return plans


def load_hf_env(path=None):
    """Load optional HF dotenv credentials, retaining process environment priority."""
    return load_env_file(path or os.getenv('HF_ENV_FILE') or DEFAULT_HF_ENV_FILE)


def connect():
    """Return no client without an official token. Never return or log the token."""
    load_hf_env()
    try:
        hub = importlib.import_module('huggingface_hub')
    except ImportError:
        return None, None, 'huggingface_hub is missing; install requirements-tracking.txt'
    try:
        if not hub.get_token():
            return None, None, 'no Hugging Face token; set HF_TOKEN, configure HF_ENV_FILE or /root/.secrets/hf.env, or run hf auth login'
        return hub.HfApi(), hub.CommitOperationAdd, None
    except Exception as exc:
        return None, None, f'Hugging Face authentication unavailable ({type(exc).__name__})'


def initialize_repositories(repositories=None, *, dry_run=False):
    repositories = repositories or DEFAULT_REPOSITORIES
    validate_repositories(repositories)
    if dry_run:
        return {'status': 'dry-run', 'repositories': [r.repo_id for r in repositories.values()], 'private': True}
    api, _, reason = connect()
    if reason:
        return dict(status='skipped', reason=reason)
    results = {}
    try:
        identity = api.whoami()
        if identity.get('auth', {}).get('accessToken', {}).get('role') == 'read':
            return dict(status='skipped', reason='Hugging Face token is read-only; configure an organization write token and rerun --init-repos')
        allowed = any(org.get('name') == ORGANIZATION and org.get('roleInOrg') in ('admin', 'write', 'contributor')
                      for org in identity.get('orgs', []))
        if not allowed:
            return dict(status='skipped', reason='organization write access not confirmed; request Graph-Recurrent-VLA access')
    except Exception as exc:
        return dict(status='skipped', reason=f'authentication/access check failed ({type(exc).__name__})')
    for key, repo in repositories.items():
        try:
            url = api.create_repo(repo_id=repo.repo_id, repo_type=repo.repo_type, private=True, exist_ok=True)
            results[key] = dict(status='ready', url=str(url))
        except Exception as exc:
            results[key] = dict(status='failed', **failure_details(exc))
    return dict(status='ready' if all(r['status'] == 'ready' for r in results.values()) else 'failed', repositories=results)


def remote_matches(item, remote):
    lfs = getattr(remote, 'lfs', None)
    lfs_sha = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
    return remote.size == item.size and (lfs_sha == item.sha256 if lfs else remote.blob_id == item.git_sha1)


def failure_details(exc):
    """Safe diagnostics: never serialize SDK exception messages or responses."""
    status = getattr(getattr(exc, 'response', None), 'status_code', None)
    reason = {401: 'Hugging Face authentication rejected; configure HF_TOKEN or hf auth login',
              403: 'Hugging Face write access denied; check token scope and organization role',
              404: 'repository missing or inaccessible; check access and run --init-repos explicitly',
              409: 'repository changed concurrently; rerun the same command'}.get(status,
              'publish failed; verify selected files, repository access and connectivity, then rerun')
    return dict(error_type=type(exc).__name__, http_status=status, reason=reason)


def publish(directory, include, *, repositories=None, dry_run=False, private=False, retries=2):
    """Post-run, per-repository atomic commits. Return errors without transport exceptions."""
    if not 0 <= retries <= 5:
        raise ValueError('retries must be between 0 and 5')
    directory = Path(directory).resolve()
    plans = plan_upload(directory, include, repositories)
    report = dict(status='dry-run' if dry_run else 'pending', total_bytes=sum(
        f.size for plan in plans.values() for f in plan['files']), repositories={})
    for key, plan in plans.items():
        report['repositories'][key] = dict(repo_id=plan['repository'].repo_id,
            repo_type=plan['repository'].repo_type, files=[dict(path=f.path, size=f.size) for f in plan['files']])
    if dry_run:
        return report
    api, operation, reason = connect()
    if reason:
        report.update(status='skipped', reason=reason)
        return report
    receipt_path = directory / 'huggingface_uploads.json'
    receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else {'schema_version': 1, 'publications': {}}
    for key, plan in plans.items():
        repo = plan['repository']
        result = report['repositories'][key]
        for attempt in range(retries + 1):
            try:
                info = api.repo_info(repo.repo_id, repo_type=repo.repo_type)
                if private and not info.private:
                    raise ValueError('repository is public; --private requires a private repository')
                try:
                    tree = api.list_repo_tree(repo.repo_id, path_in_repo=plan['prefix'], recursive=True,
                                              revision=info.sha, repo_type=repo.repo_type)
                    existing = {f.path: f for f in tree if hasattr(f, 'blob_id')}
                except Exception as exc:
                    if getattr(getattr(exc, 'response', None), 'status_code', None) != 404:
                        raise
                    existing = {}
                additions = []
                for item in plan['files']:
                    if digests(item.source)[1] != item.sha256:
                        raise ValueError('artifact changed during publishing; stop writers and rerun')
                    if item.path in existing:
                        if not remote_matches(item, existing[item.path]):
                            raise ValueError('remote immutable artifact differs; refusing overwrite')
                    else:
                        additions.append(operation(path_in_repo=item.path, path_or_fileobj=item.source))
                revision = info.sha
                if additions:
                    commit = api.create_commit(repo.repo_id, repo_type=repo.repo_type, operations=additions,
                        parent_commit=info.sha, commit_message=f'Publish experiment {directory.name} ({key})')
                    revision = commit.oid
                base = 'https://huggingface.co/' + ('datasets/' if repo.repo_type == 'dataset' else '') + repo.repo_id
                result.update(status='uploaded' if additions else 'unchanged', revision=revision,
                              url=f'{base}/tree/{revision}/{plan["prefix"]}', manifest_path=plan['manifest_path'],
                              uploaded_files=len(additions), skipped_files=len(plan['files']) - len(additions))
                receipt_key = f'{repo.repo_type}:{repo.repo_id}:{plan["manifest_path"]}'
                receipts['publications'][receipt_key] = {k: v for k, v in result.items() if k != 'files'}
                write(receipt_path, receipts)
                break
            except Exception as exc:
                # Exception strings may include credentials or request headers.
                result.update(status='failed', **failure_details(exc), attempts=attempt + 1)
                status = result['http_status']
                if isinstance(exc, ValueError) or (status and 400 <= status < 500 and status not in (409, 429)):
                    break
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 8))
        if result['status'] != 'failed':
            for field in ('error_type', 'attempts', 'http_status', 'reason'):
                result.pop(field, None)
    report['status'] = 'failed' if any(r['status'] == 'failed' for r in report['repositories'].values()) else 'complete'
    return report


def sync_wandb_links(directory):
    """Separate opt-in metadata update, no run resume/init or history replay."""
    directory = Path(directory)
    receipts = json.loads((directory / 'huggingface_uploads.json').read_text())
    metadata = json.loads((directory / 'metadata.json').read_text())
    from .experiment_logger import has_wandb_api_key, load_wandb_env
    load_wandb_env()
    if not has_wandb_api_key() or not metadata.get('entity'):
        return dict(status='skipped', reason='W&B API key and recorded entity are required; HF receipts retained')
    try:
        wandb = importlib.import_module('wandb')
        run = wandb.Api(timeout=10).run(f'{metadata["entity"]}/{metadata["project"]}/{metadata["run_id"]}')
        run.summary['huggingface'] = receipts['publications']
        run.summary.update()
        return dict(status='complete')
    except Exception as exc:
        return dict(status='failed', error_type=type(exc).__name__)
