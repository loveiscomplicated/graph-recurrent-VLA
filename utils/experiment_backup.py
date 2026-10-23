"""Explicit, byte-preserving experiment snapshots. No simulator imports."""
import importlib.metadata
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import time
import uuid

from .experiment_artifacts import (ORGANIZATION, connect, digests, failure_details,
                                  immutable_additions, UploadFile)

class BackupValidationError(ValueError):
    """A safe, content-free backup diagnostic."""


DEFAULT_REPOSITORY = f'{ORGANIZATION}/experiment-backups'
TOKEN = re.compile(rb'(?<![A-Za-z0-9_])(?:hf_[A-Za-z0-9]{20,}|(?:WANDB_API_KEY|HF_TOKEN)\s*[=:]\s*[\"\x27]?[A-Za-z0-9_-]{32,})')
EXCLUDED = {'.git', '.venv', 'venv', '__pycache__', '.cache', '.secrets', '.aws', '.ssh', 'node_modules'}


def dump(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(dump(value))
    temporary.replace(path)


def relative(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or '..' in p.parts or '\\' in value or p.as_posix() != value or value == '.':
        raise BackupValidationError('unsafe relative path')
    return p


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', value):
        raise BackupValidationError('invalid identifier')
    return value


def repository(value):
    if not re.fullmatch(r'Graph-Recurrent-VLA/[A-Za-z0-9][A-Za-z0-9._-]*', value):
        raise BackupValidationError('backup repository must belong to Graph-Recurrent-VLA')
    return value


def excluded(path):
    return any(x in EXCLUDED or x == '.env' or x.startswith('.env.') or x.endswith('.env') for x in path.parts) or path.name.endswith(('.lock', '.tmp', '.tmp.pt', '.pyc'))


def safe_file(root, name):
    rel = relative(name)
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise BackupValidationError('symlinks are not allowed')
    if not current.is_file() or not stat.S_ISREG(current.stat().st_mode):
        raise BackupValidationError('selected file is missing or not regular')
    return current


def command(root, args):
    proc = subprocess.run(args, cwd=root, capture_output=True, timeout=30)
    if proc.returncode:
        raise BackupValidationError('environment or Git inspection failed')
    return proc.stdout


def scan(path):
    # Scan opaque binaries as well as text, without deserializing or printing contents.
    tail = b''
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            data = tail + block
            if TOKEN.search(data):
                raise BackupValidationError('possible credential detected; inspect selected files locally')
            tail = data[-256:]


def inventory(spec):
    root = Path(spec['root']).resolve(strict=True)
    identifier(spec['experiment'])
    if spec['team'] not in ('graph', 'visual', 'recurrent', 'integration'):
        raise BackupValidationError('invalid team')
    names, skipped = set(), set()
    explicit_excludes = [relative(x) for x in spec.get('exclude', [])]

    def omit(path):
        return excluded(path) or any(path == x or x in path.parents for x in explicit_excludes)
    selected = list(spec['include'])
    for run in spec.get('run_ids', []):
        selected.append(f'experiments/{identifier(run)}')
    if spec.get('code_snapshot', True):
        tracked = command(root, ['git', 'ls-files', '-z']).decode().split('\0')
        # Missing tracked files are represented by Git status/diff rather than invented files.
        selected.extend(x for x in tracked if x and (root / x).exists())
    for name in selected:
        rel = relative(name)
        p = root / rel
        if omit(rel):
            skipped.add(name)
            continue
        for part in [p, *p.parents]:
            if part == root:
                break
            if part.is_symlink():
                raise BackupValidationError('selected path contains a symlink')
        if not p.exists():
            raise BackupValidationError('selected path is missing')
        candidates = [p] if p.is_file() else p.rglob('*')
        for f in candidates:
            r = f.relative_to(root)
            if omit(r):
                skipped.add(r.as_posix())
                continue
            if f.is_symlink():
                raise BackupValidationError('selected directory contains a symlink')
            if f.is_dir():
                continue
            safe_file(root, r.as_posix())
            names.add(r.as_posix())
    files = []
    for name in sorted(names):
        f = safe_file(root, name)
        before = f.stat()
        scan(f)
        size, sha, _ = digests(f)
        after = f.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise BackupValidationError('file changed during inventory; stop writers')
        if name == 'manifest.json' or name.startswith('__backup_provenance__/'):
            raise BackupValidationError('reserved backup path in selection')
        files.append(dict(path=name, size=size, sha256=sha))
    if not files:
        raise BackupValidationError('empty backup selection')
    return root, files, sorted(skipped)


def create(spec, out=None, *, dry_run=False):
    root, files, skipped = inventory(spec)
    if not isinstance(spec.get('code_snapshot', True), bool):
        raise BackupValidationError('code_snapshot must be boolean')
    report = dict(status='dry-run', files=files, excluded=skipped,
                  total_bytes=sum(x['size'] for x in files))
    if dry_run:
        return report
    if out is None:
        raise BackupValidationError('snapshot destination is required')
    out = Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise BackupValidationError('snapshot destination must not exist')
    # A staging directory must not be nested inside selected input directories.
    for name in spec['include'] + [f'experiments/{x}' for x in spec.get('run_ids', [])]:
        source = root / relative(name)
        if source.is_dir() and out.resolve().is_relative_to(source.resolve()):
            raise BackupValidationError('snapshot destination overlaps selected input')
    out.mkdir(parents=True)
    for entry in files:
        source = safe_file(root, entry['path'])
        target = out / 'files' / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digests(target)[:2] != (entry['size'], entry['sha256']) or digests(source)[:2] != (entry['size'], entry['sha256']):
            raise BackupValidationError('file changed during snapshot; discard incomplete staging and retry')
        target.chmod(stat.S_IMODE(source.stat().st_mode) & 0o777)
        entry['mode'] = stat.S_IMODE(target.stat().st_mode)
    # Verify every source again, including files copied early in the snapshot.
    for entry in files:
        if digests(safe_file(root, entry['path']))[:2] != (entry['size'], entry['sha256']):
            raise BackupValidationError('source changed during snapshot')
    provenance = out / 'provenance'
    provenance.mkdir()
    for name, args in [('git-status.txt', ['git', 'status', '--porcelain=v1']),
                       ('git-diff.patch', ['git', 'diff', 'HEAD', '--binary'])]:
        (provenance / name).write_bytes(command(root, args))
        scan(provenance / name)
    try:
        gpu = command(root, ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']).decode()
    except (ValueError, OSError, subprocess.TimeoutExpired):
        gpu = None
    environment = dict(collected_on='backup_host', python=platform.python_version(),
                       executable=os.sys.executable, platform=platform.platform(), gpu=gpu,
                       packages=sorted([{'name': d.metadata['Name'], 'version': d.version}
                                        for d in importlib.metadata.distributions()], key=lambda d: d['name'].lower()))
    try:
        import torch
        environment['torch_cuda_build'] = torch.version.cuda
    except ImportError:
        environment['torch_cuda_build'] = None
    write_json(provenance / 'environment.json', environment)
    from .experiment_logger import clean
    write_json(provenance / 'selection.json', clean(spec))
    for f in sorted(provenance.iterdir()):
        scan(f)
        size, sha, _ = digests(f)
        files.append(dict(path=f'__backup_provenance__/{f.name}', size=size, sha256=sha, mode=0o644))
    runs = []
    for run in spec.get('run_ids', []):
        meta = json.loads((out / 'files' / 'experiments' / run / 'metadata.json').read_text())
        runs.append({k: meta.get(k) for k in ('run_id', 'status', 'job_type', 'source_run_id', 'git_commit', 'git_dirty')})
    manifest = dict(schema_version=1, backup_id=uuid.uuid4().hex, team=spec['team'], experiment=spec['experiment'],
                    created_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_root=str(root),
                    git_commit=command(root, ['git', 'rev-parse', 'HEAD']).decode().strip(),
                    code_provenance='snapshot of backup host; equality to original dirty execution is unverified',
                    runs=runs, files=files, excluded=skipped)
    write_json(out / 'manifest.json', manifest)
    verify(out, snapshot=True)
    return dict(status='created', backup_id=manifest['backup_id'], directory=str(out),
                files=len(files), total_bytes=sum(x['size'] for x in files))


def load_manifest(path):
    m = json.loads(path.read_text())
    if m['schema_version'] != 1 or m['team'] not in ('graph', 'visual', 'recurrent', 'integration'):
        raise BackupValidationError('unsupported backup manifest')
    identifier(m['backup_id']); identifier(m['experiment'])
    seen = set()
    for e in m['files']:
        relative(e['path'])
        if e['path'] in ('manifest.json', '__backup_provenance__/restore-receipt.json') or e['path'] in seen or not isinstance(e['size'], int) or e['size'] < 0 or not re.fullmatch('[0-9a-f]{64}', e['sha256']):
            raise BackupValidationError('invalid file entry')
        if not isinstance(e.get('mode'), int) or e['mode'] & ~0o777:
            raise BackupValidationError('invalid file mode')
        seen.add(e['path'])
    if not seen:
        raise BackupValidationError('empty backup manifest')
    if any(parent.as_posix() in seen for name in seen for parent in PurePosixPath(name).parents):
        raise BackupValidationError('file and directory path collision')
    return m


def local_file(directory, entry, snapshot):
    name = entry['path']
    if snapshot and name.startswith('__backup_provenance__/'):
        return safe_file(directory / 'provenance', name.split('/', 1)[1])
    return safe_file(directory / 'files' if snapshot else directory, name)


def verify(directory, *, snapshot=False, manifest_path=None):
    directory = Path(directory).resolve()
    m = load_manifest(Path(manifest_path) if manifest_path else directory / 'manifest.json')
    for e in m['files']:
        if digests(local_file(directory, e, snapshot))[:2] != (e['size'], e['sha256']):
            raise BackupValidationError('backup checksum mismatch')
    return dict(status='verified', backup_id=m['backup_id'], files=len(m['files']),
                total_bytes=sum(x['size'] for x in m['files']))


def init_repo(repo=DEFAULT_REPOSITORY, *, dry_run=False):
    repository(repo)
    if dry_run:
        return dict(status='dry-run', repo_id=repo, private=True, repo_type='dataset')
    api, _, reason = connect()
    if reason:
        raise BackupValidationError(reason)
    api.create_repo(repo_id=repo, repo_type='dataset', private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type='dataset').private:
        raise BackupValidationError('backup repository must be private')
    return dict(status='ready', repo_id=repo)


def publish(directory, repo=DEFAULT_REPOSITORY, *, dry_run=False, retries=2):
    repository(repo)
    if not 0 <= retries <= 5:
        raise BackupValidationError('retries must be between 0 and 5')
    directory = Path(directory).resolve()
    checked = verify(directory, snapshot=True)
    m = load_manifest(directory / 'manifest.json')
    prefix = f"{m['team']}/{m['experiment']}/{m['backup_id']}"
    items = []
    for e in m['files']:
        source = local_file(directory, e, True)
        size, sha, git = digests(source)
        items.append(UploadFile(f"{prefix}/files/{e['path']}", source, size, sha, git))
    size, sha, git = digests(directory / 'manifest.json')
    items.append(UploadFile(f'{prefix}/manifest.json', directory / 'manifest.json', size, sha, git))
    if dry_run:
        return dict(**{**checked, 'status': 'dry-run'}, repo_id=repo, prefix=prefix,
                    upload_files=[dict(path=x.path, size=x.size) for x in items])
    api, operation, reason = connect()
    if reason:
        raise BackupValidationError(reason)
    for attempt in range(retries + 1):
        try:
            info = api.repo_info(repo, repo_type='dataset')
            if not info.private:
                raise BackupValidationError('backup repository must be private')
            try:
                existing = {x.path: x for x in api.list_repo_tree(repo, repo_type='dataset', revision=info.sha,
                            path_in_repo=prefix, recursive=True) if hasattr(x, 'blob_id')}
            except Exception as exc:
                if getattr(getattr(exc, 'response', None), 'status_code', None) != 404:
                    raise
                existing = {}
            additions = immutable_additions(items, existing, operation)
            revision = info.sha
            if additions:
                revision = api.create_commit(repo, repo_type='dataset', operations=additions,
                    parent_commit=info.sha, commit_message=f"Backup {m['backup_id']}").oid
            receipt = dict(status='complete', repo_id=repo, revision=revision, prefix=prefix,
                           manifest_path=f'{prefix}/manifest.json', backup_id=m['backup_id'],
                           uploaded_files=len(additions), skipped_files=len(items)-len(additions),
                           url=f'https://huggingface.co/datasets/{repo}/tree/{revision}/{prefix}')
            write_json(directory / 'receipt.json', receipt)
            return receipt
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if isinstance(exc, ValueError) or (status and 400 <= status < 500 and status not in (409, 429)) or attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 8))


def restore(repo, revision, prefix, out):
    repository(repo); relative(prefix)
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise BackupValidationError('restore requires an immutable full commit revision')
    api, _, reason = connect()
    if reason:
        raise BackupValidationError(reason)
    if not api.repo_info(repo, repo_type='dataset', revision=revision).private:
        raise BackupValidationError('backup repository must be private')
    out = Path(out).absolute()
    if out.is_symlink() or out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise BackupValidationError('restore destination must be absent or empty')
    # SDK cache path is never treated as a safe output tree: download only listed files.
    from huggingface_hub import hf_hub_download
    manifest_source = Path(hf_hub_download(repo, f'{prefix}/manifest.json', repo_type='dataset', revision=revision))
    m = load_manifest(manifest_source)
    if prefix != f"{m['team']}/{m['experiment']}/{m['backup_id']}":
        raise BackupValidationError('manifest does not match backup prefix')
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_source, out / 'manifest.json')
    def restore_file(entry):
        if entry['path'] == 'manifest.json':
            raise BackupValidationError('reserved restore path')
        source = Path(hf_hub_download(repo, f"{prefix}/files/{entry['path']}", repo_type='dataset', revision=revision))
        if digests(source)[:2] != (entry['size'], entry['sha256']):
            raise BackupValidationError('download checksum mismatch')
        target = out / entry['path']
        if target.exists() or target.is_symlink():
            raise BackupValidationError('restore path collision')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(entry['mode'])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(restore_file, m['files']))
    result = verify(out)
    (out / '__backup_provenance__').mkdir(exist_ok=True)
    write_json(out / '__backup_provenance__' / 'restore-receipt.json',
               dict(repo_id=repo, revision=revision, prefix=prefix, **result))
    return result
