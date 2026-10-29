"""Revision-pinned HF inventory and explicit repository maintenance.

Uses the shared authentication, hashing and failure handling. No training hook,
repository creation, visibility change, history rewrite or LFS purge exists here.
"""
import json
from pathlib import Path, PurePosixPath

from .experiment_artifacts import connect, digests, failure_details
from .experiment_logger import write


REPOSITORIES = {
    ('model', 'Graph-Recurrent-VLA/graph-recurrent-models'),
    ('dataset', 'Graph-Recurrent-VLA/graph-recurrent-rollouts'),
    ('dataset', 'Graph-Recurrent-VLA/experiment-backups'),
    ('space', 'Graph-Recurrent-VLA/task09c-captures'),
}


def safe_path(value):
    p = PurePosixPath(value)
    if not value or not p.parts or p.is_absolute() or '..' in p.parts or '\\' in value or p.as_posix() != value:
        raise ValueError('maintenance requires safe explicit file paths')
    return value


def file_record(item):
    lfs = getattr(item, 'lfs', None)
    sha = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None
    return dict(path=item.path, size=item.size, sha256=sha, git_sha1=item.blob_id)


def inventory(repo_id, repo_type, *, api=None):
    if (repo_type, repo_id) not in REPOSITORIES:
        raise ValueError('repository is outside the maintenance allowlist')
    if api is None:
        api, _, reason = connect()
        if reason:
            raise ValueError(reason)
    info = api.repo_info(repo_id, repo_type=repo_type)
    files = {f.path: f for f in api.list_repo_tree(repo_id, repo_type=repo_type,
             revision=info.sha, recursive=True) if hasattr(f, 'blob_id')}
    return dict(repo_id=repo_id, repo_type=repo_type, revision=info.sha,
                private=info.private, files=[file_record(f) for f in files.values()])


def referenced_paths(snapshot, read_json):
    """Conservatively protect all published manifests and their declared members.

    Unknown legacy trees are protected by default. A backup is a complete snapshot;
    identical payloads in separate snapshots retain their original paths.
    """
    paths = {f['path'] for f in snapshot['files']}
    protected = {'.gitattributes'} & paths
    for path in paths:
        if path.endswith('/manifest.json') or path in ('bundle_manifest.json', 'results_export_manifest.json', 'catalog.json'):
            protected.add(path)
            data = read_json(path)
            for item in data.get('artifacts', []):
                if item.get('path') in paths:
                    protected.add(item['path'])
            for key in ('config_path', 'metrics_path'):
                if data.get(key) in paths:
                    protected.add(data[key])
            for key, values in data.items():
                if key.endswith('_paths') and isinstance(values, list):
                    protected.update(x for x in values if isinstance(x, str) and x in paths)
            for item in data.get('files', []):
                member = item.get('path', '')
                if data.get('backup_id'):
                    member = str(PurePosixPath(path).parent / 'files' / member)
                if member in paths:
                    protected.add(member)
    for path in paths:
        if snapshot['repo_id'].endswith('experiment-backups'):
            # Backups include provenance files outside their payload manifests.
            protected.add(path)
        elif snapshot['repo_type'] != 'space' and '/artifacts/' not in path and '/manifests/' not in path:
            protected.add(path)
    return protected


def validate_plan(plan):
    if (plan.get('schema_version') != 1 or not isinstance(plan.get('repositories'), list)
            or not plan['repositories']):
        raise ValueError('unsupported maintenance plan')
    seen = set()
    for repo in plan['repositories']:
        key = (repo['repo_type'], repo['repo_id'])
        if key not in REPOSITORIES or key in seen:
            raise ValueError('unknown or duplicate maintenance repository')
        seen.add(key)
        if not repo.get('revision') or not isinstance(repo.get('private'), bool):
            raise ValueError('maintenance requires revision and visibility')
        targets = set()
        protected = set(repo.get('protected_paths', []))
        for item in repo.get('additions', []):
            path = safe_path(item['path'])
            if '/artifacts/' in path or '/manifests/' in path:
                raise ValueError('maintenance cannot overwrite immutable experiment artifacts')
            source = Path(item['source'])
            if path in targets or source.is_symlink() or not source.is_file():
                raise ValueError('invalid maintenance addition')
            if digests(source)[1] != item['sha256']:
                raise ValueError('maintenance source hash changed')
            targets.add(path)
        deletions = repo.get('deletions', [])
        deleted = {safe_path(x['path']) for x in deletions}
        if len(deleted) != len(deletions) or targets & deleted:
            raise ValueError('maintenance targets overlap')
        for item in deletions:
            retained = safe_path(item['retained_path'])
            if item['path'] in protected or retained in deleted or retained == item['path']:
                raise ValueError('referenced deletion or missing retained copy')
            if not item.get('reason') or item.get('reference_count') != 0:
                raise ValueError('deletion requires unreferenced duplicate evidence')
            if not (item.get('sha256') or item.get('git_sha1')) or 'size' not in item:
                raise ValueError('deletion requires the original content hash and size')
    return plan


def reconcile(plan, *, apply=False, receipt=None):
    """Dry-run without network by default; apply one guarded commit per repository.

    Actual deletes independently reread manifest references at the pinned revision.
    Completed repositories persist immediately so partial failure can be retried.
    """
    validate_plan(plan)
    results = dict(status='dry-run', repositories=[])
    if not apply:
        results['repositories'] = [dict(repo_id=r['repo_id'], revision=r['revision'],
            additions=len(r.get('additions', [])), deletions=len(r.get('deletions', [])))
            for r in plan['repositories']]
        return results
    api, add, reason = connect()
    if reason:
        return dict(results, status='skipped', reason=reason)
    from huggingface_hub import CommitOperationDelete, hf_hub_download
    for repo in plan['repositories']:
        result = dict(repo_id=repo['repo_id'], repo_type=repo['repo_type'])
        results['repositories'].append(result)
        try:
            current = inventory(repo['repo_id'], repo['repo_type'], api=api)
            if current['private'] != repo['private']:
                raise ValueError('repository visibility changed')
            existing = {x['path']: x for x in current['files']}
            def matches(record, expected):
                return record['size'] == expected['size'] and (
                    record['sha256'] == expected.get('sha256') if record.get('sha256')
                    else record['git_sha1'] == expected.get('git_sha1'))
            operations = []
            for item in repo.get('additions', []):
                size, sha, git = digests(Path(item['source']))
                if sha != item['sha256']:
                    raise ValueError('maintenance source hash changed')
                if item['path'] not in existing or not matches(existing[item['path']], dict(size=size, sha256=sha, git_sha1=git)):
                    operations.append(add(path_in_repo=item['path'], path_or_fileobj=str(Path(item['source']))))
            deletions = [d for d in repo.get('deletions', []) if d['path'] in existing]
            for item in repo.get('deletions', []):
                retained = existing.get(item['retained_path'])
                if retained is None or not matches(retained, item):
                    raise ValueError('retained duplicate is missing or different')
            if current['revision'] != repo['revision'] and (operations or deletions):
                raise ValueError('remote revision changed; audit and regenerate the plan')
            if deletions:
                def read_json(path):
                    local = hf_hub_download(repo['repo_id'], path, repo_type=repo['repo_type'], revision=current['revision'])
                    return json.loads(Path(local).read_text())
                protected = referenced_paths(current, read_json)
                for item in deletions:
                    if item['path'] in protected or not matches(existing[item['path']], item):
                        raise ValueError('deletion is referenced or its content changed')
                    retained = existing.get(item['retained_path'])
                    if retained is None or not matches(retained, item):
                        raise ValueError('retained duplicate is missing or different')
                    operations.append(CommitOperationDelete(path_in_repo=item['path'], is_folder=False))
            revision = current['revision']
            if operations:
                revision = api.create_commit(repo['repo_id'], repo_type=repo['repo_type'],
                    parent_commit=current['revision'], operations=operations,
                    commit_message='Organize research catalogue and verified redundant files').oid
            result.update(status='complete' if operations else 'unchanged', revision=revision,
                          additions=len(operations)-len(deletions), deletions=len(deletions),
                          rollback_revision=repo['revision'])
        except Exception as exc:
            result.update(status='failed', **failure_details(exc))
        results['status'] = 'failed' if any(x['status'] == 'failed' for x in results['repositories']) else 'complete'
        if receipt:
            write(Path(receipt), results)
        if result['status'] == 'failed':
            break
    return results
