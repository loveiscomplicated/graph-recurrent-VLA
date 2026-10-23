"""Explicit experiment disaster-recovery snapshots and HF restoration."""
import argparse
import json
from pathlib import Path

from utils import experiment_backup as backup
from utils.experiment_artifacts import failure_details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('create')
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('publish')
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--repo-id', default=backup.DEFAULT_REPOSITORY)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--retries', type=int, default=2)
    p = sub.add_parser('init-repo')
    p.add_argument('--repo-id', default=backup.DEFAULT_REPOSITORY)
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('restore')
    p.add_argument('--repo-id', default=backup.DEFAULT_REPOSITORY)
    p.add_argument('--revision', required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('verify')
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--snapshot', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'create' and not args.dry_run and args.out is None:
        parser.error('create requires --out unless --dry-run')
    try:
        if args.command == 'create':
            report = backup.create(json.loads(args.spec.read_text()), args.out, dry_run=args.dry_run)
        elif args.command == 'publish':
            report = backup.publish(args.snapshot, args.repo_id, dry_run=args.dry_run, retries=args.retries)
        elif args.command == 'init-repo':
            report = backup.init_repo(args.repo_id, dry_run=args.dry_run)
        elif args.command == 'restore':
            report = backup.restore(args.repo_id, args.revision, args.prefix, args.out)
        else:
            report = backup.verify(args.directory, snapshot=args.snapshot)
    except Exception as exc:
        report = dict(status='failed', **failure_details(exc))
        # Only our validation messages are safe; SDK exception strings can contain secrets.
        if isinstance(exc, backup.BackupValidationError):
            report['reason'] = str(exc)
    print(json.dumps(report, indent=2))
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
