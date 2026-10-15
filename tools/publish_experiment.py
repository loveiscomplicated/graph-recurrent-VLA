"""Publish explicitly selected experiment artifacts, or initialize the two HF repos."""
import argparse
import json
import os
from pathlib import Path

from utils.experiment_artifacts import (DEFAULT_REPOSITORIES, Repository, initialize_repositories,
                                       publish, sync_wandb_links, validate_repositories)
from utils.experiment_logger import REPO


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run-id')
    mode.add_argument('--init-repos', action='store_true', help='explicitly create/reuse private repositories')
    mode.add_argument('--sync-wandb', metavar='RUN_ID', help='retry W&B link sync independently of HF')
    parser.add_argument('--experiments-dir', type=Path,
                        default=Path(os.getenv('EXPERIMENTS_DIR', REPO / 'experiments')))
    parser.add_argument('--include', nargs='+', choices=['checkpoint', 'rollouts', 'metrics'])
    parser.add_argument('--dry-run', action='store_true', help='validate files and show bytes without network calls')
    parser.add_argument('--private', action='store_true', help='require private upload targets (initialization is always private)')
    parser.add_argument('--repo-id', action='append', default=[], metavar='KIND=ORG/REPO',
                        help='override checkpoint or rollouts repository within Graph-Recurrent-VLA')
    parser.add_argument('--retries', type=int, default=2)
    args = parser.parse_args(argv)
    if args.run_id and not args.include:
        parser.error('--run-id requires an explicit --include selection')
    if not args.run_id and args.include:
        parser.error('--include requires --run-id')
    repositories = dict(DEFAULT_REPOSITORIES)
    try:
        for value in args.repo_id:
            key, repo_id = value.split('=', 1)
            if key not in repositories:
                raise ValueError('unknown repository kind')
            repositories[key] = Repository(repo_id, repositories[key].repo_type)
        validate_repositories(repositories)
        if args.init_repos:
            report = initialize_repositories(repositories, dry_run=args.dry_run)
        else:
            run_id = args.run_id or args.sync_wandb
            directory = args.experiments_dir.resolve() / run_id
            if directory.parent != args.experiments_dir.resolve():
                raise ValueError('invalid run_id')
            if args.sync_wandb:
                report = {'status': 'dry-run'} if args.dry_run else sync_wandb_links(directory)
            else:
                report = publish(directory, args.include, repositories=repositories,
                                 dry_run=args.dry_run, private=args.private, retries=args.retries)
    except Exception as exc:
        # Avoid printing raw SDK exceptions, JSON contents or credentials.
        report = dict(status='failed', error_type=type(exc).__name__,
                      reason=str(exc) if isinstance(exc, ValueError) else
                      'check run_id, completed run metadata, selected files and CLI options')
    print(json.dumps(report, indent=2))
    return 1 if report['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
