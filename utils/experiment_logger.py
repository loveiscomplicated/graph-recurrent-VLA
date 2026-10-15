"""Small local-first scalar logger; W&B is an optional transport.

JSON is used for config.yaml (JSON is a YAML 1.2 subset). No model, dataset,
checkpoint format, random state, or training schedule is owned by this module.
"""
import importlib
import json
import math
import netrc
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone
from uuid import uuid4
from urllib.parse import urlparse
import warnings

REPO = Path(__file__).resolve().parents[1]
PROJECT = 'graph-recurrent-vla'
GROUPS = {'Graph-Recurrent-VLA-controller': 'controller',
          'Graph-Recurrent-VLA-vision': 'vision',
          'Graph-Recurrent-VLA-graph': 'graph'}
SECRET = re.compile(r'api.?key|token|password|secret|credential', re.I)


def has_wandb_credentials():
    """Detect an env key or CLI-saved netrc entry without prompting/network IO.

    Only return presence; never copy credentials into config or metadata. W&B
    itself reads and uses the saved key when initializing an online run.
    """
    if os.environ.get('WANDB_API_KEY', '').strip():
        return True
    host = urlparse(os.getenv('WANDB_BASE_URL', 'https://api.wandb.ai')).hostname
    if not host:
        return False
    configured = os.getenv('NETRC')
    paths = [Path(configured).expanduser()] if configured else [
        Path.home() / '.netrc', Path.home() / '_netrc']
    for path in paths:
        try:
            entries = netrc.netrc(str(path)).hosts
            # Ignore a generic default entry: it may belong to another service.
            credentials = entries.get(host)
            if credentials and credentials[2]:
                return True
        except (OSError, netrc.NetrcParseError, UnicodeError):
            continue
    return False


def clean(value):
    """Persist explicit config only; redact credential fields recursively."""
    if isinstance(value, dict):
        return {str(k): '[REDACTED]' if SECRET.search(str(k)) else clean(v)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    raise TypeError(f'Expected JSON configuration or scalar, got {type(value).__name__}')


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def git_info():
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=REPO,
                                           stderr=subprocess.DEVNULL, timeout=3).decode().strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return dict(git_commit=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')))


def scalars(values, prefix=''):
    """Flatten dictionaries only; lists/tensors/videos are deliberately excluded."""
    result = {}
    for key, value in values.items():
        name = f'{prefix}/{key}' if prefix else str(key)
        if SECRET.search(str(key)):
            continue
        if isinstance(value, dict):
            result.update(scalars(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result[name] = value
    return result


class ExperimentLogger:
    def __init__(self, *, team, config=None, seed=None, model=None, group=None,
                 job_type='train', root=None):
        if team not in ('visual', 'graph', 'recurrent', 'integration'):
            raise ValueError('unknown team')
        self.run_id = uuid4().hex
        self.directory = Path(root or os.getenv('EXPERIMENTS_DIR', REPO / 'experiments')) / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        workspace = os.getenv('VESSL_WORKSPACE_NAME', '')
        self.group = group or os.getenv('WANDB_RUN_GROUP') or GROUPS.get(workspace) or {
            'visual': 'vision', 'recurrent': 'controller'}.get(team, team)
        self.config = clean(config or {})
        self.metadata = dict(run_id=self.run_id, team=team, group=self.group,
                             model=model, seed=seed, job_type=job_type,
                             project=PROJECT, entity=os.getenv('WANDB_ENTITY'),
                             started_at=datetime.now(timezone.utc).isoformat(),
                             status='running', outputs=[], **git_info())
        self.run = None
        self.closed = False
        self._metrics = (self.directory / 'metrics.json').open('w')
        # An incrementally valid JSON array; flush each event for error preservation.
        self._metrics.write('[]\n'); self._metrics.flush()
        self._first = True
        write(self.directory / 'config.yaml', self.config)
        self._save()
        requested = os.getenv('WANDB_MODE', 'online').lower()
        mode = requested if requested in ('offline', 'disabled') else (
            'online' if has_wandb_credentials() and os.getenv('WANDB_ENTITY') else 'offline')
        self.metadata['wandb_mode'] = mode
        if mode != 'disabled':
            try:
                wandb = importlib.import_module('wandb')
                self.run = wandb.init(project=PROJECT, entity=self.metadata['entity'],
                    id=self.run_id, group=self.group, job_type=job_type,
                    config=dict(parameters=self.config, experiment=self.metadata.copy()),
                    dir=str(self.directory), mode=mode, resume='never',
                    settings=wandb.Settings(init_timeout=10, save_code=False,
                                            disable_git=True, console='off'))
                self.metadata['wandb_directory'] = self.run.dir
                self.metadata['wandb_url'] = self.run.url if mode == 'online' else None
            except Exception as exc:
                self._failure('init', exc)
        self._save()

    def _save(self):
        write(self.directory / 'metadata.json', self.metadata)

    def _failure(self, operation, exc):
        # Exception text can contain authentication details; record type only.
        self.metadata.setdefault('tracking_errors', []).append(dict(operation=operation, type=type(exc).__name__))
        warnings.warn(f'Experiment tracking {operation} failed ({type(exc).__name__}); local metrics retained.', RuntimeWarning)

    def log(self, metrics, *, step=None, prefix=''):
        if self.closed:
            raise RuntimeError('experiment already finished')
        values = scalars(metrics, prefix)
        event = dict(run_id=self.run_id, step=step, metrics=values)
        self._metrics.seek(self._metrics.tell() - 2)
        self._metrics.write(('' if self._first else ',\n') + json.dumps(event, allow_nan=False) + ']\n')
        self._metrics.flush()
        self._first = False
        if self.run is not None:
            try:
                # Let W&B assign transport steps; epoch/evaluation coordinates are data.
                self.run.log(dict(values, **({'epoch': step} if step is not None else {})))
            except Exception as exc:
                self._failure('log', exc)
                self._save()

    def log_epoch(self, row):
        values = scalars(row)
        values.update(scalars(row.get('train_metrics', {}), 'train'))
        values.update(scalars(row.get('validation_metrics', {}), 'val'))
        if 'train_mse' in row: values['train/loss'] = row['train_mse']
        if 'val_mse' in row: values['val/loss'] = row['val_mse']
        self.log(values, step=row.get('epoch'))

    def record(self, **metadata):
        self.metadata.update(clean(metadata))
        self._save()
        if self.run is not None:
            try:
                self.run.config.update({'experiment': clean(self.metadata)}, allow_val_change=True)
            except Exception as exc:
                self._failure('config', exc)
                self._save()

    def link_output(self, path, *, checkpoint=False):
        """A sidecar associates unchanged existing artifacts with this run."""
        path = Path(path).resolve()
        if str(path) not in self.metadata['outputs']:
            self.metadata['outputs'].append(str(path)); self._save()
        sidecar = path.with_name(path.name + '.tracking.json')
        if checkpoint:
            if sidecar.exists():
                try:
                    previous = json.loads(sidecar.read_text())
                    self.record(source_run_id=previous.get('run_id'))
                except (OSError, ValueError, AttributeError) as exc:
                    self._failure('artifact', exc)
                    self._save()
            return
        write(sidecar, dict(run_id=self.run_id, experiment_directory=str(self.directory.resolve()),
                            artifact=str(path)))

    def finish(self, failed=False):
        if self.closed: return
        self.closed = True
        self.metadata.update(status='failed' if failed else 'finished',
                             finished_at=datetime.now(timezone.utc).isoformat())
        self._metrics.close()
        try:
            if self.run is not None: self.run.finish(exit_code=1 if failed else 0)
        except Exception as exc:
            self._failure('finish', exc)
        finally:
            self._save()

    def __enter__(self): return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self.metadata['exception_type'] = exc_type.__name__
        self.finish(failed=exc_type is not None)
        return False
