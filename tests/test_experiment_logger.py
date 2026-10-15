"""Local and mocked transport contracts; never needs credentials or a simulator."""
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

from utils.experiment_logger import ExperimentLogger, GROUPS, has_wandb_credentials


class ExperimentLoggerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {'EXPERIMENTS_DIR': self.temp.name, 'NETRC': str(Path(self.temp.name)/'netrc')}, clear=True)
        self.env.start(); self.addCleanup(self.env.stop)

    def logger(self, **kwargs):
        return ExperimentLogger(team='graph', **kwargs)

    def save_credentials(self, host='api.wandb.ai'):
        path = Path(os.environ['NETRC'])
        path.write_text(f'machine {host}\n  login user\n  password mock-key\n')
        path.chmod(0o600)

    def test_cli_saved_login_selects_online_without_env_key(self):
        self.save_credentials()
        sdk = Mock(); sdk.init.return_value.dir = '/mock'; sdk.init.return_value.url = 'https://example.test'
        with patch.dict(os.environ, WANDB_ENTITY='central'), patch('utils.experiment_logger.importlib.import_module', return_value=sdk):
            with self.logger() as log: log.log({'loss': 1})
        self.assertEqual(sdk.init.call_args.kwargs['mode'], 'online')
        self.assertNotIn('WANDB_API_KEY', os.environ)
        for name in ('metadata.json', 'config.yaml', 'metrics.json'):
            self.assertNotIn('mock-key', (log.directory/name).read_text())

    def test_saved_login_requires_entity_and_matching_host(self):
        self.save_credentials(host='unrelated.example')
        self.assertFalse(has_wandb_credentials())
        self.save_credentials()
        sdk = Mock(); sdk.init.return_value.dir = '/mock'
        with patch('utils.experiment_logger.importlib.import_module', return_value=sdk):
            with self.logger(): pass
        self.assertEqual(sdk.init.call_args.kwargs['mode'], 'offline')
        with patch.dict(os.environ, WANDB_BASE_URL='https://unrelated.example'):
            self.assertFalse(has_wandb_credentials())

    def test_default_home_login_and_corrupt_file(self):
        Path(os.environ['NETRC']).write_text('machine')
        self.assertFalse(has_wandb_credentials())
        self.save_credentials()
        Path(os.environ['NETRC']).rename(Path(self.temp.name)/'.netrc')
        with patch.dict(os.environ):
            del os.environ['NETRC']
            with patch('utils.experiment_logger.Path.home', return_value=Path(self.temp.name)):
                self.assertTrue(has_wandb_credentials())

    def test_json_sync_accepts_saved_login(self):
        from utils.sync_experiment import sync
        with patch.dict(os.environ, WANDB_MODE='disabled'):
            with self.logger() as log: log.log({'loss': 1})
        self.save_credentials()
        sdk = Mock()
        with patch.dict(os.environ, WANDB_ENTITY='central'), patch.dict('sys.modules', wandb=sdk):
            sync(log.directory)
        self.assertEqual(sdk.init.call_args.kwargs['mode'], 'online')

    def test_online_contract(self):
        sdk = Mock(); sdk.init.return_value.dir = '/mock/run'; sdk.init.return_value.url = 'https://example.test'
        with patch.dict(os.environ, WANDB_API_KEY='not-a-real-key', WANDB_ENTITY='central'), patch('utils.experiment_logger.importlib.import_module', return_value=sdk):
            with self.logger(config={'batch_size': 8}, seed=7, model='flat') as log:
                log.log_epoch({'epoch': 1, 'train_mse': .5, 'val_mse': .6, 'learning_rate': .001})
                log.log({'success_rate': .3}, prefix='eval')
                log.record(source_run_id='source')
            args = sdk.init.call_args.kwargs
            self.assertEqual((args['mode'], args['project'], args['entity'], args['id']), ('online', 'graph-recurrent-vla', 'central', log.run_id))
            self.assertEqual(args['config']['experiment']['seed'], 7)
            self.assertEqual(len(args['config']['experiment']['git_commit']), 40)
            sdk.init.return_value.finish.assert_called_once_with(exit_code=0)
            self.assertEqual(sdk.init.return_value.log.call_count, 2)
            self.assertEqual(sdk.init.return_value.config.update.call_args.args[0]['experiment']['source_run_id'], 'source')

    def test_missing_auth_forces_offline_and_redacts(self):
        sdk = Mock(); sdk.init.return_value.dir = '/mock'
        with patch.dict(os.environ, WANDB_MODE='online'), patch('utils.experiment_logger.importlib.import_module', return_value=sdk):
            with self.logger(config={'nested': {'WANDB_API_KEY': 'secret'}}, seed=0) as log:
                log.log({'loss': 1., 'large': [1, 2], 'nan': float('nan')})
                log.log({'loss': .5})
                rows = json.loads((log.directory/'metrics.json').read_text())
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0]['metrics'], {'loss': 1.})
        self.assertEqual(sdk.init.call_args.kwargs['mode'], 'offline')
        self.assertNotIn('secret', (log.directory/'config.yaml').read_text())

    def test_no_sdk_and_error_preservation(self):
        with patch('utils.experiment_logger.importlib.import_module', side_effect=ImportError), self.assertWarns(RuntimeWarning):
            with self.assertRaisesRegex(ValueError, 'training failure'):
                with self.logger() as log:
                    log.log({'loss': 1.})
                    raise ValueError('training failure')
        metadata = json.loads((log.directory/'metadata.json').read_text())
        self.assertEqual(metadata['status'], 'failed')
        self.assertEqual(metadata['exception_type'], 'ValueError')
        self.assertEqual(len(json.loads((log.directory/'metrics.json').read_text())), 1)

    def test_transport_failures_do_not_mask_training(self):
        sdk = Mock(); sdk.init.return_value.dir = '/mock'
        sdk.init.return_value.log.side_effect = OSError('secret must not appear')
        sdk.init.return_value.finish.side_effect = RuntimeError('secret')
        with patch('utils.experiment_logger.importlib.import_module', return_value=sdk), self.assertWarns(RuntimeWarning):
            with self.logger() as log: log.log({'loss': 2.})
        self.assertEqual(json.loads((log.directory/'metadata.json').read_text())['status'], 'finished')
        self.assertNotIn('secret', (log.directory/'metadata.json').read_text())

    def test_init_failure_and_disabled(self):
        with patch('utils.experiment_logger.importlib.import_module', side_effect=RuntimeError), self.assertWarns(RuntimeWarning):
            with self.logger() as log: log.log({'loss': 1})
        with patch.dict(os.environ, WANDB_MODE='disabled'), patch('utils.experiment_logger.importlib.import_module') as imp:
            with self.logger(): pass
            imp.assert_not_called()

    def test_concurrent_unique_ids_and_groups(self):
        with patch.dict(os.environ, WANDB_MODE='disabled'):
            def create(_):
                with self.logger() as log: return log.run_id
            with ThreadPoolExecutor(max_workers=8) as pool:
                ids = list(pool.map(create, range(32)))
            self.assertEqual(len(set(ids)), 32)
            for workspace, group in GROUPS.items():
                with patch.dict(os.environ, VESSL_WORKSPACE_NAME=workspace):
                    with self.logger() as log: self.assertEqual(log.group, group)
            for team, group in [('visual', 'vision'), ('recurrent', 'controller'), ('graph', 'graph')]:
                with ExperimentLogger(team=team) as log: self.assertEqual(log.group, group)

    def test_artifact_link_and_evaluation_parent(self):
        path = Path(self.temp.name)/'best.pt'; path.write_bytes(b'unchanged')
        with patch.dict(os.environ, WANDB_MODE='disabled'):
            with self.logger() as train: train.link_output(path)
            with self.logger(job_type='evaluate') as evaluation:
                evaluation.link_output(path, checkpoint=True)
                self.assertEqual(evaluation.metadata['source_run_id'], train.run_id)
        self.assertEqual(path.read_bytes(), b'unchanged')
        self.assertEqual(json.loads(path.with_name('best.pt.tracking.json').read_text())['run_id'], train.run_id)

    def test_json_sync_preserves_identity_and_failure_status(self):
        from utils.sync_experiment import sync
        with patch.dict(os.environ, WANDB_MODE='disabled'):
            log = self.logger(config={'batch_size': 3}, seed=4)
            log.log({'loss': 1.}, step=1)
            log.finish(failed=True)
        sdk = Mock()
        with patch.dict(os.environ, WANDB_API_KEY='mock', WANDB_ENTITY='central'), patch.dict('sys.modules', wandb=sdk):
            sync(log.directory)
        self.assertEqual(sdk.init.call_args.kwargs['id'], log.run_id)
        self.assertEqual(sdk.init.call_args.kwargs['resume'], 'never')
        sdk.init.return_value.finish.assert_called_once_with(exit_code=1)
        sdk.init.return_value.log.assert_called_once_with({'loss': 1., 'epoch': 1})

    def test_json_sync_requires_authentication_and_finished_run(self):
        from utils.sync_experiment import sync
        with patch.dict(os.environ, WANDB_MODE='disabled'), patch.dict('sys.modules', wandb=Mock()):
            with self.logger() as log:
                with self.assertRaises(ValueError): sync(log.directory)
            with self.assertRaises(ValueError): sync(log.directory)

    def test_real_offline_sdk(self):
        try: import wandb
        except ImportError: self.skipTest('optional wandb not installed')
        # Isolate the SDK service lifecycle from unittest's boolean exit code.
        script = """
from utils.experiment_logger import ExperimentLogger
with ExperimentLogger(team='graph', config={'learning_rate': .01}, seed=3) as log:
    log.log({'loss': .25})
    assert log.run is not None
assert list(log.directory.glob('wandb/offline-run-*/*.wandb'))
"""
        env = dict(os.environ, WANDB_MODE='offline', WANDB_CACHE_DIR=self.temp.name)
        result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('Traceback', result.stderr)



if __name__ == '__main__': unittest.main()
