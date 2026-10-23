"""Credential-free artifact contracts; no simulator or external services."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

from utils.experiment_logger import ExperimentLogger, write
from utils.experiment_artifacts import (DEFAULT_REPOSITORIES, Repository, connect, digests,
    create_report_bundle, initialize_repositories, load_hf_env, plan_upload, publish,
    register_existing_output, remote_matches, save_manifest, sync_wandb_links)
from tools.publish_experiment import main


class FakeHub:
    def __init__(self):
        self.files = {}
        self.commits = 0
        self.fail_before = 0
        self.fail_after = 0
        self.private = True
        self.revisions = {}

    def repo_info(self, repo_id, **kwargs):
        return SimpleNamespace(sha=self.revisions.get(repo_id, 'initial'), private=self.private)

    def list_repo_tree(self, repo_id, **kwargs):
        return [SimpleNamespace(path=path, size=len(body), blob_id=digests(body)[2], lfs=None)
                for (repo, path), body in self.files.items() if repo == repo_id]

    def create_commit(self, repo_id, operations, parent_commit, **kwargs):
        if self.fail_before:
            self.fail_before -= 1
            raise OSError('hf_SECRET must never appear')
        assert parent_commit == self.revisions.get(repo_id, 'initial')
        self.commits += 1
        for operation in operations:
            source = operation.path_or_fileobj
            self.files[repo_id, operation.path_in_repo] = source if isinstance(source, bytes) else source.read_bytes()
        revision = f'commit-{self.commits}'
        self.revisions[repo_id] = revision
        if self.fail_after:
            self.fail_after -= 1
            raise OSError('lost response')
        return SimpleNamespace(oid=revision)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        env = patch.dict(os.environ, {'WANDB_MODE': 'disabled'}, clear=True)
        env.start(); self.addCleanup(env.stop)
        os.environ['HF_ENV_FILE'] = str(self.root/'missing-hf.env')
        self.hub = FakeHub()

    def experiment(self, team='graph'):
        with ExperimentLogger(team=team, root=self.root/'experiments', seed=7, model='flat',
                              config={'batch_size': 8, 'HF_TOKEN': 'hf_SECRET'}) as log:
            log.record(environment={'env_id': 'PegInsertionSide-v1'})
            log.log({'loss': .4}, step=1)
            output = self.root / log.run_id
            output.mkdir()
            for filename, kind in [('best.pt', None), ('episode.mp4', 'rollouts'),
                                   ('episode.json', 'episode_metadata'), ('trace.h5', 'trajectory')]:
                path = output / filename
                path.write_text('{"success": true, "token": "hf_SECRET"}' if filename.endswith('.json') else filename)
                log.link_output(path, kind=kind)
        return log

    def transport(self):
        return patch('utils.experiment_artifacts.connect', return_value=(self.hub, SimpleNamespace, None))

    def test_report_bundle_keeps_relative_assets_identity_and_redacts_json(self):
        log = self.experiment('visual')
        report = self.root / 'report.html'
        report.write_text('<img src="assets/frame.png"><a href="depth.mp4">video</a>')
        image = self.root / 'frame.png'; image.write_bytes(b'PNG')
        video = self.root / 'depth.mp4'; video.write_bytes(b'VIDEO')
        data = self.root / 'data.json'; data.write_text('{"score": 1, "HF_TOKEN": "bundle-secret"}')
        selection = dict(entrypoint='report.html', publication_commit='committed-source', files={
            'report.html': str(report), 'assets/frame.png': str(image),
            'depth.mp4': str(video), 'data.json': {'source': str(data), 'role': 'input'}})
        original = json.loads((log.directory / 'metadata.json').read_text())
        with patch('utils.experiment_artifacts.connect') as connect_mock:
            archive = create_report_bundle(log.directory, selection)
            repeated = create_report_bundle(log.directory, dict(selection, files=dict(reversed(list(selection['files'].items())))))
            connect_mock.assert_not_called()
        self.assertEqual(archive, repeated)
        with ZipFile(archive) as bundle:
            self.assertEqual(bundle.read('report.html'), report.read_bytes())
            self.assertEqual(bundle.read('assets/frame.png'), b'PNG')
            self.assertEqual(bundle.read('depth.mp4'), b'VIDEO')
            self.assertNotIn(b'bundle-secret', bundle.read('data.json'))
            self.assertEqual(json.loads(bundle.read('data.json'))['HF_TOKEN'], '[REDACTED]')
            manifest = json.loads(bundle.read('bundle_manifest.json'))
            self.assertEqual(manifest['run_id'], log.run_id)
            self.assertEqual(manifest['entrypoint'], 'report.html')
            self.assertEqual(manifest['publication_commit'], 'committed-source')
            record = next(r for r in manifest['files'] if r['path'] == 'data.json')
            self.assertEqual(record['source_sha256'], digests(data)[1])
            self.assertEqual(record['sha256'], digests(bundle.read('data.json'))[1])
            self.assertNotIn(b'hf_SECRET', bundle.read('experiment/config.yaml'))
        current = json.loads((log.directory / 'metadata.json').read_text())
        for field in ('run_id', 'team', 'seed', 'git_commit', 'git_dirty', 'status'):
            self.assertEqual(current[field], original[field])
        self.assertEqual(data.read_text(), '{"score": 1, "HF_TOKEN": "bundle-secret"}')
        sidecar = json.loads(archive.with_name(archive.name + '.tracking.json').read_text())
        self.assertEqual(sidecar['run_id'], log.run_id)

    def test_reports_and_metrics_use_dataset_without_requiring_rollout_video(self):
        log = self.experiment('visual')
        report = self.root / 'report.html'; report.write_text('<p>diagnostic</p>')
        register_existing_output(log.directory, report, kind='reports')
        archive = create_report_bundle(log.directory, {'files': {'report.html': str(report)}})
        with self.transport():
            result = publish(log.directory, ['reports', 'metrics'])
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(set(result['repositories']), {'rollouts'})
        paths = [path for repo, path in self.hub.files]
        self.assertTrue(any(path.endswith(archive.name) for path in paths))
        self.assertTrue(any(path.endswith('report.html') for path in paths))
        self.assertFalse(any(path.endswith(('.mp4', '.pt', '.h5')) for path in paths))
        self.assertEqual(set(plan_upload(log.directory, ['metrics'])), {'checkpoint'})

    def test_legacy_html_metrics_migrates_without_changing_original_run(self):
        log = self.experiment('visual')
        report = self.root / 'report.html'; report.write_text('<p>HTML</p>')
        register_existing_output(log.directory, report, kind='metrics')
        with self.assertRaisesRegex(ValueError, 'metrics artifacts must be JSON'):
            plan_upload(log.directory, ['metrics'])
        register_existing_output(log.directory, report, kind='reports')
        plan = plan_upload(log.directory, ['reports', 'metrics'])
        manifest = json.loads((log.directory / 'manifest.json').read_text())
        self.assertEqual(manifest['run_id'], log.run_id)
        self.assertEqual(len(manifest['report_paths']), 1)
        self.assertNotIn(str(report), manifest['additional_metrics_paths'])
        self.assertEqual(set(plan), {'rollouts'})

    def test_bundle_rejects_unsafe_members_missing_files_and_foreign_results(self):
        log = self.experiment()
        source = self.root / 'data.json'; source.write_text('{"value": 1}')
        for name in ('../escape', '/absolute', '.', './alias', 'dir/../alias', 'a\\b', 'bundle_manifest.json', 'experiment/config.yaml'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                create_report_bundle(log.directory, {'files': {name: str(source)}})
        with self.assertRaisesRegex(ValueError, 'missing'):
            create_report_bundle(log.directory, {'files': {'missing': str(self.root / 'missing')}})
        with self.assertRaisesRegex(ValueError, 'entrypoint'):
            create_report_bundle(log.directory, {'entrypoint': 'missing', 'files': {'data.json': str(source)}})
        write(source.with_name(source.name + '.tracking.json'), {'run_id': 'foreign-run'})
        with self.assertRaisesRegex(ValueError, 'belongs to another run'):
            create_report_bundle(log.directory, {'files': {'data.json': str(source)}})
        self.assertFalse((log.directory / 'reports').exists())
        archive = create_report_bundle(log.directory, {'files': {'data.json': {'source': str(source), 'role': 'dependency'}}})
        with ZipFile(archive) as bundle:
            self.assertEqual(json.loads(bundle.read('bundle_manifest.json'))['files'][0]['source_run_id'], 'foreign-run')

    def test_bundle_rejects_symlink_and_corrupted_content_addressed_archive(self):
        log = self.experiment()
        source = self.root / 'source.bin'; source.write_bytes(b'input')
        symlink = self.root / 'symlink.bin'; symlink.symlink_to(source)
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            create_report_bundle(log.directory, {'files': {'source.bin': str(symlink)}})
        selection = {'files': {'source.bin': str(source)}}
        archive = create_report_bundle(log.directory, selection)
        archive.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'archive differs'):
            create_report_bundle(log.directory, selection)
        self.assertEqual(archive.read_bytes(), b'corrupted')

    def test_cli_report_bundle_dry_run_is_network_free(self):
        log = self.experiment('visual')
        source = self.root / 'report.html'; source.write_text('<p>report</p>')
        selection = self.root / 'selection.json'
        selection.write_text(json.dumps({'files': {'report.html': str(source)}}))
        with patch('utils.experiment_artifacts.connect') as connect_mock, redirect_stdout(io.StringIO()) as output:
            code = main(['--experiments-dir', str(log.directory.parent), '--run-id', log.run_id,
                         '--include', 'reports', 'metrics', '--report-files', str(selection), '--dry-run'])
        connect_mock.assert_not_called()
        self.assertEqual(code, 0)
        self.assertEqual(set(json.loads(output.getvalue())['repositories']), {'rollouts'})

    def test_register_source_artifact_preserves_original_sidecar_owner(self):
        log = self.experiment()
        source = Path(next(p for p in log.metadata['outputs'] if p.endswith('.pt')))
        original = source.with_name(source.name + '.tracking.json').read_bytes()
        with ExperimentLogger(team='graph', root=log.directory.parent) as evaluation:
            evaluation.record(source_run_id=log.run_id)
        register_existing_output(evaluation.directory, source, kind='checkpoint')
        self.assertEqual(source.with_name(source.name + '.tracking.json').read_bytes(), original)

    def test_manifest_generated_by_existing_logger_and_legacy_regeneration(self):
        log = self.experiment()
        manifest = json.loads((log.directory/'manifest.json').read_text())
        self.assertEqual(manifest['run_id'], log.run_id)
        self.assertEqual(manifest['wandb_run_id'], log.run_id)
        self.assertEqual(manifest['random_seed'], 7)
        self.assertEqual(manifest['model_name'], 'flat')
        self.assertEqual(manifest['environment']['env_id'], 'PegInsertionSide-v1')
        for field in ('checkpoint_paths', 'rollout_video_paths', 'trajectory_paths', 'episode_metadata_paths'):
            self.assertEqual(len(manifest[field]), 1)
            self.assertTrue((log.directory/manifest[field][0]).is_file())
        self.assertEqual(manifest['config_path'], 'config.yaml')
        (log.directory/'manifest.json').unlink()
        self.assertEqual(save_manifest(log.directory), manifest)
        self.assertEqual(json.loads((log.directory/'metrics.json').read_text())[0]['metrics']['loss'], .4)

    def test_selective_checkpoint_and_reproducibility(self):
        log = self.experiment()
        plans = plan_upload(log.directory, ['checkpoint'])
        self.assertEqual(set(plans), {'checkpoint'})
        paths = [f.path for f in plans['checkpoint']['files']]
        self.assertTrue(all(p.startswith(f'graph/{log.run_id}/') for p in paths))
        self.assertFalse(any('.mp4' in p or '.h5' in p or 'metrics.json' in p for p in paths))
        manifest = json.loads(plans['checkpoint']['files'][-1].source)
        self.assertTrue(manifest['config_path'].startswith(f'graph/{log.run_id}/'))
        self.assertNotIn('hf_SECRET', b''.join(f.source for f in plans['checkpoint']['files'] if isinstance(f.source, bytes)).decode())

    def test_rollouts_metadata_and_metrics_selection(self):
        log = self.experiment('visual')
        with self.transport():
            report = publish(log.directory, ['rollouts', 'metrics'])
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(set(report['repositories']), {'rollouts'})
        paths = [path for repo, path in self.hub.files]
        self.assertTrue(any('.mp4' in p for p in paths))
        self.assertTrue(any('/episode_metadata/' in p for p in paths))
        self.assertTrue(any('metrics.json' in p for p in paths))
        self.assertFalse(any('.pt' in p or '.h5' in p for p in paths))
        self.assertTrue(all(repo == DEFAULT_REPOSITORIES['rollouts'].repo_id for repo, path in self.hub.files))
        json_bodies = [body for (repo, path), body in self.hub.files.items() if path.endswith(('.json', '.yaml'))]
        self.assertNotIn('hf_SECRET', b''.join(json_bodies).decode())
        receipt = json.loads((log.directory/'huggingface_uploads.json').read_text())
        publication = next(iter(receipt['publications'].values()))
        self.assertEqual(publication['revision'], 'commit-1')
        self.assertIn('/datasets/Graph-Recurrent-VLA/', publication['url'])

    def test_metrics_only_and_missing_optional_artifacts(self):
        with ExperimentLogger(team='recurrent', root=self.root) as log:
            log.log({'loss': 1})
        self.assertEqual(set(plan_upload(log.directory, ['metrics'])), {'checkpoint'})
        with self.assertRaisesRegex(ValueError, 'no registered rollouts'):
            plan_upload(log.directory, ['rollouts'])
        with self.assertRaises(ValueError):
            plan_upload(log.directory, [])

    def test_missing_selected_file_but_unselected_missing_does_not_block(self):
        log = self.experiment()
        manifest = save_manifest(log.directory)
        (log.directory/manifest['rollout_video_paths'][0]).unlink()
        with self.assertRaisesRegex(ValueError, 'missing selected rollouts'):
            plan_upload(log.directory, ['rollouts'])
        plan_upload(log.directory, ['checkpoint'])
        (log.directory/manifest['checkpoint_paths'][0]).write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'empty selected'):
            plan_upload(log.directory, ['checkpoint'])

    def test_dry_run_no_sdk_or_network_and_cli_requires_selection(self):
        log = self.experiment()
        with patch('utils.experiment_artifacts.connect') as client:
            report = publish(log.directory, ['checkpoint'], dry_run=True)
            self.assertGreater(report['total_bytes'], 0)
            self.assertEqual(report['status'], 'dry-run')
            self.assertEqual(initialize_repositories(dry_run=True)['status'], 'dry-run')
            client.assert_not_called()
        self.assertFalse((log.directory/'huggingface_uploads.json').exists())
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['--run-id', log.run_id, '--experiments-dir', str(log.directory.parent),
                                   '--include', 'checkpoint', '--dry-run']), 0)
        self.assertEqual(json.loads(output.getvalue())['status'], 'dry-run')
        with patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit):
            main(['--run-id', log.run_id])

    def test_missing_authentication_no_api_calls_and_no_sdk(self):
        hub = Mock(); hub.get_token.return_value = None
        with patch('utils.experiment_artifacts.importlib.import_module', return_value=hub):
            self.assertIn('no Hugging Face token', connect()[2])
            self.assertEqual(initialize_repositories()['status'], 'skipped')
            self.assertEqual(publish(self.experiment().directory, ['checkpoint'])['status'], 'skipped')
        hub.HfApi.assert_not_called()
        with patch('utils.experiment_artifacts.importlib.import_module', side_effect=ImportError):
            self.assertIn('missing', connect()[2])

    def test_hf_dotenv_loading_and_environment_precedence(self):
        env_file = self.root/'hf.env'
        env_file.write_text('HF_TOKEN=file-test-token\n')
        with patch.dict(os.environ, HF_ENV_FILE=str(env_file)):
            self.assertTrue(load_hf_env())
            self.assertEqual(os.environ['HF_TOKEN'], 'file-test-token')
        with patch.dict(os.environ, HF_ENV_FILE=str(env_file), HF_TOKEN='environment-test-token'):
            load_hf_env()
            self.assertEqual(os.environ['HF_TOKEN'], 'environment-test-token')
        with patch('utils.experiment_artifacts.DEFAULT_HF_ENV_FILE', env_file), patch.dict(os.environ):
            os.environ.pop('HF_ENV_FILE', None)
            load_hf_env()
            self.assertEqual(os.environ['HF_TOKEN'], 'file-test-token')
        self.assertFalse(load_hf_env(self.root/'absent.env'))
        self.assertNotIn('HF_TOKEN', os.environ)

    def test_hf_dotenv_loaded_before_sdk_auth_without_persisting_token(self):
        log = self.experiment()
        env_file = self.root/'hf.env'
        env_file.write_text('HF_TOKEN=dotenv-test-secret\n')
        hub = Mock()
        hub.get_token.side_effect = lambda: os.environ.get('HF_TOKEN')
        with patch.dict(os.environ, HF_ENV_FILE=str(env_file)), patch(
                'utils.experiment_artifacts.importlib.import_module', return_value=hub):
            api, operation, reason = connect()
            self.assertIsNone(reason)
            hub.HfApi.assert_called_once()
        for name in ('config.yaml', 'metadata.json', 'manifest.json', 'metrics.json'):
            self.assertNotIn('dotenv-test-secret', (log.directory/name).read_text())

    def test_auth_check_script_reports_presence_without_revealing_token(self):
        from scripts.check_huggingface_auth import main as check_auth
        with patch.dict(os.environ, HF_TOKEN='script-test-secret'), patch.dict(
                'sys.modules', {'huggingface_hub': SimpleNamespace(get_token=lambda: 'script-test-secret')}), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(check_auth(), 0)
        self.assertIn('사용 가능한 HF token: True', output.getvalue())
        self.assertNotIn('script-test-secret', output.getvalue())

    def test_auth_check_script_missing_token_returns_failure(self):
        from scripts.check_huggingface_auth import main as check_auth
        with patch.dict('sys.modules', {'huggingface_hub': SimpleNamespace(get_token=lambda: None)}), redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(check_auth(), 1)
        self.assertIn('사용 가능한 HF token: False', output.getvalue())
        self.assertIn('설정을 확인하세요', errors.getvalue())

    def test_auth_check_script_exception_does_not_reveal_token(self):
        from scripts.check_huggingface_auth import main as check_auth
        hub = SimpleNamespace(get_token=Mock(side_effect=OSError('script-test-secret')))
        with patch.dict('sys.modules', {'huggingface_hub': hub}), redirect_stdout(io.StringIO()) as output, redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(check_auth(), 1)
        self.assertIn('OSError', errors.getvalue())
        self.assertNotIn('script-test-secret', output.getvalue() + errors.getvalue())

    def test_duplicate_upload_checked_remotely_even_without_local_receipt(self):
        log = self.experiment()
        with self.transport():
            publish(log.directory, ['checkpoint', 'rollouts', 'metrics'])
            (log.directory/'huggingface_uploads.json').unlink()
            report = publish(log.directory, ['checkpoint', 'rollouts', 'metrics'])
        self.assertEqual(self.hub.commits, 2)
        self.assertTrue(all(r['status'] == 'unchanged' for r in report['repositories'].values()))

    def test_lfs_deduplication_supports_sdk_objects_and_dicts(self):
        item = plan_upload(self.experiment().directory, ['checkpoint'])['checkpoint']['files'][0]
        for lfs in ({'sha256': item.sha256}, SimpleNamespace(sha256=item.sha256)):
            remote = SimpleNamespace(size=item.size, lfs=lfs, blob_id='lfs-pointer-git-id')
            self.assertTrue(remote_matches(item, remote))
            remote.size += 1
            self.assertFalse(remote_matches(item, remote))

    def test_failure_then_rerun_and_retry_after_lost_commit_response(self):
        log = self.experiment()
        self.hub.fail_before = 1
        with self.transport():
            failed = publish(log.directory, ['checkpoint'], retries=0)
            self.assertEqual(failed['status'], 'failed')
            self.assertNotIn('hf_SECRET', json.dumps(failed))
            self.assertFalse((log.directory/'huggingface_uploads.json').exists())
            self.assertEqual(publish(log.directory, ['checkpoint'], retries=0)['status'], 'complete')
            self.hub.fail_after = 1
            with patch('utils.experiment_artifacts.time.sleep'):
                report = publish(log.directory, ['rollouts'], retries=1)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(report['repositories']['rollouts']['status'], 'unchanged')
        self.assertEqual(self.hub.commits, 2)

    def test_partial_success_receipt_survives_other_repo_failure(self):
        log = self.experiment()
        original = self.hub.create_commit
        def commit(repo_id, **kwargs):
            if repo_id == DEFAULT_REPOSITORIES['rollouts'].repo_id:
                raise OSError('hf_SECRET')
            return original(repo_id, **kwargs)
        with self.transport(), patch.object(self.hub, 'create_commit', side_effect=commit):
            report = publish(log.directory, ['checkpoint', 'rollouts'], retries=0)
        self.assertEqual(report['status'], 'failed')
        receipts = json.loads((log.directory/'huggingface_uploads.json').read_text())
        self.assertEqual(len(receipts['publications']), 1)
        with self.transport():
            report = publish(log.directory, ['checkpoint', 'rollouts'], retries=0)
        self.assertEqual(report['repositories']['checkpoint']['status'], 'unchanged')
        self.assertEqual(report['status'], 'complete')

    def test_no_overwrite_and_new_content_and_different_runs(self):
        log = self.experiment()
        with self.transport():
            publish(log.directory, ['checkpoint'])
            old = dict(self.hub.files)
            artifact = next(p for p in log.metadata['outputs'] if p.endswith('.pt'))
            Path(artifact).write_bytes(b'new weights')
            publish(log.directory, ['checkpoint'])
            self.assertTrue(all(self.hub.files[k] == v for k, v in old.items()))
            another = self.experiment()
            other_plan = plan_upload(another.directory, ['checkpoint'])
            self.assertNotEqual(other_plan['checkpoint']['prefix'], f'graph/{log.run_id}')
            file_key = next(k for k in self.hub.files if k[1].endswith('best.pt') and self.hub.files[k] == b'new weights')
            self.hub.files[file_key] = b'corrupt'
            report = publish(log.directory, ['checkpoint'], retries=0)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(self.hub.files[file_key], b'corrupt')

    def test_stale_owner_rejected_but_evaluation_source_checkpoint_supported(self):
        log = self.experiment()
        artifact = next(p for p in log.metadata['outputs'] if p.endswith('.pt'))
        with ExperimentLogger(team='graph', root=log.directory.parent) as evaluation:
            evaluation.link_output(artifact, checkpoint=True)
        plan_upload(evaluation.directory, ['checkpoint'])
        write(Path(artifact + '.tracking.json'), {'run_id': 'different-run'})
        with self.assertRaisesRegex(ValueError, 'belongs to another run'):
            plan_upload(log.directory, ['checkpoint'])

    def test_reject_running_run_symlink_and_outside_organization(self):
        with ExperimentLogger(team='graph', root=self.root) as log:
            with self.assertRaisesRegex(ValueError, 'finish the experiment'):
                plan_upload(log.directory, ['metrics'])
        repos = dict(DEFAULT_REPOSITORIES, checkpoint=Repository('someone/model', 'model'))
        with self.assertRaises(ValueError):
            plan_upload(log.directory, ['metrics'], repos)
        with self.assertRaises(ValueError):
            initialize_repositories({'trajectory': Repository('Graph-Recurrent-VLA/later', 'dataset')})
        log = self.experiment()
        path = Path(next(p for p in log.metadata['outputs'] if p.endswith('.pt')))
        path.unlink(); path.symlink_to(log.directory/'metrics.json')
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            plan_upload(log.directory, ['checkpoint'])

    def test_private_check_does_not_create_or_change_repository(self):
        self.hub.private = False
        with self.transport():
            report = publish(self.experiment().directory, ['checkpoint'], private=True, retries=0)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(self.hub.commits, 0)

    def test_initialization_only_two_private_repos_and_auth_failure(self):
        api = Mock()
        api.whoami.return_value = {'orgs': [{'name': 'Graph-Recurrent-VLA', 'roleInOrg': 'write'}]}
        api.create_repo.side_effect = ['https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models', OSError('hf_SECRET')]
        with patch('utils.experiment_artifacts.connect', return_value=(api, None, None)):
            result = initialize_repositories()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['repositories']['checkpoint']['status'], 'ready')
        self.assertNotIn('hf_SECRET', json.dumps(result))
        self.assertEqual(api.create_repo.call_count, 2)
        for call in api.create_repo.call_args_list:
            self.assertTrue(call.kwargs['private'])
            self.assertTrue(call.kwargs['exist_ok'])
        api.reset_mock(); api.whoami.return_value = {'orgs': []}
        with patch('utils.experiment_artifacts.connect', return_value=(api, None, None)):
            self.assertEqual(initialize_repositories()['status'], 'skipped')
        api.create_repo.assert_not_called()
        api.whoami.side_effect = None
        api.whoami.return_value = {'orgs': [{'name': 'Graph-Recurrent-VLA', 'roleInOrg': 'admin'}],
                                  'auth': {'accessToken': {'role': 'read'}}}
        with patch('utils.experiment_artifacts.connect', return_value=(api, None, None)):
            result = initialize_repositories()
        self.assertEqual(result['status'], 'skipped')
        self.assertIn('read-only', result['reason'])
        api.create_repo.assert_not_called()
        api.whoami.side_effect = OSError('hf_SECRET')
        with patch('utils.experiment_artifacts.connect', return_value=(api, None, None)):
            self.assertEqual(initialize_repositories()['status'], 'skipped')
        api.create_repo.assert_not_called()

    def test_wandb_sync_is_independent_retryable_and_uses_original_identity(self):
        log = self.experiment()
        with self.transport():
            publish(log.directory, ['checkpoint'])
        result = sync_wandb_links(log.directory)
        self.assertEqual(result['status'], 'skipped')
        metadata = log.metadata.copy(); metadata['entity'] = 'team'
        write(log.directory/'metadata.json', metadata)
        sdk = Mock(); sdk.Api.return_value.run.return_value.summary = Mock()
        sdk.Api.return_value.run.return_value.summary.__setitem__ = Mock()
        with patch.dict(os.environ, WANDB_API_KEY='wandb-secret'), patch(
                'utils.experiment_artifacts.importlib.import_module', return_value=sdk):
            self.assertEqual(sync_wandb_links(log.directory)['status'], 'complete')
            sdk.Api.return_value.run.assert_called_once_with(f'team/graph-recurrent-vla/{log.run_id}')
            sdk.init.assert_not_called()
            sdk.Api.return_value.run.side_effect = OSError('wandb-secret')
            self.assertEqual(sync_wandb_links(log.directory)['status'], 'failed')
        self.assertTrue((log.directory/'huggingface_uploads.json').exists())


if __name__ == '__main__':
    unittest.main()
