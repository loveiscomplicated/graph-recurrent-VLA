"""Credential-free contracts for explicit static report deployment."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from utils.experiment_artifacts import deploy_static_space


class StaticSpaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / 'original-run'
        self.directory.mkdir()
        for name, value in [('metadata.json', dict(run_id='original-run', team='integration', status='finished')),
                            ('config.yaml', {}), ('metrics.json', {})]:
            (self.directory / name).write_text(json.dumps(value))
        self.selection = dict(entrypoint='index.html', files={})
        for name, body in [('index.html', '<html>selected report</html>'),
                           ('README.md', '---\nsdk: static\napp_file: index.html\n---\n'),
                           ('data.json', '{"HF_TOKEN":"must-be-redacted"}')]:
            path = self.directory.parent / name
            path.write_text(body)
            self.selection['files'][name] = str(path)
        self.repo = 'Graph-Recurrent-VLA/test-viewer'

    def client(self, private=True):
        api = Mock()
        api.whoami.return_value = dict(orgs=[dict(name='Graph-Recurrent-VLA', roleInOrg='write')])
        api.repo_info.return_value = SimpleNamespace(private=private, sdk='static', sha='old')
        api.create_commit.return_value = SimpleNamespace(oid='new')
        return api

    def test_dry_run_is_offline_and_keeps_original_identity(self):
        with patch('utils.experiment_artifacts.connect') as connect:
            result = deploy_static_space(self.directory, self.selection, self.repo, dry_run=True)
        connect.assert_not_called()
        self.assertTrue(result['private'])
        self.assertEqual(result['status'], 'dry-run')
        metadata = json.loads((self.directory / 'metadata.json').read_text())
        self.assertEqual(metadata['run_id'], 'original-run')
        self.assertEqual(len(metadata['outputs']), 1)

    def test_deploys_only_selected_sanitized_files_and_saves_receipt(self):
        api = self.client()
        with patch('utils.experiment_artifacts.connect', return_value=(api, SimpleNamespace, None)):
            result = deploy_static_space(self.directory, self.selection, self.repo)
        self.assertEqual(result['status'], 'complete')
        api.create_repo.assert_called_once_with(repo_id=self.repo, repo_type='space', space_sdk='static', private=True, exist_ok=True)
        members = {o.path_in_repo: o.path_or_fileobj for o in api.create_commit.call_args.kwargs['operations']}
        self.assertEqual(set(members), {*self.selection['files'], 'bundle_manifest.json'})
        self.assertNotIn(b'must-be-redacted', members['data.json'])
        self.assertEqual(json.loads(members['bundle_manifest.json'])['run_id'], 'original-run')
        self.assertTrue((self.directory / 'static_space_uploads.json').is_file())

    def test_visibility_mismatch_and_bad_namespace_do_not_commit(self):
        api = self.client(private=False)
        with patch('utils.experiment_artifacts.connect', return_value=(api, SimpleNamespace, None)):
            result = deploy_static_space(self.directory, self.selection, self.repo)
        self.assertEqual(result['status'], 'failed')
        api.create_commit.assert_not_called()
        with self.assertRaises(ValueError):
            deploy_static_space(self.directory, self.selection, 'someone/test')

    def test_running_run_rejected_before_network(self):
        path = self.directory / 'metadata.json'
        value = json.loads(path.read_text()); value['status'] = 'running'; path.write_text(json.dumps(value))
        with patch('utils.experiment_artifacts.connect') as connect, self.assertRaises(ValueError):
            deploy_static_space(self.directory, self.selection, self.repo)
        connect.assert_not_called()
