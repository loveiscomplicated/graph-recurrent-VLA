import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.publish_experiment import main
from utils.experiment_artifact_maintenance import referenced_paths, reconcile


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'README.md'; self.source.write_text('catalogue')
        self.body = self.source.read_bytes()
        self.addition = dict(path='README.md', source=str(self.source),
                             sha256=hashlib.sha256(self.body).hexdigest())
        self.repo = dict(repo_id='Graph-Recurrent-VLA/graph-recurrent-models', repo_type='model',
                         revision='before', private=True, additions=[self.addition], deletions=[])
        self.plan = dict(schema_version=1, repositories=[self.repo])
        self.api = SimpleNamespace(create_commit=lambda *a, **kw: SimpleNamespace(oid='after'))

    def snapshot(self, files=(), revision='before'):
        return dict(self.repo, files=list(files), revision=revision)

    def record(self, path, body=b'weights'):
        return dict(path=path, size=len(body), sha256=None,
                    git_sha1=hashlib.sha1(f'blob {len(body)}\0'.encode()+body).hexdigest())

    def run_apply(self, snapshot, protected=()):
        with patch('utils.experiment_artifact_maintenance.connect', return_value=(
                self.api, lambda **kw: kw, None)), patch(
                'utils.experiment_artifact_maintenance.inventory', return_value=snapshot), patch(
                'utils.experiment_artifact_maintenance.referenced_paths', return_value=set(protected)):
            return reconcile(self.plan, apply=True, receipt=self.root/'receipt.json')

    def test_default_cli_is_offline_dry_run(self):
        path=self.root/'plan.json';path.write_text(json.dumps(self.plan))
        with patch('utils.experiment_artifact_maintenance.connect') as connect, redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['--maintenance-plan', str(path)]), 0)
        connect.assert_not_called()
        self.assertEqual(json.loads(out.getvalue())['status'], 'dry-run')
        self.assertFalse((self.root/'plan.receipt.json').exists())

    def test_revision_conflict_and_source_mutation(self):
        self.assertEqual(self.run_apply(self.snapshot(revision='changed'))['status'], 'failed')
        self.source.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash changed'):reconcile(self.plan)

    def test_idempotent_after_commit_response_loss(self):
        self.assertEqual(self.run_apply(self.snapshot())['status'], 'complete')
        result=self.run_apply(self.snapshot([self.record('README.md', self.body)], 'after'))
        self.assertEqual(result['repositories'][0]['status'], 'unchanged')

    def test_delete_duplicate_but_reject_references_or_missing_retained(self):
        old=self.record('integration/a/artifacts/reports/hash/extra.txt')
        keep=self.record('integration/a/artifacts/reports/hash/keep.txt')
        self.repo['deletions']=[dict(old, retained_path=keep['path'], reference_count=0, reason='identical unreferenced copy')]
        snapshot=self.snapshot([old,keep])
        self.assertEqual(self.run_apply(snapshot, [old['path']])['status'], 'failed')
        self.assertEqual(self.run_apply(self.snapshot([old]))['status'], 'failed')
        result=self.run_apply(snapshot)
        self.assertEqual(result['repositories'][0]['deletions'], 1)
        self.assertEqual(self.run_apply(self.snapshot([keep,self.record('README.md',self.body)],'after'))['repositories'][0]['status'], 'unchanged')

    def test_partial_success_receipt_preserved(self):
        other=dict(self.repo, repo_id='Graph-Recurrent-VLA/graph-recurrent-rollouts', repo_type='dataset')
        self.plan['repositories'].append(other)
        with patch('utils.experiment_artifact_maintenance.connect', return_value=(self.api,lambda **kw:kw,None)), patch(
                'utils.experiment_artifact_maintenance.inventory', side_effect=[self.snapshot(),OSError('private-error')]):
            result=reconcile(self.plan,apply=True,receipt=self.root/'receipt.json')
        self.assertEqual(result['status'],'failed')
        saved=json.loads((self.root/'receipt.json').read_text())
        self.assertEqual(saved['repositories'][0]['revision'],'after')
        self.assertNotIn('private-error',json.dumps(saved))

    def test_manifest_and_backup_members_protected(self):
        artifact='integration/a/artifacts/checkpoint/hash/selected.pt'
        manifest='integration/a/manifests/hash/manifest.json'
        snapshot=self.snapshot([self.record(artifact),self.record(manifest)])
        self.assertIn(artifact,referenced_paths(snapshot,lambda p:dict(artifacts=[dict(path=artifact)])))
        backup=dict(snapshot,repo_id='Graph-Recurrent-VLA/experiment-backups')
        self.assertEqual(referenced_paths(backup,lambda p:{}),{artifact,manifest})

    def test_reject_unsafe_paths_and_immutable_overwrite(self):
        for value in ['.','../README.md','/README.md','integration/a/artifacts/config/hash/config.yaml']:
            self.addition['path']=value
            with self.assertRaises(ValueError):reconcile(self.plan)


if __name__=='__main__':unittest.main()
