"""Byte-preserving backup contracts; no simulator, GPU, or credentials needed."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from utils import experiment_backup as b
from tests.test_experiment_artifacts import FakeHub
from utils.experiment_artifacts import digests


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'repo'
        self.root.mkdir()
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        subprocess.run(['git', '-C', str(self.root), '-c', 'user.name=Test', '-c', 'user.email=test@example.test',
                        'commit', '--allow-empty', '-qm', 'initial'], check=True)
        (self.root / 'data').mkdir()
        (self.root / 'data' / 'partial.jsonl').write_bytes(b'{"metric":NaN}\n{"partial":')
        (self.root / 'data' / 'empty.log').touch()
        (self.root / 'data' / 'best.pt').write_bytes(b'opaque checkpoint')
        (self.root / 'data' / '.writer.lock').touch()
        (self.root / 'data' / 'hf.env').write_text('HF_TOKEN=hf_' + 'x'*30)
        self.spec = dict(root=str(self.root), team='recurrent', experiment='main', include=['data'],
                         run_ids=[], code_snapshot=True)
        self.snapshot = self.base / 'snapshot'

    def create(self):
        with patch.object(b.importlib.metadata, 'distributions', return_value=[]):
            return b.create(self.spec, self.snapshot)

    def test_dry_run_is_read_only(self):
        result = b.create(self.spec, self.snapshot, dry_run=True)
        self.assertFalse(self.snapshot.exists())
        self.assertEqual(len(result['files']), 3)
        self.assertIn('data/hf.env', result['excluded'])

    def test_snapshot_preserves_partial_bytes_and_running_metadata(self):
        run = 'original_run'
        path = self.root / 'experiments' / run
        path.mkdir(parents=True)
        metadata = dict(run_id=run, status='running', git_dirty=True)
        (path / 'metadata.json').write_text(json.dumps(metadata))
        self.spec['run_ids'] = [run]
        self.create()
        m = b.load_manifest(self.snapshot / 'manifest.json')
        self.assertEqual(m['runs'][0]['status'], 'running')
        for name in ['data/partial.jsonl', 'experiments/original_run/metadata.json']:
            self.assertEqual((self.root / name).read_bytes(), (self.snapshot / 'files' / name).read_bytes())
        self.assertEqual(b.verify(self.snapshot, snapshot=True)['status'], 'verified')
        (self.snapshot / 'files/data/best.pt').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            b.verify(self.snapshot, snapshot=True)

    def test_missing_link_escape_and_secret_rejected(self):
        for path in ['missing', '../escape', '/absolute']:
            self.spec['include'] = [path]
            with self.assertRaises((ValueError, FileNotFoundError)):
                b.create(self.spec, dry_run=True)
        self.spec['include'] = ['data']
        link = self.root / 'data/link'
        link.symlink_to(self.root / 'data/best.pt')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            b.create(self.spec, dry_run=True)
        link.unlink()
        (self.root / 'data/secret.txt').write_text('hf_' + 'a' * 30)
        with self.assertRaisesRegex(ValueError, 'credential'):
            b.create(self.spec, dry_run=True)

    def test_manifest_rejects_traversal_duplicates_and_modes(self):
        self.create()
        original = json.loads((self.snapshot / 'manifest.json').read_text())
        for change in ['escape', 'duplicate', 'mode']:
            m = json.loads(json.dumps(original))
            if change == 'escape': m['files'][0]['path'] = '../oops'
            if change == 'duplicate': m['files'].append(m['files'][0])
            if change == 'mode': m['files'][0]['mode'] = 0o4777
            p = self.base / 'bad.json'
            p.write_text(json.dumps(m))
            with self.assertRaises(ValueError): b.load_manifest(p)

    def hub(self):
        hub = FakeHub()
        hub.revisions[b.DEFAULT_REPOSITORY] = 'a' * 40
        def operation(**kwargs):
            return SimpleNamespace(**kwargs)
        return hub, operation

    def test_publish_duplicate_conflict_public_and_retry(self):
        self.create()
        hub, operation = self.hub()
        with patch.object(b, 'connect', return_value=(hub, operation, None)), patch.object(b.time, 'sleep'):
            hub.fail_before = 1
            first = b.publish(self.snapshot)
            self.assertEqual(first['status'], 'complete')
            second = b.publish(self.snapshot)
            self.assertEqual(second['uploaded_files'], 0)
            self.assertEqual(hub.commits, 1)
            key = next(iter(hub.files))
            hub.files[key] = b'changed remotely'
            with self.assertRaisesRegex(ValueError, 'immutable'):
                b.publish(self.snapshot)
            hub.private = False
            with self.assertRaisesRegex(ValueError, 'private'):
                b.publish(self.snapshot)

    def test_publish_response_lost_is_recovered(self):
        self.create()
        hub, operation = self.hub()
        hub.fail_after = 1
        with patch.object(b, 'connect', return_value=(hub, operation, None)), patch.object(b.time, 'sleep'):
            self.assertEqual(b.publish(self.snapshot)['status'], 'complete')
        self.assertEqual(hub.commits, 1)

    def test_restore_roundtrip_pinned_revision_and_collision(self):
        self.create()
        m = b.load_manifest(self.snapshot / 'manifest.json')
        prefix = f"recurrent/main/{m['backup_id']}"
        revision = 'a' * 40
        hub = SimpleNamespace(repo_info=lambda *a, **kw: SimpleNamespace(private=True))
        def download(repo, name, **kwargs):
            self.assertEqual(kwargs['revision'], revision)
            if name.endswith('/manifest.json'):
                return str(self.snapshot / 'manifest.json')
            entry = next(e for e in m['files'] if name == prefix + '/files/' + e['path'])
            return str(b.local_file(self.snapshot, entry, True))
        with patch.object(b, 'connect', return_value=(hub, None, None)), patch('huggingface_hub.hf_hub_download', side_effect=download):
            out = self.base / 'restored'
            result = b.restore(b.DEFAULT_REPOSITORY, revision, prefix, out)
            self.assertEqual(result['status'], 'verified')
            for entry in m['files']:
                self.assertEqual(digests(out / entry['path'])[:2], (entry['size'], entry['sha256']))
            with self.assertRaisesRegex(ValueError, 'empty'):
                b.restore(b.DEFAULT_REPOSITORY, revision, prefix, out)
            with self.assertRaisesRegex(ValueError, 'revision'):
                b.restore(b.DEFAULT_REPOSITORY, 'main', prefix, self.base / 'other')

    def test_explicit_link_exclusion_preserves_target(self):
        (self.root / 'data/latest-run').symlink_to(self.root / 'data/best.pt')
        self.spec['exclude'] = ['data/latest-run']
        result = b.create(self.spec, dry_run=True)
        self.assertIn('data/latest-run', result['excluded'])
        self.assertIn('data/best.pt', [e['path'] for e in result['files']])

    def test_source_change_during_copy_rejected(self):
        original = b.shutil.copyfile
        def copy(source, target):
            result = original(source, target)
            if source.name == 'best.pt': source.write_bytes(b'changed')
            return result
        with patch.object(b.shutil, 'copyfile', side_effect=copy):
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.create()
        self.assertFalse((self.snapshot / 'manifest.json').exists())

    def test_restore_hash_mismatch_rejected(self):
        self.create()
        m = b.load_manifest(self.snapshot / 'manifest.json')
        prefix = f"recurrent/main/{m['backup_id']}"
        bad = self.base / 'bad-download'
        bad.write_bytes(b'wrong bytes')
        def download(repo, name, **kwargs):
            return str(self.snapshot / 'manifest.json') if name.endswith('/manifest.json') else str(bad)
        hub = SimpleNamespace(repo_info=lambda *a, **kw: SimpleNamespace(private=True))
        with patch.object(b, 'connect', return_value=(hub, None, None)), patch('huggingface_hub.hf_hub_download', side_effect=download):
            with self.assertRaisesRegex(ValueError, 'checksum'):
                b.restore(b.DEFAULT_REPOSITORY, 'a'*40, prefix, self.base / 'restored')

    def test_existing_snapshot_and_overlapping_destination_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            b.create(self.spec, self.root / 'data/snapshot')
        self.create()
        with self.assertRaisesRegex(ValueError, 'must not exist'):
            self.create()

    def test_credentials_missing_and_changed_snapshot(self):
        self.create()
        with patch.object(b, 'connect', return_value=(None, None, 'no token')):
            with self.assertRaisesRegex(ValueError, 'no token'):
                b.publish(self.snapshot)
        (self.snapshot / 'files/data/best.pt').unlink()
        with self.assertRaises(ValueError): b.publish(self.snapshot, dry_run=True)


if __name__ == '__main__':
    unittest.main()
