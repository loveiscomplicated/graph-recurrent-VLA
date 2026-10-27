"""Release-specific isolation contracts (the historical 54 run separately)."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .integration_guard import load_candidate, validate_source


class ReleaseContracts(unittest.TestCase):
    def test_disabled_does_not_read_or_import_candidate(self):
        self.assertIsNone(load_candidate('/missing', '/missing'))

    def test_separate_dependency_and_port_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dependency, extension = root / 'dep', root / 'port'
            dependency.mkdir(); extension.mkdir()
            (dependency / 'source.py').write_text('frozen')
            (extension / 'port.py').write_text('ported')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({
                'required_dependency_source_hashes': {'source.py': hashlib.sha256(b'frozen').hexdigest()},
                'extension_source_hashes': {'port.py': hashlib.sha256(b'ported').hexdigest()}}))
            self.assertEqual(validate_source(dependency, manifest, extension_root=extension)['checked_files'], 2)
            (dependency / 'source.py').write_text('stale')
            with self.assertRaisesRegex(RuntimeError, 'source identity mismatch'):
                validate_source(dependency, manifest, extension_root=extension)

    def test_guard_rejects_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'required_dependency_source_hashes': {'../outside': 'a'},
                                            'extension_source_hashes': {}}))
            with self.assertRaises(RuntimeError):
                validate_source(root, manifest)

    def test_loaded_common_modules_must_match_validated_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'required_dependency_source_hashes': {},
                                            'extension_source_hashes': {}}))
            with self.assertRaisesRegex(RuntimeError, 'loaded module outside validated source'):
                load_candidate(root, manifest, enabled=True)
