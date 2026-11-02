"""Negative controls for a deceptively green partial completion inventory."""
import json
from pathlib import Path
import tempfile
import unittest
from .independent_evidence_readback import sha, verify_stages


class EvidenceInventoryTests(unittest.TestCase):
    def test_green_existing_receipts_do_not_hide_missing_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'stages').mkdir()
            data = root / 'masks.bin'; data.write_bytes(b'complete mask artifact')
            receipt = dict(status='PASS', files={'masks.bin': sha(data)})
            (root / 'stages/baseline.json').write_text(json.dumps(receipt))
            with self.assertRaises(AssertionError):
                verify_stages(root, ('baseline', 'candidate'))
            (root / 'stages/candidate.json').write_text(json.dumps(receipt))
            self.assertEqual(len(verify_stages(root, ('baseline', 'candidate'))), 2)

    def test_conflicting_shared_file_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'stages').mkdir()
            data = root / 'shared.bin'; data.write_bytes(b'immutable artifact')
            for name, digest in (('a', sha(data)), ('b', '0' * 64)):
                (root / 'stages' / (name + '.json')).write_text(json.dumps(
                    dict(status='PASS', files={'shared.bin': digest})))
            with self.assertRaises(AssertionError):
                verify_stages(root, ('a', 'b'))

    def test_external_file_cannot_substitute_for_stage_output(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory); root = parent / 'run'
            (root / 'stages').mkdir(parents=True)
            external = parent / 'other.bin'; external.write_bytes(b'wrong run')
            (root / 'stages/a.json').write_text(json.dumps(dict(
                status='PASS', files={'../other.bin': sha(external)})))
            with self.assertRaises(AssertionError):
                verify_stages(root, ('a',))


if __name__ == '__main__':
    unittest.main()
