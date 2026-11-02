"""Lossless task-owned JSON compression preserves historical logical SHA."""
import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest
from .run_pipeline import read,sha,logical_bytes

class StorageContract(unittest.TestCase):
    def test_compressed_fallback_preserves_logical_bytes_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace.json';raw=b'{"example": [1, 2, 3]}\n'
            with gzip.open(str(p)+'.gz','wb') as f:f.write(raw)
            self.assertEqual(logical_bytes(p),raw);self.assertEqual(read(p),{'example':[1,2,3]})
            self.assertEqual(sha(p),hashlib.sha256(raw).hexdigest())
    def test_plain_authoritative_path_wins_when_both_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace.json';p.write_bytes(b'{"value": 1}')
            with gzip.open(str(p)+'.gz','wb') as f:f.write(b'{"value": 2}')
            self.assertEqual(read(p),{'value':1})
    def test_corrupt_compressed_payload_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace.json';Path(str(p)+'.gz').write_bytes(b'invalid')
            with self.assertRaises((OSError,EOFError)):read(p)
if __name__=='__main__':unittest.main()
