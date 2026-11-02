"""A complete-looking receipt cannot make truncated NPZ/JSON reusable."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from .resumption_audit import npz_arrays,sha,read

class ResumptionAudit(unittest.TestCase):
    def test_npz_reads_all_members_without_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'masks.npz';a=np.arange(24,dtype=np.uint8).reshape(3,8)
            np.savez_compressed(p,mask_bits=a,timestep=np.arange(3))
            got=npz_arrays(p);np.testing.assert_array_equal(got['mask_bits'],a)
            self.assertEqual(sha(p),hashlib.sha256(p.read_bytes()).hexdigest())
    def test_disk_full_truncated_npz_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'masks.npz';np.savez_compressed(p,mask_bits=np.zeros((2,8192),np.uint8));p.write_bytes(p.read_bytes()[:-30])
            with self.assertRaises((zipfile.BadZipFile,EOFError,OSError)):npz_arrays(p)
    def test_object_pickle_payload_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'unsafe.npz';np.savez(p,obj=np.array([{'not':'a numeric tensor'}],dtype=object))
            with self.assertRaises(ValueError):npz_arrays(p)
    def test_incomplete_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'receipt.json';p.write_text('{"completed":')
            with self.assertRaises(json.JSONDecodeError):read(p)
if __name__=='__main__':unittest.main()
