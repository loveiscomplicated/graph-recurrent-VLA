import tempfile
import unittest
from pathlib import Path
from .verify import compare, inventory, resolve, sha, write, read


class TransferContracts(unittest.TestCase):
    def test_all_remote_failure_classes_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'ok').write_bytes(b'raw\x00binary');(r/'bad').write_bytes(b'original')
            expected=inventory(r);expected['missing']=dict(bytes=1,sha256='0'*64)
            (r/'bad').write_bytes(b'corrupt!');(r/'extra').write_text('extra')
            actual=inventory(r);actual['unreadable']=dict(unreadable='PermissionError')
            out=compare(expected,actual)
            self.assertEqual(out['counts'],dict(MATCHED=1,MISSING=1,MISMATCHED=1,EXTRA=1,UNREADABLE=1))

    def test_path_mapping_and_escape_guard(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'raw.npz').write_bytes(b'state')
            self.assertEqual(resolve(r,'raw.npz'),r.resolve()/'raw.npz')
            self.assertEqual(resolve(r,'/Users/source/raw.npz','/Users/source'),r.resolve()/'raw.npz')
            with self.assertRaises(AssertionError):resolve(r,'../outside')
            with self.assertRaises(ValueError):resolve(r,'/another/raw.npz','/Users/source')

    def test_partial_reuse_copies_preserve_source(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);source=r/'canonical';source.mkdir();(source/'f').write_bytes(b'original')
            before=inventory(source);stage=r/'stage';shutil.copytree(source,stage)
            (stage/'f').write_bytes(b'interrupted-transfer')
            self.assertEqual(inventory(source),before)
            self.assertEqual(compare(before,inventory(stage))['counts']['MISMATCHED'],1)
            shutil.copy2(source/'f',stage/'f')
            self.assertEqual(inventory(stage),before)

    def test_atomic_audit_readback(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'audit.json';write(p,dict(status='PREPARED'));write(p,dict(status='PASS'))
            self.assertEqual(read(p),dict(status='PASS'));self.assertFalse(p.with_name(p.name+'.tmp').exists())


if __name__=='__main__':unittest.main()
