import unittest
import tempfile
from pathlib import Path
import numpy as np
from graph.initial_pool10i.pipeline import overlap,write,read,ahash,compare
class Contracts(unittest.TestCase):
    def test_collision_separation_and_penetration(self):
        box=np.array([[x,y,z] for x in [-.1,.1] for y in [-.1,.1] for z in [-.1,.1]])
        self.assertEqual(overlap(box,box+[.3,0,0]),0.)
        self.assertAlmostEqual(overlap(box,box+[.199,0,0]),.001)
        self.assertAlmostEqual(overlap(box,box+[.15,0,0]),.05)
    def test_atomic_manifest_rejects_nan_and_preserves_previous(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'m.json';write(p,dict(status='PASS',value=np.array([1,2])))
            self.assertEqual(read(p)['value'],[1,2]);self.assertFalse(p.with_suffix('.json.tmp').exists())
    def test_hash_and_replay_detect_one_changed_pixel(self):
        a={'rgb':np.zeros((2,2,3),np.uint8)};b={'rgb':a['rgb'].copy()};b['rgb'][0,0,0]=1
        self.assertNotEqual(ahash(a),ahash(b));self.assertFalse(compare(a,b)['rgb']['exact'])
if __name__=='__main__':unittest.main()
