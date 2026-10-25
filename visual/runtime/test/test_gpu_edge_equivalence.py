"""Task05-D exact B1 tests; GPU skips never count as exact PASS."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from visual.runtime.test import test_extended_delaunay_edges as old
from visual.runtime.test.test_triangle_intra_object_edges import fixture, oracle
from visual.runtime.test.ede_exact_optimization_eval import exact, semantic
from visual.runtime.test.ede_gpu_bbox_sat import B1Backend, host_adapter, integer_guard
from visual.runtime.triangle_intra_object_edges import prepare_validation, validate_cover, adjacent_pairs
from visual.runtime.extended_delaunay_edges import _segment_supercover_single_sat


def backend():
    path=os.getenv('TASK05D_SUPPORT')
    if not path: raise unittest.SkipTest('separate-process MPS probe not supplied')
    support=json.loads(Path(path).read_text())
    if support['status']!='PASS': raise unittest.SkipTest('EXACT_GPU_PATH_UNSUPPORTED')
    return B1Backend(support)


class HostGuards(unittest.TestCase):
    def test_unsupported_path_stops(self):
        with self.assertRaisesRegex(RuntimeError,'EXACT_GPU_PATH_UNSUPPORTED'):
            B1Backend(dict(status='EXACT_GPU_PATH_UNSUPPORTED'))

    def test_guard_and_scratch_no_fallback_or_chunk(self):
        self.assertTrue(integer_guard(2**30-1));self.assertFalse(integer_guard(2**30))
        with self.assertRaises(ValueError):host_adapter(np.full((1,2,2),2**30,np.int64))
        with self.assertRaises(MemoryError):host_adapter(np.array([[[0,0],[1000,1000]]],np.int64))
        with self.assertRaises(ValueError):host_adapter(np.zeros((1,2,2),np.int32))


class GPUCoverTests(unittest.TestCase):
    def setUp(self):self.gpu=backend()

    def test_fraction_closed_boundary_corner_reverse_degenerate_negative_order(self):
        points=list(np.ndindex(5,5))
        endpoints=np.array([[a,b] for a in points for b in points],np.int64)-2
        cover,offsets=self.gpu.cover(endpoints)
        lengths=[]
        for i,uv in enumerate(endpoints):
            expected=np.array([(u,v) for v in range(-3,4) for u in range(-3,4)
                               if oracle(uv,u,v)],np.int64)
            exact(cover[offsets[i]:offsets[i+1]],expected)
            exact(expected,_segment_supercover_single_sat(uv))
            lengths.append(len(expected))
        exact(offsets,np.r_[np.int64(0),np.cumsum(lengths,dtype=np.int64)])
        reverse,roff=self.gpu.cover(endpoints[:,::-1].copy())
        exact(cover,reverse);exact(offsets,roff)

    def test_empty_large_translation_guard_boundary_and_repeat(self):
        cover,offsets=self.gpu.cover(np.empty((0,2,2),np.int64))
        exact(cover,np.empty((0,2),np.int64));exact(offsets,np.array([0],np.int64))
        m=2**30-1
        endpoints=np.array([[[m,m],[m-1,m]], [[-m,-m],[-m+1,-m+1]]],np.int64)
        for _ in range(3):
            c,o=self.gpu.cover(endpoints)
            for i,p in enumerate(endpoints):exact(c[o[i]:o[i+1]],_segment_supercover_single_sat(p))

    def test_all_reasons_count_top4_pair_evidence_and_image_boundary(self):
        ctx=prepare_validation(fixture())[0]
        ctx['mask'][1,2:8]=False;ctx['valid'][1,2:8]=False;ctx['overlap'][1,2:8]=True
        ctx['support'][1,2:4]=False;ctx['right'][1,4]=-1
        ctx['pairs'].signed_depth_delta_m[ctx['right'][1,5]]=np.nan
        ctx['pairs'].depth_jump[ctx['right'][1,6]]=True
        uv=np.array([[[-1,1],[8,1]]],np.int64)
        c,o=self.gpu.cover(uv);a=_segment_supercover_single_sat(uv[0])
        exact(c,a);exact(validate_cover(c,ctx),validate_cover(a,ctx))
        exact(adjacent_pairs(c,(9,9)),adjacent_pairs(a,(9,9)))
        result=validate_cover(c,ctx)
        self.assertEqual(result['reasons'],['image_outside','mask_outside','invalid_depth',
            'overlap','pair_unsupported','pair_missing','delta_invalid','depth_jump'])
        self.assertEqual(result['counts']['mask_outside'],6)
        self.assertEqual(len(result['witnesses']['mask_outside']),4)

    def test_threshold_equality_and_nextafter_uses_existing_evidence(self):
        # Create only synthetic saved-style PairEvidence; B1 never recomputes it.
        z=np.ones((9,9),np.float32);threshold=np.float32(.01+.02)
        for delta in (np.nextafter(threshold,np.float32(0)),threshold,
                      np.nextafter(threshold,np.float32(np.inf))):
            e=fixture(z);ctx=prepare_validation(e)[0];ix=ctx['right'][1,1]
            ctx['pairs'].signed_depth_delta_m[ix]=delta
            ctx['pairs'].depth_jump[ix]=bool(delta>threshold)
            uv=np.array([[[1,1],[2,1]]],np.int64);c,o=self.gpu.cover(uv)
            exact(validate_cover(c,ctx),validate_cover(_segment_supercover_single_sat(uv[0]),ctx))


class GPUExtendedDelaunayContracts(old.ExtendedDelaunayContracts):
    """Every original EDE fixture, including fallback and receiver, runs B1."""
    def setUp(self):
        self.gpu=backend()
        self.patch_build=patch.object(old,'build_extended_delaunay_edges',side_effect=self.gpu.build)
        self.patch_raster=patch.object(old,'supercover',side_effect=lambda p:self.gpu.cover(np.asarray(p,np.int64)[None])[0])
        self.patch_build.start();self.patch_raster.start()
        self.addCleanup(self.patch_build.stop);self.addCleanup(self.patch_raster.stop)


if __name__=='__main__':unittest.main()
