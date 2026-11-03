"""Final FPS preserves string-ID ties, short-cloud order and merged sources."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from graph.persistent_geometry.tensor_records import (
    current_records_tensor,deduplicate_tensor,budget_tensor,numpy_records,_identity_ranks)
from graph.plan_a_geometry.core import budget,deduplicate
from .test_tensor_records import fixture,same


class TorchFinalFPSContracts(unittest.TestCase):
    device='cpu'

    def compare(self,xyz,count=128,*,dedup=False,**kw):
        obs,rows=fixture(xyz,device=self.device,**kw)
        table=current_records_tensor(obs,coordinate_frame=kw.get('frame','world'))
        if dedup:
            table,_=deduplicate_tensor(table);rows,_=deduplicate(rows)
        saved=[a.clone() for a in (table.xyz,table.pixels,table.source_pixels,table.source_offsets)]
        result=budget_tensor(table,count)
        same(numpy_records(result),budget(rows,count))
        for a,b in zip((table.xyz,table.pixels,table.source_pixels,table.source_offsets),saved):
            self.assertTrue(torch.equal(a,b))
        for field in ('xyz','normal_valid','f5','feature_valid','structural','source_pixels','source_offsets'):
            self.assertEqual(getattr(result,field).device.type,self.device)
        return table,result

    def test_empty_short_exact_cap_and_both_objects_frames(self):
        rng=np.random.default_rng(950)
        for frame in ('ee','world'):
            for n in (0,1,127,128,129,300):
                xyz=rng.normal(size=(2*n,3))*.02
                self.compare(xyz,frame=frame)

    def test_seed_ties_are_string_order_and_short_input_stays_in_order(self):
        xyz=np.zeros((3,3));kw=dict(objects=np.zeros(3,int),cameras=np.full(3,'base_camera'),pixels=np.array([2,10,9]))
        _,selected=self.compare(xyz,2,**kw)
        self.assertEqual(selected.pixels.cpu().tolist(),[10,2])
        _,short=self.compare(xyz,3,**kw)
        self.assertEqual(short.pixels.cpu().tolist(),[2,10,9])

    def test_all_coincident_no_reselection_and_permutations(self):
        rng=np.random.default_rng(11);pixels=rng.permutation(301)
        self.compare(np.zeros((301,3)),objects=np.zeros(301,int),pixels=pixels)
        xyz=rng.integers(-5,6,(400,3)).astype(np.float64)*.004
        for _ in range(3):
            order=rng.permutation(len(xyz))
            self.compare(xyz[order],pixels=order*11,objects=order%2,
                         cameras=np.where(order%3==0,'hand_camera','base_camera'))

    def test_identity_keys_unicode_prefixes_and_full_int64_decimal_pixels(self):
        values=np.array([0,1,2,9,10,11,99,100,101,999,1000,10**15,10**18,2**63-1],np.int64)
        cameras=np.array(['a','a0','a:0','á','손목카메라','base_camera','hand_camera'])
        pixels=np.tile(values,len(cameras));names=np.repeat(cameras,len(values))
        obs,rows=fixture(np.zeros((len(pixels),3)),device=self.device,objects=np.zeros(len(pixels),int),
                         pixels=pixels,cameras=names)
        table=current_records_tensor(obs,coordinate_frame='world')
        rank=_identity_ranks(table).cpu().numpy()
        expected=np.argsort(np.array([r['persistent_node_id'] for r in rows]),kind='stable')
        np.testing.assert_array_equal(np.argsort(rank),expected)
        same(numpy_records(budget_tensor(table,30)),budget(rows,30))

    def test_large_cloud_chunk_path_and_near_ties(self):
        rng=np.random.default_rng(18)
        self.compare(rng.normal(size=(2301,3))*.02,objects=np.zeros(2301,int))
        points=rng.normal(size=(90,3));xyz=np.concatenate((points,-points))
        xyz[::3]=np.nextafter(xyz[::3],np.inf)
        self.compare(xyz,objects=np.zeros(len(xyz),int))

    def test_merged_provenance_and_zero_budget(self):
        rng=np.random.default_rng(6);xyz=np.round(rng.normal(0,.05,(600,3)),3)
        xyz[1::4]=xyz[::4][:len(xyz[1::4])]+[.001,0,0]
        self.compare(xyz,dedup=True,objects=np.zeros(600,int),frame='ee')
        self.compare(xyz,count=0,dedup=True)

    def test_invalid_count_frame_reference_or_xyz(self):
        obs,_=fixture(np.array([[0.,0.,0.],[.1,0.,0.]]),device=self.device)
        records=current_records_tensor(obs,coordinate_frame='world')
        for count in (-1,1.5,None):
            with self.assertRaises(ValueError):budget_tensor(records,count)
        for bad in (replace(records,coordinate_frame='bad'),
                    replace(records,coordinate_frame='ee',reference_ee_pose=None),
                    replace(records,xyz=records.xyz*float('nan'))):
            with self.assertRaises(ValueError):budget_tensor(bad)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA host required')
class CUDAFinalFPSContracts(TorchFinalFPSContracts):
    device='cuda'

    def test_no_point_array_readback(self):
        obs,_=fixture(np.random.default_rng(8).normal(size=(400,3))*.02,device='cuda',frame='ee')
        records,_=deduplicate_tensor(current_records_tensor(obs,coordinate_frame='ee'))
        with patch.object(torch.Tensor,'cpu',side_effect=AssertionError('point readback forbidden')), \
             patch.object(torch.Tensor,'numpy',side_effect=AssertionError('NumPy forbidden')), \
             patch.object(torch.Tensor,'tolist',side_effect=AssertionError('point list forbidden')):
            result=budget_tensor(records,128)
            self.assertTrue(result.xyz.is_cuda)
            self.assertTrue(result.source_pixels.is_cuda)


if __name__=='__main__':unittest.main()
