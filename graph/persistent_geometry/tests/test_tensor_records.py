"""Greedy representatives, numeric source order, provenance, frames and device."""
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from graph.persistent_geometry.contracts import SurfaceObservation
from graph.persistent_geometry.tensor_fusion import tensor_observations
from graph.persistent_geometry.tensor_records import current_records_tensor, deduplicate_tensor, numpy_records
from graph.persistent_geometry.map import current_records
from graph.plan_a_geometry.core import deduplicate


def same(a,b):
    if isinstance(a,np.ndarray):
        assert isinstance(b,np.ndarray) and a.dtype==b.dtype
        np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b,(a,b)


def fixture(xyz, *, frame='world', device='cpu', objects=None, cameras=None, pixels=None, structural=None):
    n = len(xyz)
    if objects is None: objects = np.arange(n) % 2
    if cameras is None: cameras = np.where(np.arange(n) % 3 == 0, 'hand_camera', 'base_camera')
    if pixels is None: pixels = np.arange(n, dtype=np.int64)*11
    if structural is None: structural = np.arange(n) % 4 == 0
    observations, flags = [], []
    for name in sorted(set(cameras) | {'base_camera','hand_camera'}):
        for oid in (0,1):
            take = np.flatnonzero((cameras == name) & (objects == oid))
            normal = np.tile([0.,0.,1.], (len(take),1))
            valid = np.ones((len(take),3), bool)
            f5 = np.c_[normal, take*.001, take*.0001]
            obs = SurfaceObservation('ep', 7, name, oid, np.asarray(xyz,dtype=np.float64).reshape(-1,3)[take],
                normal, valid[:,0], f5, valid, np.asarray(pixels)[take], np.zeros(3), frame,
                (0.,0.,0.,1.,0.,0.,0.) if frame=='ee' else None).validate()
            observations.append(obs); flags.append(np.asarray(structural)[take])
    tensors = [replace(o, structural=torch.as_tensor(f,device=device))
               for o,f in zip(tensor_observations(observations,device=device),flags)]
    rows = []
    for o, f in zip(observations,flags):
        made = current_records([o],fuse_views=False,coordinate_frame=frame)
        for r, s in zip(made,f): r['structural'] = bool(s)
        rows.extend(made)
    return tensors, rows


class TorchRecordsContracts(unittest.TestCase):
    device = 'cpu'

    def compare(self, xyz, **kw):
        obs, rows = fixture(xyz,device=self.device,**kw)
        before = deepcopy(rows)
        records = current_records_tensor(obs,coordinate_frame=kw.get('frame','world'))
        a, audit = deduplicate_tensor(records)
        expected, trace = deduplicate(rows)
        same(numpy_records(a),expected); same(audit,trace); same(rows,before)
        self.assertEqual(a.device.type,self.device)
        for name in ('xyz','f5','structural','source_camera','source_pixels','source_offsets'):
            self.assertEqual(getattr(a,name).device.type,self.device)
        return a

    def test_empty_singleton_random_permutations_and_both_frames(self):
        rng = np.random.default_rng(874)
        for frame in ('ee','world'):
            for n in (0,1,80,350,2300):
                xyz = np.round(rng.normal(0,.02,(n,3)),3)
                xyz[1::4] = xyz[::4][:len(xyz[1::4])]+[.001,0,0]
                self.compare(xyz,frame=frame)
                perm = rng.permutation(n)
                self.compare(xyz[perm],frame=frame,pixels=perm*13,
                    structural=perm%4==0,cameras=np.where(perm%3==0,'hand_camera','base_camera'),objects=perm%2)

    def test_greedy_is_not_transitive_and_multiple_hits_choose_first(self):
        xyz = np.array([[0,0,0],[.002,0,0],[.004,0,0],[.002,0,0.]])
        a = self.compare(xyz,objects=np.zeros(4,int),structural=np.zeros(4,bool),
            cameras=np.full(4,'base_camera'),pixels=np.array([2,9,11,20]))
        self.assertEqual(len(a),2)
        self.assertEqual(a.source_offsets.cpu().tolist(),[0,3,4])

    def test_structural_priority_numeric_pixel_order_and_distinct_objects(self):
        xyz = np.zeros((8,3)); xyz[4:] = [.0001,0,0]
        self.compare(xyz,objects=np.array([0,0,0,0,1,1,1,1]),
            cameras=np.array(['hand_camera','base_camera']*4),pixels=np.array([10,10,2,2,10,10,2,2]),
            structural=np.array([1,0,1,0,0,0,0,0],bool))

    def test_strict_radius_nextafter_and_negative_grid_cells(self):
        radii = [.003,np.nextafter(.003,0),np.nextafter(.003,np.inf)]
        for r in radii:
            for sign in (-1,1):
                self.compare(np.array([[0.,0.,0.],[sign*r,0.,0.]]),objects=np.zeros(2,int),
                    cameras=np.full(2,'base_camera'),structural=np.zeros(2,bool))

    def test_diagonal_distances_at_ulp_boundary(self):
        rng=np.random.default_rng(9021)
        for _ in range(48):
            delta=rng.normal(size=3);delta*=.003/np.linalg.norm(delta)
            self.compare(np.array([[0.,0.,0.],delta]),objects=np.zeros(2,int),
                cameras=np.full(2,'base_camera'),structural=np.zeros(2,bool))

    def test_missing_features_and_input_observations_preserved(self):
        obs,rows=fixture(np.array([[0.,0.,0.],[.001,0.,0.],[.02,0.,0.]]),
            device=self.device,objects=np.zeros(3,int),cameras=np.full(3,'base_camera'))
        changed=[]
        for o in obs:
            changed.append(replace(o,normals=torch.zeros_like(o.normals),
                normal_valid=torch.zeros_like(o.normal_valid),f5=torch.full_like(o.f5,torch.nan),
                feature_valid=torch.zeros_like(o.feature_valid)))
        for r in rows:
            r.update(normal=np.zeros(3),normal_valid=False,last_f5=np.full(5,np.nan),
                     last_feature_valid=np.zeros(3,bool))
        saved=[o.xyz.clone() for o in changed]
        result,_=deduplicate_tensor(current_records_tensor(changed,coordinate_frame='world'))
        same(numpy_records(result),deduplicate(rows)[0])
        for o,x in zip(changed,saved):self.assertTrue(torch.equal(o.xyz,x))

    def test_invalid_identity_reference_distance_and_values(self):
        obs,_=fixture([[0.,0.,0.]],device=self.device)
        for bad in (replace(obs[0],episode='foreign'),replace(obs[0],coordinate_frame='ee'),
                    replace(obs[0],structural=None)):
            with self.assertRaises(ValueError): current_records_tensor([bad,*obs[1:]],coordinate_frame='world')
        records=current_records_tensor(obs,coordinate_frame='world')
        for distance in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError): deduplicate_tensor(records,distance)
        with self.assertRaises(ValueError):deduplicate_tensor(replace(records,xyz=records.xyz*float('nan')))
        result,_=deduplicate_tensor(records)
        with self.assertRaises(ValueError):deduplicate_tensor(result)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA host required')
class CUDARecordsContracts(TorchRecordsContracts):
    device='cuda'

    def test_no_host_arrays_in_record_construction_or_dedup(self):
        obs,_=fixture(np.random.default_rng(3).normal(size=(1105,3))*.01,device='cuda',frame='ee')
        with patch.object(torch.Tensor,'cpu',side_effect=AssertionError('array readback forbidden')), \
             patch.object(torch.Tensor,'numpy',side_effect=AssertionError('NumPy forbidden')), \
             patch.object(torch.Tensor,'tolist',side_effect=AssertionError('array list forbidden')):
            records=current_records_tensor(obs,coordinate_frame='ee')
            out,_=deduplicate_tensor(records)
            self.assertTrue(out.xyz.is_cuda)


if __name__=='__main__':unittest.main()
