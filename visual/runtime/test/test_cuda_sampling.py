"""Policy equality for GPU contour ordering, centroid ties and surface FPS."""
import unittest
from unittest.mock import patch
import numpy as np
import torch

from visual.runtime.test import run_variable_rate_shape_first as reference
from visual.runtime.test import run_voxel_structural_sampling as voxel
from visual.runtime.sampling import fps_indices,extract_candidates,SamplingConfig
from visual.runtime.graph_perception import process_cuda,to_cpu_evidence
from visual.runtime.test.test_graph_perception import frame


@unittest.skipUnless(torch.cuda.is_available(),'actual CUDA required')
class CUDASelection(unittest.TestCase):
    def test_precomputed_seed_bounds_match_all_reference_distances(self):
        import triton
        from visual.runtime import cuda_sampling_kernels as kernels
        rng=np.random.default_rng(28)
        for n,seeds in ((257,1),(513,75),(1001,137)):
            points=torch.as_tensor(rng.normal(size=(n,3)).astype(np.float32),device='cuda')
            initial=torch.as_tensor(rng.choice(n,seeds,replace=False),device='cuda')
            used=torch.zeros(n,dtype=torch.bool,device='cuda');used[initial]=True
            g=triton.cdiv(n,128);old=torch.empty(n,dtype=torch.float64,device='cuda');new=torch.empty_like(old)
            bd=torch.empty(g,dtype=torch.float64,device='cuda');bi=torch.empty(g,dtype=torch.int32,device='cuda')
            bounds=torch.empty((triton.cdiv(seeds,32),6),dtype=points.dtype,device='cuda')
            kernels.fps_seed[(g,)](points,initial,used,old,bd,bi,n,seeds,128,32,num_warps=8,enable_fp_fusion=False)
            kernels.seed_tile_bounds[(triton.cdiv(seeds,32),)](points,initial,bounds,seeds,32,num_warps=1)
            kernels.fps_seed_bounded[(g,)](points,initial,used,bounds,new,bd,bi,used,n,seeds,128,32,num_warps=4,enable_fp_fusion=False)
            torch.testing.assert_close(new,old,rtol=0,atol=0)

    def test_long_connected_paths_checkerboard_and_large_solid_mask(self):
        from visual.runtime.cuda_sampling import external_contour_sources
        snake=np.zeros((65,97),bool)
        for j,row in enumerate(range(1,61,4)):
            snake[row,1:-1]=True
            if row+4<65:snake[row:row+5,95 if j%2==0 else 1]=True
        v,u=np.indices((65,97))
        for raw in (snake,(v+u)%2==0,np.ones((127,193),bool)):
            expected=reference.outer_contours(raw,raw.copy(),np.ones_like(raw),np.zeros_like(raw))['C']
            actual=external_contour_sources(torch.as_tensor(raw,device='cuda'),torch.as_tensor(raw,device='cuda'))
            np.testing.assert_array_equal(actual.cpu().numpy(),expected)

    def test_canonical_walk_ties_revisits_and_long_walk(self):
        from visual.runtime import cuda_sampling_kernels as kernels
        rng=np.random.default_rng(417)
        cases=[np.array([0]),np.array([0,1]),np.array([0,1,2,1]),
            np.array([0,3,0,2,4,2]),np.r_[0,np.arange(1,2049),np.arange(2047,0,-1)]]
        for n in (17,255,256,257,1025):
            walk=rng.integers(0,19,n,dtype=np.int64);walk[0]=0;walk[-1]=3;cases.append(walk)
        for walk in cases:
            w=torch.as_tensor(walk,device='cuda');output=torch.empty_like(w)
            offsets=torch.tensor([0,len(walk)],dtype=torch.int64,device='cuda')
            lengths=torch.tensor([len(walk)],dtype=torch.int64,device='cuda')
            kernels.canonical[(1,)](w,offsets,lengths,output,256,num_warps=4)
            np.testing.assert_array_equal(output.cpu().numpy(),reference.canonical_walk(walk))

    def test_pixel_lookup_matches_set_membership_with_strided_thinning(self):
        from visual.runtime.cuda_sampling import select_cuda,selection_output
        f=frame();d=torch.as_tensor(f.depth,device='cuda')
        m=torch.as_tensor(np.stack(list(f.masks.values())),device='cuda');k=torch.as_tensor(f.intrinsic,device='cuda')
        p,v=process_cuda(d,m,k);raw=select_cuda(p,v,d,m,k)
        fast=selection_output(raw)
        old=selection_output({key:value for key,value in raw.items() if key not in ('source_rows','point_count')})
        for name in ('candidate_evidence','selected_by_structural_source'):
            torch.testing.assert_close(fast[name],old[name],rtol=0,atol=0)

    def test_offset_roi_preserves_external_contour_order(self):
        from visual.runtime.cuda_sampling import external_contour_sources
        rng=np.random.default_rng(51)
        for offset in ((0,0),(7,13),(41,57)):
            raw=np.zeros((80,100),bool)
            raw[offset[0]:offset[0]+23,offset[1]:offset[1]+31]=rng.random((23,31))>.45
            observed=raw&(rng.random(raw.shape)>.15)
            expected=reference.outer_contours(raw,observed,np.ones_like(raw),np.zeros_like(raw))['C']
            actual=external_contour_sources(torch.as_tensor(raw,device='cuda'),torch.as_tensor(observed,device='cuda'))
            np.testing.assert_array_equal(actual.cpu().numpy(),expected)

    def test_fps_graph_reuses_shapes_with_fresh_data_and_independent_results(self):
        from visual.runtime.cuda_sampling import surface_fps,_FPS_PLANS
        rng=np.random.default_rng(491);previous=[]
        for n,s in ((5001,17),(6003,19),(5100,18)):
            xyz=rng.normal(size=(n,3)).astype(np.float32)
            initial=rng.choice(n,s,replace=False).astype(np.int64)
            expected=fps_indices(xyz,s+17,initial=initial)
            result=surface_fps(torch.as_tensor(xyz,device='cuda'),torch.as_tensor(initial,device='cuda'),17)
            np.testing.assert_array_equal(result.cpu().numpy(),expected)
            for old,reference in previous:np.testing.assert_array_equal(old.cpu().numpy(),reference)
            previous.append((result,expected))
        self.assertLessEqual(len(_FPS_PLANS),8)

    def test_default_output_cuda_and_explicit_cpu_device(self):
        from visual.runtime.cuda_sampling import sample_cuda,numpy_selection
        f=frame();depth=torch.as_tensor(f.depth,device='cuda')
        masks=torch.as_tensor(np.stack(list(f.masks.values())),device='cuda')
        k=torch.as_tensor(f.intrinsic,device='cuda')
        projected,features=process_cuda(depth,masks,k)
        original_to=torch.Tensor.to
        def checked_to(t,*args,**kwargs):
            target=kwargs.get('device',args[0] if args else None)
            if isinstance(target,(str,torch.device)) and torch.device(target).type=='cpu':
                raise AssertionError('implicit CPU device transfer')
            return original_to(t,*args,**kwargs)
        with patch.object(torch.Tensor,'cpu',side_effect=AssertionError('CPU output forbidden')),\
                patch.object(torch.Tensor,'numpy',side_effect=AssertionError('NumPy output forbidden')),\
                patch.object(torch.Tensor,'to',checked_to):
            gpu=sample_cuda(projected,features,depth,masks,k,diagnostic=True)
        cpu=sample_cuda(projected,features,depth,masks,k,device='cpu',diagnostic=True)
        def compare(a,b):
            if isinstance(a,dict):
                self.assertEqual(set(a),set(b))
                for key in a:compare(a[key],b[key])
            else:
                self.assertIsInstance(a,torch.Tensor);self.assertIsInstance(b,torch.Tensor)
                self.assertEqual(a.device.type,'cuda');self.assertEqual(b.device.type,'cpu')
                torch.testing.assert_close(a.cpu(),b,rtol=0,atol=0,equal_nan=True)
        compare(gpu,cpu)
        with self.assertRaisesRegex(ValueError,"device='cpu'"):numpy_selection(gpu)
        with self.assertRaisesRegex(ValueError,'cpu or cuda'):
            sample_cuda(projected,features,depth,masks,k,device='mps')

    def test_external_contour_order_and_nested_holes(self):
        from visual.runtime.cuda_sampling import external_contour_sources
        rng=np.random.default_rng(14)
        cases=[]
        for shape in ((1,1),(1,8),(8,1),(8,11),(25,31)):
            cases.extend([np.zeros(shape,bool),np.ones(shape,bool)])
            for _ in range(8):cases.append(rng.random(shape)<rng.uniform(.1,.9))
        raw=np.zeros((32,32),bool);raw[2:30,2:30]=True;raw[5:27,5:27]=False;raw[10:20,10:20]=True
        cases.append(raw)
        for index,raw in enumerate(cases):
            observed=raw&(rng.random(raw.shape)>.15)
            expected=reference.outer_contours(raw,observed,np.ones_like(raw),np.zeros_like(raw))['C']
            actual=external_contour_sources(torch.as_tensor(raw,device='cuda'),torch.as_tensor(observed,device='cuda'))
            np.testing.assert_array_equal(actual.cpu().numpy(),expected,err_msg=str(index))

    def test_voxel_centroid_and_first_source_ties(self):
        from visual.runtime.cuda_sampling import voxel_representatives
        rng=np.random.default_rng(42)
        cases=[rng.normal(size=(200,3)).astype(np.float32),np.zeros((17,3),np.float32),
            np.array([[0,0,1],[.2,0,1],[.1,0,1],[.3,0,1]],np.float32)]
        for xyz in cases:
            source=np.arange(len(xyz),dtype=np.int64)*2
            for a in (.1,.5,2.):
                expected=source[voxel.representatives(source,xyz,a)['representative_rows']]
                actual=voxel_representatives(torch.as_tensor(source,device='cuda'),torch.as_tensor(xyz,device='cuda'),
                    torch.tensor(a,dtype=torch.float64,device='cuda'))
                np.testing.assert_array_equal(actual.cpu().numpy(),expected)
                # Silhouette order survives representative selection.
                actual=voxel_representatives(torch.as_tensor(source[::-1].copy(),device='cuda'),
                    torch.as_tensor(xyz[::-1].copy(),device='cuda'),torch.tensor(a,dtype=torch.float64,device='cuda'))
                np.testing.assert_array_equal(actual.cpu().numpy(),expected[::-1])

    def test_fps_initial_order_duplicates_and_ties(self):
        from visual.runtime.cuda_sampling import surface_fps
        rng=np.random.default_rng(19)
        for n in (0,1,7,35,513):
            for duplicate in (False,True):
                xyz=(np.zeros((n,3)) if duplicate else rng.integers(-3,4,(n,3))).astype(np.float32)
                for size in (0,min(n,3),n):
                    initial=rng.choice(n,size,replace=False).astype(np.int64)
                    add=min(17,n-size)
                    seed=initial if size or not add else np.array([0],np.int64)
                    expected=fps_indices(xyz,size+add,initial=seed)
                    actual=surface_fps(torch.as_tensor(xyz,device='cuda'),torch.as_tensor(initial,device='cuda'),add)
                    np.testing.assert_array_equal(actual.cpu().numpy(),expected)

    def test_full_selection_and_selected_metadata(self):
        from visual.runtime.cuda_sampling import select_cuda,to_cpu_selection
        for mode in ('normal','empty','holes','line'):
            f=frame(mode=mode);depth=torch.as_tensor(f.depth,device='cuda')
            masks=torch.as_tensor(np.stack(list(f.masks.values())),device='cuda')
            k=torch.as_tensor(f.intrinsic,device='cuda')
            projected,features=process_cuda(depth,masks,k)
            evidence=to_cpu_evidence(f,projected,features)
            objects=extract_candidates(evidence,SamplingConfig(),method='S3')
            with patch.object(torch.Tensor,'cpu',side_effect=AssertionError('pre-selection array download')):
                result=select_cuda(projected,features,depth,masks,k)
            actual=to_cpu_selection(result,diagnostic=True)
            for oid in (0,1):
                obj=objects[oid]
                pca=obj.flat_index[evidence.pairs[oid].candidates(20)['orientation'].ravel()[obj.flat_index]]
                prep=reference.prepare_object(obj.flat_index,obj.points,pca,f.masks[oid],
                    evidence.frame.geometry.valid_depth,evidence.frame.geometry.overlap_mask,f.intrinsic)
                selected=reference.select_object(prep,.5,0,reference.SURFACE[oid])
                for kind in ('C','R'):
                    for name in ('pca','sil'):np.testing.assert_array_equal(actual['prepared'][oid][kind][name],prep[kind][name],err_msg=f'{mode}:{kind}:{name}')
                for name in ('F','P','S','xyz'):np.testing.assert_array_equal(actual['selected'][oid][name],selected[name],err_msg=f'{mode}:{name}')
                positions=np.searchsorted(obj.flat_index,selected['F'])
                offset=sum(len(actual['selected'][i]['F']) for i in range(oid))
                for name,a in actual['metadata'].items():
                    a=a[offset:offset+len(positions)];b=obj.metadata[name][positions]
                    if a.dtype.kind=='f':np.testing.assert_allclose(a,b,atol=2e-5,rtol=0,equal_nan=True,err_msg=name)
                    else:np.testing.assert_array_equal(a,b,err_msg=name)

    def test_default_packet_skips_cpu_evidence_and_shape_first(self):
        from visual.runtime.graph_perception import make_packet
        from graph.persistent_geometry.tests.test_contracts import camera
        c=camera(name='base_camera',z=300)
        masks=np.zeros((2,*c.depth.shape),bool);masks[0,3:20,3:12]=True;masks[1,4:18,14:21]=True
        expected,_=make_packet(c,masks,backend='cpu')
        with patch('visual.runtime.graph_perception.to_cpu_evidence',side_effect=AssertionError('CPU evidence forbidden')),\
                patch.object(reference,'prepare_object',side_effect=AssertionError('CPU Shape-First forbidden')),\
                patch.object(reference,'select_object',side_effect=AssertionError('CPU FPS forbidden')):
            actual,timing=make_packet(c,masks)
        self.assertEqual(timing['sampling_backend'],'torch_triton_cuda')
        for name in ('points','camera_points','object_id','feature_field_valid','valid'):
            np.testing.assert_array_equal(actual[name],expected[name],err_msg=name)
        self.assertEqual(actual['source_artifact_feature_fields'],expected['source_artifact_feature_fields'])
        for name in actual['provenance']:
            np.testing.assert_array_equal(actual['provenance'][name],expected['provenance'][name],err_msg=name)
        for name in actual['feature_diagnostics']:
            np.testing.assert_array_equal(actual['feature_diagnostics'][name],expected['feature_diagnostics'][name],err_msg=name)
        np.testing.assert_allclose(actual['geometric_features'],expected['geometric_features'],atol=2e-5,rtol=0,equal_nan=True)


if __name__=='__main__':unittest.main()
