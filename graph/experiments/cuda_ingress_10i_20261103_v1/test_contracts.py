"""Device ownership, exact legacy boundary and current input contracts."""
import unittest
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
class TransferAudit:
    def __init__(self):self.readbacks=[]
    def __enter__(self):
        original=torch.Tensor.cpu;audit=self
        def cpu(a,*args,**kwargs):
            if a.is_cuda:audit.readbacks.append((tuple(a.shape),a.numel()*a.element_size()))
            return original(a,*args,**kwargs)
        self.context=patch.object(torch.Tensor,'cpu',cpu);self.context.__enter__();return self
    def __exit__(self,*args):return self.context.__exit__(*args)


class ArrayBudgetContracts(unittest.TestCase):
    def test_exact_source_order_ties_duplicates_and_permutations(self):
        from .selected_records import records_after_budget
        from graph.plan_a_geometry.tests import Contracts
        from graph.persistent_geometry.contracts import SurfaceObservation
        from graph.persistent_geometry.map import current_records
        from graph.plan_a_geometry.core import deduplicate,budget
        from .benchmark import compare
        f=Contracts();f.setUp();frame=SimpleNamespace(cameras=tuple(f.cameras),tensors={c.camera:SimpleNamespace(camera=c) for c in f.cameras})
        for n in (0,8,350):
            rng=np.random.default_rng(42);host={};records=[]
            for c in f.cameras:
                xyz=np.round(rng.normal(0,.02,(n,3)),3)
                xyz[1::4]=xyz[::4][:len(xyz[1::4])]+[.001,0,0]
                normal=np.tile([0.,0.,1.],(n,1));valid=np.ones((n,3),bool);pixels=np.arange(n,dtype=np.int64)
                host[c.camera]=dict(xyz=xyz,normals=normal,f5=np.column_stack((normal,np.zeros((n,2)))),valid=valid,
                    oid=pixels%2,pixels=pixels,structural=pixels%3==0)
                e=host[c.camera]
                for oid in (0,1):
                    take=e['oid']==oid
                    obs=SurfaceObservation(c.episode,c.timestep,c.camera,oid,xyz[take],normal[take],valid[take,0],
                        e['f5'][take],valid[take],pixels[take],c.world_from_camera[:3,3])
                    rr=current_records([obs],fuse_views=False)
                    for r in rr:r['structural']=r['source_observation_ids'][0][3]%3==0
                    records.extend(rr)
            expected=budget(deduplicate(records,.003)[0],128)
            actual=records_after_budget(host,frame);compare(actual,expected)
            # Every array, including full merged provenance, must be exact on
            # this CPU arithmetic fixture, including equal FPS distances.
            for a,b in zip(actual,expected):
                for k in a:
                    if isinstance(a[k],np.ndarray):np.testing.assert_array_equal(a[k],b[k])
                    else:self.assertEqual(a[k],b[k])
            reversed_host={name:{k:a[::-1].copy() for k,a in e.items()} for name,e in host.items()}
            compare(records_after_budget(reversed_host,frame),expected)


@unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable on current host')
class TensorContracts(unittest.TestCase):
    def setUp(self):
        from graph.plan_a_geometry.tests import Contracts
        from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.gpu_filter import CudaRobotFilter
        from .ingress import TensorIngress
        fixture=Contracts();fixture.setUp();self.fixture=fixture
        self.ingress=TensorIngress(CudaRobotFilter());self.addCleanup(self.ingress.close)
        self.frame=self.ingress.stage(fixture.cameras,fixture.data,[])
        self.raw={k:torch.as_tensor(v,device='cuda') for k,v in fixture.masks.items()}

    def packet(self):
        from visual.runtime.graph_perception import make_packet_cuda
        t=self.frame.tensors['hand_camera']
        return make_packet_cuda(t.camera,self.raw['hand_camera'],depth_cuda=t.depth32,intrinsic_cuda=t.intrinsic,transform_cuda=t.ee)

    def test_sensor_snapshot_and_identity(self):
        c=self.fixture.wrist;t=self.frame.tensors[c.camera]
        np.testing.assert_array_equal(t.depth64.cpu().numpy(),c.depth)
        saved=t.rgb.clone();c.rgb[:]=255;c.depth_raw_mm[:]=-32768
        self.assertTrue(torch.equal(t.rgb,saved));self.assertTrue(bool((t.depth64>0).all()))
        for changed in ({**self.fixture.data,'future_depth':1},{**self.fixture.data,'episode':np.array('foreign')}):
            with self.assertRaises(ValueError):self.ingress.stage(self.fixture.cameras,changed,[])

    def test_gate_empty_overlap_invalid_and_conflicting_views(self):
        from .gate import wrist_gate
        from graph.plan_a_geometry.core import wrist_gate as reference
        modes=['normal','empty','overlap','invalid','conflict','fragmented']
        for mode in modes:
            cameras=self.fixture.cameras;raw={k:v.copy() for k,v in self.fixture.masks.items()}
            if mode=='empty':
                for m in raw.values():m[:]=False
            if mode=='overlap':raw['hand_camera'][1]=raw['hand_camera'][0]
            if mode=='invalid':
                cameras=[replace(c,depth_raw_mm=np.full_like(c.depth_raw_mm,-32768)) for c in cameras]
            if mode=='conflict':raw['base_camera'][1]=raw['hand_camera'][0]
            if mode=='fragmented':raw['hand_camera'][0,::2]=False
            frame=self.ingress.stage(cameras,self.fixture.data,[]);gpu={k:torch.as_tensor(v,device='cuda') for k,v in raw.items()}
            clean=self.ingress.filter(frame,gpu);cpu={k:v.cpu().numpy() for k,v in clean.items()}
            a,ga=wrist_gate(frame,gpu,clean,self.fixture.cfg['gate'])
            b,gb=reference(cameras[1],raw['hand_camera'],cpu['hand_camera'],cameras,cpu,self.fixture.cfg['gate'])
            np.testing.assert_array_equal(a.cpu().numpy(),b)
            from .benchmark import compare
            compare(ga,gb)

    def test_cuda_packet_receiver_has_no_array_readback(self):
        from visual.runtime.tensor_packet import numpy_packet
        from visual.runtime.graph_perception import make_packet
        from graph.vision_input import receive_vision_packet,validate_packet
        self.packet() # warm kernels
        with TransferAudit() as audit:
            packet=self.packet();received=receive_vision_packet(packet)
        # Existing Shape-First may read four ROI bounds; no point/image array.
        self.assertFalse([shape for shape,_ in audit.readbacks if np.prod(shape)>8])
        self.assertTrue(received.positions.is_cuda)
        host=numpy_packet(packet);validate_packet(host)
        expected,_=make_packet(self.fixture.wrist,self.fixture.masks['hand_camera'])
        from .benchmark import compare
        compare(host,expected)
        for mutate in (lambda p:p['provenance']['source_index'].fill_(-1),
            lambda p:p['feature_field_available'].fill_(False),lambda p:p['feature_valid'].logical_not_(),
            lambda p:p['geometric_features'].fill_(torch.inf)):
            bad=deepcopy(packet);mutate(bad)
            with self.assertRaises(ValueError):validate_packet(bad)
        with self.assertRaises(ValueError):receive_vision_packet(packet,robot_positions=torch.full((3,3),torch.nan,device='cuda'))

    def test_final_graph_pack_and_encoder_device(self):
        from .pipeline import TensorGeometry
        from graph.integration_baseline.graph import Graph,pack_graphs
        from graph.persistent_geometry.encoder import DiagnosticEncoder
        pipeline=TensorGeometry(self.fixture.system);self.addCleanup(pipeline.ingress.close)
        frame=pipeline.stage(self.fixture.cameras,self.fixture.data,[])
        with TransferAudit() as audit:g,_,_=pipeline.build(frame,self.raw)
        bulk=[n for shape,n in audit.readbacks if np.prod(shape)>8]
        self.assertEqual(bulk,[pipeline.last['selected_readback_bytes']])
        cpu=Graph(*[getattr(g,k).cpu() for k in ('node_input','node_type','edge_index','edge_attr','edge_type')],g.packet,{})
        reference=pack_graphs([cpu,cpu]).to('cuda');actual=pack_graphs([g,g])
        for k,a in vars(actual).items():self.assertTrue(torch.equal(a,getattr(reference,k)),k)
        torch.manual_seed(42);encoder=DiagnosticEncoder('E0').cuda().eval()
        with torch.inference_mode():a=encoder(actual);b=encoder(reference)
        # CUDA scatter aggregation has the same existing floating reduction
        # nondeterminism even for identical packed inputs.
        for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,rtol=0,atol=1e-6)

    def test_sam2_cuda_input_preprocess_and_embedding_bridge(self):
        # Installed transforms only, plus a fake backbone: no model inference,
        # checkpoint load or inference time is used in this contract test.
        from sam2.utils.transforms import SAM2Transforms
        from graph.experiments.robot_filter_gpu_io_10i_20261103_v1.sam2_bridge import set_image_cuda
        rgb=np.random.default_rng(42).integers(0,256,(512,512,3),dtype=np.uint8)
        transforms=SAM2Transforms(1024,0.)
        class Model:
            directly_add_no_mem_embed=True
            no_mem_embed=torch.full((1,1,2),.25,device='cuda')
            def forward_image(self,image):self.input=image;return image
            def _prepare_backbone_features(self,_):
                return None,[torch.ones((h*w,1,2),device='cuda') for h,w in ((4,4),(2,2),(1,1))],None,None
        predictor=SimpleNamespace(device=torch.device('cuda:0'),_transforms=transforms,
            model=Model(),_bb_feat_sizes=[(4,4),(2,2),(1,1)],reset_predictor=lambda:None)
        with TransferAudit() as audit:set_image_cuda(predictor,torch.as_tensor(rgb,device='cuda'))
        self.assertEqual(audit.readbacks,[])
        expected=transforms(rgb)[None].cuda()
        torch.testing.assert_close(predictor.model.input,expected,rtol=0,atol=2e-6)
        self.assertEqual(predictor._orig_hw,[(512,512)]);self.assertTrue(predictor._is_image_set)
        self.assertEqual(tuple(predictor._features['image_embed'].shape),(1,2,1,1))
        self.assertTrue(bool((predictor._features['image_embed']==1.25).all()))


if __name__=='__main__':unittest.main()
