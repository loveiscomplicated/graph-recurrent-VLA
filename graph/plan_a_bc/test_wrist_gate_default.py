"""CUDA-first dispatch, CPU-only fallback, and frozen host artifact boundary."""
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import os
from pathlib import Path
import unittest

import numpy as np
import torch

from graph.plan_a_geometry import tests as geometry_fixtures
from .geometry import CurrentGeometry


def current_geometry(**kwargs):
    with patch('graph.plan_a_bc.geometry.RobotFK'):
        geometry=CurrentGeometry('unused',{},**kwargs)
    geometry._check_mount=lambda *args:None
    return geometry


class GatePolicyContracts(unittest.TestCase):
    def setUp(self):
        self.fixture=geometry_fixtures.Contracts();self.fixture.setUp()

    def test_no_cuda_default_build_uses_cpu_without_importing_gpu_gate(self):
        f=self.fixture
        with patch('torch.cuda.is_available',return_value=False), \
                patch.dict('sys.modules',{'graph.persistent_geometry.tensor_gate':None}):
            geometry=current_geometry()
            self.addCleanup(geometry.close)
            graph,records,gate=geometry.build(f.cameras,f.data,[],f.masks)
        self.assertEqual(geometry.perception_backend,'cpu')
        self.assertEqual(geometry.robot_filter_backend,'cpu')
        self.assertEqual(geometry.wrist_gate_backend,'existing_cpu')
        self.assertEqual(geometry.last_wrist_gate_backend,'existing_cpu')
        self.assertEqual(graph.packet['wrist_gate_backend'],'existing_cpu')
        self.assertEqual(gate,f.system.prepare(f.cameras,f.data,f.masks,[])['gate'])
        self.assertEqual(graph.node_input.device.type,'cpu')
        self.assertIsInstance(records,list)

    def test_cuda_policy_also_reaches_numpy_correction_and_cpu_graph_paths(self):
        f=self.fixture;reference=f.system.prepare(f.cameras,f.data,f.masks,[])
        accepted=reference['conditions']['C3']['hand_camera'][0]
        with patch('torch.cuda.is_available',return_value=True):
            geometry=current_geometry(perception_backend='cpu')
            self.addCleanup(geometry.close)
            with patch('graph.persistent_geometry.wrist_gate_dispatch.wrist_gate_cuda_numpy',
                    return_value=(accepted,reference['gate'])) as gpu, \
                    patch('graph.plan_a_geometry.core.wrist_gate',side_effect=AssertionError('CPU gate used')):
                prepared=geometry.system.prepare(f.cameras,f.data,f.masks,[])
        gpu.assert_called_once()
        self.assertEqual(prepared['wrist_gate_backend'],'torch_cuda')
        self.assertEqual(prepared['gate'],reference['gate'])
        self.assertEqual(geometry.wrist_gate_backend,'torch_cuda')

    def test_cuda_errors_propagate_without_cpu_retry(self):
        f=self.fixture
        for error in (RuntimeError('GPU failure'),ImportError('GPU dependency missing')):
            with self.subTest(error=type(error).__name__),patch('torch.cuda.is_available',return_value=True):
                geometry=current_geometry(perception_backend='cpu')
                try:
                    with patch('graph.persistent_geometry.wrist_gate_dispatch.wrist_gate_cuda_numpy',side_effect=error), \
                            patch('graph.plan_a_geometry.core.wrist_gate') as cpu:
                        with self.assertRaisesRegex(type(error),str(error)):
                            geometry.system.prepare(f.cameras,f.data,f.masks,[])
                        cpu.assert_not_called()
                finally:geometry.close()

    def test_frozen_memo_keeps_gate_policy_and_preclean_keyword(self):
        from graph.task10i_stage4a.perframe_cache import install
        f=self.fixture
        with patch('torch.cuda.is_available',return_value=False):
            geometry=current_geometry();self.addCleanup(geometry.close)
            memo=install(geometry)
            first=geometry.system.prepare(f.cameras,f.data,f.masks,[])
            again=geometry.system.prepare(f.cameras,f.data,f.masks,[])
            cleaned=geometry.system.prepare(f.cameras,f.data,f.masks,[],clean_masks=first['clean'])
        self.assertEqual(first['gate'],again['gate'])
        self.assertEqual(first['gate'],cleaned['gate'])
        self.assertTrue(all(value is None for value in cleaned['filters'].values()))
        self.assertEqual(memo.misses,2)
        self.assertEqual(first['wrist_gate_backend'],'existing_cpu')

    def test_frozen_artifacts_export_after_default_geometry_build(self):
        from graph.task10i_stage4a2.frozen_inference import geometry_artifacts
        f=self.fixture
        with patch('torch.cuda.is_available',return_value=False):
            geometry=current_geometry();self.addCleanup(geometry.close)
            graph,records,_=geometry.build(f.cameras,f.data,[],f.masks)
        artifact=geometry_artifacts(graph,records,'ep',0)
        self.assertEqual(artifact['wrist_gate_backend'],'existing_cpu')
        self.assertEqual(set(artifact['tensors']),{'node_input','node_type','edge_index','edge_attr','edge_type'})
        self.assertEqual(artifact['timestep'],0)

    def test_frozen_h5_can_retain_host_images_with_cuda_geometry(self):
        f=self.fixture
        raw=dict(robot_state=np.zeros((1,76)),tcp_pose=f.main.tcp_pose[None],
            robot_world_geometry=np.zeros((1,3,3)),timestamp=np.zeros(1))
        for c in f.cameras:
            for name in ('rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl'):
                raw[c.camera+'/'+name]=getattr(c,name)[None]
        with patch('torch.cuda.is_available',return_value=True):
            geometry=current_geometry();self.addCleanup(geometry.close)
        geometry.fk.surfaces.return_value=[SimpleNamespace(world_from_link=np.eye(4))]
        cameras,data,_=geometry.from_h5(raw,0,'ep',tensor_images=False)
        self.assertEqual(geometry.perception_backend,'triton_fused')
        self.assertTrue(all(isinstance(c.rgb,np.ndarray) and isinstance(c.depth_raw_mm,np.ndarray) for c in cameras))
        self.assertEqual(str(data['episode']),'ep')


@unittest.skipUnless(torch.cuda.is_available(),'CUDA host required')
class CudaEntrypointGateContracts(unittest.TestCase):
    @unittest.skipUnless(os.getenv('TASK10I_GATE_INPUT'),'saved nonempty gate input not provided')
    def test_saved_29775_point_case_uses_default_gpu_gate_with_exact_reference(self):
        from graph.experiments.wrist_gate_cuda_10i_20261104_v1.benchmark import load_case
        from graph.task10i_stage4a.perframe_cache import install
        cameras,raw,clean,_,prior,_=load_case(Path(os.environ['TASK10I_GATE_INPUT']))
        geometry=current_geometry();self.addCleanup(geometry.close)
        data=dict(episode=np.array(cameras[0].episode),timestep=np.int64(cameras[0].timestep))
        install(geometry)
        with patch('graph.plan_a_geometry.core.wrist_gate',side_effect=AssertionError('CPU gate used')):
            for _ in range(2):
                prepared=geometry.system.prepare(cameras,data,raw,[],clean_masks=clean)
                self.assertEqual(prepared['wrist_gate_backend'],'torch_cuda')
                self.assertEqual(prepared['gate'],prior['trace'])
                mask=prepared['conditions']['C3']['hand_camera'][0]
                self.assertEqual(hashlib.sha256(mask.tobytes()).hexdigest(),prior['accepted_mask_sha256'])

    def test_live_tensor_and_frozen_numpy_paths_use_same_gpu_gate(self):
        from graph.persistent_geometry.tensor_camera import tensor_camera
        from graph.persistent_geometry.tensor_records import TensorRecords
        from graph.task10i_stage4a2.frozen_inference import geometry_artifacts
        from graph.task10i_stage4a.perframe_cache import install
        f=geometry_fixtures.Contracts();f.setUp()
        reference=f.system.prepare(f.cameras,f.data,f.masks,[])
        geometry=current_geometry();self.addCleanup(geometry.close)
        with patch('graph.plan_a_geometry.core.wrist_gate',side_effect=AssertionError('CPU gate used')):
            # frozen_inference correction uses host cameras and the memoized prepare.
            install(geometry)
            for _ in range(2):
                prepared=geometry.system.prepare(f.cameras,f.data,f.masks,[])
                self.assertEqual(prepared['wrist_gate_backend'],'torch_cuda')
                self.assertEqual(prepared['gate'],reference['gate'])
            frozen,provenance,gate=geometry.build(f.cameras,f.data,[],f.masks)
            self.assertTrue(frozen.node_input.is_cuda)
            self.assertIsInstance(provenance,TensorRecords)
            artifact=geometry_artifacts(frozen,provenance,'ep',0)
            self.assertEqual(artifact['wrist_gate_backend'],'torch_cuda')
            self.assertEqual(gate,reference['gate'])
            self.assertEqual(geometry.last_wrist_gate_backend,'torch_cuda')
            # live supplies native CUDA RGB-D and raw masks via its producer.
            cameras=[tensor_camera(c) for c in f.cameras]
            raw={name:torch.as_tensor(mask,device='cuda') for name,mask in f.masks.items()}
            live,_,live_gate=geometry.build(cameras,f.data,[],lambda:raw)
        self.assertEqual(live.packet['wrist_gate_backend'],'torch_cuda')
        self.assertEqual(live_gate,gate)
        for name in ('node_input','node_type','edge_index','edge_attr','edge_type'):
            torch.testing.assert_close(getattr(live,name),getattr(frozen,name),rtol=0,atol=0)
        self.assertEqual(geometry.last_robot_schedule,'before_vision')


if __name__=='__main__':unittest.main()
