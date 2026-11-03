"""Offline contracts: coordinates, no repair, reservations, failure and timing semantics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import json
import unittest
import tempfile
from unittest.mock import patch
from common import strict_parse,reserve_job,accounted,balanced_order


class Contracts(unittest.TestCase):
    def test_jpeg_io_observer_preserves_decoded_bytes_and_restores_open(self):
        import builtins
        import types
        import numpy as np
        from PIL import Image
        from io_profile import JPEGIO
        original=builtins.open
        parent=types.ModuleType('sam2');video=types.ModuleType('sam2.sam2_video_predictor')
        video.load_video_frames=lambda:None;parent.sam2_video_predictor=video
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'0.jpg';Image.fromarray(np.arange(192,dtype=np.uint8).reshape(8,8,3)).save(path)
            expected=np.array(Image.open(path).convert('RGB'))
            with patch.dict(sys.modules,{'sam2':parent,'sam2.sam2_video_predictor':video}):
                with JPEGIO(tmp) as observation:
                    actual=np.array(Image.open(path).convert('RGB'))
                    video.load_video_frames()
            np.testing.assert_array_equal(actual,expected)
            self.assertIs(builtins.open,original)
            self.assertGreater(observation.result()['file_read_bytes'],0)
            self.assertIsNotNone(observation.preload_ms)

    def test_resolution_coordinates(self):
        for width,height in ((128,128),(512,512),(768,768),(1024,1024),(768,512)):
            p=strict_parse('{"peg":[100,200,900,800],"box":null}',width,height)
            for actual,expected in zip(p['peg']['pixels'],[width*.1,height*.2,width*.9,height*.8]):
                self.assertAlmostEqual(actual,expected)
            self.assertEqual(p['box']['status'],'NULL')

    def test_no_repairs(self):
        for text in ('```json\n{"peg":null,"box":null}\n```','{"peg":null,"peg":null,"box":null}',
                     '{"peg":null,"box":null,"extra":0}','{"peg":[NaN,0,1,1],"box":null}'):
            self.assertFalse(strict_parse(text,128,128)['json_valid'])
        for v in ([True,0,1,1],[-1,0,1,1],[0,0,1001,1000],[1,0,0,1],[0,0,0,1]):
            p=strict_parse(json.dumps(dict(peg=v,box=None)),1024,1024)
            self.assertEqual(p['peg']['status'],'INVALID_BOX')

    def test_durable_unknown_never_resent_and_budget(self):
        state=dict(jobs={})
        self.assertTrue(reserve_job(state,dict(id='a'),1.1,2))
        self.assertFalse(reserve_job(state,dict(id='a'),1.1,2))
        self.assertFalse(reserve_job(state,dict(id='b'),1.1,2))
        self.assertEqual(accounted(state),1.1)
        state['jobs']['a']['accounted_usd']=.001
        self.assertTrue(reserve_job(state,dict(id='b'),1.1,2))
        self.assertEqual(state['jobs']['a']['status'],'RESERVED')

    def test_balanced_positions(self):
        orders=[balanced_order(i) for i in range(8)]
        for pos in range(4):
            self.assertEqual(sorted(o[pos] for o in orders),[128,128,512,512,768,768,1024,1024])

    def test_estimator_unchanged(self):
        import ast
        source=Path(__file__).resolve().parents[3]/'outputs/task10i/stage4a2/graph_geometric_precision_20261102_v1/run_probe.py'
        if not source.exists():self.skipTest('original probe is verified on local host')
        a=ast.parse(source.read_text());b=ast.parse(Path(__file__).with_name('estimator.py').read_text())
        for fn in [n for n in b.body if isinstance(n,ast.FunctionDef)]:
            old=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==fn.name)
            self.assertEqual(ast.dump(old),ast.dump(fn))

    def test_empty_reference_is_not_perfect_coverage(self):
        import numpy as np
        from evaluate import coverage
        self.assertIsNone(coverage(np.empty((0,3)),np.empty((0,3))))
        self.assertEqual(coverage(np.empty((0,3)),np.array([[0.,0,1]])),0.)
        self.assertEqual(coverage(np.array([[0.,0,1]]),np.array([[0.,0,1]])),1.)

    def test_failed_estimate_never_gets_zero_error(self):
        import numpy as np
        from estimator import peg_estimate,hole_estimate
        from evaluate import score_estimate
        x=np.empty((0,3));valid=np.empty(0,bool)
        est=dict(peg=peg_estimate(x,x,valid),hole=hole_estimate(x,x,valid,np.empty(0),valid))
        state=dict(peg_pose=[0,0,0,1,0,0,0],hole_pose=[.1,0,0,1,0,0,0],peg_half=[.1,.02,.02])
        scored=score_estimate(est,state)
        self.assertFalse(scored['peg_estimable']);self.assertFalse(scored['hole_estimable'])
        self.assertIsNone(scored['peg_position_error_mm']);self.assertIsNone(scored['hole_center_error_mm'])
        self.assertFalse(scored['peg_position_below3mm'])

    def test_empty_mask_is_a_miss_not_a_good_mask(self):
        import numpy as np
        from evaluate import score_mask
        target=np.zeros((8,8),bool);target[2:6,2:6]=True;empty=np.zeros_like(target)
        scored=score_mask(empty,[target,empty,empty],0,1)
        self.assertEqual(scored['mask_iou'],0.)
        self.assertEqual(scored['recall'],0.);self.assertEqual(scored['boundary_f1'],0.)

    def test_signed_invalid_depth_stays_invalid(self):
        import numpy as np
        from graph.persistent_geometry.contracts import CameraObservation
        raw=np.array([[-32768,0],[1000,-1]],np.int16)
        c=CameraObservation('test',0,'base_camera',0.,np.zeros((2,2,3),np.uint8),raw,np.eye(3),
                            np.eye(4)[:3],np.diag([1,-1,-1,1]),np.array([0,0,0,1,0,0,0.]))
        c.validate();self.assertEqual(c.depth[1,0],1.)
        self.assertEqual(np.isfinite(c.depth).sum(),1)

    def test_interrupted_jobs_become_explicit_failed_predictions_without_resending(self):
        import api
        from common import write,read,sha,image_id,job_id
        class Log:
            def __enter__(self):return self
            def __exit__(self,*_):return False
            def link_output(self,*_):pass
            def log(self,*_):pass
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);entries=[]
            for i in range(70):
                for size in (128,512,768,1024):
                    scene=f's{i//2}';camera=('base_camera','hand_camera')[i%2]
                    entries.append(dict(id=image_id(scene,size,camera),scene=scene,parent=scene,
                        cohort='static' if i<62 else 'tracking',timestep=0,size=size,camera=camera,
                        image_file='never_read.png',image_sha256='unused'))
            jobs={}
            for repeat in range(3):
                for e in entries:
                    jid=job_id(e['id'],repeat)
                    jobs[jid]=dict(id=jid,image=e['id'],size=e['size'],repeat=repeat,status='RESERVED',
                                   accounted_usd=.001,reserved_usd=.001)
            write(root/'capture_manifest.json',dict(entries=entries,status='COMPLETE'))
            write(root/'request_ledger.json',dict(jobs=jobs,stop_reason=None,consecutive_errors=0))
            (root/'grounding_prompt.txt').write_text('test prompt')
            protocol=dict(prompt_sha256=sha(root/'grounding_prompt.txt'),request_reservation_usd=3.)
            with patch.object(api,'ROOT',root),patch.object(api,'assert_source',return_value=protocol),\
                 patch.object(api,'logger',return_value=Log()),patch('dotenv.load_dotenv'),\
                 patch.dict('os.environ',{'OPENAI_API_KEY':'offline-test-value'}),\
                 patch.object(api.requests,'post',side_effect=AssertionError('must not send')) as post:
                api.main();post.assert_not_called()
                result=read(root/'api_predictions.json')['rows'];self.assertEqual(len(result),840)
                self.assertTrue(all(r['unknown_server_outcome'] and not r['parsed']['json_valid'] for r in result))
                before=sha(root/'api_freeze.json');api.main();self.assertEqual(sha(root/'api_freeze.json'),before)

    def test_timing_metadata_does_not_enter_guarded_sensor_data(self):
        import numpy as np
        import torch
        from types import SimpleNamespace
        import graph_adapter
        import graph.plan_a_geometry.core as core
        from common import write
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'sensors/guard_test/128';folder.mkdir(parents=True)
            tcp=np.array([0,0,0,1,0,0,0.],np.float32)
            np.savez(folder/'robot.npz',robot76=np.zeros(76,np.float32),robot_world_geometry=np.zeros((3,3),np.float32),tcp_pose=tcp)
            for c in ('base_camera','hand_camera'):
                np.savez(folder/(c+'.npz'),rgb=np.zeros((128,128,3),np.uint8),depth_raw_mm=np.ones((128,128),np.int16)*1000,
                         intrinsic_cv=np.eye(3),extrinsic_cv=np.eye(4)[:3],cam2world_gl=np.diag([1,-1,-1,1]))
            write(root/'protocol.json',dict(scenes=[dict(id='guard_test',timestep=0)]))
            fk=SimpleNamespace(verify_robot=lambda *a:None,surfaces=lambda *a:[SimpleNamespace(world_from_link=np.eye(4))])
            guard=core.PlanA('base_camera','hand_camera',dict(gate={}))
            seen={}
            def native(cameras,data,surfaces,masks):
                seen.update(data)
                guard.prepare(cameras,data,masks,surfaces)
                g=SimpleNamespace(node_input=torch.zeros((3,21)),node_type=torch.arange(3),edge_index=torch.empty((2,0),dtype=torch.int64),
                                  edge_attr=torch.empty((0,4)),edge_type=torch.empty((0,),dtype=torch.int64))
                return g,[],{}
            geo=SimpleNamespace(fk=fk,build=native)
            empty=np.zeros((128,128),bool)
            with patch.object(graph_adapter,'ROOT',root),patch.object(core,'classify_robot',return_value=dict(confirmed_robot=empty,uncertain=empty)),\
                 patch.object(core,'wrist_gate',return_value=(empty,dict(accepted=False))):
                inputs=graph_adapter.load_inputs('guard_test',128,geo)
                self.assertEqual(len(inputs),5)
                result=graph_adapter.build(geo,inputs,{c:np.zeros((2,128,128),bool) for c in ('base_camera','hand_camera')})
            self.assertNotIn('input_join_ms',seen);self.assertNotIn('file_read_decode_ms',seen)
            self.assertIn('input_join_ms',result);self.assertEqual(len(result['hashes']),5)

    def test_quality_sensor_cache_is_exact_and_read_only(self):
        import numpy as np
        from graph_adapter import QualityCamera
        from graph.persistent_geometry.contracts import CameraObservation
        values=('cache',0,'base_camera',0.,np.zeros((2,2,3),np.uint8),np.array([[-32768,0],[1000,500]],np.int16),
                np.eye(3),np.eye(4)[:3],np.diag([1,-1,-1,1]),np.array([0,0,0,1,0,0,0.]))
        plain=CameraObservation(*values);cached=QualityCamera(*values)
        np.testing.assert_array_equal(plain.depth,cached.depth)
        np.testing.assert_array_equal(plain.world_from_camera,cached.world_from_camera)
        np.testing.assert_array_equal(plain.world_points(),cached.world_points())
        self.assertIs(cached.depth,cached.depth);self.assertFalse(cached.depth.flags.writeable)
        self.assertFalse(cached.world_from_camera.flags.writeable)


if __name__=='__main__':unittest.main(verbosity=2)
