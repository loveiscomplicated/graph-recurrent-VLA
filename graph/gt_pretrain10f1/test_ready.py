"""Fail-closed dataset/event contracts with isolated fixtures, no model calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from recurrent.features import build_features
from graph.integration_baseline.robot import state_schema,flatten_state
from .contract import measured_robot,object_target
from . import ready_contract as ready
from . import watch_ready


class Readiness(unittest.TestCase):
    def test_native_receipt_must_pin_the_sealed_generation_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);marker,encoder=self.fixture(root)
            (root/'native_runtime.json').write_text(json.dumps({'fixture_only':True}))
            digest=ready.sha_file(root/'native_runtime.json')
            acceptance=dict(status='PASS',mode='ALL_GRAPHS_REBUILT_IN_F1_NATIVE_RUNTIME',old576_graphs_included=False,
                graph_runtime_sha256=digest,graph_runtime_path='native_runtime.json')
            (root/'native_acceptance.json').write_text(json.dumps(acceptance))
            runtime=ready.read_json(root/'runtime.json');runtime['native_runtime_acceptance']=dict(path='native_acceptance.json',sha256=ready.sha_file(root/'native_acceptance.json'))
            (root/'runtime.json').write_text(json.dumps(runtime));receipt=ready.read_json(root/'receipt.json')
            receipt.update(graph_generation_instance=57987772,graph_runtime_sha256=digest)
            def reseal():
                (root/'receipt.json').write_text(json.dumps(receipt))
                marker['runtime_contract_sha256']=ready.sha_file(root/'runtime.json')
                marker['files']=[dict(path=p.name,bytes=p.stat().st_size,sha256=ready.sha_file(p)) for p in root.iterdir() if p.is_file() and p.name!='gnn_dataset_ready.json']
                (root/'gnn_dataset_ready.json').write_text(json.dumps(marker))
            reseal();self.assertEqual(ready.validate(root,encoder)['status'],'PASS')
            receipt['graph_runtime_sha256']='0'*64;reseal()
            with self.assertRaisesRegex(ValueError,'native graph runtime proof'):ready.validate(root,encoder)

    def test_runtime_conversion_cannot_waive_native_tensor_mismatch(self):
        import copy
        from .release import runtime_mismatches
        expected=dict(tensor_runtime={'python':'p','torch':'t','numpy':'n'},robot_schema={},geometry={},geometry_protocol_sha256='g',source_sha256={'x':'s'},
            gt38_encoder_sha256='e',quaternion_order='wxyz',simulator={'sim_backend':'physx_cpu'},camera_contract={'roles':{'main':'base_camera'}},
            vision_weight_sha256={'dino_files':{'model':'m'},'sam_sha256':'s'})
        actual=copy.deepcopy(expected);actual['collector_execution']={'versions':{'python':'old','torch':'old','numpy':'old'}}
        self.assertIn('collector_runtime_requires_complete_native_conversion',runtime_mismatches(expected,actual))
        actual['native_runtime_acceptance']=dict(status='PASS',mode='ALL_GRAPHS_REBUILT_IN_F1_NATIVE_RUNTIME',old576_graphs_included=False)
        self.assertEqual(runtime_mismatches(expected,actual),[])
        actual['tensor_runtime']['torch']='wrong'
        self.assertIn('tensor_runtime',runtime_mismatches(expected,actual))

    def test_candidate_validated_before_watched_marker_exists(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,encoder=self.fixture(root)
            candidate=root/'candidate.json';(root/'gnn_dataset_ready.json').rename(candidate)
            result=ready.validate(root,encoder,marker_path=candidate)
            self.assertEqual(result['status'],'PASS')
            self.assertFalse((root/'gnn_dataset_ready.json').exists())
            with self.assertRaises(ValueError):ready.validate(root,encoder,marker_path=root.parent/'outside.json')

    def fixture(self,root,gnn_mask=None):
        def write(name,value):
            p=root/name;p.write_text(json.dumps(value));return p
        fields=dict(qpos=np.zeros(9,'f'),qvel=np.zeros(9,'f'),gripper_position=np.ones(2,'f')*.04,
            gripper_velocity=np.zeros(2,'f'),gripper_opening=np.array([.08],'f'),gripper_opening_velocity=np.zeros(1,'f'))
        for name in ('tcp','left_finger','right_finger','robot_base'):
            fields[name+'_pose']=np.array([0,0,.2,1,0,0,0],'f')
            fields[name+'_linear_velocity']=np.zeros(3,'f');fields[name+'_angular_velocity']=np.zeros(3,'f')
        schema=state_schema(fields);robots=np.tile(flatten_state(fields,schema),(3,1));robot=measured_robot(robots,schema)
        raw={**robot,'peg_pose':np.tile(np.array([.1,0,.1,1,0,0,0],'f'),(3,1)),
            'hole_pose':np.tile(np.array([0,.2,.1,1,0,0,0],'f'),(3,1)),
            'peg_half':np.tile(np.array([.1,.02,.02],'f'),(3,1)),'hole_radius':np.ones((3,1),'f')*.023}
        with h5py.File(root/'raw.h5','w') as f:
            for name,value in dict(actions=np.zeros((2,7),'f'),robot_state=robots,timestep=np.arange(3),
                timestamp=np.arange(3)/20,tcp_pose=robot['tcp_pose'],
                robot_world_geometry=np.stack((robot['tcp_pose'][:,:3],robot['left_tip'],robot['right_tip']),1),
                supervision_mask=np.zeros(2,bool),segment=np.array([0,1],np.int64)).items():f[name]=value
            for c in ('base_camera','hand_camera'):
                for k,v in dict(rgb=np.zeros((3,128,128,3),np.uint8),depth_raw_mm=np.ones((3,128,128),np.int16)*1000,
                    intrinsic=np.tile(np.eye(3),(3,1,1)),extrinsic_cv=np.tile(np.eye(4)[:3],(3,1,1)),cam2world_gl=np.tile(np.eye(4),(3,1,1))).items():f[c+'/'+k]=v
        np.savez_compressed(root/'gt.npz',**{'raw_'+k:v for k,v in raw.items()},timestep=np.arange(3),timestamp=np.arange(3)/20,target18=object_target(raw),
            gt38=build_features(raw,np.ones(3,'f'),np.zeros(3,'f')))
        np.savez_compressed(root/'mask.npz',gnn_supervision_mask=np.ones(2,bool) if gnn_mask is None else np.asarray(gnn_mask,bool),physical_valid=np.ones(2,bool))
        np.savez_compressed(root/'graph.npz',episode_id=np.int64(1),node_ptr=np.array([0,5,10]),edge_ptr=np.array([0,0,0]),
            node_input=np.zeros((10,21),'f'),node_type=np.tile(np.arange(5),2),edge_attr=np.zeros((0,4),'f'),edge_index=np.zeros((2,0),np.int64),
            edge_type=np.zeros(0,np.int64),presence2=np.ones((2,2),bool),timestamp=np.arange(2)/20,
            robot76=robots[:2],timestep=np.arange(2),point_graph_node=np.array([3,4,8,9]),source_point=np.arange(4),source_camera=np.zeros(4,int))
        geometry=dict(condition='C3',representation='SF',point_cap_per_object=128,persistent=False,node_dimension=21,edge_dimension=4)
        encoder=ready.sha_file(Path(__import__('recurrent.features',fromlist=['build_features']).__file__))
        runtime=dict(simulator={'fixture':True},tensor_runtime={'python':'fixture','torch':'fixture','numpy':'fixture'},camera_contract={'fixture':True},robot_schema=schema,geometry=geometry,
            geometry_protocol_sha256='a'*64,vision_weight_sha256={'fixture':'b'*64},source_sha256={'fixture':'c'*64},
            gt38_encoder_sha256=encoder,quaternion_order='wxyz')
        write('runtime.json',runtime)
        write('physical.json',dict(approved=True,physical_safety=True,fixture_only=True,initial={'valid':True},checks=[{'valid':True},{'valid':True}]))
        write('receipt.json',dict(readback='PASS',graphs=2,actions=2,sha256=ready.sha_file(root/'graph.npz'),raw_episode_sha256=ready.sha_file(root/'raw.h5'),
            identity=dict(graph_config=geometry,frozen_geometry_protocol_sha256='a'*64)))
        ep=dict(episode_id=1,parent_id=30000,branch_id='fixture_P',split='train',distribution='disturbed_robot',actions=2,
            raw_path='raw.h5',gt_path='gt.npz',graph_path='graph.npz',graph_receipt_path='receipt.json',physical_audit_path='physical.json',gnn_mask_path='mask.npz')
        write('dataset.json',dict(episodes=[ep]))
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=ready.sha_file(p)) for p in root.iterdir() if p.is_file()]
        marker=dict(schema_version=1,status='DATA_READY',task='10-F',source_instance=57640710,dataset_id='ISOLATED_FIXTURE',
            collection_complete=True,graph_cache_readback='PASS',physical_validity='PASS',gt38_alignment='PASS',parent_branch_provenance='PASS',files_sealed=True,
            runtime_contract='runtime.json',runtime_contract_sha256=ready.sha_file(root/'runtime.json'),
            dataset_manifest='dataset.json',dataset_manifest_sha256=ready.sha_file(root/'dataset.json'),files=files)
        write('gnn_dataset_ready.json',marker)
        return marker,encoder

    def test_action_mask_off_allowed_only_with_gt_and_physical_mask(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,encoder=self.fixture(root);result=ready.validate(root,encoder)
            self.assertEqual(result['status'],'PASS');self.assertEqual(result['action_mask_off_eligible_frames'],2)

    def test_sampling_identity_cannot_be_relabelled_without_explicit_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);marker,encoder=self.fixture(root)
            p=root/'dataset.json';value=json.loads(p.read_text())
            value['episodes'][0]['graph_sampling_episode_id']=2;p.write_text(json.dumps(value))
            marker['dataset_manifest_sha256']=ready.sha_file(p)
            for f in marker['files']:
                if f['path']==p.name:f.update(bytes=p.stat().st_size,sha256=ready.sha_file(p))
            (root/'gnn_dataset_ready.json').write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError,'sampling episode identity'):ready.validate(root,encoder)

    def test_partial_or_changed_export_never_ready(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);marker,encoder=self.fixture(root)
            marker['files_sealed']=False;(root/'gnn_dataset_ready.json').write_text(json.dumps(marker))
            with self.assertRaises(ValueError):ready.validate(root,encoder)
            marker['files_sealed']=True;(root/'gnn_dataset_ready.json').write_text(json.dumps(marker))
            (root/'gt.npz').write_bytes(b'partial')
            with self.assertRaises(ValueError):ready.validate(root,encoder)

    def test_no_centroid_label_or_symlink_substitution(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);marker,encoder=self.fixture(root)
            (root/'gt.npz').unlink();(root/'gt.npz').symlink_to(root/'graph.npz')
            with self.assertRaises(ValueError):ready.validate(root,encoder)

    def test_failed_safety_not_approved_even_with_complete_files(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);marker,encoder=self.fixture(root)
            p=root/'physical.json';p.write_text(json.dumps(dict(approved=False,physical_safety=False)))
            for f in marker['files']:
                if f['path']==p.name:f.update(bytes=p.stat().st_size,sha256=ready.sha_file(p))
            (root/'gnn_dataset_ready.json').write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError,'physical approval'):ready.validate(root,encoder)

    def ops(self):
        repo=Path(__file__).resolve().parents[2]
        return watch_ready.inherited_ops(repo/'outputs/plan_a_operations/3788798a719f4ec4a66c224ca42f5d63/event_ops.py')

    def test_ready_event_dedup_and_normal_progress_zero_calls(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ops=self.ops();ops.write(root/'callback_verification.json',dict(status='PASS'))
            cfg=dict(max_ready_callbacks=1,preparation_handoff='fixture.md',target=dict(root='/fixture'))
            event=dict(event_id='a'*24,kind='DATA_READY');calls=[]
            for _ in range(10):
                status=watch_ready.dispatch(root,cfg,event,ops,launcher=lambda *a:calls.append(a),busy=False)
                self.assertEqual(status,'FIXTURE_DISPATCHED')
            self.assertEqual(len(calls),1);self.assertEqual(ops.read(root/'invocation_journal.json')['total_calls'],1)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ops=self.ops()
            with patch.object(watch_ready,'remote_probe',return_value=dict(status='WAITING_DATA',marker_exists=False)), \
                    patch.object(watch_ready,'dispatch',side_effect=AssertionError('normal progress invoked a model')):
                for _ in range(5):self.assertEqual(watch_ready.tick(root,{},ops),'WAITING_DATA')
            self.assertEqual(ops.read(root/'watch_state.json')['normal_progress_model_calls'],0)
            self.assertFalse((root/'invocation_journal.json').exists())

    def test_runtime_mismatch_never_releases_training(self):
        from . import release
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();self.fixture(source)
            target=root/'target';target.mkdir()
            expected=dict(read_runtime=1,gt38_encoder_sha256=ready.read_json(source/'runtime.json')['gt38_encoder_sha256'])
            with patch.object(release,'expected_runtime',return_value=expected):
                result=release.run(target,source,ready.sha_file(source/'gnn_dataset_ready.json'))
            self.assertEqual(result['status'],'WAITING_RUNTIME_COMPATIBILITY')
            self.assertFalse((target/'data_ready_release.json').exists());self.assertFalse((target/'training').exists())

    def test_verified_release_and_masked_frame_fit_selection(self):
        from . import release
        from .data import GraphPool
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();self.fixture(source,gnn_mask=[False,True])
            target=root/'target';target.mkdir();runtime=ready.read_json(source/'runtime.json')
            def write(name,value):(target/name).write_text(json.dumps(value))
            spec=dict(episode_id=2000000,seed=30001,parent_id=30001,split='train',paired=False)
            write('dataset_manifest.json',dict(episodes=[spec]));write('normalizer.json',dict(schema=runtime['robot_schema']))
            write('split_manifest.json',dict(train=[30001],validation=[30002],final=[91000],forbidden=[30199]))
            write('pretrain_config.json',dict(max_updates=12000))
            entry=dict(episode_id=2000000,seed=30001,split='train',actions=2,graphs=2,
                path=str(source/'graph.npz'),sha256=ready.sha_file(source/'graph.npz'),
                gt_path=str(source/'gt.npz'),gt_sidecar_sha256=ready.sha_file(source/'gt.npz'))
            write('graph_cache_manifest.json',dict(episodes=[entry],paired_episodes=[]))
            with patch.object(release,'expected_runtime',return_value=runtime):
                result=release.run(target,source,ready.sha_file(source/'gnn_dataset_ready.json'))
            self.assertEqual(result['status'],'READY_FOR_GNN_PRETRAINING')
            pool=GraphPool(target,'cpu')
            self.assertEqual(len(pool.meta),4);self.assertEqual(len(pool.splits['train']),3)
            self.assertNotIn(2,pool.splits['train']);self.assertIn(3,pool.splits['train'])
            self.assertEqual(pool.normalization['fitted_frames'],3)
            self.assertFalse((target/'training').exists())
            from .model import StateEncoder
            stats=pool.normalization;model=StateEncoder(stats['robot_mean'],stats['robot_std'],stats['target_mean']).freeze()
            graph,robot,_=pool.select(pool.splits['train'][:2]);features=model.extract(graph,robot,pool.schema)
            self.assertEqual(features['state38'].shape,(2,38));self.assertEqual(features['readout320'].shape,(2,320))
            self.assertTrue(features['state38_identifiability_mask'][:,:15].all())
            self.assertFalse(features['state38_identifiability_mask'][:,18:25].any())
            self.assertFalse(features['state38_identifiability_mask'][:,28:36].any())


if __name__=='__main__':unittest.main()
