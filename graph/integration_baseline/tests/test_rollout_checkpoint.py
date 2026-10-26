"""Actual direct_bc_v1 bytes, restricted unpickling, corruption and schema gates."""
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import torch

from graph.integration_baseline.direct_runtime import atomic, read, sha
from graph.integration_baseline.direct_train import save_checkpoint, optimizer
from graph.integration_baseline.model import IntegrationPolicy
from graph.integration_baseline.robot import RobotNormalizer
from graph.integration_baseline.rollout_checkpoint import (load_bundle, make_manifest, required_sources, ROOT)
from graph.integration_baseline.tests.test_baseline import robot_fixture
from graph.prototype.v2.state import StateNormalizer


def fixture(root):
    root = Path(root)
    config = read(ROOT/'graph/integration_baseline/direct_config.json')
    schema = robot_fixture()[1]
    model = IntegrationPolicy(schema)
    norm = RobotNormalizer(schema, StateNormalizer(torch.zeros(76),torch.ones(76)),[1,2])
    data = dict(dataset_id=uuid4().hex,episodes=[dict(episode_id=i,seed=i,reset_group=str(i)) for i in range(1,16)])
    atomic(root/'dataset_manifest.json',data)
    split = dict(dataset_id=data['dataset_id'],dataset_sha256=sha(root/'dataset_manifest.json'),splits=dict(train=[1,2],validation=[3],test=list(range(4,16))),
                 reset_groups=dict(train=['1','2'],validation=['3'],test=list(map(str,range(4,16)))))
    for name,value in [('config.yaml',config),('robot_state_schema.json',schema),
                       ('expert_action_schema.json',dict(id='pd_ee_delta_pose_regression7',dimension=7)),
                       ('vision_manifest.json',dict(dataset_id=data['dataset_id'],dataset_sha256=sha(root/'dataset_manifest.json'))),
                       ('split_manifest.json',split),('vision_runtime.json',{}),('executed_conditions.json',{}),
                       ('source_hashes.json',{p:sha(ROOT/p) for p in required_sources()})]:
        atomic(root/name,value)
    identity = dict(dataset_sha256=sha(root/'dataset_manifest.json'), split_sha256=sha(root/'split_manifest.json'),
        vision_sha256=sha(root/'vision_manifest.json'), config_sha256=sha(root/'config.yaml'),
        robot_schema=schema, action_schema=read(root/'expert_action_schema.json'),model_schema=config['model'],
        graph_config=config['graph'],training_config=config['training'])
    opt,scheduler = optimizer(model,config)
    receipt = save_checkpoint(root,'best-fixture.pt',model,opt,scheduler,norm,identity,
                             dict(epoch=2,completed_epoch=1,global_step=3),pointers=['best'])
    manifest = root/'checkpoint_manifest.json'
    make_manifest(root,uuid4().hex,manifest)
    return model,norm,manifest,receipt


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model,self.norm,self.manifest,self.receipt = fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def rehash(self,name):
        m=read(self.manifest)
        m['files'][name]['sha256']=sha(self.root/m['files'][name]['path'])
        atomic(self.manifest,m)

    def rewrite_checkpoint(self,change):
        path=self.root/self.receipt['path']
        # Test owns this generated fixture; production never uses unsafe load.
        cp=torch.load(path,weights_only=False)
        change(cp)
        torch.save(cp,path)
        receipt=read(self.root/'training/best.json');receipt['sha256']=sha(path)
        atomic(self.root/'training/best.json',receipt)
        atomic(path.with_suffix('.json'),receipt)
        self.rehash('checkpoint');self.rehash('receipt');self.rehash('checkpoint_receipt')

    def test_direct_writer_checkpoint_and_numpy_rng_load_exact_weights(self):
        model,norm,p=load_bundle(self.manifest,check_runtime=False)
        for k,v in self.model.state_dict().items():
            self.assertTrue(torch.equal(v,model.state_dict()[k]))
        self.assertEqual(norm.fitted_episode_ids,[1,2])
        self.assertEqual(p['identity']['robot_schema'],norm.schema)
        self.assertFalse(model.training)

    def test_partial_and_missing_and_corrupt_rejected(self):
        path=self.root/self.receipt['path']
        path.write_bytes(path.read_bytes()[:200])
        with self.assertRaisesRegex(ValueError,'SHA'):
            load_bundle(self.manifest,check_runtime=False)
        path.unlink()
        with self.assertRaises(ValueError):
            load_bundle(self.manifest,check_runtime=False)

    def test_strict_missing_weight_rejected(self):
        self.rewrite_checkpoint(lambda cp:cp['model'].pop(next(iter(cp['model']))))
        with self.assertRaises(RuntimeError):
            load_bundle(self.manifest,check_runtime=False)

    def test_schema_normalizer_and_train_only_rejected(self):
        self.rewrite_checkpoint(lambda cp:cp['normalizer'].update(fitted_episode_ids=[1,3]))
        with self.assertRaisesRegex(ValueError,'train split'):
            load_bundle(self.manifest,check_runtime=False)

    def test_receipt_and_source_config_identity_rejected(self):
        config=read(self.root/'config.yaml');config['model']['d']=32
        atomic(self.root/'config.yaml',config);self.rehash('config')
        with self.assertRaisesRegex(ValueError,'identity mismatch'):
            load_bundle(self.manifest,check_runtime=False)

    def test_missing_source_and_vision_runtime_provenance_rejected(self):
        with self.assertRaisesRegex(ValueError,'Vision weight'):
            from unittest.mock import patch
            with patch('graph.integration_baseline.rollout_checkpoint.vision_runtime_identity',return_value={'real':'identity'}):
                load_bundle(self.manifest)
        atomic(self.root/'source_hashes.json',{});self.rehash('source')
        with self.assertRaisesRegex(ValueError,'training source'):
            load_bundle(self.manifest,check_runtime=False)

    def test_arbitrary_python_payload_never_executes(self):
        class Exploit:
            def __reduce__(self):
                return (eval,("__import__('builtins').print('UNSAFE_EXECUTION')",))
        self.rewrite_checkpoint(lambda cp:cp.update(unknown=Exploit()))
        import io
        from contextlib import redirect_stdout
        stream=io.StringIO()
        with redirect_stdout(stream),self.assertRaises(Exception):
            load_bundle(self.manifest,check_runtime=False)
        self.assertNotIn('UNSAFE_EXECUTION',stream.getvalue())

    def test_offline_runtime_inventory_requests_only_inference_assets(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from graph.integration_baseline.rollout_checkpoint import vision_runtime_identity
        snapshot=self.root/'cached_revision';snapshot.mkdir()
        names=['added_tokens.json','config.json','model.safetensors','preprocessor_config.json',
               'special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.txt']
        for name in names:
            (snapshot/name).write_bytes(b'fixture')
        sam=self.root/'sam2';sam.mkdir();(sam/'__init__.py').write_text('# fixture\n')
        checkpoint=self.root/'sam2.pt';checkpoint.write_bytes(b'fixture weights')
        def cached(repo, **kwargs):
            self.assertTrue(kwargs['local_files_only'])
            self.assertEqual(set(kwargs['allow_patterns']),set(names))
            return str(snapshot)
        with patch.dict('sys.modules',{'sam2':SimpleNamespace(__file__=str(sam/'__init__.py'))}), \
             patch('huggingface_hub.snapshot_download',side_effect=cached), \
             patch('visual.runtime.tracking.DEFAULT_CHECKPOINT',checkpoint):
            identity=vision_runtime_identity()
        self.assertEqual(set(identity['dino_files']),set(names))
        self.assertFalse((snapshot/'README.md').exists())


if __name__=='__main__':
    unittest.main()
