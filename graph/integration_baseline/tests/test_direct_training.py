"""Synthetic contract fixtures; the separate tmux smoke executes real data."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from uuid import uuid4

import h5py
import numpy as np
import torch

from graph.integration_baseline.direct_runtime import atomic,read,sha,lock,Events,stop_gate,StopRequested
from graph.integration_baseline.direct_dataset import load_episode,DirectDataset,freeze_split,export_npz
from graph.integration_baseline.direct_errors import bundle,dispatch,changed
from graph.integration_baseline.direct_train import train,offline_test
from graph.integration_baseline.io import save_packet,read_packet
from graph.integration_baseline.vision import packet_from_frame
from visual.runtime.rgbd import TrackedRGBDFrame


PILOT=Path('experiments/9fec4f02fc9d4172a48e97d5eff90abd')


class FakeEvents:
    def __init__(self):self.events=[]
    def emit(self,event,**values):self.events.append((event,values))


def fixture(root):
    """Shortened real arrays + synthetic masks/terminal flags; NOT real rollout."""
    run=Path(root)/uuid4().hex;run.mkdir()
    config=read('graph/integration_baseline/direct_config.json')
    config['training'].update(sequence_length=2,max_epochs=2,patience=2)
    config['collection'].update(target_successes=4,max_attempts=4)
    config['splits']={'train':2,'validation':1,'test':1}
    atomic(run/'config.yaml',config);schema=read(PILOT/'robot_state_schema.json')
    atomic(run/'robot_state_schema.json',schema);atomic(run/'expert_action_schema.json',read(PILOT/'expert_action_schema.json'))
    source=read(PILOT/'pilot_manifest.json')['episodes'][0]
    original=load_episode(PILOT/Path(source['path']).with_suffix('.h5'))
    n=len(original['actions']);episodes=[];vision=[]
    for seed in range(90000,90004):
        a={k:v[:4].copy() if len(v)==n+1 else v[:3].copy() for k,v in original.items()}
        a['episode_id'][:]=seed;a['reset_seed'][:]=seed
        for k in ['success','terminated','truncated']:a[k][:]=False
        a['success'][-1]=True;a['terminated'][-1]=True
        path=run/f'ep{seed}.h5'
        with h5py.File(path,'x') as f:
            for k,v in a.items():f.create_dataset(k,data=v)
            f.attrs['completion']='COMPLETE';f.attrs['dataset_id']=run.name
        e=dict(episode_id=seed,seed=seed,path=path.name,sha256=sha(path),success=True,actions=3,observations=4,
               reset_group=f'fixture{seed}',phases=[dict(before='reach',after='reach')]*3)
        episodes.append(e);rows=[]
        for t in range(4):
            mask=np.zeros((128,128),bool);mask[50:56,50:56]=True
            frame=TrackedRGBDFrame(t,'base_camera',a['base_camera/rgb'][t],a['base_camera/depth'][t],
                a['base_camera/intrinsic'][t],a['base_camera/cam2world'][t],a['tcp_pose'][t],
                {0:mask,1:np.zeros_like(mask)},t,float(a['timestamp'][t]))
            p=packet_from_frame(frame,run.name+f':ep{seed}')[0]
            p['observation_provenance']=dict(dataset_id=run.name,episode_id=seed,timestep=t,episode_sha256=e['sha256'],
                rgb_sha256=__import__('hashlib').sha256(a['base_camera/rgb'][t].tobytes()).hexdigest())
            packet=run/f'ep{seed}-f{t}.npz';save_packet(packet,p)
            rows.append(dict(timestep=t,path=packet.name,sha256=sha(packet),has_action_target=t<3,object_counts=[len(p['points']),0]))
        mask=run/f'mask{seed}.npz';np.savez(mask,masks=np.zeros((4,2,128,128),bool))
        vision.append(dict(episode_id=seed,episode_sha256=e['sha256'],packets=rows,mask_path=mask.name,mask_sha256=sha(mask),
            mask_provenance=dict(future_observations_used=False,gt_segmentation_used=False,sam2_horizon='num_frames=t+1; fixture')))
    atomic(run/'dataset_manifest.json',dict(dataset_id=run.name,episodes=episodes,robot_schema_sha256=sha(run/'robot_state_schema.json')))
    freeze_split(run,config)
    atomic(run/'vision_manifest.json',dict(dataset_id=run.name,dataset_sha256=sha(run/'dataset_manifest.json'),camera='base_camera',episodes=vision))
    return run,config


class AtomicTests(unittest.TestCase):
    def test_immutable_and_partial_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';atomic(p,{'a':1},immutable=True)
            atomic(p,{'a':1},immutable=True)
            with self.assertRaises(ValueError):atomic(p,{'a':2},immutable=True)
            self.assertEqual(read(p),{'a':1});self.assertFalse(list(Path(tmp).glob('*.partial')))

    def test_lock_and_normal_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            with lock(p/'lock'):
                with self.assertRaises(RuntimeError):
                    with lock(p/'lock'):pass
            atomic(p/'STOP',{'normal':True})
            with self.assertRaises(StopRequested):stop_gate(p)

    def test_config_has_requested_unmodified_baseline(self):
        c=read('graph/integration_baseline/direct_config.json')
        self.assertEqual(c['splits'],{'train':96,'validation':12,'test':12})
        self.assertEqual(c['training']['sequence_length'],32);self.assertEqual(c['training']['batch_size'],2)
        self.assertEqual(c['model'],dict(d=64,L=2,gru_hidden=144,gru_layers=1,enc_hidden=128))


class ErrorTests(unittest.TestCase):
    def test_dispatch_dedup_budget_and_separate_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/uuid4().hex;run.mkdir()
            atomic(run/'callback_verification.json',dict(actual_cli_success=True))
            events=FakeEvents();launcher=Mock()
            first=dispatch(run,run/'errors/fixture/bundle.json',events,launcher=launcher)
            second=dispatch(run,run/'errors/fixture/bundle.json',events,launcher=launcher)
            self.assertEqual(first['attempt'],1);self.assertEqual(second['status'],'BLOCKED_REPAIR_LIMIT')
            launcher.assert_called_once();self.assertIn('repair',first['session'])

    def test_recursive_and_unverified_dispatch_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);events=FakeEvents();launcher=Mock()
            self.assertEqual(dispatch(run,run/'bundle.json',events,launcher=launcher)['status'],'BLOCKED_AUTO_CODEX_CALLBACK')
            with patch.dict(os.environ,{'GRVLA_REPAIR_SESSION':'1'}):
                self.assertEqual(dispatch(run,run/'bundle.json',events,launcher=launcher)['status'],'BLOCKED_REPAIR_LIMIT')
            launcher.assert_not_called()

    def test_nonzero_bundle_redacts_secrets_and_excludes_rgbd(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);atomic(run/'config.yaml',dict(seed=42));atomic(run/'status.json',dict(stage='training'))
            atomic(run/'stage_failure.json',dict(error_type='Fixture',stage='training',time='fixture',traceback='secret-fixture-token'))
            with patch.dict(os.environ,{'OPENAI_API_KEY':'secret-fixture-token'}),patch('graph.integration_baseline.direct_errors.source_inventory',return_value={}):
                p=bundle(run,'training',9)
            text=p.read_text();self.assertNotIn('secret-fixture-token',text)
            self.assertEqual(read(p)['exit_code'],9);self.assertNotIn('rgb',read(p))

    def test_normal_and_early_completion_never_dispatch(self):
        from graph.integration_baseline.run_control import supervise,STAGES
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);atomic(run/'config.yaml',dict(discord_required=False,fixture_only=False))
            for stage in STAGES:atomic(run/'stages'/f'{stage}.json',dict(status='COMPLETE',early_stopping=True))
            with patch('graph.integration_baseline.run_control.source_gate'),patch('graph.integration_baseline.direct_errors.dispatch') as called:
                self.assertEqual(supervise(run),0);called.assert_not_called()

    def test_source_allowlist_detects_added_deleted_files(self):
        self.assertEqual(changed({'a':'1','b':'2'},{'a':'3','c':'4'}),{'a','b','c'})

    def test_nonzero_exit_dispatches_once_and_stops_later_stages(self):
        from graph.integration_baseline.run_control import supervise
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);atomic(run/'config.yaml',dict(discord_required=False,fixture_only=False));(run/'logs').mkdir()
            with patch('graph.integration_baseline.run_control.source_gate'),patch('subprocess.Popen') as process,patch('graph.integration_baseline.direct_errors.source_inventory',return_value={}),patch('graph.integration_baseline.direct_errors.dispatch') as called:
                process.return_value.wait.return_value=7;process.return_value.pid=1234
                self.assertEqual(supervise(run),1);called.assert_called_once();process.assert_called_once()
            b=read(next((run/'errors').glob('*/bundle.json')))
            self.assertEqual(b['exit_code'],7);self.assertEqual(b['stage'],'collection')

    def test_user_stop_never_dispatches(self):
        from graph.integration_baseline.run_control import supervise
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);atomic(run/'config.yaml',dict(discord_required=False,fixture_only=False));atomic(run/'STOP',dict(user=True))
            with patch('graph.integration_baseline.run_control.source_gate'),patch('graph.integration_baseline.direct_errors.dispatch') as called:
                self.assertEqual(supervise(run),0);called.assert_not_called()
            self.assertEqual(read(run/'status.json')['status'],'STOPPED')

    def test_explicit_callback_waiver_preserves_bundle_without_dispatch(self):
        from graph.integration_baseline.run_control import supervise
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);atomic(run/'config.yaml',dict(discord_required=False,fixture_only=False,auto_callback_required=False));(run/'logs').mkdir()
            with patch('graph.integration_baseline.run_control.source_gate'),patch('subprocess.Popen') as process,patch('graph.integration_baseline.direct_errors.source_inventory',return_value={}),patch('graph.integration_baseline.direct_errors.dispatch') as called:
                process.return_value.wait.return_value=7;process.return_value.pid=1234
                self.assertEqual(supervise(run),1);called.assert_not_called()
            self.assertEqual(len(list((run/'errors').glob('*/bundle.json'))),1)
            self.assertEqual(read(run/'status.json')['status'],'FAILED')


class DatasetTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.run,self.config=fixture(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_full_join_readback_empty_object_and_train_only_normalizer(self):
        d=DirectDataset(self.run);report=d.validate()
        self.assertEqual(report['actions'],12);self.assertEqual(report['box_never_observed_fraction'],1.)
        norm=d.fit_normalizer();self.assertEqual(norm.fitted_episode_ids,sorted(d.splits['train']))
        before=norm.stats.mean.clone()
        for eid in d.splits['test']:d.raw(eid)['robot_state'][:]=123
        self.assertTrue(torch.equal(norm.stats.mean,before))
        with self.assertRaises(ValueError):d.sequence(next(iter(d.episodes)),2,2)

    def test_episode_overlap_and_manifest_tamper_rejected(self):
        split=read(self.run/'split_manifest.json');split['splits']['test']=split['splits']['train'][:1]
        atomic(self.run/'split_manifest.json',split)
        with self.assertRaises(ValueError):DirectDataset(self.run)

    def test_mask_and_packet_corruption_are_fatal(self):
        d=DirectDataset(self.run);eid=next(iter(d.episodes));path=self.run/d.packet_rows[(eid,0)]['path']
        with path.open('ab') as f:f.write(b'corrupt')
        with self.assertRaises(ValueError):d.frame(eid,0)

    def test_packet_wrong_timestep_rejected_even_with_updated_file_hash(self):
        manifest=read(self.run/'vision_manifest.json');row=manifest['episodes'][0]['packets'][0]
        path=self.run/row['path'];p=read_packet(path);p['observation_provenance']['timestep']=8
        save_packet(path,p);row['sha256']=sha(path);atomic(self.run/'vision_manifest.json',manifest)
        d=DirectDataset(self.run)
        with self.assertRaises(ValueError):d.frame(manifest['episodes'][0]['episode_id'],0)

    def test_explicit_export_no_overwrite(self):
        e=read(self.run/'dataset_manifest.json')['episodes'][0];target=self.run/'export.npz'
        export_npz(self.run/e['path'],target)
        with np.load(target,allow_pickle=False) as f:self.assertEqual(f['actions'].shape,(3,7))
        with self.assertRaises(FileExistsError):export_npz(self.run/e['path'],target)

    def test_separate_execution_preserves_source_dataset_identity(self):
        import shutil
        output=Path(self.tmp.name)/uuid4().hex;output.mkdir()
        atomic(output/'data_source.json',dict(directory=str(self.run)))
        for p in self.run.glob('*.npz'):shutil.copy2(p,output/p.name)
        shutil.copy2(self.run/'vision_manifest.json',output/'vision_manifest.json')
        frozen=sha(self.run/'dataset_manifest.json');split=sha(self.run/'split_manifest.json')
        d=DirectDataset(output);eid=next(iter(d.episodes));frame=d.frame(eid,0)
        self.assertEqual(frame.packet['observation_provenance']['dataset_id'],self.run.name)
        self.assertNotEqual(output.name,self.run.name);self.assertEqual(d.run,self.run)
        d.validate()
        self.assertTrue((output/'dataset_validation.json').exists())
        self.assertFalse((self.run/'dataset_validation.json').exists())
        self.assertEqual(sha(self.run/'dataset_manifest.json'),frozen)
        self.assertEqual(sha(self.run/'split_manifest.json'),split)

    def test_cooperative_mid_epoch_resume_matches_uninterrupted_exactly(self):
        from graph.integration_baseline import direct_train as module
        other,other_config=fixture(self.tmp.name);events=FakeEvents();logger=Mock()
        complete=train(other,other_config,DirectDataset(other),events,logger)
        real_update=module.finite_update;updates=[]
        def stop_after_first(*args,**kwargs):
            result=real_update(*args,**kwargs);updates.append(1)
            if len(updates)==1:atomic(self.run/'STOP',dict(test=True))
            return result
        with patch.object(module,'finite_update',side_effect=stop_after_first):
            with self.assertRaises(StopRequested):train(self.run,self.config,DirectDataset(self.run),events,logger)
        saved=read(self.run/'training/last_valid.json')
        cp=torch.load(self.run/saved['path'],weights_only=False)
        self.assertEqual(cp['progress']['cursor']['offset'],2)
        self.assertTrue(cp['progress']['cursor']['hidden'])
        (self.run/'STOP').unlink()
        resumed=train(self.run,self.config,DirectDataset(self.run),events,logger)
        a=torch.load(other/complete['last_valid']['path'],weights_only=False)
        b=torch.load(self.run/resumed['last_valid']['path'],weights_only=False)
        for k,v in a['model'].items():self.assertTrue(torch.equal(v,b['model'][k]),k)
        self.assertEqual(a['scheduler'],b['scheduler']);self.assertEqual(a['progress']['global_step'],b['progress']['global_step'])
        result=offline_test(self.run,self.config,DirectDataset(self.run),events,logger)
        self.assertEqual(result['test_evaluations'],1)
        self.assertEqual(offline_test(self.run,self.config,DirectDataset(self.run),events,logger),result)

    def test_checkpoint_config_and_corruption_rejected(self):
        from graph.integration_baseline.direct_train import identity,optimizer,restore_checkpoint
        from graph.integration_baseline.model import IntegrationPolicy
        d=DirectDataset(self.run);train(self.run,self.config,d,FakeEvents(),Mock())
        model=IntegrationPolicy(d.schema,**self.config['model']);opt,sched=optimizer(model,self.config)
        ident=identity(self.run,self.config,d);ident['config_sha256']='wrong'
        with self.assertRaises(ValueError):restore_checkpoint(self.run,'last_valid',model,opt,sched,ident)
        receipt=read(self.run/'training/last_valid.json')
        with (self.run/receipt['path']).open('ab') as f:f.write(b'corrupt')
        with self.assertRaises(ValueError):restore_checkpoint(self.run,'last_valid',model,opt,sched,identity(self.run,self.config,d))

    def test_missing_checkpoint_does_not_restart_training(self):
        d=DirectDataset(self.run);events=FakeEvents();logger=Mock()
        first=train(self.run,self.config,d,events,logger)
        self.assertEqual(train(self.run,self.config,d,events,logger),first)
        (self.run/'training/last_valid.json').unlink()
        before=len(list((self.run/'training/checkpoints').glob('*.pt')))
        with self.assertRaises(RuntimeError):train(self.run,self.config,d,events,logger)
        self.assertEqual(before,len(list((self.run/'training/checkpoints').glob('*.pt'))))

    def test_final_report_generation_and_relative_link_readback(self):
        from graph.integration_baseline import direct_report as report
        # Fill only synthetic reporting metadata; no real training metric claim.
        m=read(self.run/'dataset_manifest.json');m.update(successes=4,attempts=4,success_rate_all_attempts=1.,
            success_rate_completed=1.,completed_attempts=4,interrupted_attempts=[],failed_episodes=[])
        for e in m['episodes']:e['elapsed_seconds']=.1
        atomic(self.run/'dataset_manifest.json',m)
        split=read(self.run/'split_manifest.json');split['dataset_sha256']=sha(self.run/'dataset_manifest.json');atomic(self.run/'split_manifest.json',split)
        vision=read(self.run/'vision_manifest.json');vision['dataset_sha256']=split['dataset_sha256']
        for e in vision['episodes']:
            e['elapsed_seconds']=.1
            for row in e['packets']:row['missing']=[dict(reason='present'),dict(reason='not_observed')]
        atomic(self.run/'vision_manifest.json',vision)
        d=DirectDataset(self.run);d.validate();events=FakeEvents();logger=Mock()
        train(self.run,self.config,d,events,logger);offline_test(self.run,self.config,d,events,logger)
        (self.run/'plots').mkdir();atomic(self.run/'metadata.json',dict(run_id=self.run.name,status='finished',team='integration',outputs=[]))
        for name,value in [('tmux.json',dict(session='fixture')),('discord_verification.json',dict(status='USER_DISABLED')),
            ('callback_verification.json',dict(actual_cli_success=True)),('source_hashes.json',{}),('training_gate.json',dict(status='PASS'))]:
            atomic(self.run/name,value)
        with patch.object(report,'DOC',self.run/'report.md'):report.generate(self.run,self.config)
        self.assertEqual(report.verify_html(self.run/'graph_gru_training.html')['status'],'PASS')
        self.assertEqual(read(self.run/'verification.json')['test_evaluations'],1)


if __name__=='__main__':unittest.main()
