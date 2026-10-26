"""Task08 contracts. Synthetic fixtures do not establish manipulation success."""
import copy
import json
import os
import hashlib
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from graph.integration_baseline.bc08_codec import ActionCodec, EncodedAction, DecodedPolicy, native_metrics, controller_commands
from graph.integration_baseline.bc08_train import (tensor_hash, normalizer_hash, optimizer, save, load_checkpoint,
    update, chunks, evaluate, worker)
from graph.integration_baseline.direct_runtime import atomic, read
from graph.integration_baseline.direct_train import seed_all, batch, normalizer_state, load_normalizer, restore_rng
from graph.integration_baseline.model import IntegrationPolicy, bc_loss, EpisodeStream
from graph.integration_baseline.robot import RobotNormalizer
from graph.prototype.v2.state import StateNormalizer
from graph.integration_baseline.data import AlignedFrame
from graph.integration_baseline.tests.test_baseline import fixture_packet, robot_fixture, ROBOT


class Dataset:
    def __init__(self):
        self.schema=robot_fixture()[1];self.splits=dict(train=[1,2],validation=[3],test=[4])
        self.episodes={e:dict(actions=3+(e%2)) for e in range(1,5)}
        self.packet=fixture_packet();self.accesses=[]

    def sequence(self,e,start,length):
        self.accesses.append((e,start,length))
        return [AlignedFrame(e,t,self.packet,np.full(76,.001*(e+t),np.float32),ROBOT.copy(),
            np.array([.1,-.05,.02,0,.2,-.1,1 if t<2 else -1],np.float32),False,False)
            for t in range(start,start+length)]


def plan_fixture(root,ds):
    config=read('graph/integration_baseline/direct_config.json');config.update(device='cpu',threads=1)
    config['training'].update(sequence_length=2,max_epochs=2)
    seed_all(42,1);model=IntegrationPolicy(ds.schema,**config['model'])
    norm=RobotNormalizer(ds.schema,StateNormalizer(torch.zeros(76),torch.ones(76)),[1,2])
    initial=Path(root)/'initial.pt';torch.save(dict(model=model.state_dict(),normalizer=normalizer_state(norm)),initial)
    arrays={e:np.stack([r.action for r in ds.sequence(e,0,ds.episodes[e]['actions'])]) for e in [1,2]}
    plan=dict(version='task08_bc_v1',config=config,experiments_root=str(Path(root)/'experiments'),
        initial_path=str(initial),initial_weight_identity=tensor_hash(model.state_dict()),
        robot_normalizer_identity=normalizer_hash(normalizer_state(norm)),input_hashes={},source_hashes={},
        overfit_ids=[1,2],splits=ds.splits,episode_lengths={str(e):x['actions'] for e,x in ds.episodes.items()},
        episode_input_hashes={str(e):str(e) for e in ds.episodes},dataset_directory='synthetic_fixture',
        minimum_free_bytes=0,overfit_updates=2000,offline_fit_thresholds=dict(native_action7_mse=.001,native_motion6_mse=.0001),
        codecs={phase:{m:ActionCodec.fit(m,arrays,[1,2]).state for m in ['A','B','C']} for phase in ['overfit','full']})
    path=Path(root)/'plan.json';atomic(path,plan);return plan,path,model,norm


class CodecTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(5)
        self.target=torch.randn(2,4,7);self.target[...,6]=torch.where(self.target[...,6]>0,1.,-1.)
        self.arrays={10:self.target[0].numpy(),20:self.target[1].numpy()}

    def codec(self,m):return ActionCodec.fit(m,self.arrays,[10,20])

    def test_A_exact_baseline_and_padding_exclusion(self):
        p=torch.randn(2,4,7,requires_grad=True);mask=torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
        y=self.target.clone();y[~mask]=float('nan')
        actual=self.codec('A').loss(p,y,mask)
        self.assertTrue(torch.equal(actual,bc_loss(p[mask],self.target[mask],gripper_mode='regression')))
        actual.backward();self.assertEqual(float(p.grad[~mask].abs().sum()),0)

    def test_train_subset_only_stats_and_stability(self):
        with self.assertRaises(ValueError):ActionCodec.fit('B',self.arrays,[10])
        c=ActionCodec.fit('B',{10:self.arrays[10]},[10,20])
        np.testing.assert_array_equal(c.mean,self.arrays[10][:,:6].mean(0))
        np.testing.assert_array_equal(c.std,self.arrays[10][:,:6].std(0)+np.float32(1e-6))
        a=np.zeros((3,7),np.float32);a[:,6]=1
        tiny=ActionCodec.fit('C',{10:a},[10]);self.assertTrue((tiny.std>0).all())
        self.assertEqual(tiny.state['small_std_components'],list(range(6)))
        self.assertFalse(tiny.state['floor_used'])

    def test_B_roundtrip_and_continuous_gripper(self):
        c=self.codec('B');mean,std=c._stats(self.target)
        encoded=torch.cat(((self.target[...,:6]-mean)/std,torch.full_like(self.target[...,6:],.37)),dim=-1)
        native=c.decode(c.tag(encoded))
        torch.testing.assert_close(native[...,:6],self.target[...,:6],atol=1e-6,rtol=1e-6)
        self.assertTrue(torch.equal(native[...,6],encoded[...,6]))
        mask=torch.ones((2,4),dtype=torch.bool);p=torch.randn_like(encoded)
        expected=(6*F.mse_loss(p[...,:6],(self.target[...,:6]-mean)/std)+F.mse_loss(p[...,6],self.target[...,6]))/7
        self.assertTrue(torch.equal(c.loss(p,self.target,mask),expected))

    def test_C_BCE_mapping_threshold_and_wrong_codec_rejected(self):
        c=self.codec('C');p=torch.zeros_like(self.target);p[0,:,6]=torch.tensor([-1.,0.,1.,2.])
        self.assertEqual(c.decode(c.tag(p))[0,:,6].tolist(),[-1,-1,1,1])
        with self.assertRaises(ValueError):c.decode(p)
        with self.assertRaises(ValueError):c.decode(self.codec('B').tag(p))
        mean,std=c._stats(p);expected=(6*F.mse_loss(p[...,:6],(self.target[...,:6]-mean)/std)+F.binary_cross_entropy_with_logits(p[...,6],(self.target[...,6]>0).float()))/7
        self.assertEqual(float(c.loss(p,self.target,torch.ones((2,4),dtype=torch.bool))),float(expected))
        bad={10:self.arrays[10].copy()};bad[10][0,6]=.5
        with self.assertRaises(ValueError):ActionCodec.fit('C',bad,[10])

    def test_native_controller_metric_and_rotation_convention(self):
        a=np.array([[2,-2,.5,2,0,0,1],[0,0,0,0,0,0,-1]],np.float32)
        commands=controller_commands(a)
        np.testing.assert_allclose(commands[0],[.1,-.1,.05,-.1,0,0,.04],atol=1e-8)
        perfect=native_metrics(a,a);self.assertEqual(perfect['native_action7_mse'],0)
        self.assertLess(perfect['controller_rotation_geodesic_rad']['max'],1e-15)
        b=a.copy();b[0,3]=0
        self.assertAlmostEqual(native_metrics(a,b)['controller_rotation_geodesic_rad']['max'],.1,places=6)


class TrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);self.ds=Dataset()

    def test_fixed_schedule_independent_of_model_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,*_=plan_fixture(tmp,self.ds);a=chunks(plan,'full',1)
            torch.rand(100);np.random.rand(100)
            self.assertEqual(a,chunks(plan,'full',1));self.assertEqual(sum(len(x[0]) for x in a),4)

    def test_hidden_carry_detach_gradients_and_reload_resume_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,path,model,norm=plan_fixture(tmp,self.ds);codec=ActionCodec(plan['codecs']['full']['C'])
            opt,sched=optimizer(model,plan,'full');g,s,y,valid=batch(self.ds,[1,2],0,2,'cpu',plan['config'])
            before=tensor_hash(model.state_dict());out,h=model(g,s,norm)
            update(model,opt,codec,out,y,valid,1.);self.assertNotEqual(before,tensor_hash(model.state_dict()))
            for prefix in ['graph_encoder','policy.rnn','policy.head']:
                self.assertTrue(any(k.startswith(prefix) and p.grad is not None and p.grad.abs().sum()>0 for k,p in model.named_parameters()))
            ident=dict(codec_sha256=codec.identity,normalizer_sha256=normalizer_hash(normalizer_state(norm)))
            progress=dict(step=1,epoch=1,hidden={1:h[:,:1].detach(),2:h[:,1:].detach()},chunk=1)
            receipt=save(tmp,model,opt,sched,norm,codec,ident,progress)
            p1,_=model(g,s,norm,h.detach());update(model,opt,codec,p1,y,valid,1.)
            cp,_=load_checkpoint(receipt,ident)
            clone=IntegrationPolicy(self.ds.schema,**plan['config']['model']);clone.load_state_dict(cp['model'])
            opt2,sc2=optimizer(clone,plan,'full');opt2.load_state_dict(cp['optimizer']);sc2.load_state_dict(cp['scheduler']);restore_rng(cp['rng'])
            hh=torch.cat([cp['progress']['hidden'][e] for e in [1,2]],1)
            self.assertFalse(hh.requires_grad);p2,_=clone(g,s,norm,hh);update(clone,opt2,codec,p2,y,valid,1.)
            self.assertEqual(tensor_hash(clone.state_dict()),tensor_hash(model.state_dict()))
            with self.assertRaises(ValueError):load_checkpoint(receipt,dict(ident,codec_sha256='wrong'))
            cp['action_codec']['method']='B';torch.save(cp,receipt['path'])
            with self.assertRaises(ValueError):load_checkpoint(receipt)

    def test_sequence_step_decode_hidden_carry_and_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,path,model,norm=plan_fixture(tmp,self.ds);codec=ActionCodec(plan['codecs']['full']['C'])
            model.eval();g,s,y,mask=batch(self.ds,[1],0,4,'cpu',plan['config'])
            with torch.no_grad():out,h=model(g,s,norm)
            adapter=DecodedPolicy(model,codec);stream=EpisodeStream(adapter,norm);arr=[]
            with torch.no_grad():
                for t in range(4):
                    gt,st,_,_=batch(self.ds,[1],t,1,'cpu',plan['config']);arr.append(stream.step(gt,st,'ep1',t))
            torch.testing.assert_close(torch.stack(arr,1),codec.decode(codec.tag(out)),atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(h,stream.hidden,atol=1e-6,rtol=1e-5)
            stream.reset();self.assertIsNone(stream.hidden)

    def test_full_ABC_workers_same_budget_initial_weight_sequence_and_test_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,path,_,_=plan_fixture(tmp,self.ds);results={}
            with patch.dict(os.environ,{'WANDB_MODE':'disabled'}),patch('graph.integration_baseline.bc08_train.load_plan',return_value=plan),patch('graph.integration_baseline.bc08_train.DirectDataset',return_value=self.ds):
                for m in ['A','B','C']:results[m]=worker(path,'full',m)
            runs=[Path(r['directory']) for r in results.values()]
            self.assertEqual(len(set(p.name for p in runs)),3)
            self.assertEqual({r['updates'] for r in results.values()},{4})
            schedules=[(p/'batch_identity.jsonl').read_text() for p in runs];self.assertEqual(schedules[0],schedules[1]);self.assertEqual(schedules[1],schedules[2])
            initial=[]
            for p in runs:
                cp=torch.load(p/'checkpoints/step000000-epoch001.pt',weights_only=False)
                initial.append(tensor_hash(cp['model']));self.assertTrue((p/'test_started.json').exists());self.assertTrue((p/'test_metrics.json').exists())
            self.assertEqual(len(set(initial)),1)

    def test_mid_training_failure_resume_matches_uninterrupted_and_preserves_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,path,_,_=plan_fixture(tmp,self.ds)
            with patch.dict(os.environ,{'WANDB_MODE':'disabled'}),patch('graph.integration_baseline.bc08_train.load_plan',return_value=plan),patch('graph.integration_baseline.bc08_train.DirectDataset',return_value=self.ds):
                reference=worker(path,'full','C');calls=[0]
                def fail_after_epoch(*args,**kwargs):
                    calls[0]+=1
                    if calls[0]==3:raise RuntimeError('injected failure before update')
                    return update(*args,**kwargs)
                with patch('graph.integration_baseline.bc08_train.update',side_effect=fail_after_epoch):
                    with self.assertRaises(RuntimeError):worker(path,'full','C')
                failed=Path(read(Path(tmp)/'full-C-train.json')['directory'])
                self.assertTrue((failed/'error_bundle.json').exists())
                result=worker(path,'full','C',resume=failed/'last_valid.json')
                self.assertEqual(result['selected']['model_identity'],reference['selected']['model_identity'])
                self.assertEqual((Path(result['directory'])/'batch_identity.jsonl').read_text(),(Path(reference['directory'])/'batch_identity.jsonl').read_text())
                with self.assertRaises(RuntimeError):worker(path,'full','C',resume=Path(result['directory'])/'last_valid.json')

    def test_real_worker_probe_checks_sequence_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan,path,_,_=plan_fixture(tmp,self.ds)
            with patch.dict(os.environ,{'WANDB_MODE':'disabled'}),patch('graph.integration_baseline.bc08_train.load_plan',return_value=plan),patch('graph.integration_baseline.bc08_train.DirectDataset',return_value=self.ds):
                result=worker(path,'overfit','C',probe=True)
            proof=read(Path(result['directory'])/'probe.json')
            self.assertTrue(proof['sequence_step_equivalence']);self.assertTrue(proof['exact_model_optimizer_rng_hidden_resume'])


class ExecutionTests(unittest.TestCase):
    def test_tmux_runtime_environment_excludes_credentials_and_limits_jobs(self):
        from graph.integration_baseline.bc08_pipeline import tmux_environment,run_tmux_jobs
        with patch.dict(os.environ,{'HF_TOKEN':'private-test-value','DISCORD_WEBHOOK_URL':'private-webhook','CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
            args=tmux_environment()
        self.assertIn('CUBLAS_WORKSPACE_CONFIG=:4096:8',args)
        self.assertNotIn('private-test-value',' '.join(args));self.assertNotIn('private-webhook',' '.join(args))
        with patch('graph.integration_baseline.bc08_pipeline.read',return_value={}):
            with self.assertRaises(ValueError):run_tmux_jobs('/tmp/unused-task08-plan.json',[{}, {}, {}])
        with patch('graph.integration_baseline.bc08_pipeline.read',return_value={'maximum_parallel_training_jobs':3}):
            with self.assertRaises(ValueError):run_tmux_jobs('/tmp/unused-task08-plan.json',[{}, {}, {}, {}])

    def test_completion_notification_waits_for_actual_process_exit(self):
        from graph.integration_baseline.bc08_pipeline import shell_job_command
        with tempfile.TemporaryDirectory(prefix='task08 exit ') as tmp:
            root=Path(tmp);pidfile=root/'worker.pid';notification=root/'notified.json'
            notifier=root/'tmux'
            notifier.write_text('#!'+sys.executable+'\nimport os,json\nfrom pathlib import Path\n'
                +'pid=int(Path('+repr(str(pidfile))+').read_text())\n'
                +'try:\n os.kill(pid,0);alive=True\nexcept ProcessLookupError:\n alive=False\n'
                +'Path('+repr(str(notification))+').write_text(json.dumps({"worker_alive":alive}))\n')
            notifier.chmod(0o700)
            env=dict(os.environ,PATH=str(root)+os.pathsep+os.environ.get('PATH',''))
            for code in [0,7]:
                command=[sys.executable,'-c','import os,sys;from pathlib import Path;Path('+repr(str(pidfile))+').write_text(str(os.getpid()));sys.exit('+str(code)+')']
                shell=shell_job_command(command,root/'job log.txt','test-channel',root/'job.exit')
                process=subprocess.run(['bash','-c',shell],env=env,capture_output=True,text=True)
                self.assertEqual(process.returncode,code)
                self.assertEqual(int((root/'job.exit').read_text()),code)
                self.assertFalse(read(notification)['worker_alive'])

    def test_recovery_rejects_learning_source_changes(self):
        from graph.integration_baseline.bc08_recover import audit_sources
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pipeline=root/'graph/integration_baseline/bc08_pipeline.py';pipeline.parent.mkdir(parents=True)
            trainer=pipeline.with_name('bc08_train.py')
            pipeline.write_text('old pipeline');trainer.write_text('fixed learning code')
            old=dict(source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pipeline,trainer]})
            pipeline.write_text('new exit ordering')
            self.assertEqual(len(audit_sources(old,root)),1)
            trainer.write_text('different learning code')
            with self.assertRaises(ValueError):audit_sources(old,root)

    def test_adopted_checkpoint_requires_exact_receipt_and_contract(self):
        from graph.integration_baseline.bc08_train import make_identity
        from graph.integration_baseline.bc08_eval import validate_checkpoint_plan
        with tempfile.TemporaryDirectory() as tmp:
            old,path,model,norm=plan_fixture(tmp,Dataset());codec=ActionCodec(old['codecs']['overfit']['A'])
            ident=make_identity(old,path,'overfit','A');opt,sched=optimizer(model,old,'overfit')
            receipt=save(tmp,model,opt,sched,norm,codec,ident,dict(step=2000,epoch=1))
            cp,_=load_checkpoint(receipt,ident)
            validate_checkpoint_plan(cp,codec,receipt,old,path,'overfit','A')
            new=copy.deepcopy(old);new['adopted_overfit']={'A':dict(receipt=receipt,identity=ident)}
            newpath=Path(tmp)/'successor.json';atomic(newpath,new)
            validate_checkpoint_plan(cp,codec,receipt,new,newpath,'overfit','A')
            with self.assertRaises(ValueError):validate_checkpoint_plan(cp,codec,dict(receipt,sha256='wrong'),new,newpath,'overfit','A')
            with self.assertRaises(ValueError):validate_checkpoint_plan(cp,codec,receipt,new,newpath,'full','A')
            damaged=copy.deepcopy(cp);damaged['normalizer']['mean']+=1
            with self.assertRaises(ValueError):validate_checkpoint_plan(damaged,codec,receipt,new,newpath,'overfit','A')

    def test_resource_gate_preserves_existing_jobs(self):
        from graph.integration_baseline.bc08_pipeline import gate
        r=dict(gpu=[['0','NVIDIA GeForce RTX 3090','24576','20000','1']],ram_available=32*1024**3,storage_free=20*1024**3)
        self.assertEqual(gate(r,peak=1024**3,slots=2)['slots'],2)
        r['gpu'][0][3]='100'
        with self.assertRaises(RuntimeError):gate(r,peak=1024**3,slots=2)

    def test_gpu_idle_allows_completed_worker_teardown_but_blocks_persistent_pid(self):
        from graph.integration_baseline.bc08_pipeline import wait_for_gpu_idle
        active={'processes':[['123','python','900']]};idle={'processes':[]}
        with patch('graph.integration_baseline.bc08_pipeline.resources',side_effect=[active,idle]),patch('graph.integration_baseline.bc08_pipeline.time.sleep'):
            self.assertIs(wait_for_gpu_idle(timeout=5),idle)
        with patch('graph.integration_baseline.bc08_pipeline.resources',return_value=active):
            with self.assertRaisesRegex(RuntimeError,'123'):
                wait_for_gpu_idle(timeout=0)

    def test_triple_probe_residency_accepts_vast_host_pid_namespace(self):
        from graph.integration_baseline.bc08_pipeline import probe_residency
        probes=[dict(pid=i,paired=True) for i in (10,11,12)]
        same=dict(processes=[[str(i),'python','900'] for i in (10,11,12)])
        host=dict(processes=[[str(i),'[Not Found]','900'] for i in (100,101,102)])
        with patch('graph.integration_baseline.bc08_pipeline.os.kill') as live:
            self.assertEqual(probe_residency(same,probes),'same_namespace')
            self.assertEqual(probe_residency(host,probes),'host_container_pid_namespace')
            self.assertEqual(live.call_count,6)
            with self.assertRaises(RuntimeError):probe_residency(dict(processes=host['processes'][:2]),probes)
            with self.assertRaises(RuntimeError):probe_residency(dict(processes=[[str(i),'python','900'] for i in (100,101,102)]),probes)

    def test_incomplete_reset_identity_never_counted_as_failure_or_success(self):
        from graph.integration_baseline.bc08_eval import summarize
        x=summarize([dict(status='BLOCKED_RESET_IDENTITY')])
        self.assertIsNone(x['success_rate']);self.assertEqual(x['blocked_reset_identity'],1)
        y=summarize([dict(status='COMPLETE',success=False,grasp_success=True,episode_length=250,first_success_step=None,timeout=True)])
        self.assertEqual(y['success_rate'],0);self.assertEqual(y['grasp_success_rate'],1)

    def test_static_error_report_decodes_assets_and_relative_links(self):
        from graph.integration_baseline.bc08_report import render
        from utils.experiment_logger import ExperimentLogger
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'WANDB_MODE':'disabled'}):
            with ExperimentLogger(team='integration',root=Path(tmp)/'experiments',model='task08_comparison') as logger:
                root=logger.directory;plan,path,_,_=plan_fixture(root,Dataset())
            (Path(tmp)/'docs/visual-graph-integration').mkdir(parents=True)
            with patch('graph.integration_baseline.bc08_report.__file__',str(Path(tmp)/'graph/integration_baseline/bc08_report.py')):
                result=render(path,dict(status='ERROR',stages={}))
            self.assertEqual(result['questions']['Q1'],'NOT_MEASURED')
            self.assertEqual(read(root/'verification.json')['status'],'PASS')
            self.assertGreater(read(root/'verification.json')['decoded']['links'],0)
            self.assertTrue((root/'bc_training_comparison.html').exists())


if __name__=='__main__':unittest.main()
