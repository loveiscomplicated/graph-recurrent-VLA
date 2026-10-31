"""Negative controls for the execution evidence gate; no simulator/CUDA required."""
import ast,hashlib,json,tempfile,unittest
from types import SimpleNamespace
import numpy as np
from pathlib import Path


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

class PrefixEvidenceTests(unittest.TestCase):
    def setUp(self):
        tree=ast.parse(Path(__file__).with_name('runner.py').read_text())
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='paired_verify')
        ns=dict(read=lambda p:json.loads(p.read_text()),nested_hash=digest)
        exec(compile(ast.Module(body=[func],type_ignores=[]),'gate','exec'),ns)
        self.verify=ns['paired_verify'];self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.out=Path(self.temp.name)
        self.rows=[]
        for c in ('E','B0','H1','H2'):
            folder=self.out/c;folder.mkdir()
            tr=[]
            for t in range(3):
                tr.append(dict(timestep=t,input_sha256='input'+str(t),rgbd_sha256='rgb'+str(t),state_sha256='state'+str(t),hidden_before_sha256=digest(None) if t==0 else 'hidden'+str(t-1),hidden_after_sha256='hidden'+str(t),policy_action=[t],action=[t]))
            (folder/'trace.json').write_text(json.dumps(tr))
            self.rows.append(dict(seed=1,condition=c,folder=c,handoff_timestep=1 if c in ('H1','H2') else None,initial_snapshot_sha256='s',initial_input_sha256='i',initial_rgbd_sha256='r'))
    def change(self,c,t,key,value):
        p=self.out/c/'trace.json';tr=json.loads(p.read_text());tr[t][key]=value;p.write_text(json.dumps(tr))
    def test_correct_prefix_carry(self):self.assertEqual(len(self.verify(self.out,self.rows)),2)
    def test_handoff_hidden_reset_rejected(self):
        self.change('H1',1,'hidden_before_sha256',digest(None))
        with self.assertRaises(AssertionError):self.verify(self.out,self.rows)
    def test_duplicate_forward_timestep_rejected(self):
        self.change('H2',1,'timestep',0)
        with self.assertRaises(AssertionError):self.verify(self.out,self.rows)
    def test_mismatched_prefix_input_rejected(self):
        self.change('H2',1,'input_sha256','wrong')
        with self.assertRaises(AssertionError):self.verify(self.out,self.rows)
    def test_initial_state_mismatch_rejected(self):
        self.rows[1]['initial_snapshot_sha256']='wrong'
        with self.assertRaises(AssertionError):self.verify(self.out,self.rows)

class AnchorScalarTests(unittest.TestCase):
    def test_numpy_physics_values_are_json_serializable(self):
        tree=ast.parse(Path(__file__).with_name('runner.py').read_text())
        func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='anchor_metrics')
        class Pose:
            def __mul__(self,other):return self
        ns=dict(np=np,to_sapien=lambda x:Pose(),sapien=SimpleNamespace(Pose=lambda x:Pose()),numpy=lambda x:x)
        exec(compile(ast.Module(body=[func],type_ignores=[]),'anchor','exec'),ns)
        expert=SimpleNamespace(half_len=.05,lift_target=Pose(),_peg_err=lambda x:(np.float32(.001),np.float32(.01)),_err=lambda x:(np.float32(.001),np.float32(.01)))
        robot=SimpleNamespace(get_qpos=lambda:np.zeros(9),get_qvel=lambda:np.zeros(9))
        env=SimpleNamespace(unwrapped=SimpleNamespace(goal_pose=Pose(),agent=SimpleNamespace(robot=robot)))
        for phase,previous in [('align','lift'),('insert','align')]:
            metrics=ns['anchor_metrics'](env,expert,dict(grasp=True,lift_candidate=True),3,phase,previous)
            json.dumps(metrics,allow_nan=False)
            self.assertTrue(metrics['valid'])

if __name__=='__main__':unittest.main()
