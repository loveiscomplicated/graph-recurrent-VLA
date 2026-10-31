import sys,inspect
sys.path.insert(0,'/workspace/task10g_f1_abc_20261031/source')
from graph.geometry_reach10d.runtime import environment
from graph.integration_baseline.direct_train import seed_all
seed_all(42,1)
e=environment('/workspace/task10g_f1_abc_20261031/full');o,_=e.reset(seed=30003);t=e.unwrapped
print('BACKEND',t.backend)
print('STATE',t.get_state_dict())
print('RNG',[(k,type(v).__name__) for k,v in vars(t).items() if 'rng' in k or 'seed' in k])
print('LIMITS',t.agent.robot.get_qlimits())
print('CONTACTS',t.scene.get_contacts())
print('BOX',[(k,str(v)) for k,v in vars(t).items() if any(x in k for x in ['hole','half','box_size'])])
print('PHYSX',[(x.name,[(type(c).__name__) for c in x.components]) for x in t.peg._objs])
print('RESET_SOURCE',inspect.getsource(type(t)._initialize_episode));print('LOAD_SOURCE',inspect.getsource(type(t)._load_scene))
print('SET_STATE_SOURCE',inspect.getsource(t.set_state_dict))
e.close()
