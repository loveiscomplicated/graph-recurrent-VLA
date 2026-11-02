"""Byte compare preserved baseline detector tensors, without new forwards."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from graph.task10i_stage4a.run_pipeline import read,write,sha


def key(event):
    return (str(event['episode_id']),event['camera'],event['decision_timestep'],event['prompt'],event['input_rgb_sha256'])


def run(root,previous):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    current=Path('/workspace/task10i_stage4a2_20261101_v1/run/baseline_trace/baseline')
    config=dict(previous_events_sha256=sha(previous/'dino_events.json'),current_events_sha256=sha(current/'dino_events.json'),
        source_sha256=sha(Path(__file__)),new_model_forwards=0,scope='same historical101-action-frame fixture; F1 original raw tensors not presumed present')
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_baseline_raw_forward_equivalence',seed=42,
            root=root/'experiments',job_type='evaluation',config=config) as logger:
        all_left={key(e):e for e in read(previous/'dino_events.json')};right={key(e):e for e in read(current/'dino_events.json')}
        assert right.keys()<=all_left.keys() and len(right)==4
        left={k:all_left[k] for k in right}
        rows=[]
        for k in left:
            a,b=Path(left[k]['raw_path']),Path(right[k]['raw_path'])
            arrays=[]
            with np.load(a) as x,np.load(b) as y:
                assert x.files==y.files
                for name in x.files:
                    equal=x[name].shape==y[name].shape and x[name].dtype==y[name].dtype and x[name].tobytes()==y[name].tobytes()
                    arrays.append(dict(name=name,shape=list(x[name].shape),dtype=str(x[name].dtype),byte_equal=equal,
                        previous_tensor_sha256=hashlib.sha256(x[name].tobytes()).hexdigest(),
                        current_tensor_sha256=hashlib.sha256(y[name].tobytes()).hexdigest()))
                    assert equal,('baseline raw tensor mismatch',k,name)
            rows.append(dict(key=k,previous=str(a),previous_sha256=sha(a),current=str(b),current_sha256=sha(b),arrays=arrays))
        write(root/'baseline_raw_forward_equivalence.json',dict(status='PASS',run_id=logger.run_id,**config,
            events=rows,original_F1_mask_and_Graph_proof='/workspace/task10i_stage4a2_20261101_v1/run/baseline_forward_equivalence.json',
            original_F1_raw_equality='NOT_ASSERTED_WITHOUT_ORIGINAL_RAW_CACHE',whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('--previous',type=Path,default=Path('/workspace/task10i_stage4a_completion_20261101_v1/run/baseline'))
    a=p.parse_args();run(a.root,a.previous)
