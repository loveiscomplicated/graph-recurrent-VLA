"""Record whether the unchanged CPU IK solver accepts each Mamba TCP target."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

import replay
from utils.experiment_logger import ExperimentLogger, write
from data_collection.episode_dataset import file_sha256


def main():
    replay.HORIZON=250
    original=replay.eval_episode
    output=Path(sys.argv[2])
    probe=[]

    def traced(env,front,decoder,seed,condition,moving):
        arm=env.unwrapped.agent.controller.controllers['arm']
        solver=arm.kinematics.compute_ik

        def accept(*a,**kw):
            result=solver(*a,**kw)
            probe.append(dict(accepted=result is not None,
                              joint_change=float((result-arm.qpos).abs().max().item())
                              if result is not None else None))
            return result

        with patch.object(arm.kinematics,'compute_ik',new=accept):
            return original(env,front,decoder,seed,condition,moving)

    replay.eval_episode=traced
    replay.main()
    trace=json.loads((output/'trace_700806.json').read_text())['episode']['trace']
    assert len(probe)==len(trace)==250
    result={'frames':probe,'windows':{f'{lo}:{hi}':dict(
                calls=hi-lo, rejected=sum(not f['accepted'] for f in probe[lo:hi]))
                for lo,hi in ((0,105),(105,125),(125,140),(140,160),(230,250))}}
    out=output/'ik.json'
    with ExperimentLogger(team='recurrent',model='mamba_700806_ik_probe',job_type='evaluate',
                          seed=700806,config={'script_sha256':file_sha256(__file__),'horizon':250},
                          root=output/'experiments') as log:
        write(out,result)
        log.link_output(out)
        log.log({'rejected':sum(not f['accepted'] for f in probe)})
    print(result['windows'],flush=True)


if __name__=='__main__':
    main()
