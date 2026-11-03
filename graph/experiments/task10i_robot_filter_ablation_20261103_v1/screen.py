"""512 baseline-only retrospective audit. Actor labels never enter the builder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

ROOT=Path(os.environ['TASK10I_ROOT']);CODE=Path(os.environ['TASK10I_LATENCY_CODE'])
REFERENCE=Path(os.environ['TASK10I_REFERENCE_CODE'])
sys.path.insert(0,str(REFERENCE));sys.path.insert(0,str(CODE));sys.path.insert(0,str(ROOT/'code'))
import common as C
import numpy as np
import torch
from graph_adapter import geometry,load_inputs
from graph_worker import masks_for
from candidates import Candidate
from utils.experiment_logger import ExperimentLogger


def main(args):
    C.assert_source();torch.set_num_threads(4)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(args.config).read_text());size=config['screen_size']
    specs=C.read(ROOT/'protocol.json')['scenes'];manifest=C.read(ROOT/'capture_manifest.json')['entries']
    entries={(x['scene'],x['size'],x['camera']):x for x in manifest}
    identity=dict(config=config,host=platform.platform(),gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,frozen_protocol_sha256=C.sha(ROOT/'protocol.json'),manifest_sha256=C.sha(ROOT/'capture_manifest.json'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        GT_used_for_screening=True,GT_used_as_graph_input=False,training=False,rollout=False,API_calls=False)
    C.write(output/'protocol.json',identity);rows=[]
    import graph.plan_a_geometry.core as core
    with ExperimentLogger(team='integration',model='task10i_robot_filter_screen',job_type='evaluate',seed=42,
                          root=output/'experiments',config=identity) as logger:
        with Candidate(9) as context:
            for i,s in enumerate(specs):
                scene=s['id'];result=dict(scene=scene,parent=s['parent'],timestep=s['timestep'],cohort=s['cohort'],size=size)
                try:
                    context.cache={};geo=geometry();inputs=load_inputs(scene,size,geo)
                    masks=masks_for(s,size,0,'actual');items=[]
                    for camera in inputs[0]:
                        filtered=core.classify_robot(camera,inputs[2]);reject=filtered['confirmed_robot']|filtered['uncertain']
                        e=entries[scene,size,camera.camera];gtpath=ROOT/e['gt_file']
                        with np.load(gtpath,allow_pickle=False) as z:seg=z['actor_segmentation'].copy()
                        assert C.sha(gtpath)==e['gt_sha256']
                        robot=np.isin(seg,e['object_ids']['robot']);valid=np.isfinite(camera.depth)
                        for oid,name in enumerate(('peg','box')):
                            if camera.camera=='hand_camera' and oid==1:continue
                            pred=masks[camera.camera][oid]&valid
                            correct=pred&np.isin(seg,e['object_ids'][name]);wrong=pred&robot
                            items.append(dict(camera=camera.camera,object_id=oid,raw_points=int(pred.sum()),
                                correct_raw=int(correct.sum()),correct_removed=int((correct&reject).sum()),
                                robot_raw=int(wrong.sum()),robot_removed=int((wrong&reject).sum()),
                                robot_remaining=int((wrong&~reject).sum())))
                    result.update(status='PASS',masks={k:C.array_sha(v) for k,v in masks.items()},items=items)
                except Exception as exc:
                    result.update(status='FAILED',error_type=type(exc).__name__,error=str(exc));torch.cuda.empty_cache()
                rows.append(result)
                C.write(output/'screen.json',dict(status='PARTIAL',rows=rows))
                logger.log(dict(attempted=i+1,failures=sum(x['status']!='PASS' for x in rows)),step=i+1)
                if i%10==0:print('screen',i+1,'/',len(specs),scene,result['status'],flush=True)
        path=output/'screen.json';C.write(path,dict(status='COMPLETE' if all(x['status']=='PASS' for x in rows) else 'PARTIAL',rows=rows))
        logger.link_output(path);logger.link_output(output/'protocol.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config',required=True);main(p.parse_args())
