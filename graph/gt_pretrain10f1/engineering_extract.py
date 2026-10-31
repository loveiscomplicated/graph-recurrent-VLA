"""Additional real-frame frozen API smoke using discarded engineering weights."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from graph.plan_a_bc.common import read,write,sha,tensor_identity
from .model import restore_checkpoint
from .data import GraphPool
from .extract import run as extract_cli


def run(root,suffix=''):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ckpath=root/'engineering_data/engineering/resume_step1.pt'
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    pool=GraphPool(root/'engineering_data','cuda')
    ck.update(robot_schema=pool.schema,geometry_contract=read(root/'config.json')['geometry'],
        slot_order=['TCP','Left','Right','Peg','Box'],scope='DISCARDED_ENGINEERING_WEIGHTS_NOT_PRIMARY_PRETRAINING')
    path=root/('engineering_frozen_api_smoke'+suffix+'.pt')
    if path.exists():raise FileExistsError('preserve previous API smoke')
    torch.save(ck,path)
    model,_=restore_checkpoint(path,'cuda',True);before=tensor_identity(model.state_dict())
    g,r,_=pool.select([0,1,2,3]);first=model.extract(g,r,pool.schema);second=model.extract(g,r,pool.schema)
    for key in ('readout320','presence2','robot76'):torch.testing.assert_close(first[key],second[key],atol=0,rtol=0)
    np.testing.assert_array_equal(first['state38'],second['state38'])
    assert before==tensor_identity(model.state_dict())
    eid=pool.meta[0]['episode_id'];out=root/('engineering_extraction_features'+suffix+'.npz')
    extract_cli(path,root/'cache'/f'ep{eid}'/'graphs.npz',root/'normalizer.json',
        root/'cache'/f'ep{eid}'/'receipt.json',out,[0,1,2,3],'cuda')
    with np.load(out,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['state38'],first['state38'])
        np.testing.assert_array_equal(saved['readout320'],first['readout320'].cpu().numpy())
    write(root/('frozen_API_engineering_verification'+suffix+'.json'),dict(status='PASS',scope='real CUDA frames with discarded engineering weights',
        primary_pretraining_updates=0,readout320_shape=[4,320],state38_shape=[4,38],
        same_checkpoint_A_B=True,no_GT_sidecar_in_CLI=True,forward_deterministic=True,
        weight_hash_before=before,weight_hash_after=tensor_identity(model.state_dict()),
        feature_cache_sha256=sha(out),checkpoint_sha256=sha(path),
        actual_representation_accuracy='NOT_TESTED; engineering outputs are not a pretrained result'),immutable=True)
    print('FROZEN REAL-CUDA API PASS; engineering weights only',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--suffix',default='')
    a=p.parse_args();run(a.root,a.suffix)
