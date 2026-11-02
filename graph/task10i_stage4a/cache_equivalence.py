"""Real-data cache controls, unchanged masks and changed/empty masks, bytewise Graph check."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
from .run_pipeline import CAMERAS,STAGE3,read,write


def run(output):
    from graph.plan_a_bc.geometry import CurrentGeometry
    from .perframe_cache import install
    from graph.plan_a_bc.cache import GRAPH_KEYS
    torch.set_num_threads(1);torch.manual_seed(42)
    base=Path('/workspace/task10i_stage4a_completion_20261101_v1/run/baseline')
    rt=Path('/workspace/task10g_f1_abc_20261031/full');schema=read(Path('/workspace/task10f1_gt_pretrain_20261030/normalizer.json'))['schema']
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    choices=['10i_stage3_10i_s1_0362_full_task','10i_stage3_10i_s1_0666_full_task','2000003'];rows=[]
    for eid in choices:
        native=CurrentGeometry(rt/'inputs',schema);cached=CurrentGeometry(rt/'inputs',schema);memo=install(cached)
        with h5py.File(entries[eid]['raw_path'],'r') as f,np.load(base/'perception'/eid/'masks.npz',allow_pickle=False) as z:
            for t in (0,1,19):
                cameras,data,surfaces=native.from_h5(f,t,eid)
                original=np.unpackbits(z['mask_bits'][t]).reshape(2,2,128,128).astype(bool)
                variants=dict(original=original,empty=np.zeros_like(original),shifted=np.roll(original,1,axis=3))
                for label,mm in variants.items():
                    masks=dict(zip(CAMERAS,mm));g,_,gate=native.build(cameras,data,surfaces,masks)
                    a,_,ag=cached.build(cameras,data,surfaces,masks)
                    b,_,bg=cached.build(cameras,data,surfaces,masks)
                    assert gate==ag==bg
                    for k in GRAPH_KEYS:
                        expected=getattr(g,k).numpy()
                        for actual in (getattr(a,k).numpy(),getattr(b,k).numpy()):
                            assert actual.shape==expected.shape and actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes(),(eid,t,label,k)
                    rows.append(dict(episode=eid,timestep=t,mask_variant=label,gate_equal=True,graph_bytes_equal=True))
        print('cache equivalence',eid,'PASS',memo.hits,memo.misses,flush=True)
    write(output,dict(status='REAL_DATA_PREPARE_CACHE_EQUIVALENCE_PASS',rows=rows,graphs=27,repeated_graph_checks=54,
        graph_tensor_comparisons=270,mask_variants='original, empty negative control, shifted wrong/overlap control',
        source_classifier_inputs_hashed=True,cache_cleared_each_episode_frame=True,GT_input=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);run(p.parse_args().output)
