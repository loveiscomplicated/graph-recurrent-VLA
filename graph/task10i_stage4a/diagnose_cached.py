"""Stagewise preliminary development diagnosis from verified historical masks."""
from pathlib import Path
import h5py
import numpy as np
from .run_pipeline import CAMERAS,STAGE3,read,write,sha
from .stagewise import trace_geometry

def run(root):
    from graph.plan_a_bc.geometry import CurrentGeometry
    from graph.plan_a_bc.cache import graph_at,GRAPH_KEYS
    old=Path('/workspace/task10f1_gt_pretrain_20261030');runtime=Path('/workspace/task10g_f1_abc_20261031/full')
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    summaries=[];allrows=[]
    for ref in sorted((root/'replay_reference').glob('*/receipt.json')):
        rr=read(ref);eid=str(rr['episode_id']);e=entries[eid];cache=old/'cache'/('ep'+eid)/'graphs.npz'
        if not cache.exists():
            summaries.append(dict(episode_id=eid,status='BASELINE_FORWARD_NEEDED',frames=e['actions']+1));continue
        cr=read(cache.parent/'receipt.json')
        assert cr['raw_episode_sha256']==e['raw_sha256'] and sha(cache)==cr['sha256']
        geom=CurrentGeometry(runtime/'inputs',read(old/'normalizer.json')['schema']);rows=[];equivalence=[]
        with h5py.File(e['raw_path']) as f,np.load(cache,allow_pickle=False) as masks,np.load(ref.parent/'segmentation.npz',allow_pickle=False) as gt:
            for t in range(e['actions']):
                cameras,data,surfaces=geom.from_h5(f,t,eid)
                mm=np.unpackbits(masks['mask_bits'][t]).reshape(2,2,128,128).astype(bool)
                reference={c:(gt['actor_segmentation'][t,i] if gt['exact_frame'][t] else None) for i,c in enumerate(CAMERAS)}
                graph,points,trace=trace_geometry(geom,cameras,data,surfaces,dict(zip(CAMERAS,mm)),reference,
                    [rr['object_ids']['peg'],rr['object_ids']['box']],rr['robot_ids'],root/'stagewise'/eid/f't{t:04d}')
                expected=graph_at(masks,t)
                equal=all(np.array_equal(getattr(graph,k).numpy(),getattr(expected,k).numpy()) for k in GRAPH_KEYS)
                equivalence.append(equal)
                if not equal:raise AssertionError('instrumented geometry baseline mismatch '+eid+'/'+str(t))
                for r in trace:r.update(parent_id=e['parent_id'],branch=e['branch'],source=e['source'])
                rows.extend(trace)
        allrows.extend(rows);summaries.append(dict(episode_id=eid,status='PASS',frames=e['actions'],terminal_frame_status='NO_HISTORICAL_GRAPH; remains required in full investigation',
            instrumented_forward_equivalence=all(equivalence),raw_sha256=e['raw_sha256'],historical_graph_sha256=sha(cache)))
        print('stagewise diagnosed',eid,len(equivalence),flush=True)
    write(root/'stagewise_development_rows.json',allrows)
    write(root/'stagewise_development_summary.json',dict(status='PRELIMINARY_NOT_FULL_DATASET',episodes=summaries,
        trace_rows=len(allrows),full_validation=False,all_required_stages_complete=False,production_adoption=False))
