"""Re-forward an original development sequence against its immutable F1 cache."""
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from .run_pipeline import STAGE3,CAMERAS,read,write,sha
from .instrumentation import AuditedDino

def run(root):
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    from visual.runtime.tracking import SAM2Tracker
    from graph.plan_a_bc.geometry import CurrentGeometry
    from graph.plan_a_bc.cache import graph_at,GRAPH_KEYS
    # Match F1 cache.run exactly; seed_all also changes deterministic/TF32 flags.
    torch.set_num_threads(1);torch.manual_seed(42)
    entries=read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()
    dev=set(read(root/'protocol.json')['development'])
    candidates=[e for e in entries if e['source']=='existing_full_task' and e['parent_id'] in dev]
    e=min(candidates,key=lambda e:int(e['episode_id']));identity=str(e['episode_id'])
    old=Path('/workspace/task10f1_gt_pretrain_20261030');cached=old/'cache'/('ep'+identity)/'graphs.npz'
    receipt=read(cached.parent/'receipt.json')
    assert receipt['raw_episode_sha256']==e['raw_sha256'] and sha(cached)==receipt['sha256']
    runtime=Path('/workspace/task10g_f1_abc_20261031/full');paths=read(runtime/'runtime_paths.json')
    detector=AuditedDino(device='cuda',model_id=paths['dino'],output=root/'baseline/dino_raw')
    tracker=SAM2Tracker(device='cuda',checkpoint=paths['sam'])
    geometry=CurrentGeometry(runtime/'inputs',read(old/'normalizer.json')['schema'])
    adapters={c:OnlineVisionAdapter(detector,tracker) for c in CAMERAS}
    for a in adapters.values():a.reset(identity)
    rows=[];records=[]
    try:
        with h5py.File(e['raw_path']) as f,np.load(cached,allow_pickle=False) as expected:
            for t in range(e['actions']):
                start=time.perf_counter();cameras,data,surfaces=geometry.from_h5(f,t,identity);masks={}
                for c in cameras:
                    detector.context=dict(episode_id=identity,parent_id=e['parent_id'],camera=c.camera,decision_timestep=t)
                    masks[c.camera],v=adapters[c.camera].process(c.rgb,identity,t)
                    records.append(dict(camera=c.camera,**v))
                actual_bits=np.packbits(np.stack([masks[c] for c in CAMERAS]).ravel())
                mask_equal=np.array_equal(actual_bits,expected['mask_bits'][t])
                graph,points,gate=geometry.build(cameras,data,surfaces,masks)
                target=graph_at(expected,t)
                differences={k:int(np.count_nonzero(getattr(graph,k).numpy()!=getattr(target,k).numpy())) if getattr(graph,k).shape==getattr(target,k).shape else -1 for k in GRAPH_KEYS}
                rows.append(dict(timestep=t,mask_equal=mask_equal,graph_difference_values=differences,gate=gate,elapsed_ms=1000*(time.perf_counter()-start)))
                if t%20==0:print('baseline original-cache equivalence',identity,t,mask_equal,differences,flush=True)
        write(root/'baseline_forward_equivalence.json',dict(status='PASS' if all(r['mask_equal'] and not any(r['graph_difference_values'].values()) for r in rows) else 'FAIL',
            episode_id=identity,parent_id=e['parent_id'],frames=len(rows),raw_sha256=e['raw_sha256'],original_graph_sha256=sha(cached),
            original_cache_receipt=receipt,scope='one entire preserved development full-task sequence; not all dataset frames',
            torch_settings=dict(deterministic=torch.are_deterministic_algorithms_enabled(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),rows=rows))
        write(root/'baseline/vision_records.json',records);write(root/'baseline/dino_events.json',detector.events)
    finally:
        for a in adapters.values():a.close()
        detector.close()
    assert read(root/'baseline_forward_equivalence.json')['status']=='PASS','baseline reproduction mismatch: investigation required'
