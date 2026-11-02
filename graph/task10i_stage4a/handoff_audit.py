"""Read-only final raw-data/source-freeze/causal-source handoff checks."""
from pathlib import Path
import numpy as np
from .run_pipeline import STAGE3,CAMERAS,read,write,sha


def run(output,final,audits,development):
    output.mkdir(parents=True,exist_ok=False)
    manifest=read(STAGE3/'stage4_recache_input_manifest.json');protected=[]
    for e in manifest['unique_sources'].values():
        for field in ('raw','gt'):
            actual=sha(e[field+'_path']);assert actual==e[field+'_sha256'],e[field+'_path']
            protected.append(dict(episode_id=e['episode_id'],kind=field,path=e[field+'_path'],sha256=actual))
    for name,item in manifest['datasets'].items():assert sha(item['path'])==item['sha256']
    first=read(final/'development_candidate_freeze.json');frozen=read(final/'final_candidate_freeze.json')
    assert first['inference_files']==frozen['inference_files']
    source=Path(read(final/'handoff.json')['final_source_root'])
    for name,digest in frozen['inference_files'].items():assert sha(source/name)==digest,name
    roots=[('development',development),('internal_test',final/'internal_test/candidate'),('independent_validation',final/'validation/candidate')]
    counts=[]
    for partition,root in roots:
        frames=0;sources=0;nodes=0;wrist_box=0
        for p in sorted((root/'perception').glob('*/receipt.json')):
            receipt=read(p);eid=str(receipt['episode_id'])
            assert sha(p.parent/'masks.npz')==receipt['masks_sha256']
            assert sha(p.parent/'stage_rows.json')==receipt['rows_sha256']
            for t in range(receipt['frames']):
                folder=root/'traces'/eid/f't{t:04d}';trace=read(folder/'trace.json')
                with np.load(folder/'support.npz',allow_pickle=False) as z:
                    n=len(z['node_type']);edge=z['edge_index']
                    assert z['node_input'].shape==(n,21) and z['edge_attr'].shape==(edge.shape[1],4)
                    assert np.array_equal(z['node_type'][:3],np.arange(3))
                    assert np.all(z['node_input'][:,14:]==0)
                    assert np.isfinite(z['node_input']).all() and np.isfinite(z['edge_attr']).all()
                    assert not edge.size or edge.min()>=0 and edge.max()<n
                    assert all((z['node_type']==oid+3).sum()<=128 for oid in (0,1))
                    assert len(trace['representative_sources'])==len(z['point_world'])
                    for i,r in enumerate(trace['representative_sources']):
                        assert r['point']==i
                        for ep,ts,c,pixel in r['sources']:
                            assert ep==eid and ts==t and c in CAMERAS and 0<=pixel<128*128
                            assert not (r['object_id']==1 and c=='hand_camera')
                            if c=='hand_camera':assert trace['gate']['accepted']
                            sources+=1
                    nodes+=n
                frames+=1
        counts.append(dict(partition=partition,frames=frames,source_observation_ids=sources,nodes=nodes,foreign_future_or_history_sources=0,wrist_box_sources=wrist_box))
    write(output/'protection_checks.json',dict(status='PASS',original_raw_gt_files=len(protected),files=protected,
        original_dataset_manifests_unchanged=True,inference_source_unchanged_between_freezes_and_after_validation=True))
    write(output/'causal_graph_checks.json',dict(status='PASS',partitions=counts,current_only=True,cap128=True,graph21_edge4=True,production_adoption=False))
