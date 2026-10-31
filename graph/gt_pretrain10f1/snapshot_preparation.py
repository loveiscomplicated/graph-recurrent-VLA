"""Hash/readback completed artifacts while preserving active cache writes."""
import argparse
from collections import Counter
from pathlib import Path
import json

from graph.plan_a_bc.common import read,write,sha,now,tree_identity


def run(root):
    receipts=[read(p) for p in sorted((root/'raw').glob('*/receipt.json'))]
    phases=Counter();split_frames=Counter();success=Counter();pairs=Counter();hashes=[]
    for row in receipts:
        folder=root/'raw'/str(row['spec']['episode_id'])
        rp=folder/'receipt.json'
        hashes.append(dict(path=str(rp.relative_to(root)),sha256=sha(rp),bytes=rp.stat().st_size))
        for name,expected in row['hashes'].items():
            actual=sha(folder/name)
            if actual!=expected:raise ValueError('completed raw hash mismatch')
            hashes.append(dict(path=str((folder/name).relative_to(root)),sha256=actual,bytes=(folder/name).stat().st_size))
        for stage in read(folder/'expert_stages.json'):phases[stage['phase']]+=1
        split_frames[row['spec']['split']]+=row['actions'];success[row['spec']['split']]+=int(row['success'])
        pairs[row['spec']['split']+'_candidate']+=row['pair_candidates'];pairs[row['spec']['split']+'_valid']+=row['pairs_valid']
    caches=[];paired=[]
    for directory,rows in [('cache',caches),('pairs',paired)]:
        for p in sorted((root/directory).glob('*/receipt.json')):
            record=read(p);graph=p.parent/'graphs.npz'
            if sha(graph)!=record['sha256']:raise ValueError('completed graph hash mismatch')
            rows.append(record)
            for f in p.parent.iterdir():
                if f.is_file() and not f.name.endswith('.partial'):
                    hashes.append(dict(path=str(f.relative_to(root)),sha256=sha(f),bytes=f.stat().st_size))
    protected=read(root/'source_identity.json')['source_tree'];source=Path(__file__).resolve().parents[2]
    changed=[p for p,s in protected.items() if not p.startswith('graph/gt_pretrain10f1/') and sha(source/p)!=s]
    if changed:raise ValueError('inherited source changed '+str(changed))
    snapshot=dict(at=now(),run_id=read(root/'run_identity.json')['run_id'],status='WAITING_TASK10F_GT_READY_NOT_FAILURE',
        raw_episodes=len(receipts),raw_actions_by_split=dict(split_frames),insertion_successes_by_split=dict(success),
        phases=dict(phases),paired_candidates_valid=dict(pairs),GT38_oracle_max_abs=max(r['oracle_max_abs'] for r in receipts),
        raw_and_sidecar_sha_readback='PASS',cache_episodes=len(caches),cache_graphs=sum(r['graphs'] for r in caches),
        paired_cache_episodes=len(paired),paired_graphs=sum(r['graphs'] for r in paired),
        inherited_source_protection='PASS',inherited_source_files_checked=sum(not p.startswith('graph/gt_pretrain10f1/') for p in protected),
        primary_training_started=(root/'training').exists(),pretrained_checkpoint_exists=(root/'gnn_pretrained.pt').exists(),
        final_training_release_exists=(root/'data_ready_release.json').exists(),source_instance=57987772)
    write(root/'preparation_snapshot_steering_v1.json',snapshot,immutable=True)
    core=['run_identity.json','source_identity.json','gt38_feature_contract.json','target_observability_audit.json',
        'dataset_manifest.json','split_manifest.json','paired_intervention_manifest.json','pretrain_config.json',
        'expected_task10f_runtime_contract.json','existing_data_audit.json','optimization_equivalence.json',
        'concurrent_worker_benchmark.json','engineering_verification.json','preparation_snapshot_steering_v1.json',
        'gt_gru_normalizer_seed0.json','gt_gru_normalizer_seed1.json','gt_gru_normalizer_seed2.json',
        'engineering_extract.log','engineering_extract_v2.log','frozen_API_engineering_verification_v2.json',
        'engineering_frozen_api_smoke.pt','engineering_frozen_api_smoke_v2.pt','engineering_extraction_features_v2.npz',
        'engineering_extraction_features_v2.identity.json']
    for name in core:
        path=root/name
        if path.is_file():hashes.append(dict(path=name,sha256=sha(path),bytes=path.stat().st_size))
    inventory=dict(at=now(),scope='completed files only; active cache folders excluded; source package preserved separately',files=hashes)
    write(root/'preparation_artifact_index.json',inventory,immutable=True)
    (root/'preparation_backup.files').write_text('\n'.join([r['path'] for r in hashes]+['preparation_artifact_index.json'])+'\n')
    print(json.dumps(snapshot,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
