"""Independent completion audit; writes a separate receipt, never edits a run.

The reference distance/greedy implementation is separate from core.distances
and core.greedy. Run on the designated server with --raw-inputs, then on the
delivered artifacts for byte/readback verification.
"""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from graph.initial_pool10i.mirror.verify import read,sha,inventory
from .core import descriptor,fit,partition,random_order,greedy,SEEDS,canonical_hash
from .evaluate import stats


def csvrows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))


def audit(root,receipt,raw_inputs=False,prior=None):
    root=Path(root);assert read(root/'progress.json')['status']=='COMPLETED'
    idx=read(root/'artifact_index.json')['files']
    for rel,info in idx.items():
        p=root/rel;assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],rel
        if p.suffix=='.json':read(p)
        if p.suffix=='.png':
            from PIL import Image
            with Image.open(p) as im:im.verify()
        if p.suffix=='.csv':assert csvrows(p)
    for name in ['selection_identity.json','stage3_input_receipt.json']:
        for rel,h in read(root/name)['files'].items():assert sha(root/rel)==h,(name,rel)
    allstates=np.load(root/'analysis_states.npz',allow_pickle=False)
    ids=allstates['parent_id'].tolist();assert len(ids)==len(set(ids))==340
    states={cid:{k:allstates[k][i] for k in allstates.files if k!='parent_id'} for i,cid in enumerate(ids)}
    existing=read(root/'existing_valid_parent_inventory.json')['parents']
    split=read(root/'candidate_pool_split_manifest.json');newids=[i for i in ids if not i.startswith('existing_')]
    assert partition(newids)==split
    oldtrain=[r for r in existing if r['role']=='existing_train'];oldval=[r for r in existing if r['role']=='existing_validation']
    assert len(oldtrain)==33 and len(oldval)==7
    rows={i:dict(id=i,descriptor=descriptor(states[i]),role=('existing_train' if i in {r['id'] for r in oldtrain} else
        'existing_validation' if i.startswith('existing_') else 'training' if i in split['training'] else 'holdout' if i in split['holdout'] else 'validation')) for i in ids}
    for r in existing:
        for k,v in r['descriptor'].items():np.testing.assert_allclose(v,rows[r['id']]['descriptor'][k],atol=1e-12,rtol=0)
        rows[r['id']]['descriptor']=r['descriptor']
    lookup={k:read(root/n)['parents'] for k,n in [('R','dataset_R_parent_manifest.json'),('D','dataset_D_parent_manifest.json'),('V','shared_validation_parent_manifest.json')]}
    for rs in lookup.values():
        for r in rs:
            for k,v in r['descriptor'].items():np.testing.assert_allclose(v,rows[r['id']]['descriptor'][k],atol=1e-12,rtol=0)
            rows[r['id']]['descriptor']=r['descriptor']
    common=random_order(split['training'],SEEDS['common'])[:44]
    rtail=random_order(set(split['training'])-set(common),SEEDS['random'])[:19]
    base=[r['id'] for r in oldtrain]+common
    assert [r['id'] for r in lookup['R']]==base+rtail
    assert [r['id'] for r in lookup['V']]==[r['id'] for r in oldval]+split['validation'][:9]
    fitids=[r['id'] for r in oldtrain]+sorted(split['training'])
    scale=fit([rows[i] for i in fitids],fitids)
    frozen_scale=read(root/'feature_normalization.json')
    assert scale['fit_ids']==frozen_scale['fit_ids']
    for k in ['g1','g2','g3p','g3q','g4','g5']:
        for field,value in scale[k].items():np.testing.assert_allclose(value,frozen_scale[k][field],atol=1e-12,rtol=0)
    scale=frozen_scale
    a=[rows[i]['descriptor'] for i in ids];weights=np.array([.30,.20,.25,.15,.10])
    def vector(key):
        v=np.array([r[key] for r in a]);return np.linalg.norm((v[:,None]-v[None,:])/scale[key]['scale'],axis=-1)/np.sqrt(v.shape[1])
    def rotation(key):
        q=np.array([r[key] for r in a]);rr=Rotation.from_quat(q)
        return np.stack([(rr.inv()*r).magnitude() for r in rr])/scale[key]['scale']
    groups=np.stack([vector('g1'),rotation('g2'),.5*(vector('g3p')+rotation('g3q')),vector('g4'),vector('g5')],axis=-1)
    distance=np.sum(groups*weights,axis=-1);ind={cid:i for i,cid in enumerate(ids)}
    tr=sorted(split['training']);ti=[ind[i] for i in tr]
    nearest=distance[np.ix_(ti,[ind[i] for i in base])].min(1);used=set(base);order=[]
    trace=read(root/'coverage_selection_order.json')['trace']
    approximate_ties=[];max_reference_suboptimality=0.
    for step in range(176):
        scored=[]
        for cid in tr:
            if cid in used:continue
            scored.append((float(np.minimum(nearest,distance[ti,ind[cid]]).mean()),cid))
        reference_score,reference_id=min(scored)
        cid=trace[step]['id']
        score=next(value for value,candidate in scored if candidate==cid)
        gap=score-reference_score;max_reference_suboptimality=max(max_reference_suboptimality,gap)
        assert gap<1e-7,('independent objective non-optimal',step,gap)
        if cid!=reference_id:approximate_ties.append(dict(rank=step+1,frozen_id=cid,reference_id=reference_id,objective_gap=gap))
        order.append(cid);used.add(cid)
        nearest=np.minimum(nearest,distance[ti,ind[cid]])
        assert abs(score-trace[step]['objective_after'])<1e-7
    repeated=greedy([rows[i] for i in tr],[rows[i] for i in base],scale)
    repeat_exact=repeated==trace
    if raw_inputs:assert repeat_exact, 'separate-process same-runtime float-contract repeat failed'
    assert [r['id'] for r in lookup['D']]==base+order[:19]
    for label,pop,file in [('selection',split['training'],'random_vs_diversity_coverage.csv'),
                          ('holdout',split['holdout'],'coverage_holdout_metrics.csv')]:
        measured=csvrows(root/file);sets=dict(existing33=[r['id'] for r in oldtrain],base77=base,
                                            R=[r['id'] for r in lookup['R']],D=[r['id'] for r in lookup['D']])
        for result in measured:
            expected=stats(distance[np.ix_([ind[i] for i in pop],[ind[i] for i in sets[result['dataset']]])].min(1))
            for key,value in expected.items():assert abs(float(result[key])-value)<1e-7,(label,result['dataset'],key)
    assert [r['parent_id'] for r in csvrows(root/'validation_reserve.csv')]==split['validation'][9:]
    assert [r['parent_id'] for r in csvrows(root/'coverage_reserve.csv')]==order[19:]
    assert [r['parent_id'] for r in csvrows(root/'common_reserve.csv')]==random_order(set(split['training'])-set(common),SEEDS['common_reserve'])
    assert [r['parent_id'] for r in csvrows(root/'random_reserve.csv')]==random_order(set(split['training'])-set(common)-set(rtail),SEEDS['random_reserve'])
    trainset=set(sets['R'])|set(sets['D']);v=set(r['id'] for r in lookup['V']);h=set(split['holdout'])
    assert len(sets['R'])==len(sets['D'])==96 and len(v)==16
    assert not trainset&v and not (trainset|v)&h
    # Recompute all 200 conditional draws and their coverage metrics.
    remaining=sorted(set(split['training'])-set(common));rng=np.random.Generator(np.random.PCG64(SEEDS['monte_carlo']))
    samples=[rng.choice(remaining,19,replace=False).tolist() for _ in range(200)]
    assert samples==read(root/'random_monte_carlo_sampling_trace.json')['samples']
    for row in csvrows(root/'random_monte_carlo_metrics.csv'):
        pop=split['training'] if row['cohort']=='selection' else split['holdout']
        selected=base+samples[int(row['repeat'])]
        expected=stats(distance[np.ix_([ind[i] for i in pop],[ind[i] for i in selected])].min(1))
        for k,value in expected.items():assert abs(float(row[k])-value)<1e-7
    # Validate actual physical distances in every exported near-duplicate row.
    for row in csvrows(root/'near_duplicate_audit.csv'):
        sa,sb=states[row['left']],states[row['right']]
        for key,col in [('peg_pose','peg_translation_m'),('hole_pose','hole_translation_m'),('tcp_pose','tcp_translation_m')]:
            value=np.linalg.norm(sa[key][:3].astype(float)-sb[key][:3])
            assert abs(value-float(row[col]))<1e-10
        for key,col in [('peg_pose','peg_angle_rad'),('hole_pose','hole_angle_rad'),('tcp_pose','tcp_angle_rad')]:
            qa=Rotation.from_quat(sa[key][[4,5,6,3]]);qb=Rotation.from_quat(sb[key][[4,5,6,3]])
            assert abs((qa.inv()*qb).magnitude()-float(row[col]))<1e-7
    plan=read(root/'stage3_demonstration_plan.json');alloc=plan['per_parent_allocations']
    assert [sum(a['planned_logical_episodes'] for a in alloc if a['dataset']==k) for k in ['R','D','validation']]==[190,190,25]
    assert {a['new_parent'] for a in alloc}=={r['id'] for r in read(root/'stage3_parent_collection_union.json')['parents']}
    assert len(list((root/'figures').glob('*.png')))==12
    assert len(list((root/'gallery').glob('*_raw.png')))==340
    assert len(list((root/'gallery').glob('*_obb.png')))==340
    text=(root/'10i_stage2_diversity_aware_dataset_selection.md').read_text()
    for ref in re.findall(r'\]\(([^)]+)\)',text):assert (root/ref).is_file(),ref
    prior_match=None
    if prior:
        for name in ['selection_identity.json','coverage_selection_order.json','feature_normalization.json',
                     'candidate_pool_split_manifest.json','random_R_selected.csv','diversity_D_selected.csv']:
            assert sha(root/name)==sha(Path(prior)/name),name
        prior_match='EXACT_SHA256_MATCH'
    raw_count=0
    if raw_inputs:
        from .load_and_verify import POOL
        assert inventory(POOL)==read(root/'input_pool_inventory.json')
        for entry in read(POOL/'eligible_candidate_pool.json'):
            cid=entry['candidate_id']
            with np.load(POOL/entry['initial_npz'],allow_pickle=False) as z:
                for key,a in states[cid].items():np.testing.assert_array_equal(np.asarray(z[key]).reshape(a.shape),a)
            raw_count+=1
        import h5py
        for r in existing:
            ep=r['episodes'][0]
            assert sha(ep['gt_path'])==ep['gt_sha256'] and sha(ep['raw_path'])==ep['raw_sha256']
            with np.load(ep['gt_path'],allow_pickle=False) as z,h5py.File(ep['raw_path']) as f:
                for k in ['peg_pose','tcp_pose','hole_pose','peg_half','hole_radius']:
                    a=states[r['id']][k];np.testing.assert_array_equal(z['raw_'+k][0].reshape(a.shape),a)
                np.testing.assert_array_equal(f['robot_state'][0,:9],states[r['id']]['qpos'])
                np.testing.assert_array_equal(f['robot_state'][0,9:18],states[r['id']]['qvel'])
            raw_count+=1
    result=dict(status='PASS',artifacts=len(idx),parent_state_readback=340,raw_source_parents_checked=raw_count,
        independent_distance='SciPy relative Rotation magnitude + RMS; agrees within 1e-7',
        independent_greedy='176/176 choices optimal within 1e-7 in independent SO3/RMS reference',
        separate_process_exact_repeat=repeat_exact,
        cross_platform_numeric_tolerance=1e-12,
        independent_reference_roundoff_ties=approximate_ties,max_reference_suboptimality=max_reference_suboptimality,
        all_coverage_and_MC_metrics='PASS',reserve_order='PASS',
        per_parent_branch_budget=[190,190,25],prior_failed_run_selection=prior_match,
        selected_manifest_counts=[96,96,16],new_collection_union=read(root/'stage3_parent_collection_union.json')['count'],
        source_sha256=sha(Path(__file__)),stage3_started=False)
    Path(receipt).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('receipt',type=Path)
    p.add_argument('--raw-inputs',action='store_true');p.add_argument('--prior',type=Path);a=p.parse_args()
    audit(a.root,a.receipt,a.raw_inputs,a.prior)
