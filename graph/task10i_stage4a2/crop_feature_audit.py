"""Matched patch-pooling crop control plus actual practical proposal negatives.

GT-mask features are explicitly ORACLE_DIAGNOSTIC. Predicted crops receive no GT
input. GT supplies only post-forward labels/purity; different-parent same-category
objects are never instance negatives. No candidate threshold is selected here.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import torch
from .dinov2_runtime import CurrentFeatures
from .dinov2_matching_audit import csv_write
from .prompt_audit import inputs
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS

MODES=('full_patch448','bbox_patch448','padded_patch448','masked_padded_patch448')


def actor_label(mask,seg,rr):
    n=int(mask.sum())
    if not n:return None,{}
    fractions={name:float(np.isin(seg,rr['robot_ids'] if name=='robot' else rr['object_ids'][name])[mask].mean())
               for name in ('peg','box','robot')}
    fractions['background']=1-sum(fractions.values())
    best=max(fractions,key=fractions.get)
    return best if fractions[best]>=.8 else None,fractions


def run(root,causal):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(42)
    chosen=inputs();metadata=[];vectors=[];source_identities=[];duplicate_proposals=0
    dino_events=read('/workspace/task10i_stage4a2_controls_20261101_v1/prompt/dino_events.json')
    raw_events=defaultdict(list)
    for event in dino_events:
        if event['prompt'] in ('red and white block','cardboard box'):
            raw_events[str(event['episode_id']),event['camera']].append(event)
    write(root/'registered_protocol.json',dict(modes=MODES,feature='last_hidden_state, CLS excluded, area-mask patch pooling, L2',
        resolution=[448,448],source_resolution=[128,128],new_sensor_information=False,
        sample='historical24 first source per parent: t0, midpoint, terminal plus actual C/BC proposal-event times',
        selection_by_GT_mask_quality=False,threshold_selection=False,source_sha256=sha(Path(__file__))))
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_matched_patch_crop_audit',seed=42,
         root=root/'experiments',job_type='evaluation',config=dict(modes=MODES,partition='historical_development')) as logger:
        extractor=CurrentFeatures('/workspace/task10i_stage4a2_models/dinov2-small')
        for parent,(e,receipt) in sorted(chosen.items()):
            eid=str(e['episode_id']);refpath=Path(receipt['corrected_reference']);rr=read(refpath)
            assert sha(refpath)==receipt['corrected_reference_sha256']
            with np.load(refpath.parent/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
            p=Path('/workspace/task10i_stage4a_completion_20261101_v1/run/baseline/perception')/eid/'masks.npz'
            assert sha(p)==receipt['masks_sha256']
            with np.load(p) as z:predicted=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
            proposals=defaultdict(list)
            seen_proposals=set()
            for condition in ('C','BC'):
                path=causal/'episodes'/eid/condition/'candidate_masks.json'
                if path.exists():
                    source_identities.append(dict(path=str(path),sha256=sha(path)))
                    for r in read(path):
                        key=(r['timestep'],r['camera'],r['object_id'],tuple(r['mask_bits']))
                        if key in seen_proposals:
                            duplicate_proposals+=1;continue
                        seen_proposals.add(key)
                        proposals[r['timestep'],r['camera']].append(dict(r,condition=condition))
            sample={0,e['actions']//2,e['actions']}|{t for t,c in proposals}
            extractor.reset(eid)
            with h5py.File(e['raw_path']) as f:
                for t in range(e['actions']+1):
                    for ci,camera in enumerate(CAMERAS):
                        rgb=f[camera+'/rgb'][t];extractor.observe(eid,camera,t,rgb)
                        if t not in sample or not exact[t]:continue
                        masks=[]
                        if t in (0,e['actions']//2,e['actions']):
                            for oid,name in enumerate(('peg','box','robot')):
                                gt=np.isin(seg[t,ci],rr['robot_ids'] if name=='robot' else rr['object_ids'][name])
                                masks.append((name,'ORACLE_DIAGNOSTIC',gt,None))
                                if oid<2:masks.append((name,'PREDICTION_CROP_DIAGNOSTIC',predicted[t,ci,oid],None))
                        if t==0:
                            for event in raw_events[eid,camera]:
                                path=Path(event['raw_path']);source_identities.append(dict(path=str(path),sha256=sha(path)))
                                with np.load(path) as z:boxes=z['selected_boxes'].copy()
                                seen=set()
                                # Registered top6 postprocessed candidates per
                                # original phrase, without any GT success filter.
                                for rank,box in enumerate(boxes[:6]):
                                    x0,y0=np.floor(box[:2]).astype(int);x1,y1=np.ceil(box[2:]).astype(int)
                                    x0,y0=np.maximum([x0,y0],0);x1,y1=np.minimum([x1,y1],128)
                                    if x1<=x0 or y1<=y0:continue
                                    mask=np.zeros((128,128),bool);mask[y0:y1,x0:x1]=True
                                    key=tuple(np.packbits(mask.ravel()))
                                    if key in seen:continue
                                    seen.add(key)
                                    masks.append(('peg' if event['prompt']=='red and white block' else 'box',
                                        'ACTUAL_DINO_BOX_CROP',mask,rank))
                        for r in proposals[t,camera]:
                            mask=np.unpackbits(np.asarray(r['mask_bits'],np.uint8)).reshape(128,128).astype(bool)
                            masks.append((('peg','box')[r['object_id']],'ACTUAL_DINO_SAM_PROPOSAL',mask,r.get('post_rank',r.get('raw_rank'))))
                        for requested,branch,mask,rank in masks:
                            if not mask.any():continue
                            for mode in MODES:
                                vector=extractor.embed(camera,t,mask,mode)
                                if vector is None:continue
                                # Evaluation labels are computed after extraction,
                                # never used as crop or matching acceptance inputs.
                                label,fractions=actor_label(mask,seg[t,ci],rr)
                                metadata.append(dict(index=len(metadata),parent_id=parent,episode=eid,timestep=t,
                                    camera=camera,requested=requested,branch=branch,mode=mode,post_rank=rank,
                                    actual_actor=label,actor_fractions=fractions,
                                    target_purity=fractions[requested],mask_pixels=int(mask.sum()),
                                    reference_kind='EXACT_GT',reference_bank_used=False))
                                vectors.append(vector)
            write(root/'partial_provenance.json',metadata)
            print('matched crop audit',parent,len(metadata),flush=True)
        array=np.stack(vectors);np.savez_compressed(root/'embeddings.npz',embeddings=array)
        pairs=[]
        groups=defaultdict(list)
        for row in metadata:groups[row['parent_id'],row['mode'],row['branch']].append(row)
        for (parent,mode,branch),rows in groups.items():
            for i,a in enumerate(rows):
                for b in rows[i+1:]:
                    if a['actual_actor'] is None or b['actual_actor'] is None:continue
                    positive=a['actual_actor']==b['actual_actor']
                    if positive and a['actual_actor'] not in ('peg','box'):continue
                    if positive and a['timestep']==b['timestep'] and a['camera']==b['camera']:continue
                    pairs.append(dict(parent_id=parent,mode=mode,branch=branch,a=a['index'],b=b['index'],
                        positive=positive,similarity=float(array[a['index']]@array[b['index']]),
                        cross_camera=a['camera']!=b['camera'],hard_negative=not positive and
                        any(x in ('robot','background') for x in (a['actual_actor'],b['actual_actor'])),
                        unit='same physical parent/actor; not cross-episode instance classification'))
        # Separate category-only comparisons across physical parents; never mix
        # these with the instance Re-ID denominator.
        category=[]
        category_rows=[r for r in metadata if r['branch']=='ORACLE_DIAGNOSTIC' and r['timestep']==0 and
                       r['camera']=='base_camera' and r['actual_actor'] in ('peg','box')]
        for i,a in enumerate(category_rows):
            for b in category_rows[i+1:]:
                if a['parent_id']==b['parent_id'] or a['mode']!=b['mode']:continue
                category.append(dict(a=a['index'],b=b['index'],mode=a['mode'],
                    category_positive=a['actual_actor']==b['actual_actor'],
                    similarity=float(array[a['index']]@array[b['index']]),instance_label='NOT_DEFINED'))
        metrics=[]
        grouped=defaultdict(list)
        for row in pairs:grouped[row['mode'],row['branch'],row['cross_camera']].append(row)
        for (mode,branch,cross),rows in grouped.items():
            score=np.asarray([r['similarity'] for r in rows]);positive=np.asarray([r['positive'] for r in rows])
            for threshold in np.linspace(-1,1,201):
                accept=score>=threshold;tp=int((accept&positive).sum());fp=int((accept&~positive).sum())
                p=int(positive.sum());n=len(rows)-p
                metrics.append(dict(mode=mode,branch=branch,cross_camera=cross,threshold=float(threshold),
                    positives=p,negatives=n,tpr=tp/p if p else None,fpr=fp/n if n else None,
                    precision=tp/(tp+fp) if tp+fp else None,false_rejection_rate=1-tp/p if p else None))
        csv_write(root/'dinov2_positive_negative_pairs.csv',pairs)
        csv_write(root/'dinov2_category_matching_pairs.csv',category)
        csv_write(root/'dinov2_crop_feature_ablation.csv',metrics)
        write(root/'dinov2_reference_provenance.json',metadata)
        write(root/'dinov2_call_trace.json',extractor.calls)
        write(root/'scope.json',dict(status='MATCHED_PATCH_CROP_AND_PROPOSAL_AUDIT_COMPLETE',run_id=logger.run_id,
            parents=len(chosen),embeddings=len(metadata),instance_pairs=len(pairs),category_pairs=len(category),
            model_forwards=sum(r['model_forward'] for r in extractor.calls),source_proposal_files=source_identities,
            identical_same_frame_proposal_masks_deduplicated=duplicate_proposals,
            baseline_dinov2_active=False,actual_reference_bank_analysis_in_causal_ablation=True,
            GT_mask_forward_branches='ORACLE_DIAGNOSTIC only',independent_test=False,threshold_selected=False))
        logger.link_output(root/'scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('causal',type=Path)
    a=p.parse_args();run(a.root,a.causal)
