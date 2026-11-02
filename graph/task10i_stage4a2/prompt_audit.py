"""Controlled prompt forwards on preselected historical Development t0 inputs."""
import argparse
import hashlib
from pathlib import Path
import h5py
import numpy as np
import torch
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read, write, sha, STAGE3, CAMERAS
from graph.task10i_stage4a.instrumentation import AuditedDino
from visual.runtime.initialization import box_iou, assign

PROMPTS={'peg':['red and white peg','peg','rectangular rod','red and white block'],
         'box':['cardboard box','box','box with a hole']}
AUDITS=Path('/workspace/task10i_stage4a_completion_20261101_v1/run')


def bbox(mask):
    y,x=np.nonzero(mask)
    return [int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1] if len(x) else None


def inputs():
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    chosen={}
    for p in sorted((AUDITS/'corrected/perception').glob('*/receipt.json')):
        e=entries[p.parent.name]
        chosen.setdefault(e['parent_id'], (e,read(p)))
    return chosen


def run(root):
    from utils.experiment_logger import ExperimentLogger
    root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.manual_seed(42)
    chosen=inputs()
    write(root/'registered_inputs.json',dict(source_sha256=sha(Path(__file__)),
          episodes=[str(e['episode_id']) for e,r in chosen.values()],selection='first ID per historical Development parent',
          prompts=PROMPTS,selection_before_forward=True,new_final_access=False))
    rt=read('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json')
    rows=[];assignments=[]
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_prompt_control',seed=42,
         root=root/'experiments',job_type='evaluation',config=dict(partition='historical_development',prompts=PROMPTS)) as logger:
        dino=AuditedDino(device='cuda',model_id=rt['dino'],output=root/'raw_queries')
        try:
            for parent,(e,receipt) in sorted(chosen.items()):
                eid=str(e['episode_id']);refpath=Path(receipt['corrected_reference']);reference=read(refpath)
                assert sha(refpath)==receipt['corrected_reference_sha256']
                with np.load(refpath.parent/'segmentation.npz') as z:
                    assert z['exact_frame'][0];seg=z['actor_segmentation'][0].copy()
                with h5py.File(e['raw_path']) as f:
                    for ci,camera in enumerate(CAMERAS):
                        rgb=f[camera+'/rgb'][0];pool={}
                        for name,prompts in PROMPTS.items():
                            gtmask=np.isin(seg[ci],reference['object_ids'][name]);target=bbox(gtmask)
                            for prompt in prompts:
                                dino.context=dict(episode_id=eid,parent_id=parent,camera=camera,decision_timestep=0)
                                boxes,scores=dino(rgb,prompt)
                                event=dino.events[-1]
                                with np.load(event['raw_path']) as z:
                                    normalized=z['pred_boxes'][0]
                                    rawboxes=np.concatenate((normalized[:,:2]-normalized[:,2:]/2,
                                                            normalized[:,:2]+normalized[:,2:]/2),axis=1)*128
                                    tokens=dino.proc.tokenizer.convert_ids_to_tokens(z['input_ids'][0].tolist())
                                    logits=z['logits'][0]
                                eligible=[(b,float(s)) for b,s in zip(boxes,scores) if (b[2]-b[0])*(b[3]-b[1])<.25*128*128][:5]
                                pool[prompt]=eligible
                                local=lambda b:target is not None and box_iou(b,target)>=.5
                                hitranks=[i for i,b in enumerate(boxes) if local(b)]
                                rows.append(dict(parent_id=parent,episode=eid,camera=camera,object=name,prompt=prompt,
                                    gt_pixels=int(gtmask.sum()),raw_query_count=len(rawboxes),token_count=logits.shape[1],
                                    raw_localized_count=sum(local(b) for b in rawboxes),
                                    post_localized_count=sum(local(b) for b in boxes),
                                    first_post_localized_rank=hitranks[0] if hitranks else None,
                                    top5_localized=sum(local(b) for b,s in eligible),
                                    top5_localized_above_missing=sum(local(b) and s>.3 for b,s in eligible),
                                    maximum_localized_score=max((s for b,s in zip(boxes,scores) if local(b)),default=None),
                                    tokens=' '.join(tokens),raw_path=event['raw_path'],
                                    input_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                                    reference_kind='EXACT_GT',semantic_identity_confirmed=False))
                        for pegprompt in PROMPTS['peg']:
                            for boxprompt in PROMPTS['box']:
                                selected=assign(pool[pegprompt],pool[boxprompt])
                                for oid,name in enumerate(('peg','box')):
                                    target=bbox(np.isin(seg[ci],reference['object_ids'][name]));choice=selected[oid]
                                    assignments.append(dict(parent_id=parent,episode=eid,camera=camera,object=name,
                                        peg_prompt=pegprompt,box_prompt=boxprompt,chosen_box=choice[0] if choice else None,
                                        chosen_score=choice[1] if choice else None,
                                        gt_box_iou=box_iou(choice[0],target) if choice and target else None,
                                        localized=bool(choice and target and box_iou(choice[0],target)>=.5),
                                        semantic_identity_confirmed=False))
                write(root/'progress.json',dict(completed_parents=len({r['parent_id'] for r in rows}),last_parent=parent))
                write(root/'dino_events.json',dino.events)
                csv_write(root/'dino_prompt_comparison.csv',rows)
                write(root/'prompt_joint_assignment.json',assignments)
                print('prompt parent',parent,len(rows),flush=True)
        finally:dino.close()
        write(root/'scope.json',dict(status='HISTORICAL_PROMPT_CONTROL_COMPLETE',run_id=logger.run_id,
            parents=len(chosen),model_forwards=len(dino.events),rows=len(rows),new_prompt_changes_forward=True,
            threshold=.1,area_fraction=.25,topk=5,missing_score=.3,exclusion_iou=.3,
            thresholds_unchanged=True,semantic_and_sam2_evaluation_pending=True))
        logger.link_output(root/'scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
