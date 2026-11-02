"""Temporal/case diagnostics, keeping unsupported references and tiny support distinct."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from .run_pipeline import CAMERAS,read,write
from .instrumentation import support
from .temporal_audit import sequence_metrics


def run(output,baseline,corrected,candidate):
    output.mkdir(parents=True,exist_ok=False);episodes=[];cases=[];frame_rows=[]
    for p in sorted((corrected/'perception').glob('*/receipt.json')):
        cr=read(p);eid=str(cr['episode_id']);rrpath=Path(cr['corrected_reference']);rr=read(rrpath)
        with np.load(rrpath.parent/'segmentation.npz',allow_pickle=False) as z:seg,exact=z['actor_segmentation'].copy(),z['exact_frame'].copy()
        for condition,root in [('baseline',baseline),('candidate',candidate)]:
            with np.load(root/'perception'/eid/'masks.npz',allow_pickle=False) as z:
                masks=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
            for ci,camera in enumerate(CAMERAS):
                for oid in (0,1):
                    visible=[];correct=[];present=[];labels=[];xor=[]
                    for t,mask in enumerate(masks[:,ci,oid]):
                        m=support(mask,seg[t,ci],oid,rr['robot_ids'],[rr['object_ids']['peg'],rr['object_ids']['box']]) if exact[t] else None
                        visible.append(m['input_gt_support'] if m else -1);correct.append(m['true_positive'] if m else -1);present.append(bool(mask.any()))
                        identity=max((m['true_positive'],'object'),(m['wrong_object_pixels'],'wrong_object'),(m['robot_pixels'],'robot'),(m['background_pixels'],'background'))[1] if m and mask.any() else 'empty' if m else 'GT_UNSUPPORTED'
                        labels.append(identity)
                        flicker=float(np.logical_xor(mask,masks[t-1,ci,oid]).sum()/max(1,np.logical_or(mask,masks[t-1,ci,oid]).sum())) if t else None
                        if flicker is not None:xor.append(flicker)
                        status='INSUFFICIENT_REFERENCE' if m is None else 'PHYSICALLY_NOT_VISIBLE' if m['input_gt_support']==0 else 'GT_VISIBLE_BUT_NO_CORRECT_MASK' if m['input_gt_support']>=8 and m['true_positive']==0 else 'TINY_VISIBLE_SUPPORT' if m['input_gt_support']<8 else 'CORRECT_MASK_SUPPORT'
                        row=dict(condition=condition,episode=eid,parent_id=cr['parent_id'],timestep=t,camera=camera,object_id=oid,status=status,dominant_actor=identity,metrics=m,raw_mask_xor_union=flicker,reference_kind='EXACT_GT' if m else 'GT_UNSUPPORTED')
                        frame_rows.append(row)
                        if status not in ('CORRECT_MASK_SUPPORT','TINY_VISIBLE_SUPPORT') or identity in ('wrong_object','robot','background'):
                            cases.append(row)
                    temporal=sequence_metrics(visible,correct,present)
                    episodes.append(dict(condition=condition,episode=eid,parent_id=cr['parent_id'],camera=camera,object_id=oid,
                        **temporal,total_frames=len(exact),unsupported_frames=int((~exact).sum()),
                        tiny_frames=sum(0<n<8 for n in visible),mask_xor_union_mean=float(np.mean(xor)) if xor else None,
                        dominant_actor_switches=sum(a!=b for a,b in zip(labels,labels[1:]) if a not in ('empty','GT_UNSUPPORTED') and b not in ('empty','GT_UNSUPPORTED')),
                        tracker_slot_ids_fixed=True,unknown_frames_break_visible_runs=True))
    grouped=defaultdict(list)
    for r in episodes:grouped[r['condition'],r['parent_id'],r['camera'],r['object_id']].append(r)
    parent=[]
    for (condition,pid,camera,oid),rows in grouped.items():
        rate=[r['visible_missing_frames']/r['visible_frames'] for r in rows if r['visible_frames']]
        parent.append(dict(condition=condition,parent_id=pid,camera=camera,object_id=oid,
            visible_missing_rate=float(np.mean(rate)) if rate else None,episodes=len(rows),
            reappearances=sum(len(r['reappearances']) for r in rows),
            recovered=sum(sum(x['recovered'] for x in r['reappearances']) for r in rows),
            temporal_fragment_count=sum(r['temporal_fragment_count'] for r in rows),
            unsupported_frames=sum(r['unsupported_frames'] for r in rows)))
    write(output/'episode_metrics.json',episodes);write(output/'parent_metrics.json',parent)
    write(output/'frame_metrics.json',frame_rows);write(output/'failure_cases.json',cases)
    write(output/'scope.json',dict(status='TEMPORAL_PAIR_FINISHED',future_used=False,
        reappearance='>=8 visible support immediately following true zero; no joining across unsupported frames',
        raw_mask_xor_is_not_motion_compensated=True,dominant_actor_switch_is_not_internal_slot_switch=True,
        identical_inputs_and_complete_denominator=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('baseline',type=Path);p.add_argument('corrected',type=Path);p.add_argument('candidate',type=Path)
    a=p.parse_args();run(a.output,a.baseline,a.corrected,a.candidate)
