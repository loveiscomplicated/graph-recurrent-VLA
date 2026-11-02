"""Read-only DINO observer and exact-reference stage support metrics."""
from pathlib import Path
import time
import numpy as np
from visual.runtime.grounding import GDino

class AuditedDino(GDino):
    """Observe the SAME forward/postprocessing, including subthreshold queries.

    No proposal truncation, changed thresholds, or extra forward. Raw logits
    and normalized cxcywh boxes retain the installed model's query/token axes.
    """
    def __init__(self,*args,output,**kwargs):
        super().__init__(*args,**kwargs)
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.events=[];self.context={};self.pending=None
        self.handle=self.model.register_forward_hook(self.observe,with_kwargs=True)
    def observe(self,module,args,kwargs,outputs):
        self.pending=dict(logits=outputs.logits.detach().cpu().numpy(),
            pred_boxes=outputs.pred_boxes.detach().cpu().numpy(),input_ids=kwargs['input_ids'].detach().cpu().numpy(),
            pixel_values_shape=np.array(kwargs['pixel_values'].shape))
    def __call__(self,rgb,prompt):
        self.pending=None
        boxes,scores=super().__call__(rgb,prompt)
        assert self.pending is not None
        index=len(self.events);path=self.output/f'call{index:07d}.npz'
        np.savez_compressed(path,**self.pending,selected_boxes=np.asarray(boxes).reshape(-1,4),selected_scores=np.asarray(scores))
        self.events.append(dict(**self.context,call=index,prompt=prompt,normalized_text=prompt.lower().strip()+'.',
            box_threshold=self.threshold,text_threshold=self.threshold,ms=self.last_ms.copy(),raw_path=str(path),
            input_rgb_sha256=__import__('hashlib').sha256(np.asarray(rgb).tobytes()).hexdigest(),
            selected_count=len(boxes),score_semantics='uncalibrated model score'))
        return boxes,scores
    def close(self):self.handle.remove()

def support(mask,reference,oid,robot_ids,object_ids):
    mask=np.asarray(mask,bool);reference=np.asarray(reference)
    if mask.shape!=reference.shape:raise ValueError('pixel alignment')
    target=np.isin(reference,object_ids[oid]);other=np.isin(reference,object_ids[1-oid]);robot=np.isin(reference,robot_ids)
    tp=int((mask&target).sum());fp=int((mask&~target).sum());fn=int((~mask&target).sum())
    return dict(input_gt_support=int(target.sum()),output_support=int(mask.sum()),true_positive=tp,false_positive=fp,false_negative=fn,
        robot_pixels=int((mask&robot).sum()),wrong_object_pixels=int((mask&other).sum()),
        background_pixels=int((mask&~target&~other&~robot).sum()),
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
        iou=tp/(tp+fp+fn) if tp+fp+fn else None)

def reference_kind(comparisons):
    required={'rgb','depth_raw_mm','intrinsic','extrinsic_cv','cam2world_gl'}
    complete=set(comparisons)=={'base_camera','hand_camera'} and all(set(c)==required for c in comparisons.values())
    return 'EXACT_GT' if complete and all(v.get('exact') is True for c in comparisons.values() for v in c.values()) else 'GT_UNSUPPORTED'
