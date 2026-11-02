"""Observers for exposed installed SAM2 outputs; never alter returned values."""
from pathlib import Path
import numpy as np
import torch


class SAMTrace:
    def __init__(self,tracker,logits):
        self.tracker=tracker;self.logits=Path(logits);self.calls=[];self.modules=[];self.handles=[]
        predictor=tracker.predictor
        self.original=predictor._run_single_frame_inference;self.original_get=predictor._get_image_feature
        def get(*args,**kwargs):
            frame=kwargs.get('frame_idx',args[1] if len(args)>1 else None)
            context=getattr(tracker,'research_context',{})
            assert context and context['anchor_t']+frame<=context['timestep'],'unobserved SAM image'
            return self.original_get(*args,**kwargs)
        def infer(*args,**kwargs):
            result=self.original(*args,**kwargs)
            context=getattr(tracker,'research_context',{}).copy()
            row=dict(**context,frame_idx=kwargs.get('frame_idx'),output_keys=list(result[0]),
                object_score_logits=result[0]['object_score_logits'].detach().float().cpu().reshape(-1).tolist(),
                is_init_cond_frame=kwargs.get('is_init_cond_frame'),run_mem_encoder=kwargs.get('run_mem_encoder'))
            state=kwargs.get('inference_state',{})
            row['memory_frame_keys']={str(k):{name:sorted(value[name]) for name in ('cond_frame_outputs','non_cond_frame_outputs')}
                for k,value in state.get('output_dict_per_obj',{}).items()}
            if context.get('kind')=='candidate_prompt':
                self.logits.mkdir(parents=True,exist_ok=True);path=self.logits/f'call{len(self.calls):07d}.npz'
                np.savez_compressed(path,mask_logits=result[0]['pred_masks'].detach().float().cpu().numpy())
                row['raw_mask_logits']=str(path)
            self.calls.append(row)
            return result
        predictor._get_image_feature=get;predictor._run_single_frame_inference=infer
        def hook(name):
            def observe(module,args,output):
                row=dict(**getattr(tracker,'research_context',{}),module=name)
                if name=='sam_mask_decoder':
                    assert len(output)==4,'installed decoder contract changed'
                    row.update(internal_iou_predictions=output[1].detach().float().cpu().tolist(),
                        object_score_logits=output[3].detach().float().cpu().tolist(),quality_is_semantic_confidence=False)
                self.modules.append(row)
            return observe
        for name in ('image_encoder','memory_encoder','sam_mask_decoder'):
            self.handles.append(getattr(predictor,name).register_forward_hook(hook(name)))

    def clear(self):
        self.calls.clear();self.modules.clear()

    def close(self):
        for handle in self.handles:handle.remove()
        self.tracker.predictor._run_single_frame_inference=self.original
        self.tracker.predictor._get_image_feature=self.original_get
