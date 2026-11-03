"""SAM2 device-output connections; no model/threshold/Graph policy changes.

The image bridge follows the installed SAM2ImagePredictor.predict body through
_prep_prompts/_predict and omits only its final .cpu().numpy() materialization.
Tracking uses SAM2Tracker.track(mask_filter=...) before attach_rgbd's existing
CPU consumer boundary. No predictor construction, grounding, or inference runs
at import time. Keep the caller's existing autocast/model settings.
"""
import numpy as np
import torch


@torch.inference_mode()
def image_masks_cuda(predictor,rgb,boxes):
    if len(boxes)!=2:raise ValueError('exactly Peg/Box prompts required')
    device=torch.device(predictor.device)
    if device.type!='cuda':raise ValueError('CUDA SAM2 predictor required')
    predictor.set_image(rgb)
    h,w=rgb.shape[:2];masks=torch.zeros((2,h,w),device=device,dtype=torch.bool);scores=[None,None]
    for oid,box in enumerate(boxes):
        if box is None:continue
        prompt=np.asarray(box,np.float32)
        if prompt.shape!=(4,):raise ValueError('XYXY prompt required')
        mask_input,coords,labels,bb=predictor._prep_prompts(None,None,prompt,None,True)
        logits,iou,_=predictor._predict(coords,labels,bb,mask_input,False,return_logits=True)
        logits=logits.squeeze(0).float().detach();iou=iou.squeeze(0).float().detach()
        if logits.device!=masks.device or tuple(logits.shape)!=(1,h,w) or iou.device!=masks.device:
            raise ValueError('unexpected SAM2 single-image CUDA output contract')
        masks[oid]=logits[0]>0;scores[oid]=iou[0]
    return masks,scores


class SAM2RobotFilter:
    """Callable for SAM2Tracker.track(mask_filter=...).

    observation_for_frame(frame_index) must explicitly join the current camera
    calibration/depth and current FK surfaces. It must respect any source-frame
    subsampling already used by the caller; frame/time mappings are not inferred.
    """
    def __init__(self,observation_for_frame,robot_filter=None):
        self.observation_for_frame=observation_for_frame
        if robot_filter is None:
            from .gpu_filter import CudaRobotFilter
            robot_filter=CudaRobotFilter()
        self.robot_filter=robot_filter

    @torch.inference_mode()
    def __call__(self,frame_index,masks):
        if set(masks)-{1,2}:raise ValueError('unknown tracker object ID')
        if not masks:return {}
        sample=next(iter(masks.values()))
        if not isinstance(sample,torch.Tensor) or not sample.is_cuda or sample.dtype!=torch.bool:
            raise ValueError('SAM2 CUDA bool masks required')
        for mask in masks.values():
            if not isinstance(mask,torch.Tensor) or mask.device!=sample.device or mask.dtype!=torch.bool or mask.shape!=sample.shape:
                raise ValueError('unaligned SAM2 CUDA masks')
        raw=torch.stack([masks[i] if i in masks else torch.zeros_like(sample) for i in (1,2)])
        camera,surfaces=self.observation_for_frame(frame_index)
        clean=self.robot_filter(camera,surfaces,raw)
        return {i:clean[i-1] for i in masks}


@torch.inference_mode()
def image_clean_masks_cuda(predictor,rgb,boxes,camera,surfaces,*,robot_filter=None):
    """Image SAM2 -> filter -> CUDA clean masks; scores remain CUDA tensors."""
    if robot_filter is None:
        from .gpu_filter import CudaRobotFilter
        robot_filter=CudaRobotFilter()
    raw,scores=image_masks_cuda(predictor,rgb,boxes)
    return robot_filter(camera,surfaces,raw),scores
