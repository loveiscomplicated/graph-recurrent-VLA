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
        from .preparation import RobotFilterPreparation
        self.preparation=RobotFilterPreparation(robot_filter)
        self._job=None;self._frame_index=None

    def prepare_frame(self,frame_index):
        """Called before forward SAM2 propagation by SAM2Tracker.track."""
        if self._job is not None:raise RuntimeError('previous SAM2 frame has not been filtered')
        camera,surfaces=self.observation_for_frame(frame_index)
        self._job=self.preparation.submit((camera,),surfaces);self._frame_index=int(frame_index)

    def cancel_prepared(self):
        if self._job is not None:self.preparation.discard(self._job)
        self._job=None;self._frame_index=None

    def close(self):
        self.cancel_prepared();self.preparation.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    @torch.inference_mode()
    def __call__(self,frame_index,masks):
        try:return self._filter(frame_index,masks)
        except BaseException:
            self.cancel_prepared();raise

    def _filter(self,frame_index,masks):
        if set(masks)-{1,2}:raise ValueError('unknown tracker object ID')
        if not masks:self.cancel_prepared();return {}
        sample=next(iter(masks.values()))
        if not isinstance(sample,torch.Tensor) or not sample.is_cuda or sample.dtype!=torch.bool:
            raise ValueError('SAM2 CUDA bool masks required')
        for mask in masks.values():
            if not isinstance(mask,torch.Tensor) or mask.device!=sample.device or mask.dtype!=torch.bool or mask.shape!=sample.shape:
                raise ValueError('unaligned SAM2 CUDA masks')
        raw=torch.stack([masks[i] if i in masks else torch.zeros_like(sample) for i in (1,2)])
        if self._job is None:self.prepare_frame(frame_index)
        if self._frame_index!=int(frame_index):
            self.cancel_prepared();raise ValueError('prepared frame does not match SAM2 output')
        job=self._job;self._job=None;self._frame_index=None
        clean=self.preparation.filter(job,{job.cameras[0].camera:raw})[job.cameras[0].camera]
        return {i:clean[i-1] for i in masks}


@torch.inference_mode()
def image_clean_masks_cuda(predictor,rgb,boxes,camera,surfaces,*,robot_filter=None,preparation=None):
    """Overlap current CPU geometry preparation with image SAM2; then filter."""
    name=camera.camera
    clean,scores=images_clean_masks_cuda({name:predictor},{name:rgb},{name:boxes},(camera,),surfaces,
        robot_filter=robot_filter,preparation=preparation)
    return clean[name],scores[name]


@torch.inference_mode()
def images_clean_masks_cuda(predictors,rgb,boxes,cameras,surfaces,*,robot_filter=None,preparation=None):
    """Prepare once for synchronized cameras while their image SAM2 runs.

    Pass a persistent RobotFilterPreparation context to reuse its CPU worker.
    A temporary context keeps the existing single-call API self-contained.
    """
    from .preparation import RobotFilterPreparation
    cameras=tuple(cameras);names={c.camera for c in cameras}
    if any(set(mapping)!=names for mapping in (predictors,rgb,boxes)):raise ValueError('SAM2 camera identity')
    if preparation is not None:
        if robot_filter is not None and robot_filter is not preparation.robot_filter:raise ValueError('preparation/filter ownership')
        robot_filter=preparation.robot_filter
    if robot_filter is None:
        from .gpu_filter import CudaRobotFilter
        robot_filter=CudaRobotFilter()
    owned=preparation is None
    if owned:preparation=RobotFilterPreparation(robot_filter)
    job=None
    try:
        job=preparation.submit(cameras,surfaces)
        raw={};scores={}
        for c in cameras:raw[c.camera],scores[c.camera]=image_masks_cuda(predictors[c.camera],rgb[c.camera],boxes[c.camera])
        return preparation.filter(job,raw),scores
    finally:
        if job is not None:preparation.discard(job)
        if owned:preparation.close()
