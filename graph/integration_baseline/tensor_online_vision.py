"""Native CUDA RGB -> existing causal SAM2 state -> CUDA raw masks.

Terra/DINO initialization explicitly reads only requested RGB images on scan
frames. Tracking frames perform no image download or JPEG filesystem roundtrip.
The legacy CPU adapter remains available via the inherited NumPy branch.
"""
from time import perf_counter
import numpy as np
import torch
import torch.nn.functional as F
from .online_vision_adapter import OnlineVisionAdapter
from visual.runtime.initialization import detect_initial,SCAN_EVERY


class HostScanFrames:
    """Lazy, scan-only CPU encoding boundary for grounding providers."""
    def __init__(self,frames):self.frames=frames;self.host={}
    def __len__(self):return len(self.frames)
    def __getitem__(self,i):
        if not isinstance(i,int) or not 0<=i<len(self):raise IndexError('unobserved grounding frame')
        if i not in self.host:self.host[i]=self.frames[i].detach().cpu().numpy()
        return self.host[i]


class TensorFrames:
    def __init__(self):self.frames=[]
    def __len__(self):return len(self.frames)
    def __getitem__(self,i):
        if not isinstance(i,int) or not 0<=i<len(self):raise IndexError('unobserved SAM2 frame')
        return self.frames[i]
    def append(self,rgb,predictor):
        # Video predictor's loader resizes with PIL bicubic, then normalizes.
        # Native RGB intentionally bypasses the legacy lossy JPEG encode/decode.
        image=rgb.permute(2,0,1)[None].to(torch.float32)
        image=F.interpolate(image,size=(predictor.image_size,)*2,mode='bicubic',align_corners=False,antialias=True)[0]
        image=image.clamp(0,255).round()/255.
        mean=image.new_tensor((.485,.456,.406))[:,None,None];std=image.new_tensor((.229,.224,.225))[:,None,None]
        self.frames.append((image-mean)/std)


class TensorOnlineVisionAdapter(OnlineVisionAdapter):
    @torch.inference_mode()
    def process(self,rgb,episode_identity,timestep,*,mask_device='cuda'):
        if not isinstance(rgb,torch.Tensor):return super().process(rgb,episode_identity,timestep,mask_device=mask_device)
        if mask_device!='cuda':raise ValueError('tensor RGB path returns CUDA masks; select legacy CPU RGB path explicitly')
        if not rgb.is_cuda or rgb.dtype!=torch.uint8 or rgb.ndim!=3 or rgb.shape[-1]!=3:raise ValueError('CUDA uint8 HWC RGB required')
        if episode_identity!=self.identity or timestep!=len(self.rgb):raise ValueError('online vision episode/timestep discontinuity; explicit reset required')
        if self.rgb and rgb.shape!=self.rgb[0].shape:raise ValueError('camera dimensions changed within episode')
        if not self.rgb:self.images=TensorFrames()
        if not isinstance(self.images,TensorFrames):raise ValueError('cannot change image device within episode')
        tick=perf_counter();self.rgb.append(rgb.clone());self.images.append(rgb,self.tracker.predictor)
        def detect(image,prompt):self.detector_calls+=1;return self.detector(image,prompt)
        new={}
        if timestep%SCAN_EVERY==0 and len(self.found)<2:
            self.scan_events+=1;joint=callable(getattr(self.detector,'initialize',None))
            if joint:self.detector_calls+=1
            initial=detect_initial(HostScanFrames(self.rgb),self.detector if joint else detect)
            new={i+1:initial[name]['box'] for i,name in enumerate(('peg','box')) if name not in self.found and initial[name]['frame']==timestep}
            if new:
                state=self.tracker.start_frames(self.images,tuple(rgb.shape[:2]),new,frame_idx=timestep)
                self.states[tuple(new)]=state
                for i,name in enumerate(('peg','box')):
                    if i+1 in new:self.found[name]=dict(initial[name],decision_observation=timestep,prefix_length=timestep+1)
        masks=torch.zeros((2,*rgb.shape[:2]),device=rgb.device,dtype=torch.bool)
        for group,state in self.states.items():
            state['num_frames']=timestep+1;iterator=self.tracker.track(state,start_frame=timestep)
            try:frame,values=next(iterator)
            finally:iterator.close()
            if frame!=timestep or set(values)!=set(group):raise ValueError('SAM2 current frame/object identity mismatch')
            for oid in group:
                value=values[oid]
                if value.device!=masks.device or value.dtype!=torch.bool or value.shape!=rgb.shape[:2]:raise ValueError('SAM2 CUDA mask alignment')
                masks[oid-1]=value
        record=dict(timestep=timestep,prefix_length=len(self.images),elapsed_ms=(perf_counter()-tick)*1000,
            mask_pixels=masks.sum(dim=(1,2)).tolist(),new_object_ids=list(new),detector_calls=self.detector_calls,
            initialization_scan_events=self.scan_events,tracking_groups=[list(g) for g in self.states],image_backend='native_cuda_rgb')
        if callable(getattr(self.detector,'status',None)):record['grounding']=self.detector.status()
        self.records.append(record);return masks,record
