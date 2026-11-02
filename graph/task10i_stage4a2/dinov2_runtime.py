"""Pinned DINOv2-S masked-patch features with explicit causal camera cache keys."""
from collections import OrderedDict
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from graph.task10i_stage4a.run_pipeline import read,sha

CHECKPOINT_SHA256='ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1'


class CurrentFeatures:
    def __init__(self,folder):
        from transformers import AutoModel
        folder=Path(folder);identity=read(folder/'download_identity.json')
        assert identity['model']=='facebook/dinov2-small'
        assert identity['files']['model.safetensors']==CHECKPOINT_SHA256
        assert all(sha(folder/p)==h for p,h in identity['files'].items())
        self.identity=identity
        self.model=AutoModel.from_pretrained(folder,local_files_only=True).cuda().eval()
        assert self.model.config.hidden_size==384 and self.model.config.patch_size==14
        self.calls=[];self.reset(None)

    def reset(self,episode):
        self.episode=str(episode);self.observed={};self.cache=OrderedDict()

    def observe(self,episode,camera,t,rgb):
        if str(episode)!=self.episode:raise ValueError('explicit feature episode reset required')
        if camera not in ('base_camera','hand_camera'):raise ValueError('unknown feature camera')
        previous=self.observed.get(camera)
        if t!=(previous['t']+1 if previous else 0):raise ValueError('feature observation clock discontinuity')
        rgb=np.asarray(rgb)
        if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3:raise ValueError('uint8 RGB required')
        self.observed[camera]=dict(t=t,rgb=rgb.copy(),sha256=hashlib.sha256(rgb.tobytes()).hexdigest())

    @torch.inference_mode()
    def embed(self,camera,t,mask,mode='full_patch448'):
        current=self.observed.get(camera)
        if current is None or current['t']!=t:raise ValueError('features require currently observed RGB')
        rgb=current['rgb'];mask=np.asarray(mask)
        if mask.dtype!=bool or mask.shape!=rgb.shape[:2]:raise ValueError('current predicted bool mask required')
        if not mask.any():return None
        image=rgb;support=mask;box=None
        if mode not in ('full_patch448','bbox_patch448','padded_patch448','masked_padded_patch448'):
            raise ValueError('unknown crop/feature contract')
        if mode!='full_patch448':
            y,x=np.nonzero(mask);box=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
            image=rgb[box[1]:box[3],box[0]:box[2]].copy()
            support=mask[box[1]:box[3],box[0]:box[2]].copy()
            if mode=='masked_padded_patch448':image[~support]=127
            if mode in ('padded_patch448','masked_padded_patch448'):
                h,w=support.shape;side=max(h,w)
                pad=np.full((side,side,3),127,np.uint8);sm=np.zeros((side,side),bool)
                yy,xx=(side-h)//2,(side-w)//2
                pad[yy:yy+h,xx:xx+w]=image;sm[yy:yy+h,xx:xx+w]=support
                image,support=pad,sm
        crop_hash=hashlib.sha256(image.tobytes()).hexdigest()
        weights=F.interpolate(torch.as_tensor(support,device='cuda').float()[None,None],
                              size=(32,32),mode='area').reshape(-1)
        if float(weights.sum())<.3:
            self.calls.append(dict(episode=self.episode,camera=camera,timestep=t,mode=mode,
                model_forward=False,rejection='insufficient_mask_patch_weight',mask_pixels=int(mask.sum()),
                mask_sha256=hashlib.sha256(np.packbits(mask.ravel()).tobytes()).hexdigest()))
            return None
        key=(self.episode,camera,t,current['sha256'],mode,crop_hash)
        hit=key in self.cache;forward_ms=0.
        if not hit:
            x=torch.from_numpy(np.ascontiguousarray(image)).cuda().permute(2,0,1)[None].float()/255
            x=F.interpolate(x,(448,448),mode='bilinear',align_corners=False)
            mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
            std=x.new_tensor([.229,.224,.225])[None,:,None,None]
            x=(x-mean)/std
            torch.cuda.synchronize();tick=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=self.model(pixel_values=x).last_hidden_state[0,1:].float()
            torch.cuda.synchronize();forward_ms=1000*(time.perf_counter()-tick)
            assert output.shape==(1024,384)
            self.cache[key]=output;self.cache.move_to_end(key)
            while len(self.cache)>8:self.cache.popitem(last=False)
        patches=self.cache[key]
        self.cache.move_to_end(key)
        embedding=F.normalize((patches*weights[:,None]).sum(0)/weights.sum(),dim=0)
        assert embedding.shape==(384,) and bool(torch.isfinite(embedding).all())
        assert abs(float(embedding.norm())-1)<1e-4
        self.calls.append(dict(episode=self.episode,camera=camera,timestep=t,mode=mode,
            rgb_sha256=current['sha256'],crop_sha256=crop_hash,crop_box=box,input_shape=[1,3,448,448],
            feature_shape=[1024,384],embedding_shape=[384],norm=float(embedding.norm()),
            model_forward=not hit,forward_ms=forward_ms,mask_pixels=int(mask.sum()),
            mask_sha256=hashlib.sha256(np.packbits(mask.ravel()).tobytes()).hexdigest()))
        return embedding.cpu().numpy()
