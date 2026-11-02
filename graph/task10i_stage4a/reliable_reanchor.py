"""Opt-in RGB-D-only Wrist Peg re-anchoring; production defaults unchanged."""
from pathlib import Path
import tempfile
import numpy as np
import cv2
from scipy.ndimage import binary_dilation
from graph.integration_baseline.online_vision_adapter import ObservedFrames
from graph.persistent_geometry.contracts import transform
from visual.runtime.initialization import PROMPTS


def positive_main_witness(cameras,clean_masks):
    main=next(c for c in cameras if c.camera=='base_camera');wrist=next(c for c in cameras if c.camera=='hand_camera')
    good=clean_masks['base_camera'][0]&np.isfinite(main.depth)
    world=main.world_points()[good];out=np.zeros(wrist.depth.shape,bool)
    if not len(world):return out
    xyz=transform(world,np.linalg.inv(wrist.world_from_camera));k=wrist.intrinsic
    uv=np.floor(xyz[:,:2]/np.maximum(xyz[:,2,None],1e-12)*np.array([k[0,0],k[1,1]])+k[:2,2]).astype(int)
    h,w=out.shape;inside=(xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    rows=np.flatnonzero(inside);u,v=uv[rows].T;agree=np.isfinite(wrist.depth[v,u])&(np.abs(wrist.depth[v,u]-xyz[rows,2])<.004)
    out[v[agree],u[agree]]=True
    return out


def raw_support_guard(mask,wrist,box_mask,robot_result,cfg):
    valid=mask&np.isfinite(wrist.depth)
    clean=valid&~robot_result['confirmed_robot']&~robot_result['uncertain']&~box_mask
    valid_fraction=float(valid.sum()/mask.sum()) if mask.any() else 0.
    clean_fraction=float(clean.sum()/valid.sum()) if valid.any() else 0.
    passed=valid_fraction>=cfg['minimum_valid_depth_fraction'] and clean_fraction>=cfg['minimum_clean_fraction']
    return passed,dict(valid_depth_fraction=valid_fraction,clean_fraction=clean_fraction,minimum_valid_depth_fraction=cfg['minimum_valid_depth_fraction'],minimum_clean_fraction=cfg['minimum_clean_fraction'],passed=passed)


def refine_shared_surface(mask,cameras,clean_masks,robot_result):
    """Keep unknown/occluded Main surfaces; prune depth-confirmed disagreements.

    Main clean Peg dilation is one pixel at the actual input resolution. No GT,
    object dimensions, simulator actor identity or pose is accepted here.
    """
    main=next(c for c in cameras if c.camera=='base_camera');wrist=next(c for c in cameras if c.camera=='hand_camera')
    output=np.asarray(mask,bool).copy()
    robot=robot_result['confirmed_robot']|robot_result['uncertain'];invalid=~np.isfinite(wrist.depth)
    robot_removed=int((output&robot).sum());output&=~robot&~invalid
    shared_removed=0
    if int(clean_masks['base_camera'][0].sum())>=6:
        pixel=np.flatnonzero(output);xyz=transform(wrist.world_points().reshape(-1,3)[pixel],np.linalg.inv(main.world_from_camera));k=main.intrinsic
        uv=np.floor(xyz[:,:2]/np.maximum(xyz[:,2,None],1e-12)*np.array([k[0,0],k[1,1]])+k[:2,2]).astype(int)
        h,w=main.depth.shape;inside=(xyz[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        rows=np.flatnonzero(inside);u,v=uv[rows].T;agree=np.isfinite(main.depth[v,u])&(np.abs(main.depth[v,u]-xyz[rows,2])<.004)
        allowed=binary_dilation(clean_masks['base_camera'][0],structure=np.ones((3,3),bool))
        rejected=rows[agree&~allowed[v,u]];output.ravel()[pixel[rejected]]=False;shared_removed=len(rejected)
    return output,dict(robot_removed=robot_removed,shared_depth_confirmed_disagreement_removed=shared_removed,output_pixels=int(output.sum()))


class AnchorState:
    def __init__(self,tracker,rgb,box,actual_t):
        self.tracker=tracker;self.anchor_t=actual_t;self.last_t=actual_t-1
        self.directory=tempfile.TemporaryDirectory(prefix='task10i_reanchor_');self.images=ObservedFrames();self.state=None
        self.append(rgb,actual_t)
        self.state=tracker.start(self.directory.name,{1:box},frame_idx=0)
        self.state['images']=self.images;self.state['num_frames']=1
    def append(self,rgb,actual_t):
        if actual_t!=self.last_t+1:raise ValueError('correction-state timestamp discontinuity')
        local=actual_t-self.anchor_t;path=Path(self.directory.name)/f'{local:06d}.jpg'
        if not cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)):raise OSError('correction JPEG write')
        self.images.append_jpeg(path,self.tracker.predictor);self.last_t=actual_t
    def current(self):
        local=self.last_t-self.anchor_t;self.state['num_frames']=local+1
        it=self.tracker.track(self.state,start_frame=local)
        try:frame,values=next(it)
        finally:it.close()
        if frame!=local or set(values)!={1}:raise ValueError('correction object/local frame identity')
        return values[1].detach().cpu().numpy().astype(bool)
    def close(self):
        if self.state is not None:self.tracker.reset(self.state);self.state=None
        self.directory.cleanup()


class WristPegReanchor:
    """Current observations/proprioception only; GT labels are not accepted.

    Re-anchor opportunity every5 observed frames, only when baseline or current
    correction fails the unchanged Wrist gate. Low-score/large proposals require
    >=6 depth-agreeing positive Main witness pixels and >=0.6 coverage. Baseline
    masks and other object/camera slots remain available as fallback.
    """
    def __init__(self,detector,tracker,geometry,interval=5,topk=8,refine=False):
        self.detector=detector;self.tracker=tracker;self.geometry=geometry;self.interval=interval;self.topk=topk
        self.refine=refine
        self.identity=None;self.last_t=-1;self.active=None;self.events=[];self.proposal_masks=[];self.validated_proposal_masks=[];self.active_raw_masks=[]
    def reset(self,episode):
        if self.active:self.active.close()
        self.identity=episode;self.last_t=-1;self.active=None;self.events=[];self.proposal_masks=[];self.validated_proposal_masks=[];self.active_raw_masks=[]
    def close(self):
        if self.active:self.active.close();self.active=None
    def process(self,cameras,data,surfaces,baseline_masks):
        t=int(data['timestep']);identity=str(data['episode'])
        if identity!=self.identity or t!=self.last_t+1:raise ValueError('reanchor episode/branch continuity')
        self.last_t=t;wrist=next(c for c in cameras if c.camera=='hand_camera')
        base={c:m.copy() for c,m in baseline_masks.items()}
        prepared=self.geometry.system.prepare(cameras,data,base,surfaces)
        event=dict(episode=identity,timestep=t,baseline_gate=prepared['gate'],correction_used=False,attempted=False,proposals=[])
        if self.active:
            self.active.append(wrist.rgb,t);candidate=self.active.current()
            self.active_raw_masks.append((t,np.packbits(candidate.ravel())))
            guard=True
            if self.refine:guard,event['raw_support_guard']=raw_support_guard(candidate,wrist,base['hand_camera'][1],prepared['filters']['hand_camera'],self.geometry.system.protocol['gate'])
            if self.refine:candidate,event['propagation_refinement']=refine_shared_surface(candidate,cameras,prepared['clean'],prepared['filters']['hand_camera'])
            trial={c:m.copy() for c,m in base.items()};trial['hand_camera'][0]=candidate
            pp=self.geometry.system.prepare(cameras,data,trial,surfaces)
            event['propagation_gate']=pp['gate']
            if pp['gate']['accepted'] and guard:
                base=trial;prepared=pp;event['correction_used']=True;event['anchor_t']=self.active.anchor_t
            else:self.active.close();self.active=None
        if not prepared['gate']['accepted'] and t%self.interval==0:
            witness=positive_main_witness(cameras,prepared['clean']);event['witness_pixels']=int(witness.sum())
            if witness.sum()>=6:
                event['attempted']=True;boxes,scores=self.detector(wrist.rgb,PROMPTS['peg']);best=None
                for rank,(box,score) in enumerate(zip(boxes[:self.topk],scores[:self.topk])):
                    x0,y0,x1,y1=box;vv,uu=np.nonzero(witness)
                    box_support=float(((uu+.5>=x0)&(uu+.5<=x1)&(vv+.5>=y0)&(vv+.5<=y1)).mean())
                    row=dict(rank=rank,box=box,detector_uncalibrated_score=float(score),positive_witness_box_fraction=box_support)
                    if box_support<.6:row['reject']='insufficient_main_witness_in_box';event['proposals'].append(row);continue
                    state=AnchorState(self.tracker,wrist.rgb,box,t)
                    try:
                        mask=state.current();raw_mask=mask.copy();guard=True
                        if self.refine:guard,row['raw_support_guard']=raw_support_guard(mask,wrist,base['hand_camera'][1],prepared['filters']['hand_camera'],self.geometry.system.protocol['gate'])
                        if self.refine:mask,row['refinement']=refine_shared_surface(mask,cameras,prepared['clean'],prepared['filters']['hand_camera'])
                        trial={c:m.copy() for c,m in base.items()};trial['hand_camera'][0]=mask
                        pp=self.geometry.system.prepare(cameras,data,trial,surfaces)
                        fraction=float((pp['conditions']['C3']['hand_camera'][0]&witness).sum()/witness.sum())
                        row['proposal_mask_index']=len(self.proposal_masks);self.proposal_masks.append(np.packbits(raw_mask.ravel()));self.validated_proposal_masks.append(np.packbits(mask.ravel()))
                        row.update(gate=pp['gate'],positive_witness_mask_fraction=fraction,mask_pixels=int(mask.sum()))
                        accepted=pp['gate']['accepted'] and fraction>=.6 and guard;row['accepted']=accepted
                        if accepted:
                            key=(fraction,float(score))
                            if best is None or key>best[0]:
                                if best:best[1].close()
                                best=(key,state,trial,pp);state=None
                    finally:
                        if state:state.close()
                    event['proposals'].append(row)
                if best:
                    if self.active:self.active.close()
                    _,self.active,base,prepared=best;event.update(correction_used=True,anchor_t=t,reanchored=True)
        self.events.append(event)
        return base,event
