"""ORACLE_DIAGNOSTIC: native SAM with exact current t0 actor boxes, then RGB only.

Uses the original JPEG/ObservedFrames and native video state API. Consumer
filters are bypassed deliberately. Only graph-relevant initially visible slots
are initialized. This is never a practical path or an adoption measurement.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import time
import cv2
import h5py
import numpy as np
import torch
from .evaluation_metrics import pixel_metrics,hierarchical_paired
from .dinov2_matching_audit import csv_write
from .prompt_audit import bbox
from graph.task10i_stage4a.run_pipeline import read,write,sha,CAMERAS


class InitialBoxOracle:
    def __init__(self,tracker,boxes):
        from graph.integration_baseline.online_vision_adapter import ObservedFrames
        self.tracker=tracker;self.boxes=boxes;self.images=ObservedFrames();self.directory=tempfile.TemporaryDirectory(prefix='10i4a2_ORACLE_init_')
        self.state=None;self.last=-1

    def process(self,rgb,t):
        assert t==self.last+1;self.last=t
        path=Path(self.directory.name)/f'{t:06d}.jpg'
        assert cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        assert self.images.append_jpeg(path,self.tracker.predictor)==rgb.shape[:2]
        if t==0 and self.boxes:
            self.state=self.tracker.start(self.directory.name,self.boxes,frame_idx=0)
            self.state['images']=self.images
        masks=np.zeros((2,*rgb.shape[:2]),bool)
        if self.state is not None:
            self.state['num_frames']=t+1
            iterator=self.tracker.track(self.state,start_frame=t)
            try:frame,values=next(iterator)
            finally:iterator.close()
            assert frame==t and set(values)==set(self.boxes)
            for oid,value in values.items():masks[oid-1]=value.detach().cpu().numpy()
        return masks

    def close(self):
        if self.state is not None:self.tracker.reset(self.state)
        self.state=None;self.directory.cleanup()


def run(root,expected,references,baseline,challenger,criteria,candidate):
    from utils.experiment_logger import ExperimentLogger
    from visual.runtime.tracking import SAM2Tracker
    for source in (baseline,challenger):assert read(source/'pipeline_state.json')['status']=='FROZEN_LIVE_CONDITION_COMPLETE'
    root.mkdir(parents=True,exist_ok=True)
    with (root/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        paths=read('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json')
        identity=read('/workspace/task10i_stage4a_20261031_v5/run/model_identity.json')
        assert sha(paths['sam'])==identity['sam_sha256']
        torch.set_num_threads(1);torch.manual_seed(42)
        assert not torch.are_deterministic_algorithms_enabled() and not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        protocol=dict(source_sha256=sha(Path(__file__)),expected_sha256=sha(expected),SAM_checkpoint_sha256=sha(paths['sam']),
            candidate=candidate,condition='ORACLE_CORRECT_T0_BOX',GT_only_at_initial_prompt=True,
            future_GT_or_correction_prompts=False,consumer_filters_bypassed=True,JPEG_same_as_B0=True,
            initialized_slots='Main Peg/Main Box and Wrist Peg if exact t0 visible pixels>=8',
            Wrist_Box_not_evaluated_or_initialized=True,subsequent_inputs='strict current RGB prefix only',
            model_failure_interpretation='failure after GT box initialization includes SAM ambiguity/propagation; does not prove an unavoidable intrinsic model limit',
            not_practical_improvement=True,production_latency_measurement=False)
        frozen=root/'protocol_frozen.json'
        if frozen.exists():assert read(frozen)==protocol
        else:write(frozen,protocol)
        with ExperimentLogger(team='integration',model='Task10I_Stage4A2_ORACLE_independent_correct_box_initialization',seed=42,
                root=root/'experiments',job_type='evaluation',config=protocol) as logger:
            tracker=SAM2Tracker('cuda',checkpoint=paths['sam']);progress=[]
            for e in read(expected)['entries']:
                eid=str(e['episode_id']);folder=root/'episodes'/eid
                if (folder/'receipt.json').exists():
                    receipt=read(folder/'receipt.json');assert receipt['protocol_sha256']==sha(frozen) and receipt['raw_sha256']==e['raw_sha256']
                    assert all(sha(folder/p)==h for p,h in receipt['files'].items());progress.append(receipt);continue
                if folder.exists():folder.rename(folder.with_name(eid+'.partial_'+str(time.time_ns())))
                folder.mkdir(parents=True)
                assert sha(e['raw_path'])==e['raw_sha256']
                ref=references/'references'/eid;rr=read(ref/'receipt.json')
                assert rr['raw_sha256']==e['raw_sha256'] and all(sha(ref/p)==h for p,h in rr['files'].items())
                with np.load(ref/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
                assert exact[0],'this control needs exact initial boxes'
                boxes={};initial=[]
                for ci,camera in enumerate(CAMERAS):
                    boxes[camera]={}
                    for oid,name in enumerate(('peg','box')):
                        if camera=='hand_camera' and oid==1:continue
                        target=np.isin(seg[0,ci],rr['object_ids'][name]);visible=int(target.sum());box=bbox(target)
                        initialized=visible>=8
                        if initialized:boxes[camera][oid+1]=box
                        initial.append(dict(camera=camera,object_id=oid,initial_GT_pixels=visible,
                            initialized_from_GT_box=initialized,oracle_box=box if initialized else None))
                write(folder/'initial_oracle_prompts.json',initial)
                adapters={camera:InitialBoxOracle(tracker,value) for camera,value in boxes.items()};masks=[]
                write(root/'pipeline_state.json',dict(status='RUNNING',episode=eid,completed_episodes=len(progress),pid=__import__('os').getpid()))
                try:
                    with h5py.File(e['raw_path']) as raw:
                        for t in range(e['actions']+1):
                            # Future segmentation is not read by this loop.
                            current=np.stack([adapters[c].process(raw[c+'/rgb'][t],t) for c in CAMERAS])
                            masks.append(np.packbits(current.ravel()))
                    np.savez_compressed(folder/'ORACLE_CORRECT_T0_BOX.npz',mask_bits=np.stack(masks))
                    write(folder/'receipt.json',dict(episode=eid,parent_id=e['parent_id'],stratum=e.get('stratum','internal'),
                        frames=len(masks),raw_sha256=e['raw_sha256'],protocol_sha256=sha(frozen),
                        reference_receipt=str(ref/'receipt.json'),reference_sha256=sha(ref/'receipt.json'),
                        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
                    progress.append(read(folder/'receipt.json'));write(root/'progress.json',progress)
                    print('ORACLE correct initial box',len(progress),'/',read(expected)['episodes'],eid,len(masks),flush=True)
                finally:
                    for adapter in adapters.values():adapter.close()
                    torch.cuda.empty_cache()
            assert len(progress)==read(expected)['episodes'] and sum(r['frames'] for r in progress)==read(expected)['frames']
            # All oracle predictions are complete before evaluating any suffix.
            rows=[];initial_rows=[]
            for e in read(expected)['entries']:
                eid=str(e['episode_id']);folder=root/'episodes'/eid;rr=read(references/'references'/eid/'receipt.json')
                with np.load(references/'references'/eid/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
                initial={(r['camera'],r['object_id']):r for r in read(folder/'initial_oracle_prompts.json')}
                for condition,source in [('B0',baseline/'perception'/eid/'B0.npz'),(candidate,challenger/'perception'/eid/(candidate+'.npz')),
                        ('ORACLE_CORRECT_T0_BOX',folder/'ORACLE_CORRECT_T0_BOX.npz')]:
                    with np.load(source) as z:pred=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
                    for t in range(e['actions']+1):
                        for ci,camera in enumerate(CAMERAS):
                            for oid,name in enumerate(('peg','box')):
                                if camera=='hand_camera' and oid==1:continue
                                row=dict(episode=eid,parent_id=e['parent_id'],stratum=e.get('stratum','internal'),condition=condition,
                                    camera=camera,object_id=oid,timestep=t,reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED',
                                    initially_GT_visible=initial[camera,oid]['initialized_from_GT_box'])
                                if exact[t]:row.update(pixel_metrics(pred[t,ci,oid],np.isin(seg[t,ci],rr['object_ids'][name]),
                                    np.isin(seg[t,ci],rr['robot_ids']),np.isin(seg[t,ci],rr['object_ids'][('box','peg')[oid]])))
                                rows.append(row)
                                if t==0:initial_rows.append(row)
            csv_write(root/'sam2_correct_init_independent_metrics.csv',rows)
            csv_write(root/'sam2_initialization_audit.csv',initial_rows)
            kwargs=read(criteria)['bootstrap'];summary=[]
            eligible=[r for r in rows if r['initially_GT_visible']]
            for metric in ('precision','recall','iou','correct_mask_availability','false_presence'):
                for comparator in (candidate,'ORACLE_CORRECT_T0_BOX'):
                    summary.extend(dict(comparator=comparator,**r) for r in hierarchical_paired(eligible,metric,
                        ('stratum','camera','object_id'),candidate=comparator,bootstrap=kwargs['replicates'],seed=kwargs['seed']))
            write(root/'correct_initialization_parent_metrics.json',summary)
            write(root/'scope.json',dict(status='ORACLE_INDEPENDENT_CORRECT_BOX_INITIALIZATION_COMPLETE',run_id=logger.run_id,**protocol,
                frames=read(expected)['frames'],episodes=len(progress),parents=read(expected)['parents'],
                primary_comparison_limited_to_initially_GT_visible_slots=True,never_claimed_as_practical_gain=True,
                files={p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name not in ('pipeline_state.json','progress.json')},whole_goal_complete=False))
            write(root/'pipeline_state.json',dict(status='ORACLE_INDEPENDENT_CORRECT_BOX_INITIALIZATION_COMPLETE',whole_goal_complete=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','expected','references','baseline','challenger','criteria'):p.add_argument(name,type=Path)
    p.add_argument('--candidate',required=True);a=p.parse_args()
    try:run(**vars(a))
    except BaseException as exc:
        if a.root.exists():write(a.root/'pipeline_state.json',dict(status='FAILED',error=repr(exc),partial_preserved=True))
        raise
