"""Same128 RGB information: original JPEG versus lossless SAM2 image container.

No source sensor, resolution, model, prompt, initializer or memory algorithm is
changed. JPEG-named temporary frames contain PNG data only in the explicit
lossless condition; the installed PIL loader's actual readback is verified.
"""
import argparse
from pathlib import Path
import cv2
import h5py
import numpy as np
import torch
from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter,ObservedFrames
from .prompt_audit import AUDITS
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha,STAGE3,CAMERAS


class LosslessFrames(ObservedFrames):
    def append_jpeg(self,path,predictor):
        rgb=self.pending_rgb;self.pending_rgb=None
        ok,encoded=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        if not ok:raise OSError('lossless PNG encoding failed')
        Path(path).write_bytes(encoded.tobytes())
        # Exact original pixels, including channel order, before same loader.
        decoded=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(decoded,rgb)
        from PIL import Image
        with Image.open(path) as image:np.testing.assert_array_equal(np.asarray(image.convert('RGB')),rgb)
        return super().append_jpeg(path,predictor)


class LosslessAdapter(OnlineVisionAdapter):
    def reset(self,episode_identity):
        super().reset(episode_identity);self.images=LosslessFrames()
    def process(self,rgb,episode_identity,timestep):
        self.images.pending_rgb=np.asarray(rgb).copy()
        return super().process(rgb,episode_identity,timestep)


WINDOWS=[dict(episode='10i_stage3_10i_s1_0924_full_task',last=20),
         dict(episode='10i_stage3_10i_s1_0121_full_task',last=106),dict(episode='2000003',last=70)]


def run(root):
    from utils.experiment_logger import ExperimentLogger
    from graph.task10i_stage4a.instrumentation import AuditedDino,support
    from graph.task10i_stage4a.development_perception import Memo
    from visual.runtime.tracking import SAM2Tracker
    from graph.plan_a_bc.geometry import CurrentGeometry
    root.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);torch.manual_seed(42)
    assert not torch.are_deterministic_algorithms_enabled() and not torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    write(root/'protocol.json',dict(windows=WINDOWS,conditions=['JPEG_B0','LOSSLESS_PNG'],raw_sensor=[128,128],
        changed='temporary SAM2 image encoding only',unchanged='RGB source, DINO forward/prompt/selection, processor loader, SAM checkpoint, tracking',
        graph_diagnostic_frames='every5th and terminal; not whole-frame geometry population',
        independent_test=False,GT_practical_input=False,source_sha256=sha(Path(__file__))))
    rt=Path('/workspace/task10g_f1_abc_20261031/full');paths=read(rt/'runtime_paths.json')
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_same128_preprocessing_control',seed=42,
         root=root/'experiments',job_type='evaluation',config=read(root/'protocol.json')) as logger:
        dino=AuditedDino(device='cuda',model_id=paths['dino'],output=root/'dino_raw');memo=Memo(dino)
        tracker=SAM2Tracker(device='cuda',checkpoint=paths['sam']);distortion=[]
        for w in WINDOWS:
            eid=w['episode'];e=entries[eid];folder=root/'episodes'/eid;folder.mkdir(parents=True)
            for condition,adapter_type in [('JPEG_B0',OnlineVisionAdapter),('LOSSLESS_PNG',LosslessAdapter)]:
                adapters={c:adapter_type(memo,tracker) for c in CAMERAS}
                for a in adapters.values():a.reset(eid)
                values=[]
                try:
                    with h5py.File(e['raw_path']) as raw,np.load(AUDITS/'baseline/perception'/eid/'masks.npz') as old:
                        for t in range(w['last']+1):
                            masks=[]
                            for c in CAMERAS:
                                rgb=raw[c+'/rgb'][t];dino.context=dict(episode_id=eid,camera=c,decision_timestep=t,condition=condition)
                                mask,_=adapters[c].process(rgb,eid,t);masks.append(mask)
                                if condition=='JPEG_B0':
                                    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                                    _,encoded=cv2.imencode('.jpg',bgr)
                                    _,quality95=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
                                    decoded=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
                                    error=decoded.astype(float)-rgb.astype(float);mse=float(np.mean(error**2))
                                    distortion.append(dict(episode=eid,camera=c,timestep=t,
                                        different_values=int(np.count_nonzero(error)),max_abs=float(np.max(abs(error))),mse=mse,
                                        psnr=10*np.log10(255**2/mse) if mse else None,
                                        original_encoder_equals_explicit_quality95=bool(np.array_equal(encoded,quality95))))
                            packed=np.packbits(np.stack(masks).ravel())
                            if condition=='JPEG_B0':assert np.array_equal(packed,old['mask_bits'][t]),('JPEG_B0_MISMATCH',eid,t)
                            values.append(packed)
                            if t%20==0:print('preprocessing',eid,condition,t,flush=True)
                    np.savez_compressed(folder/(condition+'.npz'),mask_bits=np.stack(values))
                finally:
                    for a in adapters.values():a.close()
                    adapters.clear();torch.cuda.empty_cache()
            memo.cache.clear()
        # No reference is opened until all prediction files have been saved.
        pixel=[];geometry_rows=[]
        schema=read('/workspace/task10f1_gt_pretrain_20261030/normalizer.json')['schema']
        for w in WINDOWS:
            eid=w['episode'];e=entries[eid];refpath=Path(read(AUDITS/'corrected/perception'/eid/'receipt.json')['corrected_reference']);rr=read(refpath)
            with np.load(refpath.parent/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
            with h5py.File(e['raw_path']) as raw:
                for condition in ('JPEG_B0','LOSSLESS_PNG'):
                    g=CurrentGeometry(rt/'inputs',schema)
                    with np.load(root/'episodes'/eid/(condition+'.npz')) as z:
                        masks=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
                    for t,mm in enumerate(masks):
                        for ci,c in enumerate(CAMERAS):
                            for oid in (0,1):
                                row=dict(episode=eid,condition=condition,timestep=t,camera=c,object_id=oid,
                                    reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED')
                                if exact[t]:row.update(support(mm[ci,oid],seg[t,ci],oid,rr['robot_ids'],[rr['object_ids']['peg'],rr['object_ids']['box']]))
                                pixel.append(row)
                        if t%5==0 or t==w['last']:
                            cameras,data,surfaces=g.from_h5(raw,t,eid)
                            graph,points,gate=g.build(cameras,data,surfaces,dict(zip(CAMERAS,mm)))
                            out=root/'graphs'/eid/condition;out.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(out/f't{t:04d}.npz',**{k:getattr(graph,k).numpy() for k in
                                ('node_input','node_type','edge_index','edge_attr','edge_type')})
                            for oid in (0,1):
                                selected=[r for r in points if r['object_id']==oid];labels=[]
                                if exact[t]:
                                    for r in selected:
                                        _,_,c,p=r['source_observation_ids'][0];actor=int(seg[t,CAMERAS.index(c)].ravel()[p])
                                        labels.append('correct' if actor in rr['object_ids'][('peg','box')[oid]] else 'robot' if actor in rr['robot_ids'] else 'other')
                                geometry_rows.append(dict(episode=eid,condition=condition,timestep=t,object_id=oid,points=len(selected),
                                    correct_points=labels.count('correct') if exact[t] else None,robot_points=labels.count('robot') if exact[t] else None,
                                    reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED'))
        csv_write(root/'jpeg_pixel_distortion.csv',distortion);csv_write(root/'preprocessing_pixel_metrics.csv',pixel)
        csv_write(root/'preprocessing_graph_control.csv',geometry_rows);write(root/'dino_events.json',dino.events)
        write(root/'scope.json',dict(status='SAME128_JPEG_LOSSLESS_CONTROL_COMPLETE',run_id=logger.run_id,
            windows=WINDOWS,JPEG_B0_masks_byte_equal=True,lossless_pixels_readback_equal=True,
            original_sensor_resolution_unchanged=True,DINO_receives_original_raw_RGB_in_both=True,
            no_new_high_resolution_information=True,independent_test=False,GT_evaluation_after_saved_inference=True))
        logger.link_output(root/'scope.json',kind='metrics');dino.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
