"""Oracle timing controls for corrected installed-SAM2 API ordering and rollback."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
from .sam2_correction import propose,commit
from .prompt_audit import bbox,AUDITS
from .dinov2_matching_audit import csv_write
from graph.task10i_stage4a.run_pipeline import read,write,sha,STAGE3,CAMERAS

CONDITIONS=('RETURN_AND_PREFLIGHT','CONDITIONING_CORRECTION','REJECTED_TRANSACTION')
WINDOWS=[dict(episode='10i_stage3_10i_s1_0121_full_task',camera='base_camera',oid=1,t=95,last=106),
         dict(episode='2000003',camera='base_camera',oid=0,t=5,last=25)]


def run(root):
    from utils.experiment_logger import ExperimentLogger
    from graph.integration_baseline.online_vision_adapter import OnlineVisionAdapter
    from graph.task10i_stage4a.development_perception import Memo
    from graph.task10i_stage4a.instrumentation import support
    from visual.runtime.grounding import GDino
    from visual.runtime.tracking import SAM2Tracker
    root.mkdir(parents=True,exist_ok=False)
    write(root/'registered_windows.json',dict(windows=WINDOWS,conditions=CONDITIONS,
        oracle_timing=True,source_sha256=sha(Path(__file__)),correction_sha256=sha(Path(__file__).with_name('sam2_correction.py'))))
    torch.set_num_threads(1);torch.manual_seed(42)
    entries={str(e['episode_id']):e for e in read(STAGE3/'stage4_recache_input_manifest.json')['unique_sources'].values()}
    rt=read('/workspace/task10g_f1_abc_20261031/full/runtime_paths.json');events=[];evaluation=[]
    with ExperimentLogger(team='integration',model='Task10I_Stage4A2_SAM2_API_order_and_transaction',
         root=root/'experiments',seed=42,job_type='evaluation',config=dict(oracle_control=True,windows=WINDOWS)) as logger:
        dino=GDino(device='cuda',model_id=rt['dino']);dino.context={};memo=Memo(dino)
        tracker=SAM2Tracker(device='cuda',checkpoint=rt['sam'])
        write(root/'predictor_flags.json',{key:getattr(tracker.predictor,key) for key in
            ('add_all_frames_to_correct_as_cond','clear_non_cond_mem_around_input')})
        for w in WINDOWS:
            eid=w['episode'];e=entries[eid];refpath=Path(read(AUDITS/'corrected/perception'/eid/'receipt.json')['corrected_reference'])
            rr=read(refpath)
            with np.load(refpath.parent/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
            assert exact[w['t']]
            box=bbox(np.isin(seg[w['t'],CAMERAS.index(w['camera'])],rr['object_ids'][('peg','box')[w['oid']]]))
            for condition in CONDITIONS:
                adapters={c:OnlineVisionAdapter(memo,tracker) for c in CAMERAS}
                for adapter in adapters.values():adapter.reset(eid)
                masks=[]
                try:
                    with h5py.File(e['raw_path']) as raw,np.load(AUDITS/'baseline/perception'/eid/'masks.npz') as old:
                        for t in range(w['last']+1):
                            values=[]
                            for c in CAMERAS:
                                dino.context=dict(episode_id=eid,camera=c,decision_timestep=t)
                                m,_=adapters[c].process(raw[c+'/rgb'][t],eid,t);values.append(m)
                            values=np.stack(values)
                            if t<w['t'] or condition=='REJECTED_TRANSACTION':
                                assert np.array_equal(np.packbits(values.ravel()),old['mask_bits'][t]),(eid,condition,t)
                            if t==w['t']:
                                transaction,detail=propose(adapters[w['camera']],w['oid']+1,t,box,
                                    conditioning=condition=='CONDITIONING_CORRECTION',verify_isolation=True)
                                assert transaction is not None
                                prompt_folder=root/'prompt_outputs'/eid;prompt_folder.mkdir(parents=True,exist_ok=True)
                                np.savez_compressed(prompt_folder/(condition+'.npz'),
                                    logits=transaction['prompt_logits'],
                                    object_ids=np.asarray(list(transaction['masks'])),oracle_box=np.asarray(box))
                                accepted=condition!='REJECTED_TRANSACTION'
                                commit(adapters[w['camera']],transaction,accepted)
                                if accepted:
                                    for oid,mask in transaction['masks'].items():values[CAMERAS.index(w['camera']),oid-1]=mask
                                events.append(dict(**w,condition=condition,accepted=accepted,detail=detail))
                            masks.append(np.packbits(values.ravel()))
                            if t%20==0:print('API control',eid,condition,t,flush=True)
                    folder=root/'episodes'/eid;folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/(condition+'.npz'),mask_bits=np.stack(masks))
                finally:
                    for adapter in adapters.values():adapter.close()
                    adapters.clear();memo.cache.clear();torch.cuda.empty_cache()
            write(root/'sam2_correction_event_log.json',events)
        # All inference output files exist before GT metric computation begins.
        for w in WINDOWS:
            eid=w['episode'];rrpath=Path(read(AUDITS/'corrected/perception'/eid/'receipt.json')['corrected_reference']);rr=read(rrpath)
            with np.load(rrpath.parent/'segmentation.npz') as z:seg=z['actor_segmentation'].copy();exact=z['exact_frame'].copy()
            for condition in CONDITIONS:
                with np.load(root/'episodes'/eid/(condition+'.npz')) as z:
                    masks=np.unpackbits(z['mask_bits'],axis=1).reshape(-1,2,2,128,128).astype(bool)
                for t,mm in enumerate(masks):
                    for ci,c in enumerate(CAMERAS):
                        for oid in (0,1):
                            row=dict(episode=eid,condition=condition,timestep=t,camera=c,object_id=oid,
                                correction_t=w['t'],reference_kind='EXACT_GT' if exact[t] else 'GT_UNSUPPORTED')
                            if exact[t]:row.update(support(mm[ci,oid],seg[t,ci],oid,rr['robot_ids'],
                                [rr['object_ids']['peg'],rr['object_ids']['box']]))
                            evaluation.append(row)
        csv_write(root/'sam2_memory_correction_ablation.csv',evaluation)
        write(root/'scope.json',dict(status='SAM2_API_ORDER_AND_TRANSACTION_CONTROLS_COMPLETE',run_id=logger.run_id,
            rollback_original_state_bytes_verified=True,oracle_timing=True,practical_trigger_evaluated=False,
            episodes=len(WINDOWS),rows=len(evaluation),source_sha256=sha(Path(__file__))))
        logger.link_output(root/'scope.json',kind='metrics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)
