"""Observer-free frozen-input timing, separate from concurrent quality construction."""
from common import *
import argparse
import subprocess
import time
import numpy as np
import torch
from PIL import Image
from sam_worker import setup,image_predict,video_predict


def static(arm):
    assert_source()
    predictions=read(ROOT/('api_predictions.json' if arm=='actual' else 'oracle_jobs.json'))['rows']
    jobs={p['image']:p for p in predictions if p['repeat']==0}
    entries=[e for e in read(ROOT/'capture_manifest.json')['entries'] if e['cohort']!='tracking']
    lookup={(e['scene'],e['camera'],e['size']):e for e in entries}
    pairs=list(dict.fromkeys((e['scene'],e['camera']) for e in entries));rows=[];details=[]
    predictor,identity=setup()
    def one(e,detail=False):
        p=jobs[e['id']];rgb=np.asarray(Image.open(ROOT/e['image_file']).convert('RGB')).copy()
        boxes=[p['parsed'][o]['pixels'] for o in ('peg','box')]
        # RGB loading above is outside the predictor timer.
        masks,timing,_=image_predict(predictor,rgb,boxes,detail)
        frozen=read(ROOT/'masks'/arm/'static'/p['id']/'result.json')
        assert array_sha(masks)==frozen['mask_sha256'],'timed image mask differs from frozen quality output'
        return timing
    with logger('timing_static_'+arm) as log:
        for size in SIZES:
            for _ in range(3):one(lookup[*pairs[0],size])
        for repeat in range(8):
            for i,pair in enumerate(pairs):
                for pos,size in enumerate(balanced_order(i+repeat)):
                    e=lookup[*pair,size];t=one(e)
                    rows.append(dict(arm=arm,scene=e['scene'],parent=e['parent'],camera=e['camera'],size=size,
                                     repeat=repeat,position=pos,**t))
            write(ROOT/('timing_sam_static_'+arm+'.json'),dict(rows=rows,identity=identity,status='PARTIAL'))
            print('TIME_STATIC',arm,repeat+1,'/8',flush=True)
        for e in entries:details.append(dict(scene=e['scene'],parent=e['parent'],camera=e['camera'],size=e['size'],**one(e,detail=True)))
        write(ROOT/('timing_sam_static_'+arm+'.json'),dict(rows=rows,details=details,identity=identity,status='COMPLETE',masks_equal=True))
        csvwrite(ROOT/('timing_sam_static_'+arm+'.csv'),rows)
        log.link_output(ROOT/('timing_sam_static_'+arm+'.json'))


def tracking(arm):
    assert_source()
    predictions=read(ROOT/('api_predictions.json' if arm=='actual' else 'oracle_jobs.json'))['rows']
    jobs={p['image']:p for p in predictions if p['repeat']==0}
    entries=[e for e in read(ROOT/'capture_manifest.json')['entries'] if e['cohort']=='tracking' and e['timestep']==0]
    lookup={(e['parent'],e['camera'],e['size']):e for e in entries};pairs=list(dict.fromkeys((e['parent'],e['camera']) for e in entries))
    model,identity=setup(video=True);frames=[];sequences=[];details=[]
    def one(e):
        p=jobs[e['id']];boxes=[p['parsed'][o]['pixels'] for o in ('peg','box')]
        torch.cuda.reset_peak_memory_stats()
        masks,timing=video_predict(model,ROOT/'jpeg'/e['parent']/str(e['size'])/e['camera'],boxes)
        frozen=read(ROOT/'masks'/arm/'tracking'/p['id']/'result.json')
        assert array_sha(masks)==frozen['mask_sha256'],'timed causal tracking masks differ'
        timing['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        timing['prompt_count']=sum(b is not None for b in boxes)
        return timing
    with logger('timing_tracking_'+arm) as log:
        for size in SIZES:one(lookup[*pairs[0],size])
        for repeat in range(4):
            for i,pair in enumerate(pairs):
                for pos,size in enumerate(balanced_order(i+repeat)):
                    e=lookup[*pair,size];t=one(e);meta=dict(arm=arm,parent=e['parent'],camera=e['camera'],size=size,repeat=repeat,position=pos)
                    frames.extend(dict(meta,**f) for f in t.pop('frames'))
                    sequences.append(dict(meta,**t))
            write(ROOT/('timing_sam_tracking_'+arm+'.json'),dict(frames=frames,sequences=sequences,identity=identity,status='PARTIAL'))
            print('TIME_TRACK',arm,repeat+1,'/4',flush=True)
        from functools import wraps
        from io_profile import JPEGIO
        for e in entries:
            spans=[];original=[];context={'t':-1}
            def hook(obj,name,label):
                fn=getattr(obj,name);original.append((obj,name,fn))
                @wraps(fn)
                def call(*a,**kw):
                    torch.cuda.synchronize();start=time.perf_counter();out=fn(*a,**kw);torch.cuda.synchronize()
                    spans.append(dict(timestep=context['t'],stage=label,ms=(time.perf_counter()-start)*1000));return out
                setattr(obj,name,call)
            for obj,name,label in [(model,'forward_image','image_encoder'),(model.sam_prompt_encoder,'forward','prompt_encoder'),
                    (model.sam_mask_decoder,'forward','mask_decoder'),(model.memory_attention,'forward','memory_attention'),
                    (model.memory_encoder,'forward','memory_encoder'),(model,'_get_orig_video_res_output','restore_original_size')]:hook(obj,name,label)
            p=jobs[e['id']];boxes=[p['parsed'][o]['pixels'] for o in ('peg','box')]
            try:
                folder=ROOT/'jpeg'/e['parent']/str(e['size'])/e['camera']
                with JPEGIO(folder) as file_io:
                    masks,t=video_predict(model,folder,boxes,before_frame=lambda t:context.update(t=t))
                t['file_io_profile']=file_io.result()
            finally:
                for obj,name,fn in reversed(original):setattr(obj,name,fn)
            assert array_sha(masks)==read(ROOT/'masks'/arm/'tracking'/p['id']/'result.json')['mask_sha256']
            details.append(dict(parent=e['parent'],camera=e['camera'],size=e['size'],spans=spans,timing=t))
        write(ROOT/('timing_sam_tracking_'+arm+'.json'),dict(frames=frames,sequences=sequences,details=details,identity=identity,status='COMPLETE',masks_equal=True))
        csvwrite(ROOT/('timing_sam_tracking_'+arm+'.csv'),frames)
        log.link_output(ROOT/('timing_sam_tracking_'+arm+'.json'))


def graph():
    from graph_adapter import geometry,load_inputs,build
    from graph_worker import masks_for,resolve_folder,timeout_handler
    import signal
    import resource
    resource.setrlimit(resource.RLIMIT_DATA,(80*1024**3,80*1024**3))
    torch.set_num_threads(4);signal.signal(signal.SIGALRM,timeout_handler)
    assert_source();p=read(ROOT/'protocol.json');index=read(ROOT/'graph_index_actual.json')['rows']
    path=ROOT/'timing_graph_actual.json';rows=read(path)['rows'] if path.exists() else []
    profile_path=ROOT/'timing_graph_profiles.json';profiles=read(profile_path)['rows'] if profile_path.exists() else []
    profile_scenes={s['id'] for s in p['scenes'][:6]}|{'seed98202_t000'}
    done={(r['scene'],r['size']) for r in rows};geo=geometry()
    with logger('timing_graph_actual') as log:
        # One cold/warm reference per size; excluded from official frame distributions.
        warmups=[]
        for size in SIZES:
            s=p['scenes'][0];inputs=load_inputs(s['id'],size,geo);masks=masks_for(s,size,0,'actual')
            signal.alarm(180)
            try:warmups.append(dict(size=size,graph_ms=build(geo,inputs,masks)['graph_ms']))
            except Exception as e:warmups.append(dict(size=size,status=type(e).__name__))
            finally:signal.alarm(0);geo.reset()
        for si,s in enumerate(p['scenes']):
            for pos,size in enumerate(balanced_order(si)):
                if (s['id'],size) in done:continue
                meta=dict(scene=s['id'],parent=s['parent'],cohort=s['cohort'],timestep=s['timestep'],size=size,position=pos)
                expected=next(r for r in index if r['scene']==s['id'] and r['size']==size and r['repeat']==0)
                signal.alarm(180)
                try:
                    inputs=load_inputs(s['id'],size,geo);masks=masks_for(s,size,0,'actual')
                    # Linux resets only this process's high-water RSS, not memory or caches.
                    Path('/proc/self/clear_refs').write_text('5\n')
                    rss_before=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
                    result=build(geo,inputs,masks)
                    peak_rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
                    if expected['status']=='COMPLETE':
                        frozen=read(resolve_folder(expected)/'result.json');assert result['hashes']==frozen['hashes']
                    signal.alarm(0)
                    rows.append(dict(meta,status='COMPLETE',graph_ms=result['graph_ms'],fk_ms=result['fk_ms'],
                                     input_join_ms=result['input_join_ms'],file_read_decode_ms=result['file_read_decode_ms'],
                                     host_peak_rss_bytes=peak_rss,host_rss_before_build_bytes=rss_before,
                                     hashes=result['hashes'],instrumentation_equal=expected['status']=='COMPLETE'))
                    if s['id'] in profile_scenes:
                        signal.alarm(180)
                        profiled=build(geo,inputs,masks,trace=True);assert profiled['hashes']==result['hashes']
                        profiles.append(dict(meta,native_ms=result['graph_ms'],instrumented_ms=profiled['graph_ms'],
                            overhead_ms=profiled['graph_ms']-result['graph_ms'],spans=profiled['spans'],hashes_equal=True))
                        write(profile_path,dict(rows=profiles,threads=4,concurrent_quality_workers=0))
                except Exception as e:
                    if rows and rows[-1].get('scene')==s['id'] and rows[-1].get('size')==size:
                        profiles.append(dict(meta,status=type(e).__name__,error=str(e),hashes_equal=False))
                        write(profile_path,dict(rows=profiles,threads=4,concurrent_quality_workers=0))
                    else:rows.append(dict(meta,status=type(e).__name__,graph_ms=None,fk_ms=None,error=str(e)))
                finally:signal.alarm(0);geo.reset()
                write(path,dict(rows=rows,warmups=warmups,status='PARTIAL',threads=4,concurrent_quality_workers=0))
            print('TIME_GRAPH',si+1,'/115',flush=True)
        write(path,dict(rows=rows,warmups=warmups,status='COMPLETE',threads=4,concurrent_quality_workers=0,
                        repeats_per_frozen_frame=1,independent_units='parent; frames are correlated'))
        csvwrite(ROOT/'timing_graph_actual.csv',rows);log.link_output(path)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['static','tracking','graph']);ap.add_argument('--arm',default='actual');a=ap.parse_args()
    if a.stage=='static':static(a.arm)
    elif a.stage=='tracking':tracking(a.arm)
    else:graph()
