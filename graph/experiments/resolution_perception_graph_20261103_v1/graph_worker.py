"""Build frozen real/oracle current Graphs and preserve every planned outcome."""
from common import *
import argparse
from functools import lru_cache
import signal
import time
import numpy as np
import torch
from graph_adapter import geometry,load_inputs,build,save_result


class GraphTimeout(TimeoutError):
    pass


def timeout_handler(*_):
    raise GraphTimeout('graph exceeded its per-case execution deadline')


@lru_cache(maxsize=24)
def masks_file(path):
    with np.load(path,allow_pickle=False) as z:return z['masks'].copy()


def masks_for(s,size,repeat,arm):
    if arm=='gt_mask':
        result={}
        manifest=read(ROOT/'capture_manifest.json')['entries']
        for camera in CAMERAS:
            e=next(e for e in manifest if e['scene']==s['id'] and e['size']==size and e['camera']==camera)
            with np.load(ROOT/e['gt_file'],allow_pickle=False) as z:seg=z['actor_segmentation']
            result[camera]=np.stack([np.isin(seg,e['object_ids'][o]) for o in ('peg','box')])
        return result
    result={}
    for camera in CAMERAS:
        if s['cohort']=='tracking':
            iid=image_id(s['parent']+'_track_t000',size,camera)
            path=ROOT/'masks'/arm/'tracking'/job_id(iid,repeat)/'masks.npz'
            result[camera]=masks_file(str(path))[s['timestep']].copy()
        else:
            iid=image_id(s['id'],size,camera)
            path=ROOT/'masks'/arm/'static'/job_id(iid,repeat)/'masks.npz'
            result[camera]=masks_file(str(path)).copy()
    return result


def verify_receipt(folder):
    r=read(folder/'receipt.json')
    for name,h in r.get('files',{}).items():assert sha(folder/name)==h,(folder,name)
    return r


def quality(arm,shard=0,shards=1):
    assert_source()
    if arm!='actual':assert (ROOT/'actual_graph_freeze.json').exists()
    if arm!='gt_mask':
        for name,h in read(ROOT/(arm+'_masks_freeze.json'))['files'].items():assert sha(ROOT/name)==h
    torch.set_num_threads(1 if shards>1 else 4)
    import resource
    resource.setrlimit(resource.RLIMIT_DATA,(24*1024**3,24*1024**3))
    signal.signal(signal.SIGALRM,timeout_handler)
    p=read(ROOT/'protocol.json');rows=[];cache={};geo=geometry()
    scenes=[s for i,s in enumerate(p['scenes']) if i%shards==shard]
    suffix='' if shards==1 else f'_shard{shard}'
    index=ROOT/('graph_index_'+arm+suffix+'.json')
    repeats=3 if arm=='actual' else 1
    with logger('graph_'+arm) as log:
        for s in scenes:
            for size in balanced_order(p['scenes'].index(s)):
                inputs=load_inputs(s['id'],size,geo,cache_sensors=True)
                for repeat in range(repeats):
                    folder=ROOT/'graphs'/arm/s['id']/str(size)/f'r{repeat}'
                    meta=dict(arm=arm,scene=s['id'],parent=s['parent'],cohort=s['cohort'],timestep=s['timestep'],size=size,repeat=repeat,
                              folder=str(folder.relative_to(ROOT)))
                    if (folder/'receipt.json').exists():
                        r=verify_receipt(folder);row=dict(meta,**r.get('outcome',{}));rows.append(row)
                        if row.get('mask_hash') and row.get('status')=='COMPLETE':cache[(s['id'],size,row['mask_hash'])]=folder
                        continue
                    if folder.exists():
                        folder.rename(folder.with_name(folder.name+'.interrupted_'+str(time.time_ns())))
                    mask_hash=None
                    try:
                        masks=masks_for(s,size,repeat,arm)
                        mask_hash=digest({c:array_sha(a) for c,a in masks.items()})
                        key=(s['id'],size,mask_hash)
                        if key in cache:
                            reference=cache[key]
                            folder.mkdir(parents=True)
                            outcome=dict(status='COMPLETE',mask_hash=mask_hash,reused_from=str(reference.relative_to(ROOT)),
                                         graph_ms=None,quality_reuse_only=True)
                            write(folder/'receipt.json',dict(status='COMPLETE',files={},outcome=outcome,
                                reused_receipt_sha256=sha(reference/'receipt.json')))
                            rows.append(dict(meta,**outcome));continue
                        signal.alarm(p['graph_case_timeout_seconds']*shards)
                        # The independent latency stage checks every real r0 against an uninstrumented build.
                        if repeat==0:
                            traced=build(geo,inputs,masks,trace=True)
                            diagnostic_ms=traced['graph_ms'];plain=traced
                            plain.update(graph_ms=None,instrumented_graph_ms=diagnostic_ms)
                            if s==p['scenes'][0]:
                                native_inputs=load_inputs(s['id'],size,geo,cache_sensors=False)
                                native=build(geo,native_inputs,masks)
                                assert native['hashes']==traced['hashes'],'instrumentation changed Graph tensors'
                                plain['graph_ms']=native['graph_ms']
                        else:plain=build(geo,inputs,masks)
                        signal.alarm(0)
                        save_result(folder,plain)
                        outcome=dict(status='COMPLETE',mask_hash=mask_hash,graph_ms=plain['graph_ms'],timing_context='quality_parallel' if shards>1 else 'quality_serial',
                                     instrumented_graph_ms=plain.get('instrumented_graph_ms'),hashes=plain['hashes'])
                        receipt=read(folder/'receipt.json');receipt['outcome']=outcome;write(folder/'receipt.json',receipt)
                        cache[key]=folder;rows.append(dict(meta,**outcome))
                    except (Exception,MemoryError) as e:
                        signal.alarm(0);folder.mkdir(parents=True,exist_ok=True)
                        status='TIMEOUT' if isinstance(e,GraphTimeout) else 'OOM' if isinstance(e,MemoryError) else 'GRAPH_FAILURE'
                        outcome=dict(status=status,error_type=type(e).__name__,error=str(e),mask_hash=mask_hash,
                                     execution_deadline_seconds=p['graph_case_timeout_seconds']*shards)
                        write(folder/'receipt.json',dict(status=status,files={},outcome=outcome));rows.append(dict(meta,**outcome))
                        if isinstance(e,AssertionError) or (isinstance(e,ValueError) and str(e).startswith((
                            'forbidden non-sensor input','camera identity','sensor identity join','mask alignment','Graph21/Edge4 shape','point cap'))):
                            # An adapter/schema defect must stop before GT controls, not become an accuracy outcome.
                            raise
                    write(index,dict(rows=rows,status='PARTIAL'))
                print('GRAPH',arm,s['id'],size,rows[-1]['status'],len(rows),flush=True)
                geo.reset()
        assert len(rows)==len(scenes)*4*repeats
        write(index,dict(rows=rows,status='COMPLETE',frozen_utc=now()))
        freeze=dict(index_sha256=sha(index),
                    receipts={r['folder']+'/receipt.json':sha(ROOT/r['folder']/'receipt.json') for r in rows},
                    cases=len(rows),frozen_utc=now(),GT_input=arm!='actual')
        write(ROOT/(arm+'_graph_freeze'+suffix+'.json'),freeze)
        log.log(dict(cases=len(rows),failures=sum(r['status']!='COMPLETE' for r in rows)))
        log.link_output(ROOT/(arm+'_graph_freeze'+suffix+'.json'))


def merge(arm,shards):
    rows=[]
    for shard in range(shards):
        p=read(ROOT/(f'graph_index_{arm}_shard{shard}.json'));assert p['status']=='COMPLETE';rows.extend(p['rows'])
    assert len(rows)==len(read(ROOT/'protocol.json')['scenes'])*4*(3 if arm=='actual' else 1)
    assert len({(r['scene'],r['size'],r['repeat']) for r in rows})==len(rows)
    rows=recover_memory_cases(rows,arm)
    for r in rows:verify_receipt(ROOT/r['folder'])
    write(ROOT/('graph_index_'+arm+'.json'),dict(rows=rows,status='COMPLETE',quality_workers=shards))
    write(ROOT/(arm+'_graph_freeze.json'),dict(index_sha256=sha(ROOT/('graph_index_'+arm+'.json')),
        receipts={r['folder']+'/receipt.json':sha(ROOT/r['folder']/'receipt.json') for r in rows},cases=len(rows),
        frozen_utc=now(),GT_input=arm!='actual',
        masks_freeze_sha256=sha(ROOT/(arm+'_masks_freeze.json')) if arm!='gt_mask' else None))


def recover_memory_cases(rows,arm):
    """After all24GiB workers exit, retry only memory failures alone at80GiB.

    Primary receipts remain unchanged. This changes resource allocation, never
    inputs, thresholds, sampling, resolution or the shared Graph implementation.
    """
    import resource
    resource.setrlimit(resource.RLIMIT_DATA,(80*1024**3,80*1024**3))
    torch.set_num_threads(4);signal.signal(signal.SIGALRM,timeout_handler)
    specs={s['id']:s for s in read(ROOT/'protocol.json')['scenes']};geo=geometry();cache={};out=[];audit=[]
    with logger('isolated_memory_recovery_'+arm) as log:
        for row in rows:
            if row['status']!='OOM':out.append(row);continue
            folder=ROOT/(row['folder']+'_mem80')
            meta={k:row[k] for k in ('arm','scene','parent','cohort','timestep','size','repeat')}
            meta.update(folder=str(folder.relative_to(ROOT)),resource24_folder=row['folder'],resource24_status='OOM',resource_data_limit_gib=80)
            # A failure before mask hashing must never alias another response.
            key=(row['scene'],row['size'],row.get('mask_hash') or ('unhashed',row['repeat']))
            if (folder/'receipt.json').exists():
                receipt=verify_receipt(folder);out.append(dict(meta,**receipt['outcome']))
                if receipt['outcome']['status']=='COMPLETE':cache[key]=folder
                continue
            if folder.exists():folder.rename(folder.with_name(folder.name+'.interrupted_'+str(time.time_ns())))
            start=time.perf_counter()
            try:
                if key in cache:
                    reference=cache[key];folder.mkdir(parents=True)
                    outcome=dict(status='COMPLETE',mask_hash=row.get('mask_hash'),reused_from=str(reference.relative_to(ROOT)),graph_ms=None,quality_reuse_only=True)
                    write(folder/'receipt.json',dict(status='COMPLETE',files={},outcome=outcome,reused_receipt_sha256=sha(reference/'receipt.json')))
                else:
                    signal.alarm(180)
                    s=specs[row['scene']];inputs=load_inputs(row['scene'],row['size'],geo,cache_sensors=True)
                    masks=masks_for(s,row['size'],row['repeat'],arm)
                    result=build(geo,inputs,masks,trace=row['repeat']==0)
                    signal.alarm(0);save_result(folder,result)
                    outcome=dict(status='COMPLETE',mask_hash=row.get('mask_hash'),graph_ms=None,
                        instrumented_graph_ms=result['graph_ms'],hashes=result['hashes'],timing_context='isolated_quality_recovery')
                    receipt=read(folder/'receipt.json');receipt['outcome']=outcome;write(folder/'receipt.json',receipt);cache[key]=folder
            except Exception as e:
                signal.alarm(0);folder.mkdir(parents=True,exist_ok=True)
                outcome=dict(status='OOM' if isinstance(e,MemoryError) else 'TIMEOUT' if isinstance(e,GraphTimeout) else 'GRAPH_FAILURE',
                             mask_hash=row.get('mask_hash'),error_type=type(e).__name__,error=str(e),graph_ms=None)
                write(folder/'receipt.json',dict(status=outcome['status'],files={},outcome=outcome))
                if isinstance(e,AssertionError):raise
            finally:signal.alarm(0);geo.reset()
            out.append(dict(meta,**outcome));audit.append(dict(meta,status=outcome['status'],recovery_elapsed_ms=(time.perf_counter()-start)*1000))
            write(ROOT/('memory_recovery_'+arm+'.json'),dict(rows=audit,data_limit_gib=80,concurrent_workers=1))
            print('MEMORY_RECOVERY',arm,row['scene'],row['size'],row['repeat'],outcome['status'],flush=True)
        log.log(dict(primary_OOM_cases=sum(r['status']=='OOM' for r in rows),final_OOM_cases=sum(r['status']=='OOM' for r in out)))
    return out


def oracle_jobs():
    assert (ROOT/'actual_graph_freeze.json').exists()
    if (ROOT/'oracle_jobs.json').exists():return
    manifest=read(ROOT/'capture_manifest.json')['entries'];rows=[]
    for e in manifest:
        if e['cohort']=='tracking' and e['timestep']!=0:continue
        gt=next(g for g in manifest if g['scene']==e['scene'] and g['camera']==e['camera'] and g['size']==1024)
        with np.load(ROOT/gt['gt_file'],allow_pickle=False) as z:seg=z['actor_segmentation']
        parsed=dict(json_valid=True)
        for obj in ('peg','box'):
            yy,xx=np.where(np.isin(seg,gt['object_ids'][obj]))
            normalized=[float(xx.min()*1000/1024),float(yy.min()*1000/1024),float((xx.max()+1)*1000/1024),float((yy.max()+1)*1000/1024)] if len(xx) else None
            from common import validate_box
            parsed[obj]=validate_box(normalized,e['size'],e['size'])
        rows.append(dict(id=job_id(e['id'],0),image=e['id'],repeat=0,parsed=parsed,GT_box=True))
    assert len(rows)==280
    write(ROOT/'oracle_jobs.json',dict(rows=rows,source='native1024 exact actor bbox; one fixed normalized box across sizes; t0 only for tracking',
                                     actual_freeze_sha256=sha(ROOT/'actual_graph_freeze.json')))


def resolve_folder(row):
    folder=ROOT/row['folder'];receipt=read(folder/'receipt.json')
    if receipt.get('outcome',{}).get('reused_from'):
        folder=ROOT/receipt['outcome']['reused_from']
        assert sha(folder/'receipt.json')==receipt['reused_receipt_sha256']
    return folder


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['quality','oracle_jobs','merge']);ap.add_argument('--arm',default='actual')
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);a=ap.parse_args()
    if a.stage=='oracle_jobs':oracle_jobs()
    elif a.stage=='merge':merge(a.arm,a.shards)
    else:quality(a.arm,a.shard,a.shards)
