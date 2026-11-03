"""Task-local C++ loops, built with an existing compiler; no package installation."""
import ctypes
from copy import deepcopy
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import numpy as np


@lru_cache(maxsize=1)
def library():
    source=Path(__file__).with_name('native.cpp')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    folder=Path(os.environ.get('TASK10I_NATIVE_CACHE','/tmp/task10i_graph_native'))
    folder.mkdir(parents=True,exist_ok=True)
    output=folder/('native_'+digest+'.so')
    if not output.exists():
        temp=output.with_suffix('.'+str(os.getpid())+'.so')
        subprocess.run(['c++','-std=c++17','-O3','-shared','-fPIC','-ffp-contract=off',
                        '-fno-fast-math',str(source),'-o',str(temp)],check=True,capture_output=True)
        temp.replace(output)
    lib=ctypes.CDLL(str(output))
    ptr=ctypes.c_void_p;i=ctypes.c_int64
    lib.dedup.argtypes=[ptr,ptr,i,ctypes.c_double,ptr,ptr,i,ptr,i];lib.dedup.restype=i
    lib.fps.argtypes=[ptr,ptr,i,i,ptr,i,ptr];lib.fps.restype=None
    lib.representatives.argtypes=[ptr,ptr,ptr,i,ptr];lib.representatives.restype=None
    return lib


def pointer(a):return ctypes.c_void_p(a.ctypes.data)


def deduplicate(records,distance=.003,*,fallback,owned=False):
    from graph.plan_a_geometry.core import key
    from graph.persistent_geometry.coordinates import record_xyz, validate_record_frames
    validate_record_frames(records)
    if not records:return fallback(records,distance)
    ordered=sorted(records,key=lambda r:(r['object_id'],not r.get('structural',False),key(r)))
    xyz=np.ascontiguousarray([record_xyz(r) for r in ordered],dtype=np.float64)
    if (not np.isfinite(distance) or distance<=0 or not np.isfinite(xyz).all()
            or np.max(np.abs(xyz))/distance>=2.**62):return fallback(records,distance)
    objects=np.ascontiguousarray([r['object_id'] for r in ordered],dtype=np.int64)
    labels=np.empty(len(ordered),np.int64)
    unknown=np.empty((max(64,len(xyz)*8),2),np.int64);overrides=np.empty((0,3),np.int64)
    for _ in range(8):
        n=library().dedup(pointer(xyz),pointer(objects),len(xyz),distance,pointer(labels),
            pointer(overrides),len(overrides),pointer(unknown),len(unknown))
        if n==0:break
        if n<0:return fallback(records,distance)
        extra=np.array([(i,j,int(np.linalg.norm(xyz[j]-xyz[i])<distance)) for i,j in unknown[:n]],np.int64)
        overrides=np.ascontiguousarray(np.concatenate([overrides,extra]))
    else:return fallback(records,distance)
    out=[];lookup={}
    for i,(r,j) in enumerate(zip(ordered,labels)):
        if i==j:
            item=r if owned else deepcopy(r)
            item.setdefault('structural',False);lookup[i]=item;out.append(item)
        else:
            item=lookup[int(j)];item['source_observation_ids'].extend(r['source_observation_ids'])
            item['structural']|=r.get('structural',False)
    return out,dict(input_points=len(records),unique_points=len(out),duplicates=len(records)-len(out),
        duplicate_fraction=(len(records)-len(out))/len(records),
        representative='stable real measurement, no XYZ/F5 averaging')


def fps_ordered(xyz,order,count,initial=()):
    xyz=np.ascontiguousarray(xyz,dtype=np.float64)
    order=np.ascontiguousarray(order,dtype=np.int64)
    initial=np.ascontiguousarray(initial,dtype=np.int64)
    chosen=np.empty(count,np.int64)
    library().fps(pointer(xyz),pointer(order),len(xyz),count,pointer(initial),len(initial),pointer(chosen))
    return chosen


def farthest_rows(xyz,ids,count):
    xyz=np.asarray(xyz,float)
    if not isinstance(count,(int,np.integer)) or count<0:raise ValueError('nonnegative point count required')
    order=np.argsort(np.asarray(ids).astype(str),kind='stable')
    if not count:return np.empty(0,np.int64)
    if len(xyz)<=count:return order
    return fps_ordered(xyz,order,count)


def representatives(flat,xyz,a):
    from visual.runtime.test import run_voxel_structural_sampling as voxel
    flat,xyz=voxel.sources(flat,xyz);key=voxel.voxel_keys(xyz,a)
    cells,membership,capacity=np.unique(key,axis=0,return_inverse=True,return_counts=True)
    order=np.ascontiguousarray(np.argsort(membership,kind='stable'),dtype=np.int64)
    starts=np.ascontiguousarray(np.r_[0,capacity.cumsum()],dtype=np.int64)
    points=np.ascontiguousarray(xyz,dtype=np.float64);reps=np.empty(len(cells),np.int64)
    library().representatives(pointer(points),pointer(order),pointer(starts),len(cells),pointer(reps))
    same=reps[membership];distance=np.linalg.norm(points-points[same],axis=1)
    if len(flat) and np.max(distance)>np.sqrt(3)*a+64*np.finfo(float).eps*max(1.,np.max(np.abs(xyz))):
        raise AssertionError('same-cell bound failed')
    return dict(keys=key,cells=cells,membership=membership,capacity=capacity,cell_rep_rows=reps,
        representative_rows=np.sort(reps),same_representative_source=flat[same],same_cell_distance_m=distance)
