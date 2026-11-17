"""Audit time-limit replays and quantify 700806 command/response and contacts."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT=Path(__file__).resolve().parent
RUNS=ROOT/'source_runs'


def read(path):
    return json.loads(path.read_text())


def rotate(q,v):
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    cross=2*np.cross(q[:,1:],v)
    return v+q[:,:1]*cross+np.cross(q[:,1:],cross)


def contact_metrics(payload):
    trace=payload['episode']['trace']
    diagnostics=payload['diagnostics']
    assert len(trace)==len(diagnostics)
    gt=np.array([r['object21_gt'] for r in trace])
    pos=np.array([r['actual_relative_mm'] for r in trace])
    dep=pos[:,0]+gt[:,17]*1000
    pred=np.array([r['object21'] for r in trace])
    f=np.asarray([r['net_force_n'] for r in diagnostics])
    tcp=np.asarray([r['tcp_xyz_m'] for r in diagnostics])
    q=gt[:,13:17].copy()
    q[:,1:]*=-1
    a=np.clip(np.array([r['action'][:3] for r in trace]),-1,1)*100
    cmd=rotate(q,a)
    contact=rotate(q,f)
    local=np.broadcast_to(np.array([1.,0.,0.]),(len(gt),3))
    peg_axis=np.rad2deg(np.arccos(np.clip(rotate(q,rotate(gt[:,3:7],local))[:,0],-1,1)))
    gt_lat=np.linalg.norm(pos[:,1:],axis=1)
    # TCP movement is in robot-root/world axes for this fixed Panda base.
    tcp_axial=rotate(q[:-1],np.diff(tcp,axis=0)*1000)[:,0]
    result={}
    for lo,hi in [(105,125),(140,160),(230,250),(250,500),(len(trace)-20,len(trace))]:
        sl=slice(lo,hi)
        actual=dep[sl]
        if not len(actual):
            continue
        aligned=tcp_axial[lo:min(hi,len(tcp_axial))]
        stage=dict(frames=len(actual),depth_min=float(actual.min()),depth_max=float(actual.max()),
                   forward_command_p50_mm=float(np.median(cmd[sl,0])),
                   forward_command_positive_fraction=float((cmd[sl,0]>0).mean()),
                   lateral_command_p50_mm=float(np.median(np.linalg.norm(cmd[sl,1:],axis=1))),
                   peg_actual_lateral_p50_mm=float(np.median(gt_lat[sl])),
                   peg_axis_tilt_p50_deg=float(np.median(peg_axis[sl])),
                   peg_box_force_p50_n=float(np.median(np.linalg.norm(f[sl],axis=1))),
                   peg_box_force_positive_fraction=float((np.linalg.norm(f[sl],axis=1)>.1).mean()),
                   peg_box_force_axial_p50_n=float(np.median(contact[sl,0])),
                   tcp_forward_displacement_p50_mm=float(np.median(aligned)) if len(aligned) else None,
                   gnn_lateral_error_p50_mm=float(np.median([r['raw_lateral_mm'] for r in trace[sl]])))
        assert np.isfinite(pred).all()
        result[f'{lo}:{hi}']=stage
    return result


def main():
    rows=[]
    for p in sorted(RUNS.glob('worker_*/summary.json')):
        s=read(p)
        meta=read(p.parent/'experiments'/s['run_id']/'metadata.json')
        assert meta['status']=='finished' and not meta['git_dirty']
        assert s['config']['script_sha256']==hashlib.sha256((ROOT/'replay.py').read_bytes()).hexdigest()
        assert s['config']['horizon']==500 and s['config']['prior_horizon']==250
        rows.extend(s['results'])
    assert len(rows)==len({(r['condition'],r['seed']) for r in rows})==29
    phases={}
    for key in ('fixed','moving'):
        group=[r for r in rows if r['condition']==key]
        phases[key]=Counter()
        for row in group:
            assert not row['original']['success'] and row['original']['steps']==250
            n=row['extended']
            if n['success']:
                phases[key]['up_to_250' if n['steps']<=250 else 'after_250']+=1
            else:
                phases[key]['no_success']+=1
    force=contact_metrics(read(RUNS/'worker_-1/trace_700806.json'))
    data=dict(totals=phases,results=rows,contact_700806=force)
    (RUNS/'verified.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(phases,indent=2))
    for r in rows:
        if r['extended']['success'] or r['seed']==700806:
            print(r)
    print(json.dumps(force,indent=2))


if __name__=='__main__':
    main()
