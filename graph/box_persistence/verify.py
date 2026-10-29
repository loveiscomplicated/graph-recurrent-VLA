"""Read back every output; independently audit paired/source/graph contracts."""
import argparse,csv,json,re,time
from pathlib import Path
from collections import defaultdict
import numpy as np,cv2
from PIL import Image
from graph.integration_baseline.io import write_json,sha256
from .preflight import protection


def readback(root):
    counts=defaultdict(int);errors=[]
    for p in root.rglob('*'):
        if not p.is_file():continue
        try:
            if p.suffix=='.json':json.loads(p.read_text());counts['JSON']+=1
            elif p.suffix=='.csv':
                with p.open(newline='') as f:
                    reader=csv.DictReader(f);rows=list(reader)
                    if any(None in r for r in rows):raise ValueError('CSV schema mismatch')
                counts['CSV']+=1;counts['CSV_rows']+=len(rows)
            elif p.suffix=='.npz':
                with np.load(p,allow_pickle=False) as a:
                    for k in a.files:_=a[k].copy()
                    if p.stem.endswith('_graph'):
                        for k in a.files:
                            if not np.isfinite(a[k]).all():raise ValueError('nonfinite graph')
                        if a['node_input'].shape[1]!=21:raise ValueError('feature schema')
                counts['NPZ']+=1
            elif p.suffix=='.png':
                with Image.open(p) as im:im.load()
                counts['PNG']+=1
            elif p.suffix=='.mp4':
                cap=cv2.VideoCapture(str(p));n=0
                while True:
                    ok,im=cap.read()
                    if not ok:break
                    n+=1
                cap.release()
                if not n:raise ValueError('empty/invalid MP4')
                counts['MP4']+=1;counts['MP4_decoded_frames']+=n
        except Exception as e:errors.append(dict(path=str(p.relative_to(root)),error=type(e).__name__,detail=str(e)[:200]))
    return dict(counts),errors


def paired(geometry):
    groups=defaultdict(list);future=0;wrist=0;frames=0;graphs=0;matched=0;baseline=0
    for path in geometry.glob('geometry/seed*/paired_identity.json'):
        rr=json.loads(path.read_text())
        for r in rr:
            groups[(r['seed'],r['timestep'],r['budget'])].append(r)
            if not r['temporal_extras_zero']:raise ValueError('status feature changed')
            baseline+=r['baseline_exact']
    for key,rr in groups.items():
        if len({r['peg_hash'] for r in rr})!=1 or len({r['robot76_hash'] for r in rr})!=1:raise ValueError('paired Peg/Robot mismatch')
    for folder in geometry.glob('geometry/seed*/f*'):
        if not folder.is_dir():continue
        t=int(folder.name[1:]);frames+=1;counts=[]
        for name in ('B0','B1','B2','O1','O2','O3'):
            with np.load(folder/(name+'_native_sources.npz'),allow_pickle=False) as a:
                future+=int((a['timestep']>t).sum())
                source_rows=a['node_row'].copy();source_cameras=a['camera'].copy()
            with np.load(folder/(name+'_native.npz'),allow_pickle=False) as a:
                box=a['object_id']==1
                wrist+=int(((source_cameras!='base_camera')&box[source_rows]).sum())
                if np.any(a['last_observed_camera'][box]!='base_camera'):raise ValueError('Wrist Box')
                if np.any(a['last_seen_timestep']>t):raise ValueError('future map')
                if name=='B0' and not np.all(a['currently_observed']):raise ValueError('B0 has memory')
            for mode in ('cap128','matched_N'):
                with np.load(folder/(name+'_'+mode+'_graph.npz'),allow_pickle=False) as a:
                    graphs+=1
                    for k in a.files:
                        if not np.isfinite(a[k]).all():raise ValueError('nonfinite GNN tensor')
                    oid=a['node_type'];inp=a['node_input']
                    if inp.shape!=(len(oid),21):raise ValueError('shape')
                    if max(int((oid==3).sum()),int((oid==4).sum()))>128:raise ValueError('cap')
                    if np.any(inp[:,14:]!=0):raise ValueError('temporal extras')
                    if a['edge_index'].size and (a['edge_index'].min()<0 or a['edge_index'].max()>=len(oid)):raise ValueError('edge indices')
                    if mode=='matched_N' and name in ('B0','B1','B2'):counts.append(int((oid==4).sum()))
        if len(set(counts))!=1:raise ValueError('not matched-N')
        matched+=1
    if future:raise ValueError('future source')
    if wrist:raise ValueError('historical Wrist Box source')
    static_frames=0
    for path in geometry.glob('geometry/seed*/B1_tracking.json'):
        for row in json.loads(path.read_text()):
            if not np.array_equal(np.array(row['pose']),np.eye(4)) or row['pose_update_applied']:raise ValueError('B1 transported pose')
            static_frames+=1
    return dict(status='PASS',evaluated_frames=frames,graphs=graphs,paired_groups=len(groups),original_PlanA_cap_graph_exact_checks=baseline,
        peg_and_robot_hash_equal=True,matched_N_frames=matched,B1_static_pose_checks=static_frames,future_source_count=future,box_wrist_sources=wrist,GT_leakage_scope='sensor-only API/whitelist and practical-before-GT ordering, source tests; oracle separately labelled')


def links(root):
    errors=[];n=0
    for p in [root/'plan_b_box_persistence.html',root/'09e_plan_b_box_persistence.md']:
        if not p.exists():continue
        text=p.read_text()
        targets=re.findall(r'(?:href|src)=["\x27]([^"\x27]+)',text) if p.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',text)
        for target in targets:
            if target.startswith(('http:','https:','#','data:')):continue
            n+=1
            if not (p.parent/target.split('#')[0]).exists():errors.append(str(p.name)+': '+target)
    return dict(checked=n,errors=errors)


def hidden_maps(root):
    state=json.loads((root/'pipeline_state.json').read_text());checked=0
    for key in ('motion_id','motion_warm_id'):
        source=root/'sources'/state[key]
        controls=json.loads((source/'hidden_controls.json').read_text())
        if not controls or not all(r['hidden_sensor_indistinguishable'] for r in controls):raise ValueError('hidden RGB/depth/mask negative control')
        for t in sorted({int(r['timestep']) for r in controls}):
            for n in ('B1','B2'):
                parent=source/f't{t:04d}'
                with np.load(parent/'box_full_baseline'/n/'points.npz',allow_pickle=False) as a:baseline={k:a[k].copy() for k in ('world_xyz','status','confidence','position_uncertainty','estimated_only')}
                for condition in ('box_full_x-10','box_full_x+10','box_full_y-10','box_full_y+10'):
                    with np.load(parent/condition/n/'points.npz',allow_pickle=False) as a:
                        for k,v in baseline.items():np.testing.assert_array_equal(a[k],v)
                    checked+=1
    return dict(status='PASS',same_input_hidden_map_pairs=checked,actual_hidden_motion_not_inferred=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--remote-protection',action='store_true');a=p.parse_args();start=time.time()
    if not (a.root/'verification.json').exists():write_json(a.root/'verification.json',dict(status='NOT_EXECUTED',scope='verification in progress'))
    counts,errors=readback(a.root);contracts=paired(a.geometry);hidden=hidden_maps(a.root)
    prior=json.loads((a.root/'protection_before.json').read_text())
    audit={'status':'NOT_EXECUTED','scope':'remote protection checked separately'}
    if a.remote_protection:
        after=protection();missing=[k for k in prior if k not in after];changed=[k for k,v in prior.items() if k in after and v!=after[k]]
        audit=dict(status='PASS' if not missing and not changed else 'FAIL',protected_files=len(prior),missing=missing,changed=changed)
        identity=json.loads(Path('/workspace/task09d_plan_a/v2/model_identity.json').read_text())
        weights=[path for path,h in identity['files'].items() if not Path(path).is_file() or sha256(Path(path))!=h]
        audit['pretrained_identity_mismatches']=weights
        if weights:audit['status']='FAIL'
        write_json(a.root/'audits/protection_audit.json',audit)
    else:
        audit=json.loads((a.root/'audits/protection_audit.json').read_text())
    link_result=links(a.root)
    result=dict(status='PASS' if not errors and not link_result['errors'] and audit['status']=='PASS' else 'FAIL',
        readback=counts,errors=errors,paired_contracts=contracts,hidden_motion_contracts=hidden,relative_links=link_result,protection=audit,
        elapsed_seconds=time.time()-start,scope='executed output/source/paired/protection contracts; research geometry suitability is a separate Gate')
    write_json(a.root/'verification.json',result);print(json.dumps(result,indent=2),flush=True)
    if result['status']=='PASS':
        phase=json.loads((a.root/'phase_status.json').read_text());phase['Stage6']=dict(status='PASS',scope='report, all stored JSON/CSV/NPZ and decoded media/source/protection verified; see verification.json')
        write_json(a.root/'phase_status.json',phase)
    if result['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
