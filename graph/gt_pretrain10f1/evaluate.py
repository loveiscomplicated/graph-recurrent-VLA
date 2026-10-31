"""Independent frozen held-out evaluation; no optimizer or policy exists here."""
import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch

from graph.plan_a_bc.common import read,write,sha,npz,tensor_identity,now
from graph.integration_baseline.geometry09b import geometry_features
from recurrent.features import build_features,quat_rotate
from utils.experiment_logger import ExperimentLogger
from .contract import split_fields,measured_robot,assemble_gt38,normalize_gt38
from .data import GraphPool
from .model import restore_checkpoint,RobotOnly


def csv_write(path,rows):
    rows=list(rows)
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def qnormalize(y):
    y=y.copy()
    for s in (slice(3,7),slice(10,14)):
        norm=np.linalg.norm(y[:,s],axis=-1,keepdims=True)
        y[:,s]=np.divide(y[:,s],norm,out=np.full_like(y[:,s],np.nan),where=norm>1e-8)
        y[:,s]=np.where(y[:,s.start:s.start+1]<0,-y[:,s],y[:,s])
    return y


def errors(pred,gt):
    out={}
    for name,s in [('peg_position',slice(0,3)),('hole_position',slice(7,10))]:
        out[name]=(np.linalg.norm(pred[:,s]-gt[:,s],axis=-1)*1000,'mm',.1)
    for name,s in [('peg_orientation',slice(3,7)),('hole_orientation',slice(10,14))]:
        dot=np.abs(np.sum(pred[:,s]*gt[:,s],axis=-1)).clip(0,1)
        out[name]=(2*np.arctan2(np.sqrt(np.maximum(1-dot**2,0)),dot)*180/np.pi,'deg',1.)
    for j,name in enumerate(('peg_half_x','peg_half_y','peg_half_z','hole_radius')):
        out[name]=(abs(pred[:,14+j]-gt[:,14+j])*1000,'mm',.02)
    def tip(y):
        offset=np.zeros_like(y[:,:3]);offset[:,0]=y[:,14]
        return y[:,:3]+quat_rotate(y[:,3:7],offset)-y[:,7:10]
    out['peg_tip_hole']=(np.linalg.norm(tip(pred)-tip(gt),axis=-1)*1000,'mm',.1)
    return out


def summary(predictions,gt,meta,presence,only_split=None):
    rows=[];splits=np.asarray([r['split'] for r in meta]);isbase=np.asarray([not r['pair'] for r in meta])
    for model,pred in predictions.items():
        for field,(values,unit,scale) in errors(pred,gt).items():
            oid=1 if field.startswith('hole') else 0
            field_visible=presence.all(1) if field=='peg_tip_hole' else presence[:,oid]
            for split in ((only_split,) if only_split else ('train','validation','final')):
                for visible in ('all','present','missing'):
                    selected=isbase&(splits==split)&np.isfinite(values)
                    if visible=='present':selected &= field_visible
                    if visible=='missing':selected &= ~field_visible
                    data=values[selected];episodes={}
                    for idx in np.flatnonzero(selected):episodes.setdefault(meta[idx]['episode_id'],[]).append(values[idx])
                    rows.append(dict(model=model,split=split,field=field,stratum=visible,n=int(len(data)),episodes=len(episodes),
                        unit=unit,mean=float(data.mean()) if len(data) else None,median=float(np.median(data)) if len(data) else None,
                        p95=float(np.percentile(data,95)) if len(data) else None,
                        episode_macro_mean=float(np.mean([np.mean(v) for v in episodes.values()])) if episodes else None,
                        normalized_mean=float(np.mean(data/(1000*scale) if unit=='mm' else data*np.pi/180)) if len(data) else None,
                        normalized_unit='physical scale' if unit=='mm' else 'rad',undefined_or_invalid=int((isbase&(splits==split)&~np.isfinite(values)).sum())))
    return rows


def delta_metrics(gt,pred):
    norm=float(np.linalg.norm(gt));pn=float(np.linalg.norm(pred));den=float(gt@gt)
    return dict(actual_dx_mm=float(gt[0]*1000),actual_dy_mm=float(gt[1]*1000),actual_dz_mm=float(gt[2]*1000),
        predicted_dx_mm=float(pred[0]*1000),predicted_dy_mm=float(pred[1]*1000),predicted_dz_mm=float(pred[2]*1000),
        actual_magnitude_mm=norm*1000,predicted_magnitude_mm=pn*1000,delta_vector_error_mm=float(np.linalg.norm(pred-gt)*1000),
        direction_cosine=float(pred@gt/(pn*norm)) if norm>1e-9 and pn>1e-9 else None,
        signed_gain=float(pred@gt/den) if den>1e-18 else None,magnitude_ratio=pn/norm if norm>1e-9 else None,
        zero_response_error_mm=norm*1000,improvement_vs_zero_mm=float((norm-np.linalg.norm(pred-gt))*1000),
        direction_defined=norm>1e-9 and pn>1e-9)


def raw_features(pool):
    # Same actual Graph nodes; measured TCP rotates EE displacements to world axes.
    x=[]
    ptr=pool.ptr.cpu().numpy();typ=pool.values['node_type'].cpu().numpy()
    points=pool.values['node_input'][:,:3].cpu().numpy();robot=pool.robot.cpu().numpy()
    tcp=measured_robot(robot,pool.schema)['tcp_pose']
    for i in range(len(pool.meta)):
        lo,hi=ptr[i:i+2];pieces=[]
        for oid in (3,4):
            xyz=points[lo:hi][typ[lo:hi]==oid]
            world=quat_rotate(np.broadcast_to(tcp[i,3:],(len(xyz),4)),xyz)
            pieces.append(geometry_features(world))
        x.append(np.r_[*pieces,pool.presence[i].cpu().numpy()])
    return np.asarray(x,np.float64)


def latency(model,pool,root):
    indices=pool.splits['final'][:32];costs={'A_state38_ms':[],'B_readout320_ms':[],'A_B_shared_extract_ms':[]}
    for idx in indices:
        g,r,_=pool.select([idx])
        for _ in range(2):model(g,r)
        torch.cuda.synchronize()
        tick=time.perf_counter()
        with torch.inference_mode():
            p,z,pres=model(g,r);pred=p.cpu().numpy();robot=r.cpu().numpy()
            _=assemble_gt38(split_fields(pred),measured_robot(robot,pool.schema))
        torch.cuda.synchronize();costs['A_state38_ms'].append((time.perf_counter()-tick)*1000)
        tick=time.perf_counter()
        with torch.inference_mode():
            z,pres=model.graph_encoder(g);_=torch.cat((z,pres,r),-1)
        torch.cuda.synchronize();costs['B_readout320_ms'].append((time.perf_counter()-tick)*1000)
        tick=time.perf_counter();out=model.extract(g,r,pool.schema);torch.cuda.synchronize()
        costs['A_B_shared_extract_ms'].append((time.perf_counter()-tick)*1000)
    result=dict(device='RTX3090 CUDA',batch_size=1,frames=len(indices),
        scope='cached graph already on GPU; A includes CPU38 assembly, B remains GPU; shared API includes weight hash',
        excludes='RGB-D capture, vision, geometry, graph packing, future GRU',
        metrics={k:dict(mean=float(np.mean(v)),median=float(np.median(v)),p95=float(np.percentile(v,95))) for k,v in costs.items()})
    write(root/'frozen_extraction_cost.json',result,immutable=True)
    return result


def run(root):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    manifest=read(root/'checkpoint_manifest.json');checkpoint=root/'gnn_pretrained.pt'
    assert sha(checkpoint)==manifest['selected_checkpoint']['sha256']
    model,ck=restore_checkpoint(checkpoint,'cuda',True);before=tensor_identity(model.state_dict())
    pool=GraphPool(root,'cuda',allow_final=True);gt=pool.targets.cpu().numpy();presence=pool.presence.cpu().numpy().astype(bool)
    robotck=torch.load(root/'robot_baseline_training/best.pt',map_location='cpu',weights_only=False)
    stats=robotck['normalization'];robotmodel=RobotOnly(stats['robot_mean'],stats['robot_std'],stats['target_mean']).to('cuda')
    robotmodel.load_state_dict(robotck['model']);robotmodel.eval().requires_grad_(False)
    with ExperimentLogger(team='graph',model='Task10F1_frozen_independent_evaluation',seed=42,
            config=dict(parent_run_id=read(root/'run_identity.json')['run_id'],checkpoint_sha256=sha(checkpoint),
                        source_train_run_id=ck['run_id']),root=root/'experiments',job_type='evaluate') as log:
        preds={name:[] for name in ('GNN','RobotOnly')};readouts=[]
        with torch.inference_mode():
            for start in range(0,len(pool.meta),64):
                ids=np.arange(start,min(start+64,len(pool.meta)));g,r,y=pool.select(ids)
                p,z,pr=model(g,r)
                # Preserve a fixed cuBLAS matrix shape for exact same-input
                # Robot-only responses, including the final partial batch.
                robot_batch=torch.cat((r,r[-1:].expand(64-len(ids),-1)),0) if len(ids)<64 else r
                rp,_,_=robotmodel(None,robot_batch);rp=rp[:len(ids)]
                torch.testing.assert_close(pr,pool.presence[ids].float(),atol=0,rtol=0)
                preds['GNN'].append(p.cpu().numpy());preds['RobotOnly'].append(rp.cpu().numpy());readouts.append(z.cpu().numpy())
        preds={k:np.concatenate(v) for k,v in preds.items()};readout=np.concatenate(readouts)
        x=raw_features(pool);train=np.asarray(pool.splits['train']);xm=x[train].mean(0);xs=np.maximum(x[train].std(0),1e-6)
        xx=(x-xm)/xs;ym=gt[train].mean(0);yy=gt[train]-ym
        weight=np.linalg.solve(xx[train].T@xx[train]+np.eye(xx.shape[1]),xx[train].T@yy)
        preds['RawGeometryRidge']=qnormalize(xx@weight+ym)
        preds['TrainMean']=qnormalize(np.broadcast_to(ym,gt.shape).copy())
        # Direct centroid retains the partial-surface bias; do not call box center hole center.
        direct=np.full_like(gt,np.nan);direct[:,:3]=x[:,:3];direct[:,7:10]=x[:,15:18]
        preds['RawCentroidHoleProxy']=direct
        npz(root/'raw_geometry_decoder.npz',dict(feature_mean=xm,feature_std=xs,weight=weight,target_mean=ym,alpha=np.float64(1)))
        rows=summary(preds,gt,pool.meta,presence)
        csv_write(root/'absolute_state_metrics.csv',rows)
        csv_write(root/'robot_only_baseline.csv',[r for r in rows if r['model']=='RobotOnly'])
        csv_write(root/'raw_geometry_baseline.csv',[r for r in rows if r['model'].startswith('Raw')])
        pairrows=[];responses=[]
        for pair in pool.pairs:
            i=pair['index'];b=pair['base'];meta=pool.meta[i];oid=0 if pair['object']=='peg' else 1
            s=slice(0,3) if oid==0 else slice(7,10)
            actual=gt[i,s]-gt[b,s]
            # Two world-position subtractions in float32; 0.1 micrometre bound.
            np.testing.assert_allclose(actual,np.asarray(pair['offset_mm'])*.001,atol=1e-7,rtol=0)
            np.testing.assert_array_equal(pool.robot[i].cpu().numpy(),pool.robot[b].cpu().numpy())
            for name,pred in preds.items():
                row=dict(model=name,split=meta['split'],parent=meta['parent'],episode_id=meta['episode_id'],
                    t=meta['t'],pair_id=pair['pair_id'],object=pair['object'],physical_valid=pair['valid'],
                    intervention_group='same_robot_actual_object_motion',
                    baseline_present=bool(presence[b,oid]),moved_present=bool(presence[i,oid]),
                    **delta_metrics(actual,pred[i,s]-pred[b,s]))
                pairrows.append(row)
                if name=='RobotOnly':np.testing.assert_array_equal(pred[i],pred[b])
            responses.append(dict(split=meta['split'],parent=meta['parent'],t=meta['t'],pair_id=pair['pair_id'],
                object=pair['object'],physical_valid=pair['valid'],
                readout320_delta_norm=float(np.linalg.norm(readout[i]-readout[b])),
                robot_slots_delta_norm=float(np.linalg.norm(readout[i,:192]-readout[b,:192])),
                peg_slot_delta_norm=float(np.linalg.norm(readout[i,192:256]-readout[b,192:256])),
                box_slot_delta_norm=float(np.linalg.norm(readout[i,256:]-readout[b,256:]))))
        csv_write(root/'paired_displacement_metrics.csv',pairrows);csv_write(root/'geometry_response_comparison.csv',pairrows)
        csv_write(root/'readout_slot_responses.csv',responses)
        # Oracle contract is evaluated separately; predictions never receive these raw labels.
        oracle_max=0.;normalizer_max=0.;count=0;pred38=[];gt38=[];robot=pool.robot.cpu().numpy()
        for e in pool.manifest['episodes']:
            gtpath=Path(e['gt_path']) if e.get('gt_path') else root/'raw'/str(e['episode_id'])/'gt_supervision.npz'
            with np.load(gtpath) as label:
                for t in range(e['graphs']):
                    idx=pool.index[(e['episode_id'],t)]
                    measured=measured_robot(robot[idx],pool.schema)
                    oracle=assemble_gt38(split_fields(gt[idx]),measured)
                    original=build_features({k[4:]:label[k][t] for k in label.files if k.startswith('raw_')},np.float32(1),np.float32(0))
                    np.testing.assert_allclose(oracle,original,atol=1e-7,rtol=0)
                    oracle_max=max(oracle_max,float(abs(oracle-original).max()))
                    for seed in (0,1,2):
                        norm=read(root/f'gt_gru_normalizer_seed{seed}.json')
                        normalizer_max=max(normalizer_max,float(abs(normalize_gt38(oracle,norm)-normalize_gt38(original,norm)).max()))
                    pred38.append(assemble_gt38(split_fields(preds['GNN'][idx]),measured));gt38.append(original);count+=1
        schema=read(root/'normalizer.json')['schema']
        g,r,_=pool.select(pool.splits['final'][:8]);one=model.extract(g,r,schema);two=model.extract(g,r,schema)
        torch.testing.assert_close(one['readout320'],two['readout320'],atol=0,rtol=0)
        np.testing.assert_array_equal(one['state38'],two['state38'])
        npz(root/'feature_extraction_smoke.npz',dict(readout320=one['readout320'].cpu().numpy(),presence2=one['presence2'].cpu().numpy(),
            robot76=one['robot76'].cpu().numpy(),state38=one['state38'],checkpoint_sha256=np.array(sha(checkpoint)),weight_sha256=np.array(before)))
        after=tensor_identity(model.state_dict());assert before==after
        frozen=dict(status='PASS',weight_hash_before=before,weight_hash_after=after,frozen_parameters=True,
            checkpoint_sha256=sha(checkpoint),A_B_checkpoint_identical=True,forward_deterministic=True,
            shape=dict(readout320=[8,320],presence2=[8,2],robot76=[8,76],state38=[8,38]),
            inference_input_whitelist=['current_graph','measured_robot76'],label_sidecar_in_forward=False)
        write(root/'frozen_interface_verification.json',frozen,immutable=True)
        assembly=dict(oracle_status='PASS',frames=count,raw_gt38_max_abs=oracle_max,normalizer_max_abs=normalizer_max,
            oracle_tolerance=1e-7,predicted_provenance='PASS: object18 only model output; robot0:15 measured; head22:25 derived; freshness clean',
            predicted_raw_state_values_generated=True,full_observability_claim=False,
            unsupported_cases=read(root/'target_observability_audit.json')['unsupported'],
            strict_A_policy_readiness='BLOCKED: field errors and symmetry/missing observability require downstream decision; no policy trained')
        write(root/'state38_assembly_verification.json',assembly,immutable=True)
        npz(root/'predictions.npz',dict(target18=gt,**{'pred_'+k:v for k,v in preds.items()},readout320=readout,
            presence2=presence,episode_id=np.asarray([r['episode_id'] for r in pool.meta]),
            timestep=np.asarray([r['t'] for r in pool.meta]),split=np.asarray([r['split'] for r in pool.meta]),
            parent=np.asarray([r['parent'] for r in pool.meta]),
            distribution=np.asarray([r['distribution'] for r in pool.meta]),
            source=np.asarray([r['source'] for r in pool.meta]),
            physical_valid=np.asarray([r['physical_valid'] for r in pool.meta]),
            state38_frame_index=np.asarray([i for i,r in enumerate(pool.meta) if not r['pair']],np.int64),
            is_pair=np.asarray([r['pair'] for r in pool.meta]),predicted_state38=np.stack(pred38),original_gt38=np.stack(gt38)))
        cost=latency(model,pool,root)
        assert before==tensor_identity(model.state_dict())
        for p in root.glob('*.csv'):log.link_output(p,kind='metrics')
        for p in (root/'frozen_interface_verification.json',root/'state38_assembly_verification.json'):log.link_output(p,kind='metrics')
        write(root/'evaluation_complete.json',dict(run_id=log.run_id,status='PASS',at=now(),
            final_parent_count=12,final_frame_count=len(pool.splits['final']),
            final_valid_pairs=sum(p['split']=='final' and p['valid'] for p in pool.pairs),
            frozen=frozen,assembly=assembly,latency=cost),immutable=True)
        print('EVALUATION COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
