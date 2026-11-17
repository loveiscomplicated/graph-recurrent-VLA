"""Select using per-slot validation recall, then test the pre-correction frontend."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from data_collection.episode_dataset import file_sha256
from frontend import TRACKER
from match_recall import build
from train_tracker import CornerTransformer, evaluate, prepare
from utils.experiment_logger import ExperimentLogger, write

VAL_END=3100192
CONFIGS={'baseline':None,'normal_3d':(0.,1.,None),'rank_hint1':(.001,1.,None),
         'rank_hint3':(.003,1.,None),'rank_hint6':(.006,1.,None),
         'guard6':(.003,1.,.006),'guard4':(.003,1.,.004),'guard2':(.003,1.,.002),
         'guard1':(.003,1.,.001)}
RAGGED=('raw_xyz','raw_family')
AREAS={'peg':slice(8),'box':slice(8,16),'hole':slice(16,20)}


def read(path):
    return json.loads(path.read_text())


def load_data(root):
    parts,sources,episodes=[],[],[]
    for folder in sorted(p for p in root.glob('shard_*') if p.is_dir()):
        r=read(folder/'summary.json')
        md=read(folder/'experiments'/r['run_id']/'metadata.json')
        assert md['status']=='finished' and not md['git_dirty']
        path=folder/'raw.npz'
        assert file_sha256(path)==r['sha256']
        with np.load(path,allow_pickle=False) as f:
            parts.append(dict(f))
        episodes.extend(r['episodes'])
        sources.append(dict(run_id=r['run_id'],sha256=r['sha256']))
    assert [e['seed'] for e in episodes]==list(range(3100184,3100208))
    data={n:np.concatenate([p[n] for p in parts]) for n in parts[0]}
    assert len(data['raw_xyz'])==data['raw_count'].sum()
    return data,episodes,sources


def frame_observations(data, which):
    offsets=np.r_[0,np.cumsum(data['raw_count'])]
    xyz,clas,tcp=(data[field] for field in ('raw_xyz','raw_family','tcp'))
    result=np.zeros_like(data['xyz'])
    seen=np.zeros_like(data['oracle_seen'])
    for row in range(len(result)):
        if data['step'][row]==0:
            m=build(which)
        a,b=offsets[row:row+2]
        candidates=xyz[a:b]
        indexed=m.step(candidates,clas[a:b],tcp[row])
        hit=indexed>=0
        seen[row]=hit
        result[row,hit]=candidates[indexed[hit]]
    return result,seen


def matches(coords, seen, gt):
    err=np.linalg.norm(coords[:,:,None]-gt[:,None],axis=-1)
    good=(err.min(axis=-1)<=.005)&seen
    exact=good&(err.argmin(axis=-1)==np.arange(20))
    return dict(exact=exact,wrong=good&~exact,spurious=seen&~good)


def scores(data, prediction, base, selected, family):
    s=AREAS[family]
    point={k:v[selected,s] for k,v in prediction.items()}
    reference=data['oracle_seen'][selected,s]
    was=base['exact'][selected,s]
    return dict(matchable=int(reference.sum()),correct=int(point['exact'].sum()),
                wrong=int(point['wrong'].sum()),spurious=int(point['spurious'].sum()),
                recovered=int((point['exact']&reference).sum()),
                newly_recovered=int((point['exact']&~was&reference).sum()),
                lost=int((~point['exact']&was&reference).sum()))


def summarize(data, pred, base, mask):
    depth=data['depth_mm']
    pieces=dict(all=mask,pre=mask&(depth<=0),entry=mask&(depth>0)&(depth<=20),deep=mask&(depth>20))
    return {k:dict(frames=int(m.sum()),**{f:scores(data,pred,base,m,f) for f in AREAS})
            for k,m in pieces.items()}


def tracker_results(data, input_type, model, prior, device):
    coords,observed=input_type
    keep=(data['seed']>=VAL_END)
    subset={k:v[keep] for k,v in data.items() if k not in (*RAGGED,'raw_count')}
    observed=observed[keep]
    test_input=dict(subset,detected_xyz=coords[keep],observed=observed)
    seq=prepare(test_input,prior,np.unique(subset['seed']),device)
    output=evaluate(model,seq,6)
    np.testing.assert_array_equal(output[observed],coords[keep][observed])
    error=np.linalg.norm(output-subset['xyz'],axis=-1)*1000
    depth=subset['depth_mm']
    bands=dict(pre=depth<=0,entry=(depth>0)&(depth<=20),deep=depth>20)
    return {p:{family:dict(points=len(val),p90=float(np.percentile(val,90)) if len(val) else None)
               for family,s in AREAS.items()
               if (val:=error[sel,s][~observed[sel,s]]) is not None}
            for p,sel in bands.items()},output


def main():
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    torch.set_num_threads(2)
    data,episodes,sources=load_data(args.root)
    args.output.mkdir(parents=True,exist_ok=False)
    config=dict(capture_sources=sources,candidates=CONFIGS,validation_last_excluded_seed=VAL_END,
                min_priority='no loss among baseline correct oracle-matchable observations',
                tracker_sha256=file_sha256(TRACKER),script_sha256=file_sha256(__file__),
                matcher_sha256=file_sha256(Path(__file__).with_name('match_recall.py')))
    with ExperimentLogger(team='graph',model='gtfree_preserve_corner_observations',job_type='evaluate',
                          seed=3100184,config=config,root=args.output/'experiments') as log:
        inputs={name:frame_observations(data,params) for name,params in CONFIGS.items()}
        result={name:matches(*values,data['xyz']) for name,values in inputs.items()}
        base=result['baseline']
        val,test=data['seed']<VAL_END,data['seed']>=VAL_END
        report_val={k:summarize(data,v,base,val) for k,v in result.items()}
        eligible=[key for key,p in report_val.items() if p['all']['hole']['lost']==0]

        def rank(key):
            x=report_val[key]
            # Prioritize actual usable observations before rejecting spurious ones.
            return (x['all']['hole']['recovered'],-x['all']['hole']['wrong'],
                    -x['all']['hole']['spurious'],-x['deep']['hole']['spurious'])

        selected=max(eligible,key=rank)
        report_test={k:summarize(data,result[k],base,test) for k in dict.fromkeys(('baseline',selected))}
        weights=torch.load(TRACKER,weights_only=True,map_location='cpu')
        device=torch.device('cuda:0')
        model=CornerTransformer(spatial=True).to(device).eval()
        model.load_state_dict(weights['model'],strict=True)
        versions=dict(baseline=inputs['baseline'],selected=inputs[selected],
                      oracle=(data['oracle_xyz'],data['oracle_seen']))
        stats,outputs={},{}
        for key,batch in versions.items():
            stats[key],outputs[key]=tracker_results(data,batch,model,weights['fallback'].numpy(),device)
        file=args.output/'heldout_points.npz'
        np.savez_compressed(file,seed=data['seed'][test],step=data['step'][test],depth_mm=data['depth_mm'][test],
                            truth=data['xyz'][test],**outputs,
                            **{f'{key}_obs':value[1][test] for key,value in versions.items()},
                            **{f'{key}_detected':value[0][test] for key,value in versions.items()})
        summary=args.output/'summary.json'
        write(summary,dict(run_id=log.run_id,config=config,episodes=episodes,selected=selected,validation=report_val,
                           test=report_test,tracker=stats,heldout_sha256=file_sha256(file)))
        log.link_output(file)
        log.link_output(summary)
        log.log({'chosen':selected,'test':report_test})
    print('Selected',selected,flush=True)
    print('Completed corner-recall comparison',flush=True)


if __name__=='__main__':
    main()
