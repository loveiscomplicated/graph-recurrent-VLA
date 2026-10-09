"""Tables from actual results; missing measurements remain N/A."""
import argparse
import json
from math import sqrt
from pathlib import Path
from ..common import write_json


def wilson(k,n,z=1.96):
    if not n:return None
    p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;delta=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,center-delta),min(1.,center+delta)]


def read(path):return json.loads(path.read_text()) if path.exists() else None

def fraction(r,setting='full',field='successes'):
    if not r or setting not in r['settings']:return 'N/A'
    s=r['settings'][setting];return f"{s[field]}/{s['denominator']}"


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/graph/v2'));args=p.parse_args();root=args.root
    rows={};ablation={};missing=[]
    for name in ['flat','center','surface']:
        off=read(root/'evaluation'/f'{name}_goal.json');roll=read(root/'diagnostics'/f'{name}_goal.json');fresh=read(root/'diagnostics'/f'{name}_fresh.json')
        training=read(root/'training'/f'{name}_goal'/'training.json')
        if not all(x is not None for x in [off,roll,fresh,training]):missing.append(name)
        ci={setting:wilson(r['successes'],r['denominator']) for setting,r in roll['settings'].items()} if roll else {}
        rows[name]=dict(offline=off,rollout=({setting:{k:v for k,v in r.items() if k!='episodes'} for setting,r in roll['settings'].items()} if roll else None),fresh=({k:v for k,v in fresh['settings']['full'].items() if k!='episodes'} if fresh else None),success_wilson95=ci,fresh_wilson95=wilson(fresh['settings']['full']['successes'],fresh['settings']['full']['denominator']) if fresh else None,training=({k:v for k,v in training.items() if k!='history'} if training else None))
    for name in ['flat_v0','flat_duration','flat_proprio','flat_goal']:
        old=read(root/'diagnostics'/'offline_v0.json')
        off=read(root/'evaluation'/f'{name}.json') if name!='flat_v0' else (old['flat'] if old else None)
        roll=read(root/'diagnostics'/f'{name}.json')
        ablation[name]=dict(offline=off,full=fraction(roll),grasp_start=fraction(roll,'grasp'),alignment_start=fraction(roll,'alignment'),acquisition=(f"{roll['settings']['full']['stage_counts']['acquisition']}/{roll['settings']['full']['denominator']}" if roll else 'N/A'))
    lines=['# Graph prototype v2 measured comparison','', 'CPU, seed 0. Same 91 episodes / 73–9–9 split. Main: 30 epochs, 10,590 updates, qvel9 + geometric residual22, regression head. Validation selection only. Full and each warm-start have 250 **learner** steps; warm starts include prior expert actions.','', '## Flat ablations','', '| Configuration | Action MSE | Full SR | Acquisition | Grasp-start SR | Alignment-start SR | Params |','|---|---:|---|---|---|---|---:|']
    for name,r in ablation.items():
        off=r['offline'];lines.append(f"| {name} | {off['offline']['action_mse']:.6f} | {r['full']} | {r['acquisition']} | {r['grasp_start']} | {r['alignment_start']} | {off['parameters']} |" if off else f'| {name} | N/A | N/A | N/A | N/A | N/A | N/A |')
    lines+=['','## Same controller, three encoders','','| Model | Action MSE | Translation RMSE | Rotation RMSE | Full SR | Alignment-start SR | Fresh SR | Params | Latency ms |','|---|---:|---:|---:|---|---|---|---:|---:|']
    for name,r in rows.items():
        off=r['offline'];roll=read(root/'diagnostics'/f'{name}_goal.json');fresh=read(root/'diagnostics'/f'{name}_fresh.json')
        if off:
            m=off['offline'];lines.append(f"| {name} | {m['action_mse']:.6f} | {m['translation_rmse']:.6f} | {m['rotation_rmse']:.6f} | {fraction(roll)} | {fraction(roll,'alignment')} | {fraction(fresh)} | {off['parameters']} | {off['latency']['total_ms']:.3f} |")
        else:lines.append(f'| {name} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |')
    lines+=['','RMSE is normalized action, not realized TCP motion. Latency is geometry feature/graph + shared-state construction + network, **excludes GT extraction**. No Torch GPU training was used.','', '## Full-task stage counts (denominator: all 9 episodes)','','| Metric | Flat | Center | Surface |','|---|---|---|---|']
    for metric in ['approach','acquisition','transport','alignment','insertion']:
        cells=[]
        for name in rows:
            roll=rows[name]['rollout'];cells.append(f"{roll['full']['stage_counts'][metric]}/{roll['full']['denominator']}" if roll else 'N/A')
        lines.append('| '+('transport proxy' if metric=='transport' else metric)+' | '+' | '.join(cells)+' |')
    for metric in ['full','grasp','transport','alignment']:
        lines.append('| '+metric+'-start success | '+' | '.join(f"{r['rollout'][metric]['successes']}/{r['rollout'][metric]['denominator']}" if r['rollout'] and metric in r['rollout'] else 'N/A' for r in rows.values())+' |')
    lines+=['', '## Uncertainty','']
    for name,r in rows.items():
        lines.append(f"- {name}: held-out full Wilson 95% {r['success_wilson95'].get('full')}; fresh full {r['fresh_wilson95']}; alignment-start {r['success_wilson95'].get('alignment')}.")
    lines+=['','Thresholds are fixed in diagnostics.THRESHOLDS and reproduced in each JSON. Acquisition: Panda is_grasping (.5 N, 85° default) for 5 consecutive observations; transport uses nominal upright centre_z−local half_z>3 cm lift proxy, not exact rotated-box clearance or goal-region completion; alignment: head transverse ≤1 cm, axis ≤10°, entry distance ≤4 cm. Insertion flag means aperture entry, **success is exclusively official info.success**, which checks head x≥−.015 m and y,z within randomized hole radius. Counts indicate ever reached and are not conditional transition probabilities. Warm-start stage achievement before handoff is expert-provided.','', 'Single training seed; 9 familiar held-out configurations and 30 fresh seeds (2000–2029), separately reported. Small warm-start samples do not establish graph superiority.','']
    if missing:lines+=['N/A: incomplete model artifacts: '+', '.join(missing)+'.']
    if any(r['offline'] is None for r in ablation.values()):lines+=['N/A ablations: this output root does not contain those original/duration/proprio measurements.']
    out=root/'comparison';out.mkdir(parents=True,exist_ok=True)
    write_json(out/'results.json',dict(models=rows,ablations=ablation,missing=missing));(out/'results.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines[:28]))

if __name__=='__main__':main()
