"""Local Task08 comparison, complete artifact decoding, no publishing."""
import html
import json
from pathlib import Path
import re
import shutil
import subprocess

import cv2
import numpy as np

from graph.integration_baseline.direct_runtime import atomic, read, sha


def snapshot(run,plan):
    run=Path(run);root=Path(__file__).resolve().parents[2]
    hashes={}
    for path,expected in plan['source_hashes'].items():
        p=Path(path)
        if sha(p)!=expected:raise ValueError('source changed before snapshot')
        rel=p.resolve().relative_to(root)
        dest=run/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        hashes[str(rel)]=expected
    atomic(run/'source_hashes.json',hashes,immutable=True)
    if (root/'.git').exists():
        diff=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=root)
        for p in sorted((root/'graph/integration_baseline').glob('bc08_*.py')):
            if subprocess.run(['git','ls-files','--error-unmatch',str(p.relative_to(root))],cwd=root,capture_output=True).returncode:
                diff+=subprocess.run(['git','diff','--no-index','--binary','/dev/null',str(p)],capture_output=True).stdout
        (run/'implementation.diff').write_bytes(diff)
    elif (root/'deployment_implementation.diff').exists():
        shutil.copy2(root/'deployment_implementation.diff',run/'implementation.diff')
    else:(run/'implementation.diff').write_text('Source snapshot is authoritative; deployed without .git; see source_hashes.json.\n')
    if (root/'deployment_provenance.json').exists():shutil.copy2(root/'deployment_provenance.json',run/'deployment_provenance.json')
    for name in ['remote_contract_tests.log','simulator_example.log','protected_before.json']:
        if (root/name).exists():shutil.copy2(root/name,run/name)
    (run/'reproduction.txt').write_text('Use the immutable comparison plan.json and the same source snapshot.\n'
        'python -m graph.integration_baseline.bc08_pipeline --plan /absolute/comparison/plan.json --resume\n'
        'Resume creates new ExperimentLogger UUIDs; original checkpoints remain immutable.\n')


def verify(run,*,link_root=None):
    run=Path(run).resolve();allowed=Path(link_root).resolve() if link_root else run
    counts=dict(json=0,npz=0,png=0,mp4=0,links=0)
    for p in run.rglob('*'):
        if not p.is_file() or 'source_snapshot' in p.parts:continue
        if p.suffix=='.json':read(p);counts['json']+=1
        elif p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as a:
                for key in a.files:a[key]
            counts['npz']+=1
        elif p.suffix=='.png':
            if cv2.imread(str(p)) is None:raise ValueError('PNG decode failed')
            counts['png']+=1
        elif p.suffix=='.mp4':
            cap=cv2.VideoCapture(str(p));n=0;expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
            try:
                while cap.read()[0]:n+=1
                if not n or n!=expected or abs(fps-20)>.01:raise ValueError('MP4 decode/20Hz mismatch')
            finally:cap.release()
            counts['mp4']+=1
        elif p.suffix=='.html':
            for target in re.findall(r'(?:href|src)="([^"]+)"',p.read_text()):
                if target.startswith('#'):continue
                q=(p.parent/html.unescape(target)).resolve()
                if not q.is_relative_to(allowed) or not q.is_file():raise ValueError('invalid report relative link '+target)
                counts['links']+=1
    return dict(status='PASS',decoded=counts,scope='artifact integrity, not policy success')


def render(plan_path,state):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(plan_path).parent;plan=read(plan_path)
    result=dict(status=state['status'],stages=state.get('stages',{}),training={},evaluations={},
        questions={},prior_baseline=dict(successes=0,episodes=7,scope='historical initial snapshot only; independent from new seeds'))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for phase,methods in [('overfit',['A','C']),('full',['A','B','C'])]:
        result['training'][phase]={}
        for method in methods:
            marker=root/f'{phase}-{method}-train.json'
            if not marker.exists():continue
            run=Path(read(marker)['directory']);completion=run/'completion.json'
            if completion.exists():result['training'][phase][method]=read(completion)
            h=run/'history.jsonl'
            if h.exists():
                rows=[json.loads(line) for line in h.read_text().splitlines()]
                ax=axes[0 if phase=='overfit' else 1]
                ax.plot([r['step'] if phase=='overfit' else r['epoch'] for r in rows],
                    [r['train' if phase=='overfit' else 'validation']['native_action7_mse'] for r in rows],label=method)
                if phase=='full':ax.plot([r['epoch'] for r in rows],[r['train']['native_action7_mse'] for r in rows],':',label=method+' train')
    for ax,title,xlabel in zip(axes,['Small5 offline train Action7 MSE','Full validation decoded Action7 MSE'],['optimizer updates','epoch']):
        ax.set(title=title,xlabel=xlabel,ylabel='native MSE');ax.grid(alpha=.2)
        if ax.lines:ax.legend()
    fig.tight_layout();fig.savefig(root/'bc_native_error_curves.png',dpi=150);plt.close(fig)
    for name,stage in state.get('stages',{}).items():
        if stage.get('directory') and (Path(stage['directory'])/'evaluation.json').exists():
            result['evaluations'][name]=read(Path(stage['directory'])/'evaluation.json')
    over=result['training']['overfit'];full=result['training']['full']
    def ev(key):return result['evaluations'].get(key,{}).get('summary')
    def final(method):return ev('final_'+method)
    result['questions']['Q1']={m:dict(fit=v.get('offline_fit'),train=v.get('train'),reduction=v.get('relative_train_error_reduction')) for m,v in over.items()} or 'NOT_MEASURED'
    result['questions']['Q2']={m:dict(offline=over.get(m,{}).get('train'),train_seed_rollout=ev('train_seed_'+m)) for m in ['A','C']}
    for q,pair,claim in [('Q3',['A','B'],'motion z-score'),('Q4',['B','C'],'gripper BCE')]:
        a,b=pair
        result['questions'][q]=dict(comparison=pair,changed_component=claim,
            test_metrics={m:read(Path(full[m]['directory'])/'test_metrics.json') for m in pair if m in full},
            final_rollout={m:final(m) for m in pair},
            interpretation='NOT_MEASURED' if not all(final(m) and final(m)['status']=='COMPLETE' for m in pair) else
                ('IMPROVED_SUCCESS_IN_THIS_FIXED_PROTOCOL' if final(b)['success_rate']>final(a)['success_rate'] else
                 'NO_OBSERVED_CLOSED_LOOP_SUCCESS_IMPROVEMENT'),
            limitation='one initialization seed; report intervals and paired seeds; optimization losses are not comparable')
    result['questions']['Q5']=dict(best_validation={m:v.get('selected') for m,v in full.items()},final_rollout={m:final(m) for m in ['A','B','C']},
        limitation='single training seed; no across-training-seed stability claim')
    result['questions']['Q6']=dict(historical_snapshot='0/7',same_runtime_historical7={m:ev('historical7_'+m) for m in ['reference','A','B','C']},
        new_development={m:ev('development_'+m) for m in ['A','B','C']},new_final={m:final(m) for m in ['A','B','C']},
        distinction='Original frozen test reset mismatch is audited separately; new seeds do not replace original test identities')
    result['questions']['Q7']=dict(offline={m:read(Path(v['directory'])/'test_metrics.json') for m,v in full.items()},
        closed_loop={m:final(m) for m in ['A','B','C']},interpretation='Compare decoded offline error and autonomous insertion separately; no inference from contract PASS')
    if not all(final(m) and final(m)['status']=='COMPLETE' for m in ['A','B','C']):
        next_step='INCONCLUSIVE: complete missing measurements before a research choice'
    elif all(final(m)['successes']==0 for m in ['A','B','C']):
        fitted=[m for m,v in over.items() if v.get('offline_fit')=='OFFLINE_FIT_PASS']
        next_step=('All three measured BC methods had zero final insertion successes. '
            +('Small5 fit passed for '+','.join(fitted)+': investigate closed-loop accumulation and recovery coverage next; use the train-reset audit to limit same-initial-state claims. ' if fitted else
              'Small5 fitting remained insufficient: basic optimization/input learning is unresolved, so recovery data alone is not yet the supported next explanation. ')
            +'These results do not identify Graph representation as the cause. Recovery/DAgger collection and representation probes require separate authorization.')
    else:next_step='Use paired native/offline/grasp/insertion results and uncertainty to choose the next controlled comparison; no automatic recovery-data collection or architecture search.'
    result['questions']['Q8']=next_step
    result['resource_gate']=state.get('resource_gate') or (read(root/'resource_gate.json') if (root/'resource_gate.json').exists() else None)
    if result['resource_gate']:
        result['resource_gate']['sequential']=state.get('resource_gate',{}).get('sequential',result['resource_gate'].get('sequential',False))
    atomic(root/'comparison.json',result)
    lines=['# Task08 — BC normalization, overfit and closed-loop comparison','',f'Status: **{state["status"]}**. Comparison UUID: `{root.name}`.','',
        'A: Raw Action7 MSE. B: (6×Motion6 z-MSE + continuous Gripper MSE)/7. C: (6×Motion6 z-MSE + Gripper BCE)/7.',
        'B/C outputs are decoded into native commands before all common metrics and controller application. These loss weights do not reproduce the Recurrent team’s original weights.','',
        f'Overfit IDs (frozen before training): `{plan["overfit_ids"]}`. Initial weight identity: `{plan["initial_weight_identity"]}`.',
        'The original Train96 Robot76 normalizer is reused; Motion6 statistics use only each actual training subset.',
        'Small5 fit uses predeclared native error thresholds plus 95% reduction and 99% gripper accuracy; it is distinct from manipulation success.','',
        'Full checkpoints use validation Native Action7 MSE and a common 50-epoch budget. Test is consumed once after selection. Development10 and Final40 are disjoint and fixed before training.','',
        'Rollout: existing causal DINO/SAM2, Graph/GNN/GRU/Robot76, 20Hz control/100Hz physics/horizon250, no expert actions or rescue. Stop-the-clock success is distinct from real-time performance.','']
    def number(x):return 'NOT_MEASURED' if x is None else f'{x:.6g}'
    def rate(x):
        if not x:return 'NOT_MEASURED'
        if x.get('status')!='COMPLETE':return 'BLOCKED/INCOMPLETE ('+str(x.get('blocked_reset_identity',0))+' reset mismatches)'
        return f'{x["successes"]}/{x["episodes"]} ({100*x["success_rate"]:.1f}%)'
    lines.extend(['## Measured comparison','', '| Small5 | Native Action7 MSE | Reduction | Fit | Train-seed insertion |',
        '| --- | ---: | ---: | --- | --- |'])
    for m in ['A','C']:
        v=over.get(m,{})
        lines.append(f'| {m} | {number(v.get("train",{}).get("native_action7_mse"))} | {number(v.get("relative_train_error_reduction"))} | {v.get("offline_fit","NOT_MEASURED")} | {rate(ev("train_seed_"+m))} |')
    lines.extend(['','| Full | Best val Action7 MSE | Test Motion6 MSE | Test Action7 MSE | Test grip accuracy | Development insertion | Final insertion (95% CI) | Final grasp |',
        '| --- | ---: | ---: | ---: | ---: | --- | --- | --- |'])
    test_metrics={}
    for m in ['A','B','C']:
        v=full.get(m,{});test=read(Path(v['directory'])/'test_metrics.json') if v else {};test_metrics[m]=test
        f=final(m);ci='' if not f or not f.get('success_ci95') else ' ['+', '.join(f'{100*x:.1f}%' for x in f['success_ci95'])+']'
        grasp='NOT_MEASURED' if not f or f.get('grasp_successes') is None else str(f['grasp_successes'])+'/'+str(f['episodes'])
        lines.append(f'| {m} | {number(v.get("best_validation_native_action7_mse"))} | {number(test.get("native_motion6_mse"))} | {number(test.get("native_action7_mse"))} | {number(test.get("gripper_accuracy"))} | {rate(ev("development_"+m))} | {rate(f)+ci} | {grasp} |')
    lines.extend(['','| Full offline test | Native translation vector error | Controller translation error (mm) | Controller rotation geodesic (rad) |',
        '| --- | ---: | ---: | ---: |'])
    for m,test in test_metrics.items():
        c=test.get('controller_translation_error_m',{}).get('mean')
        lines.append(f'| {m} | {number(test.get("native_translation_vector_error",{}).get("mean"))} | {number(c*1000 if c is not None else None)} | {number(test.get("controller_rotation_geodesic_rad",{}).get("mean"))} |')
    lines.extend(['','## Q1–Q8','',
        '**Q1.** '+('; '.join(m+': '+v.get('offline_fit','NOT_MEASURED') for m,v in over.items()) or 'NOT_MEASURED')+'. Small5 thresholds are declared in plan.json; they do not establish policy success.',
        '', '**Q2.** '+ '; '.join(m+': offline MSE '+number(over.get(m,{}).get('train',{}).get('native_action7_mse'))+', train-seed rollout '+rate(ev('train_seed_'+m)) for m in ['A','C'])+'. A reset mismatch is not a measured policy failure.', ''])
    for q,a,b in [('Q3','A','B'),('Q4','B','C')]:
        ta,tb=test_metrics[a],test_metrics[b]
        delta='NOT_MEASURED' if not ta or not tb else number(tb['native_motion6_mse']-ta['native_motion6_mse'])
        lines.extend([f'**{q}.** {a}→{b}: Test Motion6 MSE delta {delta}; final insertion {rate(final(a))} → {rate(final(b))}. '
            +result['questions'][q]['interpretation']+'. Grasp and gripper accuracy are reported separately; intervals and a single training seed limit the conclusion.',''])
    measured={m:v['best_validation_native_action7_mse'] for m,v in full.items() if v.get('best_validation_native_action7_mse') is not None}
    lines.extend(['**Q5.** Lowest common validation MSE: '+(min(measured,key=measured.get) if measured else 'NOT_MEASURED')+'. Final insertion results above are independent of checkpoint selection; across-training-seed stability was not tested.','',
        '**Q6.** Original snapshot: 0/7. Current-runtime historical diagnostic7: '+ '; '.join(m+' '+rate(ev('historical7_'+m)) for m in ['reference','A','B','C'])+'. Historical41005 is diagnostic only. Development10 and Final40 are distinct new sets.','',
        '**Q7.** Compare the measured native offline table with final insertion/grasp above. Lower offline error alone is not manipulation success. No success is inferred from contract PASS.','',
        '**Q8.** '+str(result['questions']['Q8']),'',
        'Full episode error distributions, per-episode results, first success/grasp steps, timeouts and inference latency are preserved in comparison.json and linked evaluation JSON/trace/media.',''])
    if result['resource_gate']:
        r=result['resource_gate']
        lines.extend(['## Parallel resource evidence','','```json',json.dumps(dict(
            single_process_memory={m:v['memory'] for m,v in r['single'].items()},paired_gpu=r['paired']['snapshot'],
            paired_probe_pids=[v['pid'] for v in r['paired']['probes']],sequential_fallback=r.get('sequential',False)),indent=2),'```',''])
        if r.get('triple'):
            lines.extend(['','Full A/B/C concurrent resource gate (explicit user override of the original two-job limit):','',
                '```json',json.dumps(r['triple'],indent=2),'```',''])
    lines.extend(['## Execution and artifacts','', 'See `status.json`, `resources.jsonl`, `seeds_manifest.json`, `plan.json`, `comparison.json`, and per-stage UUID directories.',
        'Each worker preserves immutable checkpoint receipts, action statistics, source hashes/snapshot/diff, local logs and error bundle. No external artifact publishing or automatic Codex repair is enabled.',
        'Only executed backends are evidence: CUDA policy/Vision and CPU PhysX are intended; CUDA PhysX, MPS and real-time deadlines are not tested.',''])
    md='\n'.join(lines);(root/'08_bc_normalization_overfit_training.md').write_text(md)
    # Only this Task08 document is owned by the new pipeline.
    doc=Path(__file__).resolve().parents[2]/'docs/visual-graph-integration/08_bc_normalization_overfit_training.md'
    doc.write_text(md)
    links=[]
    for p in sorted(root.iterdir()):
        if p.is_file() and p.name!='bc_training_comparison.html':links.append((p.name,p.name))
    for name,stage in state.get('stages',{}).items():
        if stage.get('directory'):
            p=Path(stage['directory'])
            for candidate in ['completion.json','evaluation.json','error_bundle.json','verification.json','action_codec.json']:
                if (p/candidate).exists():links.append((str(Path('..')/p.name/candidate),name+'/'+candidate))
            if (p/'evaluation.json').exists():
                for episode in read(p/'evaluation.json')['episodes']:
                    if not episode.get('artifact_directory'):continue
                    folder=Path(episode['artifact_directory']).resolve()
                    if not folder.is_relative_to(p.resolve()):raise ValueError('unsafe episode artifact directory')
                    for candidate in ['episode.json','rollout.mp4','representative.png','actions.npz','hidden.npz',
                                      'observations.npz','encoded_action_output.npz','grasp_contact_lift.json','trace.json']:
                        if (folder/candidate).exists():
                            links.append((str(Path('..')/p.name/folder.relative_to(p.resolve())/candidate),name+'/'+folder.name+'/'+candidate))
    text='<!doctype html><html lang="en"><meta charset="utf-8"><title>Task08 BC comparison</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:20px}pre{white-space:pre-wrap;background:#f4f5f7;padding:16px}img{max-width:100%}</style><h1>Task08 BC comparison</h1><p>'+html.escape(state['status'])+'</p><img src="bc_native_error_curves.png" alt="Native error curves"><pre>'+html.escape(md)+'</pre><ul>'+''.join(f'<li><a href="{html.escape(p,quote=True)}">{html.escape(label)}</a></li>' for p,label in links)+'</ul></html>'
    (root/'bc_training_comparison.html').write_text(text)
    atomic(root/'verification.json',verify(root,link_root=root.parent))
    from utils.experiment_artifacts import register_existing_output
    for name,kind in [('comparison.json','metrics'),('verification.json','metrics'),('bc_training_comparison.html','reports'),
                      ('08_bc_normalization_overfit_training.md','reports'),('bc_native_error_curves.png','reports')]:
        register_existing_output(root,root/name,kind=kind)
    return result
