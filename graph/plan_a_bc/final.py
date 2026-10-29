"""Post-selection offline evaluation, static figures, evidence index and report."""
import argparse
import csv
import html
from pathlib import Path

import numpy as np
import torch

from graph.integration_baseline.bc08_codec import ActionCodec
from .common import read,write,sha,checkpoint_load,npz,now
from .cache import CachedDataset
from .model import CachedGraphPolicy,normalizer
from .train import offline


def offline_final(root):
    root=Path(root).resolve();selected=read(root/'model_selection_frozen.json');cfg=read(root/'config.json')
    ds=CachedDataset(root,allow_test=True);norm=normalizer(root);summary={}
    for variant,models in selected['models'].items():
        summary[variant]={}
        for kind,receipt in models.items():
            if sha(receipt['path'])!=receipt['sha256']:raise ValueError('selected model changed')
            cp=checkpoint_load(receipt['path']);model=CachedGraphPolicy(norm.schema,variant,**cfg['model']).cuda()
            model.load_state_dict(cp['model']);codec=ActionCodec(cp['codec']);summary[variant][kind]={}
            folder=root/'final_offline'/variant/kind
            for name,ids in {**cfg['splits'],'train_seed30003':[30003]}.items():
                metrics=offline(ds,model,norm,codec,ids,cfg,'cuda',folder/name)
                summary[variant][kind][name]=metrics
                write(folder/(name+'_metrics.json'),metrics,immutable=True)
    write(root/'final_offline_metrics.json',summary,immutable=True)


def csv_rows(path):
    if not path.exists():return []
    return list(csv.DictReader(path.open()))


def truth(value):return str(value).lower()=='true'


def report(root):
    root=Path(root).resolve();cfg=read(root/'config.json');audit=read(root/'dataset_audit.json')
    state=read(root/'pipeline_state.json') if (root/'pipeline_state.json').exists() else read(root/'status.json')
    cache=read(root/'cache_manifest.json') if (root/'cache_manifest.json').exists() else {'episodes':[]}
    figures=root/'figures';figures.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records={};allroll=[];checkpoints=[];verification={}
    for variant in ['E0','E1']:
        marker=root/(variant+'_run.json')
        if not marker.exists():
            records[variant]=dict(status='NOT_EXECUTED',updates=0,rollout_episodes=0);continue
        run=Path(read(marker)['directory']);tr=csv_rows(run/'train_metrics.csv');val=csv_rows(run/'validation_metrics.csv')
        full=csv_rows(run/'train_full_metrics.csv');roll=csv_rows(run/'rollout_metrics.csv');allroll.extend(roll)
        st=read(run/'status.json') if (run/'status.json').exists() else {'status':'RUNNING'}
        records[variant]=dict(status=st['status'],updates=int(tr[-1]['step']) if tr else 0,
            run=str(run),training=tr,validation=val,train_full=full,rollouts=roll,
            rollout_episodes=sum(truth(r['denominator_eligible']) for r in roll))
        cm=read(run/'checkpoint_manifest.json') if (run/'checkpoint_manifest.json').exists() else []
        for cp in cm:
            if sha(cp['path'])!=cp['sha256']:raise ValueError('checkpoint SHA failed')
            loaded=checkpoint_load(cp['path'])
            if loaded['progress']['step']!=cp['step']:raise ValueError('checkpoint cursor mismatch')
            checkpoints.append(cp)
        verification[variant]=dict(status='PASS' if st['status']=='COMPLETE' else 'NOT_COMPLETED',
            updates=records[variant]['updates'],checkpoint_hashes_verified=len(cm),actual_rollout_episodes=records[variant]['rollout_episodes'])
    # One common step axis for both offline action fit and actual manipulation.
    colors={'E0':'#1864ab','E1':'#d97706'}
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for variant,r in records.items():
        for key,style in [('train_full','-'),('validation','--')]:
            rows=r.get(key,[])
            if rows:
                axes[0].plot([int(a['step']) for a in rows],[float(a['optimization_loss']) for a in rows],style,color=colors[variant],label=variant+' '+key)
                axes[1].plot([int(a['step']) for a in rows],[float(a['native_action7_mse']) for a in rows],style,color=colors[variant],label=variant+' '+key)
    for ax,title in zip(axes,['Method B loss','Decoded native Action7 MSE']):
        ax.set(xlabel='Optimizer updates',ylabel=title,title=title);ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figures/'learning_curves.png',dpi=170);plt.close(fig)
    curve=[]
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for variant,r in records.items():
        rows=[x for x in r.get('rollouts',[]) if truth(x['denominator_eligible']) and x['group']=='dev' and not x['final_kind']]
        steps=sorted({int(x['step']) for x in rows})
        for step in steps:
            # Fixed paired Dev2 subset; extra Dev3 never changes this denominator.
            rr=[x for x in rows if int(x['step'])==step and int(x['seed']) in cfg['evaluation']['normal_dev_seeds']]
            if len(rr)!=2:continue
            row=dict(variant=variant,step=step,episodes=2,**{k:sum(truth(x[k]) for x in rr)/2 for k in ['reach','grasp','lift','alignment','insertion']})
            curve.append(row)
        cr=[x for x in curve if x['variant']==variant]
        for ax,key in zip(axes.ravel(),['reach','grasp','lift','insertion']):
            if cr:ax.plot([x['step'] for x in cr],[x[key] for x in cr],marker='o',color=colors[variant],label=variant)
            ax.set(title=key,xlabel='Optimizer updates',ylabel='Fixed paired Dev2 rate',ylim=(-.05,1.05));ax.grid(alpha=.2)
            if ax.lines:ax.legend()
    fig.tight_layout();fig.savefig(figures/'closed_loop_curves.png',dpi=170);plt.close(fig)
    write(root/'learning_timeline.json',dict(records=records,paired_dev2=curve))
    steps={v:{k:min([int(x['step']) for x in r.get('rollouts',[]) if truth(x['denominator_eligible']) and not x['final_kind'] and truth(x[k])],default=None)
        for k in ['reach','grasp','lift','alignment','insertion']} for v,r in records.items()}
    actual=[r for r in allroll if truth(r['denominator_eligible'])]
    tv=[e for e in cache['episodes'] if e['split'] in ['train','validation']]
    full_cache=len(tv)==108 and sum(e['graphs'] for e in tv)==audit['train_actions']+audit['validation_actions']
    for r in cache['episodes']:
        if sha(root/r['path'])!=r['sha256']:raise ValueError('cache SHA mismatch')
    for name in ['training_smoke_verification','runtime_verification','cache_pilot_verification']:
        verification[name]=read(root/(name+'.json')) if (root/(name+'.json')).exists() else {'status':'NOT_TESTED'}
    verification['full_cache']=dict(status='PASS' if full_cache else 'NOT_COMPLETED',episodes=len(tv),graphs=sum(e['graphs'] for e in tv))
    verification['test12']=dict(status='PASS' if len([r for r in actual if r['group']=='test12'])==48 else 'NOT_COMPLETED',
        episodes=len([r for r in actual if r['group']=='test12']))
    verification['periodic_rollout_schedule']={}
    for v,r in records.items():
        schedule=[]
        for step in cfg['evaluation']['schedule_steps']:
            got=[x for x in r.get('rollouts',[]) if int(x['step'])==step and not x['final_kind']]
            expected=1 if step==500 else 6 if step in cfg['evaluation']['expanded_steps'] else 3
            schedule.append(dict(step=step,planned=expected,executed=len(got),status='PASS' if len(got)==expected else 'NOT_COMPLETED'))
        verification['periodic_rollout_schedule'][v]=schedule
    write(root/'verification.json',verification)
    write(root/'checkpoint_manifest.json',checkpoints)
    offline_result=read(root/'final_offline_metrics.json') if (root/'final_offline_metrics.json').exists() else {}
    comparison=[]
    for v,r in records.items():
        for kind in ['last','validation_best']:
            metrics=offline_result.get(v,{}).get(kind,{}).get('train_seed30003')
            if metrics:comparison.append(dict(variant=v,checkpoint=kind,seed=30003,**metrics))
    write(root/'baseline_comparison.json',dict(baseline=dict(source='Task08-E 1cbed275766549fda6945456503019aa',
        seed=30003,best_native_action7_mse=.0034380406404083026,last_native_action7_mse=.003433573,
        validation_best_native_action7_mse=.00613824),new=comparison,
        comparable='same original Action7, Train96/Validation12 split, sequence32, hidden144, seed42, optimizer and 11487 update budget',
        limitations='different geometry, Graph21 vs14, cap128 and trainable parameter count; validation MethodB criterion/cadence and runtime versions differ; baseline seed metric is not all-Train96 mean'))
    better=[]
    for v in ['E0','E1']:
        rows=[r for r in curve if r['variant']==v];last=next((r for r in rows if r['step']==11487),None)
        if last:
            better.extend(dict(variant=v,step=r['step'],metric=k,earlier=r[k],last=last[k],episodes=2)
                for r in rows if r['step']<11487 for k in ['reach','grasp','lift','insertion'] if r[k]>last[k])
    write(root/'intermediate_better_than_last.json',dict(paired_dev2_diagnostic=better,selection_bias='repeated Dev2 observations; no test selection'))
    first_fail={}
    for v in ['E0','E1']:
        rr=[x for x in actual if x['variant']==v and not truth(x['insertion'])]
        first_fail[v]={k:sum(x['first_failed_stage']==k for x in rr) for k in ['reach','grasp','lift','alignment','insertion']}
    lines=['# Plan A Cached Graph BC + Periodic Closed-loop Rollout','',
        f'상태: **{state["status"]}**. 작성 시각: {now()}. 미실행 항목은 성공/실패율 분모에 포함하지 않았다.',
        '',f'Comparison UUID: `{cfg.get("comparison_uuid","3788798a719f4ec4a66c224ca42f5d63")}`. Plan B의 코드·결과·Persistent Surface Map은 사용하지 않았다.',
        '', '## 실행 범위와 고정 조건','',
        '09-D의 C3 + A-SF, cap128/object, Node21/Edge4를 사용한다. GNN 이전의 Graph만 저장하며 모든 temporal/status extras는 0이다. Method B 통계와 Robot76 정규화는 기존 Train96을 기준으로 고정했다.',
        '',f'원본 120 Episode / Train96 Action {audit["train_actions"]:,}개 / Validation12 Action {audit["validation_actions"]:,}개. 현재 Train/Validation 캐시: {len(tv)}/108 Episode, {sum(e["graphs"] for e in tv):,} Graph.',
        '', '| 모델 | 실제 Updates | 상태 | 실제 성능 평가 Episode |','|---|---:|---|---:|']
    for v,r in records.items():lines.append(f'| {v} | {r["updates"]:,} | {r["status"]} | {r["rollout_episodes"]} |')
    lines+=['','## Q1–Q10','',
        f'**Q1. 전체 action 정렬 캐시를 생성했는가?** {"예" if full_cache else "아니오 / 아직 미완료"}. {len(tv)}/108 Episode, {sum(e["graphs"] for e in tv):,}/{audit["train_actions"]+audit["validation_actions"]:,} timestep. 원본 T+1번째 terminal observation에는 action을 붙이지 않았다.',
        '',f'**Q2. GNN과 GRU를 end-to-end로 학습했는가?** 실제 Full 업데이트 E0={records["E0"]["updates"]}, E1={records["E1"]["updates"]}. Smoke 상태는 verification.json에 별도로 기록했다. Synthetic gradient PASS를 Full 학습 완료로 간주하지 않는다.',
        '', '**Q3. Full96 대비 Training Fit은?** 같은 seed30003의 decoded native Action7 MSE와 전체 Train96/Validation12 수치를 구분해 baseline_comparison.json에 기록했다. 기존 Full B seed30003 Best MSE=0.00343804, Last=0.00343357. 기존 seed 하나의 수치를 전체 Train96 평균으로 비교하지 않는다. 입력 Geometry·feature 차원·parameter 수, checkpoint 선택 기준/빈도 및 runtime 버전 차이가 있어 원인 하나의 순수 효과로 해석할 수 없다.',
        '',f'**Q4. 최초 Reach/Grasp/Lift/Insertion step은?** {steps}. None은 관찰되지 않았거나 평가 미실행을 뜻하며 각 모델의 실제 평가 수와 함께 읽어야 한다. 500-step smoke는 제외했다.',
        '', '**Q5. Validation Loss와 Closed-loop가 함께 개선됐는가?** learning_timeline.json의 같은 checkpoint와 고정 Dev2 곡선으로 비교한다. 미실행 시점 사이의 개선이나 인과관계를 추정하지 않는다. Dev 확대 평가의 추가 3 seed는 고정 Dev2 그래프의 분모를 바꾸지 않는다.',
        '',f'**Q6. 중간 checkpoint가 Last보다 나았는가?** 고정 Dev2에서 관측된 사례 {len(better)}개. intermediate_better_than_last.json에 step/지표/분모를 기록했다. 반복 평가에 의한 Dev 선택 편향이 있어 독립 Test 성능의 우위로 간주하지 않는다.',
        '', '**Q7. E0와 E1의 차이는?** 같은 캐시·ActionCodec·GRU·optimizer·update budget·seed·rollout 절차로 비교한다. E0에는 message layer가 있고 E1에는 없다. Parameter 수가 달라 message passing만의 효과로 단정하지 않는다. 실제 수치가 없는 항목은 NOT_TESTED다.',
        '',f'**Q8. 첫 실패는?** 실패 Episode의 첫 미충족 단계 집계: {first_fail}. Trace의 현재 node/edge·누락·gate, Action7·실제 grasp·robot/peg/box 궤적을 함께 보존한다. 단계 실패만으로 action fit/오염/누락/GRU 중 하나를 원인으로 확정하지 않는다.',
        '', '**Q9. Plan A의 유용성을 어디까지 지지하는가?** Offline fit은 expert 관측에서 행동 근사를 검증한다. 실제 Train seed rollout은 자기 행동에 따른 이탈을 포함하며, Dev/Test rollout은 별도 seed에서 검증한다. 단일 학습 seed·작은 평가 분모로 보편적인 우위나 trajectory memorization의 부재를 확정할 수 없다. 미완료 시 정책 능력에 대한 결론을 보류한다.',
        '', '**Q10. 향후 Memory 비교를 분리할 수 있는가?** Geometry 생성과 cached-input encoder/GRU 학습, live observation rollout이 별도 모듈이다. Cache identity는 dataset/source/geometry 설정을 포함하며 다르면 재사용을 거부한다. Plan B 연결·Memory 비교는 이번에 구현하거나 실행하지 않았다.',
        '', '## 학습 및 조작 곡선','', '![학습 곡선](figures/learning_curves.png)','', '![실제 Closed-loop 곡선](figures/closed_loop_curves.png)',
        '', '## 판정의 의미와 남은 제약','',
        'Reach는 TCP–peg 30mm 이내의 근접 진단이다. Lift는 실제 grasp와 초기 높이 대비 5mm 상승이 3 step 지속된 경우이며 안정적 4cm 운반 성공과 구분한다. Insertion은 공식 Environment success다. 전체 평가는 horizon250 또는 환경 terminal까지 실행하며 성능 실패 자체로 학습을 중단하지 않는다.',
        '', '알려진 finger-near 오제거와 Box 오염을 유지했다. 평가 결과에 맞춰 filter/sampling/LR/architecture를 바꾸지 않았다. CUDA PhysX·MPS·20Hz real-time 처리는 NOT_TESTED다. Git commit 및 HF/W&B publishing은 실행하지 않았다.',
        '',f'실행 오류/미완료 원인: `{state.get("exception", "없음 또는 진행 중")}` / `{state.get("message", "")}`.',
        '', '## 원자료와 재현성','',
        '[검증](verification.json) · [설정](config.json) · [캐시 Manifest](cache_manifest.json) · [Checkpoint Manifest](checkpoint_manifest.json) · [시간축 원자료](learning_timeline.json) · [Artifact Index](artifact_index.json) · [Pipeline 상태](pipeline_state.json)',
        '', 'Source와 원본 Dataset/Checkpoint는 in-place로 수정하지 않는다. Checkpoint 모델·optimizer·scheduler·RNG·hidden/cursor는 readback하며 step별 SHA256을 검증한다. 학습 재개 시 새 ExperimentLogger UUID를 사용하고 원본 run을 보존한다.']
    text='\n'.join(lines)+'\n';(root/'research_report.md').write_text(text)
    html_content='<!doctype html><html lang="ko"><meta charset="utf-8"><title>Plan A cached graph BC</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#172033;background:#fafbfe}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%;background:white;border:1px solid #ddd}a{color:#1864ab}</style><h1>Plan A Cached Graph BC</h1><p>상태: '+html.escape(state['status'])+'</p><p><a href="research_report.md">Markdown 연구 보고서</a> · <a href="artifact_index.json">원자료 Index</a></p><img src="figures/learning_curves.png"><img src="figures/closed_loop_curves.png"><pre>'+html.escape(text)+'</pre></html>'
    (root/'research_report.html').write_text(html_content)
    index=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.is_relative_to(root/'source') or p.is_relative_to(root/'dataset') or p.is_relative_to(root/'transfer_parts'):continue
        if p.name in ['artifact_index.json','original_dataset_published.zip'] or p.suffix in ['.log','.gz','.partial']:continue
        if p.name.endswith('.tracking.json'):continue
        index.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
    write(root/'artifact_index.json',dict(created_at=now(),files=index))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--offline-only',action='store_true');p.add_argument('--report-only',action='store_true');a=p.parse_args()
    from graph.integration_baseline.direct_train import seed_all
    seed_all(42,1)
    if a.offline_only:offline_final(a.root)
    else:report(a.root)


if __name__=='__main__':main()
