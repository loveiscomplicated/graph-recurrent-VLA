"""Task08-F static causal-branch report and artifact readback."""
import argparse
import csv
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from graph.integration_baseline.direct_runtime import atomic, read, sha
from utils.experiment_artifacts import register_existing_output

BRANCHES=(0,10,25,49)


def relative(state):
    return np.asarray(state['ee_pose'][:3],np.float64)-np.asarray(state['peg_pose'][:3],np.float64)


def read_trial(root,m,k,arm):
    return root/'branches'/m/f'k{k:03d}'/arm


def trajectory_gap(a,b,start,end):
    return [float(np.linalg.norm(relative(a[t])-relative(b[t]))) for t in range(start,end+1)]


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('href','src') and value:self.links.append(value)


def create(root):
    root=Path(root).resolve()
    status=read(root/'status.json')
    if status['status']!='COMPLETE':
        raise ValueError('branch execution not complete')
    protocol=read(root/'protocol.json')
    results=read(root/'results.json')
    expert=read(root/'reference_expert/states.json')
    fig=root/'figures';fig.mkdir(exist_ok=False)
    rows=[];curves={}
    for m in 'ABC':
        for k in BRANCHES:
            base=root/'branches'/m/f'k{k:03d}'
            if not (base/'MM/summary.json').is_file():
                rows.append(dict(method=m,branch=k,status='INVALID_PREFIX_IDENTITY'))
                continue
            gate1=read(base/'prefix_gate_em_me.json');gate2=read(base/'prefix_gate_em_mm.json')
            if not gate1['pass_gate'] or not gate2['pass_gate']:
                rows.append(dict(method=m,branch=k,status='INVALID_PREFIX_IDENTITY'))
                continue
            r=read(base/'comparison.json')
            mm=read(base/'MM/states.json');em=read(base/'EM/states.json');me=read(base/'ME/states.json')
            curves[m,k]=dict(feedback=trajectory_gap(mm,em,k+1,k+10),
                             fixed=trajectory_gap(me,expert,k+1,k+10))
            outcome=r['full_prefix_rescue']
            rows.append(dict(method=m,branch=k,status='VALID',
                d1_m=r['first_action_tcp_peg_difference_m'],
                fixed_d10_m=r['fixed_expert_suffix_difference_m'],
                feedback_d10_m=r['policy_feedback_difference_m'],
                feedback_minus_fixed_m=r['feedback_minus_fixed_m'],
                amplification=r['feedback_amplification_supported'],
                grasp=outcome['grasp'],first_grasp=outcome['first_grasp'],
                lift_4cm=outcome['lift_4cm'],success=outcome['success'],
                first_success_step=outcome['first_success_step'],steps=outcome['steps'],
                max_peg_lift_m=outcome['max_peg_lift_m']))
    if len(rows)!=12:
        raise ValueError('missing prespecified method/branch records')
    with (root/'branch_metrics.csv').open('x',newline='') as f:
        valid=[r for r in rows if r['status']=='VALID']
        names=['method','branch','status','d1_m','fixed_d10_m','feedback_d10_m',
               'feedback_minus_fixed_m','amplification','grasp','lift_4cm','success','steps']
        writer=csv.DictWriter(f,fieldnames=names,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

    fig1,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for m,ax in zip('ABC',axes):
        rr=[r for r in rows if r['method']==m and r['status']=='VALID']
        x=np.asarray([r['branch'] for r in rr],np.float64)
        ax.bar(x-1.8,[1000*r['fixed_d10_m'] for r in rr],3.6,label='same Expert suffix')
        ax.bar(x+1.8,[1000*r['feedback_d10_m'] for r in rr],3.6,label='policy feedback')
        ax.set_xticks(BRANCHES);ax.set_title(m);ax.set_xlabel('Expert prefix length k')
    axes[0].set_ylabel('TCP–Peg branch difference at 10 actions (mm)');axes[0].legend()
    fig1.tight_layout();fig1.savefig(fig/'feedback_vs_fixed.png',dpi=140);plt.close(fig1)

    fig2,axes=plt.subplots(3,4,figsize=(17,9),sharex=True)
    for i,m in enumerate('ABC'):
        for j,k in enumerate(BRANCHES):
            ax=axes[i,j]
            if (m,k) in curves:
                z=curves[m,k]
                ax.plot(range(1,11),1000*np.asarray(z['fixed']),label='Expert suffix')
                ax.plot(range(1,11),1000*np.asarray(z['feedback']),label='Policy feedback')
            else:ax.text(.1,.5,'INVALID PREFIX',transform=ax.transAxes)
            ax.set_title(f'{m} k={k}')
            if j==0:ax.set_ylabel('Difference mm')
            if i==2:ax.set_xlabel('Actions after branch')
    axes[0,0].legend();fig2.tight_layout();fig2.savefig(fig/'divergence_curves.png',dpi=140);plt.close(fig2)

    level=np.full((3,4),np.nan)
    for row in rows:
        if row['status']=='VALID':
            level['ABC'.index(row['method']),BRANCHES.index(row['branch'])]=(
                3 if row['success'] else 2 if row['lift_4cm'] else 1 if row['grasp'] else 0)
    fig3,ax=plt.subplots(figsize=(8,3.4));im=ax.imshow(level,vmin=0,vmax=3,cmap='viridis',aspect='auto')
    ax.set_xticks(range(4),BRANCHES);ax.set_yticks(range(3),list('ABC'))
    ax.set_xlabel('Expert prefix length k');ax.set_title('0 no grasp · 1 grasp · 2 lift · 3 insert')
    fig3.colorbar(im,ax=ax);fig3.tight_layout();fig3.savefig(fig/'prefix_rescue.png',dpi=140);plt.close(fig3)

    def verdict(method):
        r={row['branch']:row for row in rows if row['method']==method and row['status']=='VALID'}
        if 0 not in r or len(r)!=4:
            return dict(status='INCONCLUSIVE',reason='invalid prefix identity')
        amp=any(r[k]['amplification'] for k in (0,10))
        rescue=any((not r[0][key] and r[k][key]) for k in (10,25,49)
                   for key in ('grasp','lift_4cm','success'))
        return dict(status='SUPPORTED' if amp and rescue else 'INCONCLUSIVE',
                    early_feedback_amplification=amp,prefix_rescue=rescue,
                    late_k49_grasp=r[49]['grasp'],
                    reason='one seed, one rollout per arm; action-fit and feedback are coupled')
    verdicts={m:verdict(m) for m in 'ABC'}
    baseline=Path('/workspace/task08/experiments/1cbed275766549fda6945456503019aa')
    reproduction={}
    for m in 'ABC':
        new_path=read_trial(root,m,0,'MM')/'actions.npz'
        if not new_path.is_file():
            reproduction[m]=dict(status='NOT_TESTED',reason='invalid k=0 prefix gate')
            continue
        with np.load(baseline/'rollout'/m/'actions.npz',allow_pickle=False) as z:
            old=z['actions'][0]
        with np.load(new_path,allow_pickle=False) as z:
            new=z['predicted'][0]
        reproduction[m]=dict(action0_max_abs=float(np.max(np.abs(new-old))),
                             pass_gate=bool(np.max(np.abs(new-old))<=1e-5))
    complete=dict(status='COMPLETE',run_id=root.name,rows=rows,verdicts=verdicts,
                  task08e_action0_reproduction=reproduction,
                  reference_scope='current-runtime expert replay; original saved Robot76 diverges later',
                  limitation='single seed and no repeated arms; Expert prefix changes both physical state and GRU history; amplification criteria are diagnostic, not population significance')
    atomic(root/'comparison.json',complete,immutable=True)

    lines=['# Task 08-F — Seed 30003 error accumulation branch diagnostic','',
      f'ExperimentLogger UUID `{root.name}`. Full Best A35/B48/C38, Train Seed30003, CPU PhysX 20Hz / RTX3090 Vision-GNN-GRU. 새 학습 없음.','',
      '## 질문과 설계','',
      '기존 Full A/B/C는 정상 Expert 관측 t=0부터 제어 오차가 있고 동일 Seed 폐루프에서 파지 전 실패했다. 이번 실험은 그 오차가 물리 상태를 바꾼 뒤 정책 피드백으로 증폭되는지 확인한다. 물리 상태 snapshot 복원 없이 매 분기 독립 Reset과 동일 Expert Action prefix를 사용했다.','',
      'E/E는 한 번의 현재 런타임 Expert 재생이며 t=136 성공했다. 각 k=0,10,25,49에서 M/E와 E/M 10-action probe, M/M 최대250-step을 방법별 한 번씩 수행했다. t>0 분기 상태는 저장된 HDF5의 원래 Expert 상태와 같다고 주장하지 않는다.','',
      '## 10-action 결과와 실제 제어','',
      '| Method | k | Prefix identity | First action Δ TCP–Peg mm | Expert suffix Δ10 mm | Policy feedback Δ10 mm | Amplification | Grasp | Lift4cm | Insert | Steps |',
      '|---|---:|---|---:|---:|---:|---|---|---|---|---:|']
    for r in rows:
        if r['status']!='VALID':
            lines.append(f'| {r["method"]} | {r["branch"]} | INVALID | — | — | — | — | — | — | — | — |')
        else:
            lines.append(f'| {r["method"]} | {r["branch"]} | PASS | {1000*r["d1_m"]:.2f} | {1000*r["fixed_d10_m"]:.2f} | {1000*r["feedback_d10_m"]:.2f} | {r["amplification"]} | {r["grasp"]} | {r["lift_4cm"]} | {r["success"]} | {r["steps"]} |')
    lines+=['','Policy feedback Δ10은 M/M 대 E/M, Expert suffix Δ10은 M/E 대 E/E의 같은 시점 TCP–Peg 상대 벡터 차이다. 분기 직전 학생 hidden·관측·Graph와 물리 상태 일치 검사에 실패한 행은 효과를 계산하지 않았다.','',
            '## 해석','']
    for m in 'ABC':
        v=verdicts[m]
        lines.append(f'- {m}: **{v["status"]}**. 초기 정책 피드백 증폭 `{v.get("early_feedback_amplification")}`, Expert prefix 파지/Lift/삽입 회복 `{v.get("prefix_rescue")}`, k49 파지 `{v.get("late_k49_grasp")}`.')
    lines+=['','한 Seed의 단일 분기 결과는 일반 성공률이 아니다. Expert prefix 길이는 물리 상태와 GRU 관측 이력을 함께 바꾸므로, prefix 회복만으로 둘 중 어느 요인이 핵심인지 확정할 수 없다. 초기 모방 오차가 누적의 출발점이며, 이 실험은 피드백이 그 이탈을 추가로 키우는지를 검사한다.','',
            '## 보존과 검증','',
            f'- Prefix 무효 비교 {len(status["invalid"])}개. 원본 Full 프로세스는 재개하지 않았고 Best Checkpoint SHA256을 로딩 시 다시 검증했다.',
            '- 각 arm의 `branch_input.npz`, 적용/예측 Action, Controller 전처리, GT 진단 Sidecar, RGB-D, Graph, hidden, contact, after-action MP4는 `branches/{A,B,C}/k{000,010,025,049}/{EM,ME,MM}/`에 있다.',
            '- [Comparison JSON](comparison.json) · [CSV](branch_metrics.csv) · [Protocol](protocol.json) · [Original Expert replay](reference_expert/summary.json) · [Verification](verification.json)',
            '- [Feedback vs fixed](figures/feedback_vs_fixed.png) · [10-action divergence](figures/divergence_curves.png) · [Prefix rescue](figures/prefix_rescue.png)',
            '- Source: `source_snapshot_branch.py`, `source_snapshot_entry.py`, `source_snapshot_report.py`, `implementation.diff`. 외부 게시·Git commit·새 학습 없음.', '']
    markdown='\n'.join(lines)
    (root/'08f_error_accumulation_diagnostic.md').write_text(markdown)
    links=['comparison.json','branch_metrics.csv','protocol.json','reference_expert/summary.json','verification.json',
           'figures/feedback_vs_fixed.png','figures/divergence_curves.png','figures/prefix_rescue.png']
    body='<h1>Task 08-F Error Accumulation Branch</h1><pre>'+html.escape(markdown)+'</pre>'
    body+=''.join(f'<h2>{html.escape(Path(p).stem)}</h2><img src="{p}" width="1000">' for p in links if p.endswith('.png'))
    body+=''.join(f'<p><a href="{p}">{html.escape(p)}</a></p>' for p in links)
    for m in 'ABC':
        for k in (0,49):
            path=f'branches/{m}/k{k:03d}/MM/after_action_rgb.mp4'
            if (root/path).is_file():
                body+=f'<h3>{m} k={k}</h3><video controls width="512" src="{path}"></video>'
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>Task 08-F</title><style>body{font:16px system-ui;margin:2em;max-width:1300px}pre{white-space:pre-wrap}img{max-width:100%}</style>'+body+'</html>'
    (root/'error_accumulation_diagnostic.html').write_text(page)
    shutil.copy2(__file__,root/'source_snapshot_report.py')
    entry=Path(__file__).with_name('bc08f_entry.py')
    shutil.copy2(entry,root/'source_snapshot_entry.py')
    diff=subprocess.run(['git','diff','--no-index','--','/dev/null',str(Path(__file__).resolve())],capture_output=True,text=True)
    with (root/'implementation.diff').open('a') as f:f.write('\n'+diff.stdout)
    diff=subprocess.run(['git','diff','--no-index','--','/dev/null',str(entry.resolve())],capture_output=True,text=True)
    with (root/'implementation.diff').open('a') as f:f.write('\n'+diff.stdout)

    checks={'json':0,'npz':0,'csv':0,'png':0,'mp4':0,'html_links':0}
    for p in root.rglob('*'):
        if not p.is_file():continue
        if p.suffix=='.json':json.load(p.open());checks['json']+=1
        elif p.suffix=='.npz':
            with np.load(p,allow_pickle=False) as z:
                for key in z.files:z[key]
            checks['npz']+=1
        elif p.suffix=='.csv':
            if not list(csv.reader(p.open())):raise ValueError('empty CSV')
            checks['csv']+=1
        elif p.suffix=='.png':
            if cv2.imread(str(p)) is None:raise ValueError('PNG readback')
            checks['png']+=1
        elif p.suffix=='.mp4':
            video=cv2.VideoCapture(str(p));ok,_=video.read();video.release()
            if not ok:raise ValueError('MP4 readback '+str(p))
            checks['mp4']+=1
    parsed=Links();parsed.feed(page)
    for link in parsed.links:
        if not (root/link).is_file() and link!='verification.json':
            raise ValueError('broken HTML link '+link)
        checks['html_links']+=1
    verify=dict(status='PASS' if not status['invalid'] and all(x.get('pass_gate') for x in reproduction.values())
                else 'PARTIAL_PREFIX_OR_REPRODUCTION',checks=checks,
        valid_comparisons=len(results),invalid=status['invalid'],
        source_sha256={p:sha(root/p) for p in ('source_snapshot_branch.py','source_snapshot_report.py','source_snapshot_entry.py')},
        original_checkpoint_hashes={m:protocol['checkpoints'][m]['sha256'] for m in 'ABC'})
    atomic(root/'verification.json',verify,immutable=True)
    for p,kind in [('comparison.json','metrics'),('verification.json','metrics'),
                   ('error_accumulation_diagnostic.html','reports'),
                   ('08f_error_accumulation_diagnostic.md','reports')]:
        register_existing_output(root,root/p,kind=kind)
    return root/'08f_error_accumulation_diagnostic.md'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    print(create(p.parse_args().run))
