"""Append measured matrix and answers to the v4 technical report."""
import argparse
import json
from pathlib import Path


def successes(rate):
    return int(rate.split('/')[0])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/graph/v4'));p.add_argument('--output',type=Path,default=Path('docs/graph_prototype_v4.md'));args=p.parse_args()
    result=json.loads((args.root/'comparison/results.json').read_text());rows=result['rows'];lookup={(r['encoder'],r['history']):r for r in rows}
    if len(rows)!=8 or any(r['offline'] is None or 'N/A' in [r['heldout_full'],r['fresh_full'],r['alignment']] for r in rows):raise ValueError('complete main matrix required')
    lines=['<!-- MEASURED_RESULTS -->','## 완료된 실험 결과','',
        '| Encoder | History | Action MSE | Full SR held-out | Full SR fresh | Alignment-start SR | Params | CPU cached ms |',
        '|---|---|---:|---|---|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['encoder']} | {r['history']} | {r['offline']['action_mse']:.8f} | {r['heldout_full']} | {r['fresh_full']} | {r['alignment']} | {r['params']} | {r['latency']['total_ms']:.3f} |")
    lines+=['','Full held-out 9개와 fresh 30개는 별도 조건이다. Alignment-start는 expert prefix 이후 9개 episode의 task success이며 full-task alignment reached와 구분한다. Latency는 idle host, CPU4 threads, batch1, 100 repeats, cache commit 포함/GT extraction 및 simulator step 제외다.','', '### 질문에 대한 답변','']
    reductions=[]
    for name in ['flat','surface']:
        base=lookup[name,'current']['offline']['action_mse']
        reductions.append(name+': '+', '.join(f"{mode} {(lookup[name,mode]['offline']['action_mse']/base-1)*100:+.1f}%" for mode in ['action','state','both']))
    lines+=['1. **Offline error:** Current 대비 MSE 변화는 '+ '; '.join(reductions)+'. 음수는 감소다. Expert-history 조건의 held-out error다.']
    improved=[]
    for r in rows:
        if r['history']=='current':continue
        base=lookup[r['encoder'],'current']
        for condition in ['heldout_full','fresh_full']:
            if successes(r[condition])>successes(base[condition]):improved.append(f"{r['encoder']}/{r['history']} {condition} {r[condition]} (Current {base[condition]})")
    lines+=['2. **Closed-loop:** '+('Full success 증가 조건: '+', '.join(improved)+'. 단일 seed의 paired 결과이며 일반적 우위나 통계적 유의성을 확정하지 않는다.' if improved else 'Full-task success 개선은 확인되지 않았다. Held-out/fresh 조건에서 모든 모델이 각각 0/9, 0/30이다.')+' Warm-start 결과는 위 표와 비교표에 별도로 기록했다.']
    best=[]
    for name in ['flat','surface']:
        best_row=min([lookup[name,m] for m in ['action','state','both']],key=lambda r:r['offline']['action_mse'])
        best.append(f"{name}의 최저 offline MSE는 {best_row['history']} ({best_row['offline']['action_mse']:.8f})")
    lines+=['3. **State vs Action:** '+ '; '.join(best)+'. 이것만으로 closed-loop에서 더 효과적이라고 결론 내리지 않는다. Head input과 parameter count도 함께 변했다.']
    differences=[]
    for mode in ['current','action','state','both']:
        change=(lookup['surface',mode]['offline']['action_mse']/lookup['flat',mode]['offline']['action_mse']-1)*100
        differences.append(f'{mode} {change:+.1f}%')
    lines+=['4. **Surface vs Flat:** 같은 history 조건에서 Surface의 offline MSE 변화(Flat 대비)는 '+', '.join(differences)+'. Closed-loop 우열은 별도의 success/단계별 실패 결과로 판단해야 하며 이 소규모 실험으로 일반적인 encoder 우위를 확정하지 않는다.']
    failures=[]
    for r in rows:
        f=r['stages']['heldout_full']['failure_counts']
        failures.append(f"{r['encoder']}/{r['history']}: "+', '.join(f'{k} {v}' for k,v in f.items() if v))
    lines+=['5. **남은 실패 단계:** Full held-out 실패는 '+ '; '.join(failures)+'. 전체 warm-start/fresh failure counts는 comparison에 있다. Trace는 gripper command, contact, applied action 변화와 state 분포를 남겼지만 어느 문제가 실패의 원인인지 인과적으로 분리한 실험은 아니다.']
    lines+=['6. **GRU 근거:** History가 offline 예측에 도움이 되는 결과는 temporal 입력을 후속 연구할 근거다. 그러나 고정 window만으로 full-task 제어가 해결됐다는 근거가 없으므로 GRU 도입이 다음 병목을 해결한다고 결론 내릴 수 없다. BC의 expert/learner history 분포 차이, phase ambiguity, grasp retention 및 gripper/controller 문제를 분리해 확인하고 parameter-matched Current와 복수 seed를 비교하는 것이 후속 후보다.','',
        '### 진단과 검증','']
    for name in ['flat','surface']:
        r=lookup[name,'both']
        lines.append(f"- {name}/Both ablation: offline MSE {r['offline']['action_mse']:.8f} → {r['offline_ablated']['action_mse']:.8f}; full {r['heldout_full']} → {r['ablated_full']}; alignment-start {r['alignment']} → {r['ablated_alignment']}. 현재 representation 반복/zero action 입력은 OOD일 수 있다.")
    preserve=result['preservation']
    lines+=['- v4 tests7 + 기존 regression42 = 49 tests 통과. 별도 실제 train demonstration 8개 설정의 200-step small-batch overfit도 모두 통과했다.',
        f"- Paired initializations: {result['paired_initializations']}; learner history = 실제 applied command: {result['learner_history_matches_applied_commands']}; 이전 output artifacts SHA256 보존: {preserve['unchanged']}/{preserve['checked']}.",
        '- Current baseline은 offline prediction과 held-out 4조건의 전체 applied command sequence까지 기존 v2와 정확히 일치했다.',
        '- 성공/실패 Flat Both alignment video는 기록된 applied commands를 다시 실행해 EE/peg pose maximum error 0으로 검증했다.','',
        '### 수정 파일과 미실행 범위','',
        '`graph/prototype/v4/`에 history dataset/buffer/models, config, train/evaluate, baseline import, matrix driver, compare, distribution/latency, report/figures/video, overfit 및 tests를 추가했다. 이 보고서 외 기존 shared/Visual/Recurrent/v1/v2/v3 코드는 수정하지 않았다. Checkpoints, traces, metrics, timing, figures/videos는 `outputs/graph/v4/`에 있다.','',
        '필수 K4 matrix의 학습/offline/full/warm-start/fresh/Both ablation은 완료했다. K8/K16 성능, v3 graph+history, parameter-matched Current, 복수 seed 및 GRU 비교는 후속 실험으로 남긴다. 이번 최소 matrix는 K4로 고정했으며 window 길이나 test-informed 설정을 확장하지 않았다. CUDA는 이 호스트에서 unavailable이며 CUDA backend는 실행하지 않았다. Surface State/Both의 일부 학습은 사용자 요청으로 MPS에서 재개했으며 CPU/MPS backend 차이와 deterministic warn_only 제약을 함께 고려해야 한다. 새로운 초기화에서의 history ablation은 필수 진단 범위에 포함하지 않았고 held-out full/alignment 조건에서만 실행했다.']
    marker='<!-- MEASURED_RESULTS -->'
    existing=args.output.read_text() if args.output.exists() else '# Graph Prototype v4 results\n'
    prefix=existing.split(marker)[0].rstrip()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(prefix+'\n\n'+'\n'.join(lines)+'\n')


if __name__=='__main__':main()
