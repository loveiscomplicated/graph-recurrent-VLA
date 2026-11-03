"""Offline independent timing/cost/hash checks and final diagnostic report."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from zoneinfo import ZoneInfo
import run


def own_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_stats(jobs):
    xs = sorted(j['latency_ms']/1000 for j in jobs)
    if not xs: return {'n':0}
    q = (len(xs)-1)*.95
    lo = int(q)
    hi = min(lo+1,len(xs)-1)
    return dict(n=len(xs), mean_s=statistics.mean(xs), median_s=statistics.median(xs),
                p95_s=xs[lo]+(xs[hi]-xs[lo])*(q-lo), min_s=xs[0], max_s=xs[-1])


def main():
    root = run.ROOT
    state = run.read(root/'request_ledger.json')
    summary = run.read(root/'latency_summary.json')
    protocol = run.read(root/'protocol.json')
    freeze = run.read(root/'api_freeze.json')
    assert own_sha(root/'request_ledger.json') == freeze['ledger_sha256']
    protected = run.read(root/'protected_before.json')
    for name, expected in protected.items():
        assert own_sha(run.REPO/name) == expected, name
    prompt = (root/'grounding_prompt.txt').read_text()
    manifest = run.read(root/'input_manifest.json')['images']
    jobs = list(state['jobs'].values())
    images = {im['id']:im for im in manifest}
    usage_cost = 0.
    unknown = 0.
    cached_input_tokens = 0
    for j in jobs:
        image = images[j['id']]
        assert own_sha(image['image_path']) == image['image_sha256']
        request = run.request_for(prompt, Path(image['image_path']).read_bytes())
        assert run.base.digest(request) == j['request_sha256']
        if j['status'] == 'RESERVED':
            unknown += j['accounted_usd']
            continue
        assert own_sha(root/j['raw_file']) == j['raw_sha256'] == freeze['raw_files'][j['raw_file']]
        raw = run.read(root/j['raw_file'])
        assert raw['status'] == j['status']
        assert raw['latency_ms'] == j['latency_ms']
        body = raw.get('raw') or {}
        assert body.get('service_tier') == j['served_tier']
        parsed = run.base.strict_parse(raw['final_text']) if raw['status'] == 'OK' else run.base.strict_parse('')
        assert parsed == j['parsed']
        u = body.get('usage')
        if u and u.get('input_tokens') is not None and u.get('output_tokens') is not None:
            multiplier = 1 if body.get('service_tier') == 'default' else 2
            cost = (u['input_tokens'] + 5*u['output_tokens'])*multiplier/1e6
            assert abs(cost-j['usage_estimated_usd']) < 1e-12
            usage_cost += cost
            cached_input_tokens += (u.get('input_tokens_details') or {}).get('cached_tokens',0)
        else:
            unknown += j['accounted_usd']
    fast = [j for j in jobs if j['status']=='OK' and j['served_tier'] in {'fast','priority'}]
    standard = [j for j in run.read(run.base.ROOT/'request_ledger.json')['jobs'].values()
                if j['provider']=='terra' and j['status']=='OK']
    for name, rows in [('fast_ok',fast),('standard_historical_ok',standard)]:
        recalc = independent_stats(rows)
        assert recalc['n'] == summary[name]['n']
        for k in recalc:
            if k!='n': assert abs(recalc[k]-summary[name][k]) < 1e-10, (name,k)
    assert abs(usage_cost-summary['usage_estimated_usd']) < 1e-10
    assert abs(unknown-summary['unknown_usage_reservation_usd']) < 1e-10
    assert len(jobs) <= 60
    assert protocol['prior_accounted_usd']+usage_cost+unknown <= 20
    # Read only local dotenv values, never persist or print their contents.
    from dotenv import dotenv_values
    secrets = [v for k,v in dotenv_values(run.REPO/'.env').items()
               if any(s in k for s in ['KEY','TOKEN','SECRET']) and v and len(v)>=8]
    scanned = 0
    for p in list(root.rglob('*')) + list(run.SOURCE.glob('*.py')):
        if p.is_file():
            data = p.read_bytes()
            assert not any(s.encode() in data for s in secrets), 'Credential leakage detected'
            scanned += 1
    kst = ZoneInfo('Asia/Seoul')
    start = datetime.fromisoformat(state['started_utc']).astimezone(kst)
    end = datetime.fromisoformat(state['finished_utc']).astimezone(kst)
    powerlog = subprocess.run(['pmset','-g','log'],capture_output=True,text=True,check=True).stdout
    transitions=[]
    for line in powerlog.splitlines():
        if not any(s in line for s in ['Entering Sleep','Wake from','DarkWake from']): continue
        try: stamp=datetime.strptime(line[:19],'%Y-%m-%d %H:%M:%S').replace(tzinfo=kst)
        except ValueError: continue
        if start <= stamp <= end: transitions.append(line)
    run.write(root/'host_sleep_check.json',dict(start_kst=start.isoformat(),end_kst=end.isoformat(),
        sleep_wake_transition_lines=transitions,source='pmset -g log; Entering Sleep/Wake from/DarkWake from during interval',
        clock_discrepancies=summary['timing_clock_discrepancies']))
    verification = dict(status='PASS', verified_utc=run.now(), requests=len(jobs),
        protected_files_verified=len(protected), raw_files_verified=len(freeze['raw_files']),
        input_and_request_hashes_verified=len(jobs), independent_timing_and_cost_recalculation=True,
        no_new_GT_or_SAM2=True, secret_scan_files=scanned,
        cached_input_tokens=cached_input_tokens, sleep_wake_transitions=len(transitions))
    run.write(root/'independent_verification.json',verification)
    std, fst = summary['standard_historical_ok'],summary['fast_ok']
    def cells(s):
        return f"{s['n']} | {s['mean_s']:.3f} | {s['median_s']:.3f} | {s['p95_s']:.3f} | {s['min_s']:.3f}–{s['max_s']:.3f}" if s['n'] else '0 | N/A | N/A | N/A | N/A'
    ratio = std['median_s']/fst['median_s'] if fst['n'] else None
    reduction = (1-fst['median_s']/std['median_s'])*100 if fst['n'] else None
    first = summary['fast_first']
    later = summary['fast_after_first']
    first_note = f"{first['median_s']:.3f}초" if first['n'] else 'N/A'
    later_note = (f"n={later['n']}, 평균 {later['mean_s']:.3f}초, 중앙값 {later['median_s']:.3f}초, p95 {later['p95_s']:.3f}초"
                  if later['n'] else 'N/A')
    paired_ratio = (f"{summary['paired_median_standard_over_fast']:.3f}배"
                    if summary['paired_n'] else 'N/A')
    lead = (f"Terra Fast {fst['n']}건의 HTTP E2E 중앙값은 **{fst['median_s']:.3f}초**였다. "
            f"기존 Standard {std['median_s']:.3f}초와 비교하면 중앙값 기준 **{ratio:.2f}배**, **{reduction:.1f}% 감소**다.") if fst['n'] else 'Fast tier의 정상 완료 표본을 확보하지 못했다.'
    rel = '../../'+str(root.relative_to(run.REPO))
    text = f'''# Task 10-I Stage 4A-2: Terra Fast API 소요 시간

{lead}
이는 같은 알려진 60장에 대한 **서로 다른 시점의 관측 비교**다. Tier만의 인과적 속도 향상이나 실시간 제어 성능을 뜻하지 않는다.

## 승인 범위와 고정 계약

- 사용자 요청: “terra fast의 소요 시간을 측정하고 싶어.” 기존 Terra Standard의 원본 60장과 동일한 순서·Prompt·HTTP Adapter를 재사용했다. Standard 유료 재실행은 0회다.
- Train Parent 15개 × 초기/후기 × Main/Wrist, native512 PNG. 미사용 Parent 일반화 실험이 아니다.
- 모델 `gpt-6.1-terra`, Responses API, `reasoning.effort=high`, `detail=original`, `max_output_tokens=5120`, `store=False`. 유일한 요청 변경은 `service_tier=fast`다.
- Prompt SHA256 `{protocol['prompt_sha256']}`. 한 요청은 전체 RGB 한 장과 영어 외형 Prompt뿐이다. 이전 대화·GT·DINO·SAM2·Parent ID·도구·Schema 강제 없음.
- 순차 요청, 동시성 1, Warm-up 0회, 자동 Retry/실패 대체 0회. 응답 tier 미확인, 종료 오류, 연속 서비스 오류 3회 또는 예산 위반 시 중단한다. Durable RESERVED 요청은 재전송하지 않는다.
- 기존 `adapters.payload/call`과 엄격한 좌표 파서, 공통 `ExperimentLogger`(W&B disabled)를 재사용했다. API 응답·Ledger·요청과 이미지 SHA를 저장하고 동결했다.
- `caffeinate -i -w PID`는 측정 프로세스 수명에만 적용했다. 영구 전원 설정은 변경하지 않았다.

## 실제 요청과 지연

실행: **{start.strftime('%Y-%m-%d %H:%M:%S')}–{end.strftime('%H:%M:%S')} KST**. 측정 루프 전체 경과 **{summary['elapsed_s']:.2f}초 ({summary['elapsed_s']/60:.2f}분)**. 상태 `{summary['status']}`, 중단 사유 `{summary['stopped']}`.

| 경로 | 정상 완료 n | 평균 s | 중앙값 s | p95 s | 최소–최대 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기존 Standard (`default`) | {cells(std)} |
| 이번 Fast (`fast`/`priority` 확인) | {cells(fst)} |

- 요청 상태: `{summary['statuses']}`. 실제 응답 tier: `{summary['served_tiers']}`.
- Fast 첫 요청 {first_note}. 첫 요청 제외: {later_note}. 첫 요청을 Cold Start라고 단정하지 않는다.
- 대응 이미지 {summary['paired_n']}장 중 Fast가 더 짧은 이미지 {summary['paired_faster_count']}장. 이미지별 Standard/Fast 비율의 중앙값 {paired_ratio}. 이는 전체 중앙값의 비와 다른 통계다.
- 엄격한 JSON 성공 {summary['strict_json_valid']}/{len(jobs)}, null 허용 좌표 계약 통과 {summary['coordinate_contract_including_null']}/{len(jobs)}. 좌표 수정·Clipping·Box 확대 없음. **이번 Fast Box의 Grounding/Mask 정확도는 채점하지 않았다.**
- 기록된 입력 캐시 토큰 합계 {cached_input_tokens}. 캐시 상태와 출력·추론 토큰 수가 달라질 수 있으며 이미지별 출력 토큰 수는 대응 CSV에 보존했다.
- 호스트 전원 로그의 측정 구간 Sleep/Wake 전환 {len(transitions)}건, wall clock와 perf_counter 차이 2초 초과 요청 {summary['timing_clock_discrepancies']}건.

## 비용과 공식 조건

[공식 Fast 문서](https://developers.openai.com/api/docs/guides/fast-mode)는 요청 `service_tier=fast` 설정 및 응답 tier 확인을 안내한다. [Terra 공식 모델 문서](https://developers.openai.com/api/docs/models/gpt-6.1-terra)는 high 추론·이미지 입력을 지원하고 Fast 단가를 Standard의 2배로 안내한다. 2026-11-03 KST 확인 기준 Fast 입력 $2 / 출력 $10, Standard 입력 $1 / 출력 $5 (100만 토큰당)로 동결했다.

- 이번 Usage 기반 추정 **${summary['usage_estimated_usd']:.6f}**, Usage 미확인 예약 **${summary['unknown_usage_reservation_usd']:.6f}**, 보수 합계 **${summary['conservative_usd']:.6f}**.
- 기존 누적 **${protocol['prior_accounted_usd']:.8f}**에 이번 비용을 더한 누적 **${summary['cumulative_conservative_usd']:.8f}/$20**. 이전 3개 주 Ledger와 최초 이미지 반복 Ledger를 한 번씩 합산했다.
- 사전 신규 60회 최대 예약액 ${protocol['additional_reserved_max_usd']:.5f}. 실제 추정 지출과 구분한다. 캐시 할인을 적용하지 않은 Usage 추정이며 확정 청구서가 아니다. Usage 미확인 요청은 최대 예약을 유지한다. 실제 default 강등 응답은 Standard 단가로 추정한다.
- 신규 GPU 사용은 0회이며 기존 3090 사용료는 이 API 계상에 포함하지 않는다.

## 해석·검증과 범위 보존

HTTP Adapter는 비스트리밍 단일 `requests.post` 시작부터 응답 JSON 파싱까지를 `perf_counter`로 측정한다. Payload 이미지 인코딩·로컬 저장·SAM2는 개별 요청 지연에서 제외한다. 기존과 같이 지속 HTTP Session은 사용하지 않아 연결 설정·네트워크 지연을 포함한다. Vision/Prefill/Decode·TTFT는 분리 관측하지 못했다.

기존 Standard는 Terra/Luna를 교대로 호출한 실행이고 이번에는 Terra Fast만 순차 호출했다. 호출 시각·부하·캐싱·출력 길이·확률적 추론량이 다를 수 있어 관측 비율을 Fast tier만의 효과로 확정하지 않는다. 이 비교를 위해 추가 Standard 호출을 실행하지 않았다. SAM2 초기화·Tracking·Hole 정밀도·Object Graph·Closed-loop 품질로 확장해 주장하지 않는다.

오프라인 보호 검사 5개 PASS. 별도 Python statistics/선형 보간 p95 및 raw Usage 산식으로 시간·비용을 독립 재계산했다. 기존 산출물·Source·PAUSED_SCOPE_REVIEW 자료 {len(protected)}개 SHA, 이번 원시 응답 {len(freeze['raw_files'])}개 SHA, 입력 및 요청 Hash {len(jobs)}개를 대조했다. 비밀 검사 {scanned}개 파일 PASS. 이는 과거 SAM2 정확도를 재실행한 검사가 아니다.

변경 범위는 신규 격리 실험 코드·출력과 이 보고서다. Production·기존 결과·공통 Logger/Publisher·PAUSED_SCOPE_REVIEW·Stop Stub은 유지했다. 로컬 macOS에서 API 측정과 CPU 분석만 실행했다. CUDA/MPS, 원격 Vast.ai, SAM2, 신규 Simulator Render/Rollout, Graph 생성, 학습, Controller, Commit, 외부 게시를 실행하지 않았다. DINO 제거와 Production 준비 판정은 여전히 INSUFFICIENT_SUPPORT다.

산출물: [요약 JSON]({rel}/latency_summary.json) · [대응 이미지 CSV]({rel}/paired_latency.csv) · [요청 Ledger]({rel}/request_ledger.json) · [실행 계약]({rel}/protocol.json) · [독립 검증]({rel}/independent_verification.json) · [호스트 전원 검사]({rel}/host_sleep_check.json) · [Artifact Index]({rel}/artifact_index.json).
기존 비교: [OpenAI 계열 최종 보고서](10i_stage4a2_api_openai_family_grounding_comparison.md).
'''
    run.REPORT.write_text(text)
    paths = [p for p in root.rglob('*') if p.is_file() and '.execution' not in p.parts and p.name!='artifact_index.json']
    index = {str(p.relative_to(run.REPO)):own_sha(p) for p in sorted(paths)}
    index[str(run.REPORT.relative_to(run.REPO))] = own_sha(run.REPORT)
    index.update({str(p.relative_to(run.REPO)):own_sha(p) for p in run.SOURCE.iterdir() if p.is_file()})
    run.write(root/'artifact_index.json',dict(created_utc=run.now(),files=index))
    print(json.dumps(verification))
    print('REPORT',run.REPORT)


if __name__=='__main__': main()
