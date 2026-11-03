"""Post-freeze supplementary diagnostics and user-requested timing repeat receipt."""
from common import *
import numpy as np
from PIL import Image
from adapters import payload
from evaluate import box_iou, bbox, rate

def main():
    v=verify_frozen();assert v['stage']=='COMPLETE'
    if not (ROOT/'remote_preservation_verification.json').exists():
        from audit_remote import main as audit_remote
        audit_remote()
    jobs=read(ROOT/'request_ledger.json')['jobs']
    cases=read(ROOT/'evaluation_manifest.json')['images']
    diagnostic=[]
    primary_metrics=read(ROOT/'grounding_evaluation_metrics.json')['models']
    for c in cases:
        with np.load(c['GT_path'],allow_pickle=False) as z:seg=z['actor_segmentation'].copy()
        for p in MODELS:
            j=jobs.get(p+'__'+c['id'])
            if j:
                expected=digest(payload(p,(ROOT/'grounding_prompt.txt').read_text(),Path(c['image_path']).read_bytes()))
                assert expected==j['request_sha256']
            for o in ['peg','box']:
                target=np.isin(seg,c['object_ids'][o]);pixels=int(target.sum());gt=bbox(target)
                recovery=j.get('diagnostic',{}).get(o,{}) if j else {}
                b=recovery.get('pixels')
                iou=box_iou(b,gt)
                strict=j.get('parsed',{}).get(o,{}) if j else {}
                diagnostic.append(dict(provider=p,image_id=c['id'],object=o,visible_pixels=pixels,
                    attempted=j is not None,api_completed=j is not None and j['status']=='OK',
                    strict_valid=strict.get('status')=='VALID',diagnostic_valid=recovery.get('status')=='VALID',
                    diagnostic_IoU=iou,diagnostic_box05=bool(pixels>=16 and iou>=.5),
                    primary_acceptance_unchanged=True))
    metrics={p:{o:dict(visible_box05=rate([r for r in diagnostic if r['provider']==p and r['object']==o and r['visible_pixels']>=16],'diagnostic_box05'),
                           diagnostic_boxes=sum(r['diagnostic_valid'] for r in diagnostic if r['provider']==p and r['object']==o),
                           strict_boxes=sum(r['strict_valid'] for r in diagnostic if r['provider']==p and r['object']==o))
                  for o in ['peg','box']} for p in MODELS}
    completed_diagnostics={p:{o:rate([r for r in diagnostic if r['provider']==p and r['object']==o
                           and r['visible_pixels']>=16 and r['api_completed']],'diagnostic_box05')
                           for o in ['peg','box']} for p in MODELS}
    write(ROOT/'diagnostic_coordinate_metrics.json',dict(models=metrics,
        completed_API_only=completed_diagnostics,
        warning='Post-hoc JSON extraction from final text only; never accepted as primary grounding or used in SAM2.'))
    csvwrite(ROOT/'diagnostic_coordinate_cases.csv',diagnostic)
    repeat=read(ROOT/'first_request_repeat/latency_comparison.json')
    extras=read(ROOT/'first_request_repeat/ledger.json')['jobs']
    first=read(ROOT/'input_manifest.json')['images'][0]
    for p,j in extras.items():
        assert j['request_sha256']==jobs[p+'__'+first['id']]['request_sha256']
        assert j['raw_sha256']==sha(ROOT/'first_request_repeat'/(p+'_raw.json'))
    combined=ledger_total(jobs)+ledger_total(extras)
    assert combined<=20 and len(jobs)+len(extras)<=183
    timing=[]
    timing_summary={}
    for p,c in MODELS.items():
        a=repeat['models'][p]
        timing.append(f"| {c['model']} | {a['first_seconds']:.2f}s | {a['repeat_seconds']:.2f}s | {a['repeat_status']} |")
        timing_summary[p]={}
        selected=[j for j in jobs.values() if j['provider']==p and 'latency_ms' in j]
        for status in sorted({j['status'] for j in selected}):
            values=np.asarray([j['latency_ms']/1000 for j in selected if j['status']==status])
            timing_summary[p][status]=dict(n=len(values),mean_s=float(values.mean()),
                median_s=float(np.median(values)),p95_s=float(np.quantile(values,.95)),max_s=float(values.max()))
    starts=[datetime.fromisoformat(j['started_utc']) for j in jobs.values()]
    ends=[datetime.fromisoformat(j['finished_utc']) for j in jobs.values() if 'finished_utc' in j]
    api_wall_s=(max(ends)-min(starts)).total_seconds() if starts and ends else None
    pipeline_wall_s=(datetime.fromisoformat(v['completed_utc'])-datetime.fromisoformat(v['prepared_utc'])).total_seconds()
    write(ROOT/'observed_timing_summary.json',dict(primary_by_provider_and_status=timing_summary,
        primary_inference_wall_s=api_wall_s,prepared_to_first_completed_report_wall_s=pipeline_wall_s,
        first_repeat=repeat,server_model_cold_load_not_observed=True))
    sleep_info=read(ROOT/'host_sleep_overlap.json') if (ROOT/'host_sleep_overlap.json').exists() else None
    sleep_table=[];clean_stats={}
    if sleep_info:
        for p,c in MODELS.items():
            values=np.asarray([j['monotonic_HTTP_s'] for j in sleep_info['jobs']
                               if j['provider']==p and j['status']=='OK' and j['sleep_overlap_s']==0])
            clean_stats[p]=dict(n=len(values),mean_s=float(values.mean()),median_s=float(np.median(values)),
                                 p95_s=float(np.quantile(values,.95))) if len(values) else {'n':0}
            if len(values):sleep_table.append(f"| {c['model']} | {len(values)} | {values.mean():.2f}s | {np.median(values):.2f}s | {np.quantile(values,.95):.2f}s |")
        timing_doc=read(ROOT/'observed_timing_summary.json')
        timing_doc.update(runtime_status='HOST_SLEEP_CONTAMINATED',no_sleep_OK_latency=clean_stats,
            observed_host_sleep_s=sleep_info['experiment_sleep_s'],
            measured_primary_HTTP_duration_sum_s=sum(j['monotonic_HTTP_s'] for j in sleep_info['jobs']))
        write(ROOT/'observed_timing_summary.json',timing_doc)
    diagnosis=[]
    for p,c in MODELS.items():
        a=metrics[p]
        fmt=lambda r:f"{r['numerator']}/{r['denominator']} ({100*r['rate']:.1f}%)" if r['denominator'] else 'N/A'
        diagnosis.append(f"| {c['model']} | {a['peg']['strict_boxes']} / {a['peg']['diagnostic_boxes']} | {fmt(a['peg']['visible_box05'])} | {fmt(a['box']['visible_box05'])} |")
    fmt=lambda a:f"{a['numerator']}/{a['denominator']} ({100*a['rate']:.1f}%)" if a['denominator'] else 'N/A (분모 0)'
    breakdown=[];object_table=[];json_table=[];conditional_table=[];cost_table=[];service_errors=[]
    for p,c in MODELS.items():
        a=primary_metrics[p]
        breakdown.append(f"| {c['model']} | {fmt(a['by_camera']['base_camera']['scene']['both_mask08'])} | {fmt(a['by_camera']['hand_camera']['scene']['both_mask08'])} | {fmt(a['by_time']['F0']['scene']['both_mask08'])} | {fmt(a['by_time']['F1']['scene']['both_mask08'])} |")
        jj=[j for j in jobs.values() if j['provider']==p]
        valid_json=sum(j.get('parsed',{}).get('json_valid',False) for j in jj)
        json_table.append(f"| {c['model']} | {valid_json}/{len(jj)} | {fmt(a['overall']['peg']['validity_attempted'])} | {fmt(a['overall']['peg']['abstention'])} | {fmt(a['overall']['box']['abstention'])} |")
        conditional_table.append(f"| {c['model']} | {fmt(completed_diagnostics[p]['peg'])} | {fmt(completed_diagnostics[p]['box'])} |")
        succeeded=a['overall']['scene']['both_mask08']['numerator']
        known_cost=sum(j.get('usage_estimated_usd') or 0 for j in jj)
        unknown=sum(j['accounted_usd'] for j in jj if j.get('usage_estimated_usd') is None)
        per_success=f"${sum(j['accounted_usd'] for j in jj)/succeeded:.5f}" if succeeded else 'N/A (주 성공 0건)'
        cost_table.append(f"| {c['model']} | ${known_cost:.6f} | ${unknown:.6f} | {per_success} |")
        for j in jj:
            if j['status'] in ['TIMEOUT','CONNECTION_ERROR','SERVICE_ERROR','AUTH_ERROR','REQUEST_ERROR','MODEL_UNAVAILABLE']:
                service_errors.append(f"- {c['model']} / {j['image_id']}: `{j['status']}`, {j.get('latency_ms',0)/1000:.2f}초. 재시도 없음.")
        for o in ['peg','box']:
            z=a['overall'][o]
            object_table.append(f"| {c['model']} / {o} | {fmt(z['box05'])} | {fmt(z['mask08'])} | {fmt(z['wrong_object'])} | {z['mask_IoU_including_misses']['mean']:.3f} |")
    old_ids={c['id'] for c in read(OLD/'input_manifest.json')['cases']}
    assert old_ids=={c['id'] for c in cases}
    baseline=read(OLD/'sam2_initialization_metrics.json')['methods']['A']['per_parent']
    parents=sorted(baseline);rng=np.random.default_rng(42);draws=rng.integers(0,len(parents),size=(10000,len(parents)))
    historical={};historical_table=[]
    for p,c in MODELS.items():
        deltas=np.array([primary_metrics[p]['per_parent'][pid]['scene']['both_mask08']['rate']-
                          baseline[pid]['scene']['both_mask08']['rate'] for pid in parents])
        ci=np.quantile(deltas[draws].mean(axis=1),[.025,.975])
        historical[p]=dict(parent_n=len(parents),macro_delta=float(deltas.mean()),CI95=ci.tolist())
        historical_table.append(f"| {c['model']} − 기존 DINO+Qwen A | {100*deltas.mean():+.2f} pp | [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] pp |")
    write(ROOT/'matched_historical_baseline_comparison.json',dict(comparison=historical,
        matched_images=60,metric='parent-macro both Mask IoU>=0.8',
        warning='Operational pipeline comparison, not model-size-only or detector-only causal effect.'))
    appendix=f'''

## 7. 사용자 요청 — 첫 이미지 1회 추가 반복과 소요 시간

주 평가의 이미지당 1회 계약은 그대로 유지하고, 사용자의 후속 지시에 따라 첫 이미지에 모델별 1회씩 **추가 {len(extras)}요청**을 실행했다. 입력 PNG·Prompt·요청 설정의 SHA가 최초 요청과 정확히 같다. 추가 응답은 주 정확도·SAM2 표본에 포함하지 않았다.

| 모델 | 최초 요청 왕복 시간 | 같은 입력 1회 반복 | 반복 상태 |
|---|---:|---:|---|
{chr(10).join(timing)}

각 조건 n=1이며 반복 진단 중 주 평가도 실행 중이었다. 서비스 캐시·서버 부하·네트워크·추론 길이가 달라질 수 있으므로 순수한 Cold/Warm 효과나 일반 Latency 분포로 단정하지 않는다. 상세 Usage 및 실제 생성 토큰은 `first_request_repeat/latency_comparison.json`에 기록했다.

주 평가 첫 요청 시작부터 마지막 응답 저장까지의 실제 경과 시간은 **{api_wall_s/60:.2f}분**, 준비 동결부터 첫 완료 보고서 생성까지는 **{pipeline_wall_s/60:.2f}분**이다. 최초 호출은 클라이언트의 첫 요청이며 서버 모델 Cold Load가 관측됐다는 뜻은 아니다. 정상 응답과 TOKEN_LIMIT/서비스 오류별 Mean·Median·p95·최댓값은 `observed_timing_summary.json`에 분리했다.

주 평가 {len(jobs)}요청 + 추가 {len(extras)}요청 = 총 **{len(jobs)+len(extras)}요청**이다. 추가 진단을 포함한 보수적 예산 계상액은 **${combined:.6f} / $20**이며, 청구 확정값은 아니다.

## 8. Strict JSON과 사후 좌표 진단의 구분

최종 응답이 Markdown Fence 등으로 Strict JSON에 실패한 경우, 응답에 좌표가 있는지 별도로 분석했다. 이 값은 **사후 진단**이고 SAM2에 전달하거나 주 성공률에 합산하지 않았다. 추론 텍스트에서 좌표를 추출하지 않았다.

| 모델 | Peg Strict 유효 Box / 진단 유효 Box | 진단 Peg Box IoU≥.5 | 진단 Fixture Box IoU≥.5 |
|---|---:|---:|---:|
{chr(10).join(diagnosis)}

진단 Box 지표는 최종 SAM2 Mask 성능을 대신하지 않는다. 형식 위반과 시각적 위치 오류를 분리해 해석하기 위한 자료다.

미시도·서비스 오류·TOKEN_LIMIT를 제외하고 정상 최종 응답에 조건화한 진단 Box 결과도 별도로 보고한다. 이 조건부 값은 전체 운영 성공률을 대신하지 않는다.

| 모델 | 정상 완료 응답의 진단 Peg Box IoU≥.5 | 정상 완료 응답의 진단 Fixture Box IoU≥.5 |
|---|---:|---:|
{chr(10).join(conditional_table)}

배포 단계에서 원격 Python Package 깊이를 유지하도록 Launcher만 수정한 내역을 `verification.json`의 `source_amendments`에 전후 SHA와 함께 기록했다. 진행 중인 API Adapter·Prompt·모델 설정·입력은 변경하지 않았다. 추가 반복·사후 진단·오프라인 상태 테스트의 Source는 `source_additions`에 별도 기록했다.

원격 SAM2 Checkpoint·설치 Source 26개 파일과 Package 138개의 전후 보존 검사도 PASS했다. 실제 전후 목록은 `remote_preservation_verification.json`에 기록했다.

## 9. 카메라·시점·오인식 분해 및 동일 이미지 기준 비교

아래는 주 평가의 두 물체 동시 Mask IoU≥.8 성공률이다.

| 모델 | Main | Wrist | 초기 F0 | 후기 F1 |
|---|---:|---:|---:|---:|
{chr(10).join(breakdown)}

| 모델 / 물체 | Box IoU≥.5 | Mask IoU≥.8 | Wrong-object / 비어 있지 않은 예측 | Mask IoU 평균(누락 포함) |
|---|---:|---:|---:|---:|
{chr(10).join(object_table)}

| 모델 | Strict JSON / 실제 시도 | Peg 유효 좌표 또는 Null | Peg Null | Fixture Null |
|---|---:|---:|---:|---:|
{chr(10).join(json_table)}

기존 DINO+Qwen A와 동일 60개 Image ID를 확인했다. 아래 차이는 Parent-macro 동시 Mask 성공률 기준이며, 15 Parent를 Cluster로 Bootstrap 10,000회/seed42로 집계했다. 원시 응답 형식과 운영 실패를 포함한 경로 비교이지 모델 크기만의 효과를 뜻하지 않는다.

| 비교 | Parent-macro 차이 | 95% CI |
|---|---:|---:|
{chr(10).join(historical_table)}

| 모델 | Usage 확인 요청의 비용 추정 | Usage 미확인 예약액 | 주 초기화 성공 1건당 보수적 계상액 |
|---|---:|---:|---:|
{chr(10).join(cost_table)}

성공 1건당 계상액은 Usage 미확인 요청의 최대 예약액까지 포함한 운영 지표이며 Invoice 청구 확정값이 아니다. 추가 첫 요청 반복은 위 주 평가 성공당 비용에서 제외하고 전체 예산에는 포함했다.

실제 서비스 오류 관측:

{chr(10).join(service_errors) if service_errors else '서비스 오류 없음.'}
'''
    if sleep_info:
        appendix+=f'''

## 10. 중요 실행 제한 — HOST_SLEEP_CONTAMINATED

macOS `pmset -g log`에서 실행 중 Sleep/Wake/DarkWake 시각을 읽기 전용으로 확인했다. 실험과 겹치는 Sleep 구간의 합계는 **{sleep_info['experiment_sleep_s']/60:.2f}분**, Sleep과 겹친 요청은 **{sum(j['sleep_overlap_s']>0 for j in sleep_info['jobs'])}개**다. 연결 오류/타임아웃 **6개 모두 Sleep 구간과 겹쳤다.** 해당 오류의 직접 원인이 전부 절전이라고 확정할 수는 없지만, 이번 기록으로 API 제공자의 가용성을 분리 평가할 수 없다. GLM 중단 역시 이 실행 환경의 영향을 포함한다.

UTC 기준 경과 {api_wall_s/60:.2f}분과 monotonic 기반 HTTP 측정 합계 {sum(j['monotonic_HTTP_s'] for j in sleep_info['jobs'])/60:.2f}분을 구분한다. 전체 경과 시간을 모델의 순수 추론 시간으로 사용하지 않는다. `host_sleep_overlap.json`에 요청별 UTC 경과·monotonic 측정·Sleep 겹침을 보존했다. 운영 실패를 포함한 주 정확도는 원래대로 유지하며, 정상 응답의 시각 정확도와 분리한다.

Sleep과 겹치지 않은 정상 최종 응답만의 소요 시간:

| 모델 | n | Mean | Median | p95 |
|---|---:|---:|---:|---:|
{chr(10).join(sleep_table)}

사용자가 추가 요청한 최초/반복 3쌍은 모두 Sleep 겹침 0초다. 그 비교는 유지하지만 각 n=1이고 주 평가와 동시 실행돼 순수 Warm-up 효과로 해석하지 않는다.

로컬 장시간 실행에 절전 방지를 적용하지 않은 실행 관리상의 누락을 확인했다. 측정 종료 후 Launcher에 프로세스 수명 동안의 `caffeinate -i -w PID` 보호를 추가했다. Dry-run에는 적용하지 않고, 영구 전원 설정은 바꾸지 않는다. Lid 닫기·수동 Sleep 방지까지 보장하지 않는다. 비정상 무한 좌표의 저장 실패도 방어했으며, 저장된 149개 원시 응답의 파싱 결과가 모두 동일함을 확인했다. 이 보강 뒤 유료 API 평가를 재실행하거나 실패 응답을 대체하지 않았다.
'''
    # Label API success separately from the strict JSON contract on the scientific figure.
    from PIL import ImageDraw
    figpath=ROOT/'full_image_grounding_comparison.png'
    fig=Image.open(figpath).convert('RGB');draw=ImageDraw.Draw(fig)
    preview=read(ROOT/'input_manifest.json')['images'][:6]
    picks={(a['provider'],a['image_id']):a for a in read(ROOT/'full_image_grounding_predictions.json')['rows']}
    for y,c in enumerate(preview):
        for x,p in enumerate(['RGB',*MODELS]):
            status='ORIGINAL'
            if p!='RGB':
                z=picks[p,c['id']]
                status=z['status'] if z['status']!='OK' else 'OK_JSON' if z['parsed']['json_valid'] else 'JSON_INVALID'
            draw.rectangle((x*300,y*330,x*300+299,y*330+29),fill='white')
            draw.text((x*300+3,y*330+1),p+' / '+status,fill='black')
            draw.text((x*300+3,y*330+15),c['id'].replace('_native512',''),fill='black')
    fig.save(figpath)
    for path in [REPORT,ROOT/'10i_stage4a2_api_full_image_grounding_comparison.md']:
        text=path.read_text().replace('API 요청은 최대 180개 중','주 평가 API 요청은 최대 180개 중')
        text='\n'.join(line.split(' 응답:')[0]+' 최종 JSON 없이 생성 한도로 종료; 원시 응답 파일 참조.'
                       if line.startswith('- ') and 'HTTP 200, `TOKEN_LIMIT`' in line else line
                       for line in text.splitlines())
        if '\n## 7. 사용자 요청' in text:text=text.split('\n## 7. 사용자 요청')[0]
        path.write_text(text.rstrip()+appendix)
    v.update(total_API_requests_including_user_repeat=len(jobs)+len(extras),combined_accounted_usd=combined,
        repeat_excluded_from_primary=True,source_additions={p.name:sha(p) for p in SOURCE.iterdir()
             if p.is_file() and p.name not in v['source_before']},report_sha256=sha(REPORT))
    write(ROOT/'verification.json',v)
    write(ROOT/'artifact_index.json',dict(created_utc=now(),files={str(p.relative_to(ROOT)):sha(p)
        for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_index.json' and '.execution' not in p.parts},
        full_report={'path':str(REPORT),'sha256':sha(REPORT)}))
    print('REVIEW_COMPLETE',combined,len(diagnostic),'diagnostic object rows')

if __name__=='__main__':main()
