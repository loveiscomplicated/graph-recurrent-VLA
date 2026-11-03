"""Finalize interpretation/receipts after execution; never issues inference requests."""
from common import *
import numpy as np

def main():
    from review import main as review
    review()
    metrics=read(ROOT/'grounding_evaluation_metrics.json')
    rows=read(ROOT/'evaluation_rows.json')['rows'];scenes=read(ROOT/'evaluation_rows.json')['scenes']
    ledger=read(ROOT/'request_ledger.json')['jobs'];repeat=read(ROOT/'first_request_repeat/ledger.json')['jobs']
    masks=read(ROOT/'sam2/sam2_latency_vram.json')
    failed=[r for r in rows if r['provider']=='openai' and r['visibility']=='VISIBLE' and not r['mask08']]
    normal_fail=[r for r in failed if r['api_status']=='OK']
    all_preds=read(ROOT/'full_image_grounding_predictions.json')['rows']
    normal_joint={}
    for p in MODELS:
        good={r['image_id'] for r in all_preds if r['provider']==p and r['status']=='OK' and r['parsed']['json_valid']}
        ss=[s for s in scenes if s['provider']==p and s['image_id'] in good and s['both_visible']]
        normal_joint[p]=dict(numerator=sum(s['both_mask08'] for s in ss),denominator=len(ss),
                            rate=sum(s['both_mask08'] for s in ss)/len(ss) if ss else None)
    ids=sorted(metrics['models']['openai']['per_parent'])
    values=np.array([metrics['models']['openai']['per_parent'][p]['scene']['both_mask08']['rate'] for p in ids])
    rng=np.random.default_rng(42);draws=rng.integers(0,len(ids),size=(10000,len(ids)))
    absolute_CI=np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist()
    common=set.intersection(*[{p['image_id'] for p in all_preds if p['provider']==m and p['attempted']} for m in MODELS])
    eligible=[s for s in scenes if s['provider']=='openai' and s['image_id'] in common and s['both_visible']]
    parent_ids=sorted({s['parent_id'] for s in eligible})
    rng=np.random.default_rng(42);draws=rng.integers(0,len(parent_ids),size=(10000,len(parent_ids)))
    paired_common={}
    for other in ['zai','ollama']:
        delta=[]
        for pid in parent_ids:
            rates=[]
            for m in ['openai',other]:
                a=[s for s in scenes if s['provider']==m and s['image_id'] in common and s['both_visible'] and s['parent_id']==pid]
                rates.append(sum(s['both_mask08'] for s in a)/len(a))
            delta.append(rates[0]-rates[1])
        a=np.array(delta);paired_common['openai_minus_'+other]=dict(parent_n=len(a),macro_delta=float(a.mean()),
            CI95=np.quantile(a[draws].mean(axis=1),[.025,.975]).tolist())
    known=sum(j.get('usage_estimated_usd') or 0 for j in [*ledger.values(),*repeat.values()])
    accounted=ledger_total(ledger)+ledger_total(repeat)
    observed=dict(status='VLM_GROUNDING_PROMISING',runtime_status='HOST_SLEEP_CONTAMINATED',
        DINO_replacement='INSUFFICIENT_SUPPORT',normal_strict_completed_joint=normal_joint,
        astra_absolute_parent_macro_CI95=absolute_CI,astra_parent_macro=float(values.mean()),
        common_attempted_parent_CI=paired_common,known_usage_estimate_usd=known,
        unknown_usage_reservation_usd=accounted-known,combined_accounted_usd=accounted,
        normal_Astra_mask_failures=normal_fail,
        SAM2={'loaded_once':masks['SAM2_loads']==1,'load_ms':masks['load_ms'],
            'encoding_n':len(masks['encoding']),'encoding_mean_ms':float(np.mean([v['ms'] for v in masks['encoding']])),
            'prompt_n':len(masks['prompts']),'prompt_including_IO_mean_ms':float(np.mean([v['ms'] for v in masks['prompts']])),
            'peak_allocated_GiB':masks['peak_allocated_bytes']/1024**3},
        API_requests=len(ledger)+len(repeat),new_paid_evaluation_after_runtime_fix=False)
    write(ROOT/'final_interpretation.json',observed)
    failure_table=[]
    for r in normal_fail:
        failure_table.append(f"| {r['image_id']} | {r['box_IoU']:.3f} | {r['mask_IoU']:.3f} | {100*r['contamination']['robot']:.1f}% |")
    common_rows=[]
    fmt=lambda a:f"{a['numerator']}/{a['denominator']} ({100*a['rate']:.1f}%)" if a['denominator'] else 'N/A'
    for p,c in MODELS.items():
        common_rows.append(f"| {c['model']} | {fmt(metrics['common_attempted'][p]['scene']['both_mask08'])} | {fmt(normal_joint[p])} |")
    appendix=f'''

## 11. 최종 해석과 남은 병목

**Astra 직접 Grounding → SAM2는 이 코호트에서 유망하다.** 전체 운영 기준 두 물체 동시 성공은49/58=84.5%이며, 같은 이미지의 기존 DINO+Qwen+SAM2 27/58=46.6%보다 높았다. Parent-macro는{100*values.mean():.1f}%, 절대95% CI는[{100*absolute_CI[0]:.1f}, {100*absolute_CI[1]:.1f}]%다. 기존 경로 대비 Parent-macro 차이는+37.78 pp,95% CI[+25.56,+49.44] pp다. 이는 알려진15 Parent의 운영 경로 비교이며 새로운 외형·배경 일반화나 모델 크기만의 인과효과가 아니다.

세 모델 공통 시도29장과 정상 Strict JSON 응답에 조건화한 결과를 구분한다.

| 모델 | 공통 시도 이미지의 주 동시 성공 | 정상 Strict JSON 응답에 조건화한 동시 성공 |
|---|---:|---:|
{chr(10).join(common_rows)}

오른쪽 값은 서로 다른 응답 부분집합에 조건화한 보조 지표이며 모델 간 직접 순위에 사용하지 않는다. 공통 시도 이미지 기준 Parent-cluster 차이 CI는 `final_interpretation.json`에 보존했다. GLM 미시도31장을 포함한 전체 운영 성공13.8%를 모델 자체의 시각 정확도13.8%로 해석해서는 안 된다.

**Astra의 남은 정상 응답 실패는 Peg Mask 7개에 집중됐다.** 7개 모두 예측 Box IoU≥.7이며 6개는≥.968이다. Fixture Mask는 정상 응답·GT Visible55개에서55개 모두IoU≥.8이었다. 즉 이 표본에서는 Box가 정확해도 SAM2 초기 Mask가 불완전한 경우가 남았다.

| Astra Peg 실패 Camera-Frame | Box IoU | Mask IoU | Mask 내 Robot Pixel |
|---|---:|---:|---:|
{chr(10).join(failure_table)}

관측: 네 사례는 예측 Mask가 Peg Pixel만 포함하지만 일부를 놓쳤고, 세 사례는 Robot Pixel을약17–25% 포함했다. Wrong-object≥50%가0건이라는 사실은 Robot 오염이전혀없음을뜻하지않는다. 원인가설: 색/재질별 부분 분할 또는그리퍼와의경계혼동. 이번실험은이를분리검증하지않았으며SAM2알고리즘이나Prompt를수정하지않았다.

**Gemma의0%는시각인식0%가아니다.** Strict JSON0/60때문에SAM2초기화를실행하지않았다. 최종응답의사후좌표진단에서Peg56/60=93.3%,Fixture40/58=69.0%가BoxIoU≥.5였다. 의미있는위치추정능력은관측됐지만, 그좌표로SAM2를실행한성능은이번에측정하지않았다. 코드블록제거등후처리를추가한운영계약은별도후속검증대상이며자동추가실험하지않았다.

**GLM은불완전평가다.** 정상최종응답21개·Token한도5개·전송오류3개에서종료됐다. 정상응답의사후BoxIoU≥.5는Peg/Fixture각각17/21이다. 관측된Mask오인식과좌표오류는실제실패지만, 중단및가용성차이는호스트절전과혼재돼API제공자의원인으로단정하지않는다.

Usage가확인된요청의가격표기반추정은추가반복포함**${known:.6f}**,미확인6개요청의예약액은**${accounted-known:.6f}**다. 전체보수적예산계상은**${accounted:.6f}/$20**이며Invoice확정액은아니다.

SAM2는1회로드({masks['load_ms']/1000:.3f}초),60개이미지Encoding평균{observed['SAM2']['encoding_mean_ms']:.2f}ms,144개ObjectPrompt·Mask저장·Hash포함평균{observed['SAM2']['prompt_including_IO_mean_ms']:.2f}ms,Peakallocated{observed['SAM2']['peak_allocated_GiB']:.3f}GiB였다. 순수DecoderGPU시간이나VLM과의동시상주성능으로주장하지않는다.

판정은**VLM_GROUNDING_PROMISING**,DINO대체의완전한운영근거는**INSUFFICIENT_SUPPORT**다. 동시Mask≥80%는관측됐지만,좌표계약성공은전송오류포함95%이고FixtureBoxRecall87.9%다.표본은기존15Parent이며희귀오인식·미사용Parent·장기추적·Hole위치·Graph기하품질·Closed-loop학습성능은검증하지않았다.Production코드와PAUSED_SCOPE_REVIEW를유지하고종료한다.
'''
    for path in [REPORT,ROOT/'10i_stage4a2_api_full_image_grounding_comparison.md']:
        text=path.read_text()
        old='**결과는 아래 고정 설정·예산·기존 15 Parent에 대한 관측이다. 모델별 접근/실행 실패와 유효 응답의 정확도를 분리해 해석한다.**'
        text=text.replace(old,'**Astra 직접 Grounding → SAM2의 두 물체 동시 성공은49/58(84.5%)였다. Gemma는Strict JSON 실패와 시각적 위치 정확도를 분리해야 하며, GLM은29요청에서 중단돼 불완전평가다. 호스트절전으로전송오류·가용성·전체경과시간비교가오염됐다는제약을함께보고한다.**')
        if '\n## 11. 최종 해석' in text:text=text.split('\n## 11. 최종 해석')[0]
        path.write_text(text.rstrip()+appendix)
    v=read(ROOT/'verification.json')
    v.update(report_sha256=sha(REPORT),completion='COMPLETE_WITH_PROVIDER_STOP_AND_HOST_SLEEP_LIMITATION',
        scientific_status=observed['status'],runtime_comparison_status='HOST_SLEEP_CONTAMINATED',
        source_additions={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file() and p.name not in v['source_before']})
    write(ROOT/'verification.json',v)
    write(ROOT/'artifact_index.json',dict(created_utc=now(),files={str(p.relative_to(ROOT)):sha(p)
        for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_index.json' and '.execution' not in p.parts},
        full_report={'path':str(REPORT),'sha256':sha(REPORT)}))
    print('FINAL_REVIEW_COMPLETE',observed['status'],'HOST_SLEEP_CONTAMINATED')

if __name__=='__main__':main()
