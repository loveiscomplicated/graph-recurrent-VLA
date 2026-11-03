from common import *
from collections import Counter
import numpy as np

def pct(rate):
    return f"{rate['numerator']}/{rate['denominator']} ({100*rate['rate']:.1f}%)" if rate['denominator'] else 'N/A (분모 0)'

def stats(xs):
    a=np.asarray(xs,float)
    return dict(n=len(xs),mean_ms=float(a.mean()),median_ms=float(np.median(a)),p95_ms=float(np.quantile(a,.95))) if len(a) else dict(n=0)

def figure():
    from PIL import Image,ImageDraw
    imgs=read(ROOT/'input_manifest.json')['images'][:6]
    predictions={(r['provider'],r['image_id']):r for r in read(ROOT/'full_image_grounding_predictions.json')['rows']}
    masks={(r['provider'],r['image_id'],r['object']):r for r in read(ROOT/'sam2/sam2_predictions.json')['rows']}
    canvas=Image.new('RGB',(4*300,len(imgs)*330),'white')
    for y,c in enumerate(imgs):
        for x,p in enumerate(['RGB',*MODELS]):
            im=Image.open(c['image_path']).convert('RGB');d=ImageDraw.Draw(im)
            status='Original RGB'
            if p!='RGB':
                r=predictions[p,c['id']];status=r['status']
                for o,color in [('peg',(255,40,40)),('box',(20,150,255))]:
                    box=r['parsed'][o]['pixels']
                    m=masks.get((p,c['id'],o))
                    if m:
                        mask=Image.open(ROOT/'sam2'/m['mask_file']).convert('L').point(lambda a:80 if a else 0)
                        im=Image.composite(Image.new('RGB',im.size,color),im,mask);d=ImageDraw.Draw(im)
                    if box:d.rectangle(box,outline=color,width=3)
                # GT overlays are post-scoring visualization only, never inference input.
                rr=[r for r in read(ROOT/'evaluation_rows.json')['rows'] if r['provider']==p and r['image_id']==c['id']]
                for a in rr:
                    if a['GT_visible_box']:d.rectangle(a['GT_visible_box'],outline=(40,240,40),width=1)
            canvas.paste(im.resize((300,300)),(x*300,y*330+30))
            ImageDraw.Draw(canvas).text((x*300+3,y*330+2),p+' '+status[:24],fill='black')
    canvas.save(ROOT/'full_image_grounding_comparison.png')

def main():
    v=verify_frozen(); assert v['stage'] in ['MASKS_FROZEN','COMPLETE']
    metrics=read(ROOT/'grounding_evaluation_metrics.json')
    ledger=read(ROOT/'request_ledger.json');jobs=list(ledger['jobs'].values())
    costs={}
    for p in MODELS:
        jj=[j for j in jobs if j['provider']==p];u=[j['usage_estimated_usd'] for j in jj if j.get('usage_estimated_usd') is not None]
        lat=[j['latency_ms'] for j in jj if 'latency_ms' in j]
        success=metrics['models'][p]['overall']['scene']['both_mask08']['numerator']
        costs[p]=dict(requests=len(jj),usage_known_requests=len(u),usage_based_upper_estimate_usd=sum(u),
            budget_accounted_usd=sum(j['accounted_usd'] for j in jj),
            missing_usage_reserved_usd=sum(j['accounted_usd'] for j in jj if j.get('usage_estimated_usd') is None),
            invoice_confirmed=False,latency=stats(lat),warm_latency=stats(lat[1:]),cold_first_latency_ms=lat[0] if lat else None,
            mean_estimated_cost_per_request=sum(u)/len(u) if u else None,
            estimated_cost_per_joint_success=sum(u)/success if success and len(u)==len(jj) else None)
    write(ROOT/'grounding_latency_cost.json',dict(providers=costs,accounted_usd=ledger_total(ledger['jobs']),
        budget_usd=20.,billing_note='Usage-based conservative estimates without cache discounts; missing usage retains reservation, not a claimed charge. No invoice verification.'))
    figure()
    independent=read(ROOT/'independent_verification.json')
    table=[];runtime=[];actual=[];errors=[]
    for p,c in MODELS.items():
        m=metrics['models'][p];overall=m['overall'];counts=m['api_status'];cost=costs[p]
        usable=sum(j['status']=='OK' for j in jobs if j['provider']==p)
        table.append(f"| {c['model']} | {cost['requests']}/60 | {usable} | {pct(overall['peg']['mask08'])} | {pct(overall['box']['mask08'])} | {pct(overall['scene']['both_mask08'])} |")
        runtime.append(f"| {c['model']} | {json.dumps(counts,ensure_ascii=False)} | {ledger['stopped'].get(p,'없음')} | ${cost['usage_based_upper_estimate_usd']:.6f} / {cost['usage_known_requests']}건 | ${cost['missing_usage_reserved_usd']:.6f} |")
        if usable:actual.append(p)
        for j in [j for j in jobs if j['provider']==p and j['status']!='OK'][:3]:
            raw=read(j['raw_path']) if j.get('raw_path') else {}
            msg=json.dumps(raw.get('raw',{}).get('error', raw.get('raw',{})),ensure_ascii=False)[:1800]
            errors.append(f"- {c['model']}: HTTP {j.get('http_status')}, `{j['status']}`. 응답: {msg}")
    conclusions=('세 모델의 비교 가능한 시각 정확도가 확보되지 않았다. 아래 전체 배정 기준 실패율은 API 접근·실행 실패를 포함하며 모델의 물체 인식 능력 0%를 뜻하지 않는다.'
                 if not actual else '결과는 아래 고정 설정·예산·기존 15 Parent에 대한 관측이다. 모델별 접근/실행 실패와 유효 응답의 정확도를 분리해 해석한다.')
    content=f'''# Task10I Stage4A-2 — 세 API 모델 Full-image Grounding·SAM2 초기화 비교

실험일: 2026-11-03 KST. Identity: `20261103_v1`.

**{conclusions}**

## 1. 목적과 실제 범위

원본 native512 RGB 한 장과 공통 외형 Prompt만으로 Peg·Fixture의 Bounding Box를 찾고, 유효 Box로 기존 SAM2.1 Hiera Small을 초기화한다. 기존 Train 15 Parent의 Main/Wrist·F0/F1 60장이다. 0134는 F1=t78, 나머지는 t80이다. 이미 분석한 코호트이며 미사용 Parent 일반화 평가가 아니다. 신규 DINO Forward, Simulator Render/Reset/Rollout, Tracking, Hole 추정, Graph 생성 및 학습은 0회다.

API 요청은 최대 180개 중 **{len(jobs)}개**, SAM2 Mask는 **{v['sam2_masks']}개** 생성했다. GT는 API 및 Mask 예측 동결 후 정확도 평가에만 사용했다. 원시 RGB의 파일·Pixel SHA와 기존 GT 파일의 보존 SHA를 확인했다.

## 2. 동결 입력·모델·출력 계약

Prompt SHA256: `{sha(ROOT/'grounding_prompt.txt')}`. 원문은 [grounding_prompt.txt](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/grounding_prompt.txt)에 보존했다.

빨강·흰색 직사각형 PEG, 황갈색·갈색 Fixture, 회색·흰색 Robot 및 어두운 Finger의 외형 설명을 공통 사용했다. Functional role, Few-shot, 위치 힌트, DINO Proposal/Score/Label, GT Mask/Pose, SAM2 Mask 및 대화 누적 입력은 없다. 0–1000 정규화 좌표를 512/1000으로 변환하며 clipping/보정/확대하지 않는다. Strict JSON 실패와 사후 진단을 분리하며 복구 좌표는 SAM2 입력으로 승격하지 않는다.

| API 모델 | 고정 설정 |
|---|---|
| gpt-6-astra | Responses, original detail, reasoning high, 전체 생성 최대 5120 |
| glm-5.3-flash | Z.ai 일반 API, thinking enabled/max, temperature 1, top_p .95, 최대 8192 |
| gemma4:31b | Ollama Cloud, think=true, temperature 1/top_p .95/top_k 64, 최대 8192 |

Schema 강제·Tools·Sampling 반복·자동 Retry·대체 모델은 사용하지 않았다. API 모델 alias는 제공 서비스에서 고정 Revision을 보장하지 않아 요청/반환 Model ID·UTC 시각·원시 응답을 기록했다. 계정 접근권한은 키 설정 여부만으로 보장되지 않는다. 위 토큰 한도는 예산 제한 실험이며 무제한 최고 성능 측정이 아니다.

## 3. API 실행 상태와 비용

| 모델 | 응답/미시도 상태 분포 | 중단 이유 | Usage 기반 비용 추정 / 해당 요청 수 | Usage 미확인 예약액 |
|---|---|---|---:|---:|
{chr(10).join(runtime)}

오류 응답 관측(인증값은 제거):

{chr(10).join(errors) if errors else 'API 오류 응답 없음.'}

합계 예산 계상액은 **${ledger_total(ledger['jobs']):.6f} / $20**다. Usage 미확인 예약액은 보수적인 예산 점유이며 실제 청구액으로 주장하지 않는다. 캐시 할인을 적용하지 않은 Usage 추정과 Invoice 확인을 구분한다. 이번 실행에서 Invoice 확인은 하지 않았다. 3090 사용료는 API 예산과 별도다.

Latency의 표본 수·Mean/Median/p95·첫 요청·후속 요청은 [grounding_latency_cost.json](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/grounding_latency_cost.json)에 보존했다. 서비스 오류의 응답 시간은 유효 추론 Latency와 구분해 원시 Job 상태로 확인한다. 서버 Vision/Prefill/Decode의 개별 GPU 시간은 노출되지 않아 추정하지 않는다.

## 4. 전체 배정 이미지 기준 초기화 결과

| 모델 | 실제 시도 | 정상 API 최종 응답 | Peg Mask IoU≥.8 | Fixture Mask IoU≥.8 | 두 물체 동시 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

분모는 GT 가시 Pixel≥16이며 Peg 60개, Fixture 58개, 두 물체 동시 58장이다. Tiny 1–15와 Absent 0은 별도 보고한다. Null·형식 오류·미시도·API 실행 실패·유효 Box가 없는 경우는 전체 초기화 성공에서 실패다. **API가 실행되지 않은 모델의 위 수치는 시각 인식 정확도가 아니다.** 서비스가 실제 반환한 유효 응답 조건부 지표 및 모델 간 공통 시도 이미지는 [grounding_evaluation_metrics.json](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/grounding_evaluation_metrics.json)에서 분리했다.

Box IoU≥.3/.5/.7, 중심 오차, Strict JSON/좌표 유효율, Null, 비가시 대상 Hallucination, Mask Precision/Recall, Actor/Background 오염 및 Wrong-object는 사례 CSV와 지표 JSON에 저장했다. Box IoU≥.5를 Visible Detection Recall의 기준으로 사용한다. 다른 Actor가 Mask Pixel의 50% 이상일 때 Wrong-object로 기록하며 낮은 IoU·배경 오염과 구분한다.

Main/Wrist, F0/F1 및 Parent별 결과를 함께 제공한다. Parent 15개를 독립 단위로 하여 Macro와 paired Parent Bootstrap 10,000회/seed42를 계산했다. 접근 실패 모델 간의 동일 0 결과가 능력의 동등성을 의미하지 않는다. 탐색 기준은 동시 Mask≥80%, Wrong-object≤5%, 유효 좌표≥99%이며 분모 부족 시 INSUFFICIENT_SUPPORT다.

## 5. 기존 기준과 해석

같은 60장·같은 SAM2 Mask 기준의 기존 DINO+Qwen 조건 A는 Peg33/60, Fixture48/58, 동시27/58(46.6%)였다. 이 비교는 서비스/관측 경로 비교이며 모델 크기만의 효과가 아니다. 128 RGB Qwen 단독 Grounding과 직접 수치 비교하지 않았다. Fixture 전체 Mask는 Hole 입구 위치·삽입 방향 정확도를 대신하지 않는다.

{conclusions}

전체 RGB Grounding 자체가 가능한지, 유효 Box가 실제 Mask로 이어지는지, 초기화 성공 한 건당 비용이 얼마인지 구분한다. 접근/Runtime 문제로 평가할 수 없으면 GROUNDING_RUNTIME_BLOCKED이며, 이를 모델의 시각 능력 부족으로 해석하지 않는다. 결과 기반 Prompt 수정·모델 대체·추가 이미지/실험은 실행하지 않았다. Production 또는 DINO 제거 판정은 하지 않는다.

## 6. 검증·보존·산출물

독립 PIL 기반 검증: **{independent['status']}**, Object 지표 {independent['object_metrics_recomputed']}개 재계산, 보호 파일 {independent['protected_files_verified']}개·Source {independent['source_files_verified']}개·예측 파일 Hash {independent['artifact_hashes_verified']}개 확인.

공통 ExperimentLogger를 W&B disabled로 재사용했다. API Key는 로컬 .env에서만 읽고 원격 전송·로그/Manifest 기록하지 않았다. API 모델·서버의 전체 내부 Weight 및 환경은 관측할 수 없다. SAM2 CUDA가 실행된 경우 기존 Vast58576054에서만 실행하며 macOS의 CUDA 실행으로 주장하지 않는다. 기존 Checkpoint/환경/데이터·PAUSED_SCOPE_REVIEW·Stop Stub을 보존했고 Commit·외부 게시는 없다.

주요 Artifact: `protocol.json`, `input_manifest.json`, `model_identity.json`, `request_ledger.json`, `full_image_grounding_raw_responses.json`, `full_image_grounding_predictions.csv`, `grounding_evaluation_cases.csv`, `grounding_failure_cases.csv`, `scene_cases.csv`, `sam2/sam2_predictions.json`, `grounding_latency_cost.json`, `independent_verification.json`, `verification.json`, `artifact_index.json`.

![RGB·예측 Box·사후 GT Box·실제 생성 Mask 비교](../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/full_image_grounding_comparison.png)

빨강: 예측 Peg, 파랑: 예측 Fixture, 초록 얇은 선: 사후 GT Visible Box. Mask는 실제 생성한 경우만 표시한다. Hash 순서 첫 6장으로 결과 기반 성공 사례 선택이 아니다.
'''
    REPORT.write_text(content)
    (ROOT/'10i_stage4a2_api_full_image_grounding_comparison.md').write_text(content.replace('../../outputs/task10i/stage4a2/api_full_image_grounding_20261103_v1/',''))
    v.update(stage='COMPLETE',completed_utc=now(),independent_verification='PASS',
             report_sha256=sha(REPORT),visual_accuracy_available_providers=actual)
    write(ROOT/'verification.json',v)
    write(ROOT/'artifact_index.json',dict(created_utc=now(),files={str(p.relative_to(ROOT)):sha(p)
        for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_index.json' and '.execution' not in p.parts},
        full_report={'path':str(REPORT),'sha256':sha(REPORT)}))
    print('COMPLETE',REPORT)

if __name__=='__main__':main()
