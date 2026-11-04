# 주 live 파이프라인 보고서 빠른 찾기

이 레포를 사용하는 다른 채팅에서도 같은 원문을 찾기 위한 안내다.
마지막 색인 갱신:2026-11-04 KST. 각 결과의 관측 시점·코드 snapshot은 원문을 따른다.

## Repository cleanup

- [현재 helper 분리: 파일별 이관·의존성 차단·CPU/CUDA 검증](repo_cleanup_helpers_20261104.md)
- [Vast58752018: 남은 CUDA8개 테스트 통과·skip 해소](repo_cleanup_cuda_58752018_20261104.md)
- [현재 주 live 기준 모듈화: 백업·실제 변경·검증·남은 작업](repo_cleanup_modular_live_20261104.md)
- [실행 경로·Git 밖 보존 대상·사용자 결정 D01–D12](repo_cleanup_preflight_20261104.md)
- [전체 목표 디렉터리 구조 제안](repo_cleanup_structure_proposal_20261104.md)
- [현재 실행 방법](../usage.md) / [component 교체 계약](../architecture.md)

## 주 live 비용 개선을 다시 시작할 때

먼저 [주 live 최적화 연결·잔여 비용 보고서](10i_live_nonrobot_coverage_audit_20261104.md)를 읽는다.

| 찾는 내용 | 바로 읽을 구간 |
| --- | --- |
| SAM2 포함50ms 목표·SAM2 조사 보류·그 외 추가 최적화 후보 | [50ms 목표와 SAM2 외 최적화 현황](10i_live_50ms_non_sam2_optimization_status_20261104.md) |
| GPU 전환 전 CPU·NumPy와 현재 GPU의 동일 입력 단계별 비교 | [CPU 구버전 복원·재측정](10i_live_preopt_cpu_comparison_58752018_20261104.md) |
| 렌더링→Terra/SAM2→로봇 필터·gate→Graph의 512 단계별 실측 | [주 live 단계별 시간 측정](10i_live_stage_timing_58752018_20261104.md) |
| 새 마스크 점 수에서 발생하는 Triton 재컴파일 지연 | [추적 중 발생한 Triton 재컴파일](10i_live_stage_timing_58752018_20261104.md#추적-중-발생한-triton-재컴파일) |
| 영상용 마스크 CPU 복사와 그 밖의 남은 비용 | [주 live에 남은 부가비용 구체화](10i_live_nonrobot_coverage_audit_20261104.md#주-live에-남은-부가비용-구체화) |
| 예전 평가 파일의 네 문제가 주 live 내부인지 | [주 live 내부인지 재확인](10i_live_nonrobot_coverage_audit_20261104.md#주-live-내부인지-재확인) |
| GPU 최적화들이 기본 실행에 연결됐는지 | [주 live 경로의 연결 목록](10i_live_nonrobot_coverage_audit_20261104.md#주-live-경로의-연결-목록) |
| 실제 검증·trace·원자료 | [원자료·실험 식별](10i_live_nonrobot_coverage_audit_20261104.md#원자료실험-식별) |

검색어: **주 live 비용**, **main live**, **잔여 비용**, **마스크 CPU 복사**,
**mask readback**, **영상 인코딩**, **진단 저장**, **GPU 동기화**,
**nonrobot coverage audit**.

시간 측정 검색어: **단계별 시간**, **512 live timing**, **SAM2 시간**,
**Triton 재컴파일**, **동적 점 수**, **live stage timing**.
후속 목표 검색어: **50ms**, **20Hz**, **SAM2 외 최적화**, **약80ms**, **SAM2 조사 보류**.
새 시간 측정 보고서는 로봇 필터와 wrist gate를 포함한다.
같은 관측 trace의 커널 준비 이후 시간과 새 점 수의 컴파일 지연을 구분해 읽는다.

보고서의 기준 진입점은 `graph.plan_a_bc.rollout.evaluate → run_episode`다.
로봇 필터·손목 gate는 그 보고서의 조사 범위에서 제외했다.
별도 평가 실행 파일의 문제와 주 live 내부의 비용을 구분해서 읽는다.

잔여 비용 목록은 다음 일곱 항목을 다룬다. 상세 근거와 발생 시점은 원문에 있다.

- 모델 호출 전 영상용 마스크 CPU 복사
- 액션 후 RGB CPU 복사·영상 합성·인코딩
- hidden/command/raw 기록 및 주기적 Graph export·저장
- 기록용·검증용 GPU scalar 동기화
- 작은 로봇 상태·카메라 보정 메타데이터 왕복과 반복 계산
- GPU snapshot·특징 복사·포장·임시 할당
- 초기화/미발견 물체 탐색 시 grounding 이미지 인코딩·API 대기

**시간상 병목 순위는 이 목록만으로 정하지 않는다.** 항목별 시간이 분리 측정된
보고서는 아니다. 코드가 바뀌면 원문의 과거 결함을 현재 상태로 단정하지 말고
현재 호출 경로를 재확인한다. 레포 정리 시 재사용 중인 부모 클래스·helper의
의존성도 확인한다.

## 관련 구현·검증 보고서

| 주제 | 보고서 |
| --- | --- |
| 두 카메라512×512 CUDA 출력 | [live512 변경·검증](10i_live_512_20261104.md) |
| native CUDA 관측과 live 출력 wrapper | [live GPU 관측 출력](10i_live_gpu_observation_output_20261104.md) |
| GPU 엣지·최종 Graph 및 시간 측정 | [Graph CUDA 전환](10i_graph_cuda_58752018_20261103.md) |
| Terra primary와 인증 fallback | [live Terra 연결](10i_live_terra_primary_20261103.md) |
| 현재 코드의128/512 대응 Graph 정확도·연속 데이터 보존 | [초기1초 Graph 정확도 파일럿](10i_current_graph_128_512_accuracy_20261104.md) |
| 512 Graph의 이득이 사라지는 단계·오점 확대·FPS 진단 | [Frozen mask 단계별 진단](10i_current_graph_128_512_sampling_diagnosis_20261104.md) |
| 손실 구간 분리·Peg 양끝/Box rim·priority 압축·edge witness 참조 대조 | [512 정보 손실 위치 특정](10i_512_information_loss_localization_20261104.md) |

전체 과거 연구 원문은 [연구 색인](../../research_context_update/RESEARCH_INDEX.md)을 참고한다.
