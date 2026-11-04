# Task10-I — 128/512 Graph 정확도 비교 재사용 사전 조사

상태: **NOT_RUN**. 기존 자료와 코드의 재사용 가능성을 확인했다. 이 문서는 정확도 결과나 실행 완료 보고서가 아니다.

> 후속 실행 완료: 서버 재시작 후 완성된 연속 캡처를 전수 확인하고 현재 코드의128/512
> paired 파일럿을 실행했다. 최신 결과와 검증은
> [현재 Graph 정확도 보고서](10i_current_graph_128_512_accuracy_20261104.md)를 따른다.
> 아래의 접속 상태·인증·미완료 설명은 사전 조사 당시의 기록이다.

## 추가 확인: tracking 원자료는 존재한다

2026-11-04 후속 조사에서 앞선 정적 자료 중심 설명을 보완했다. **tracking 평가용 데이터가 없다는 뜻은 아니다.**

- HF `Graph-Recurrent-VLA/experiment-backups`의 현재 파일 목록을 인증된 `HfApi.list_repo_files`로 확인했다. `task10i_stage3`, `task10f1_data` 및 Stage3 복원 catalogue가 게시되어 있다. 현재 공개 범위는 기존 비공개 설정을 따른다. `resolution`/`tracking` 명칭의 별도 Graph 백업 prefix는 목록에서 발견하지 못했다. 다른 이름의 archive 내부 전체를 조사한 것은 아니다.
- 로컬에도 HF 복원본이 있다. `outputs/task10i/stage3/hf_completion_20261102/restored/outputs/task10i/stage3/run/`의 R Train manifest를 열었으며, `parents/10i_s1_0537/nominal/episode.h5`에서 두 카메라 RGB `[35,128,128,3]`, Main depth `[35,128,128]`, robot state `[35,76]`를 직접 확인했다. 인접 `gt_supervision.npz`에 physics qpos/qvel·Peg/Box/Hole pose가 있다. 이 표본의 확인을 모든 episode의 복원 완전성 판정으로 확대하지 않는다. Actor pixel segmentation이 이 GT state 파일에 포함된다고 가정하지 않는다.
- [SAM2 해상도 시간 비교](10i_sam2_resolution_latency_128_512_768_20261103.md)는 4 Parent × 두 카메라의 **t0..20 연속 시퀀스**를 128/512/768에서 각각 native capture하여 사용했다. 로컬 input manifest의 528행과 시점 t0..20,t80 및 해상도를 확인했다. 당시 원격 보존 경로는 `/workspace/task10i_sam2_resolution_latency_20261103_v1/run/inputs/`다. 이는 짧은 tracking 비교의 출발점이며, 과거 결과는 Oracle bbox 초기화의 시간 측정이다. 이 자료에 RGB-D 및 매 프레임 actor pixel GT가 완비되었다는 전수 확인은 이번에 하지 않았다.
- `resolution_perception_graph_20261103_v1`은 4 Parent × t0..20의 native 128/512/768/1024 RGB-D·pixel GT capture와 graph 평가를 구현한다. 그러나 로컬 capture manifest 280행에는 tracking t0만 있고 t1..20은 없다. 계획 920 camera observation이나 README만으로 원격 전체 capture 완료를 선언하지 않는다. 원격 최종 manifest의 확인이 필요하다.
- 후속 확인 시 Vast API에서 58576054와 58752018 모두 `actual_status=exited`였으며, 58576054 SSH는 connection refused였다. 앞선 접속 때의 running 관측과 시간적으로 다르다. 서버를 자동으로 시작하지 않았다.

따라서 **긴 128 tracking 원자료와 HF 백업은 있고**, **짧은 해상도별 paired RGB tracking 자료도 보존 기록이 있다**. 현재 필요한 full-episode native128/512 RGB-D와 매 프레임 정확한 actor reference의 완비 여부는 아직 확인되지 않았다. Stage3 원자료의 저장 physics state를 재사용해 대응 캡처를 확장할 수 있지만, 렌더링 및 reference 검증은 새로 수행해야 한다.

## 확인한 자료

2026-11-04에 실행 중인 인스턴스 58576054에서 `/workspace/task10i_stage3_20261031_v5_finalization/run/`과 Stage3 원자료 경로가 존재함을 확인했다. `/workspace/task10i_stage4a2_qwen35_native_resolution_prompt_20261102_v1/`에는 기존에 정적으로 같은 상태를 복원해 캡처한 4 Parent, t0/t80, 두 카메라, 128/512의 32개 RGB 이미지 및 대응 sidecar가 있다. sidecar 두 개를 직접 열어 각 해상도의 depth, actor segmentation, calibration 배열 크기를 확인했다. 나머지 sidecar의 전수 해시 확인은 이번에 하지 않았다. 이 캡처는 시간적으로 연속인 trajectory가 아니다.

기존 캡처의 물리 상태 및 카메라 검증과 원래 RGB와의 차이는 [기존 Native 해상도 실험](10i_stage4a2_qwen35_native_resolution_prompt.md)을 따른다. 이 과거 결과는 선택된 소수 Train Parent의 정적 파일럿이고, 현재 live graph의 성능이나 장기 tracking 결과가 아니다.

## 재사용 가능 코드와 차이

- [paired_evaluation.py](../../graph/task10i_stage4a2/paired_evaluation.py)는 현재 `rollout.py`와 같은 `CurrentGeometry`를 호출하고, point source의 actor 일치, correct-source coverage, 누락과 양끝 support를 평가한다. 다만 저장 mask와 pixel index를 128×128로 고정하고 이전 실험의 directory·receipt에 의존한다. 512 입력으로 그대로 실행할 수 없다.
- [evaluation_metrics.py](../../graph/task10i_stage4a2/evaluation_metrics.py)는 pixel metric, 4mm correct-source coverage, Parent 단위 paired 집계를 제공한다. 해상도가 다른 GT depth 표본은 샘플 밀도와 관측 영역이 다르므로 coverage 차이는 별도 공통 참조를 사용해 해석해야 한다.
- [capture.py](../../graph/experiments/qwen35_native_resolution_prompt_20261102_v1/capture.py)는 원본 해상도별 카메라 및 GT를 동시에 생성하고 pose/calibration을 검증하는 기존 도구다. 정적 복원과 동적 action replay는 다르다.
- 현재 [rollout.py](../../graph/plan_a_bc/rollout.py)는 512 카메라와 `LiveGrounding`, `SAM2Tracker`, `CurrentGeometry`를 쓴다. 두 해상도의 mask는 각각 새로 만들어야 하며 초기화뿐 아니라 episode prefix의 tracker memory를 별도로 재현해야 장기 tracking을 비교할 수 있다.

## 실행 전 남은 작업

현재 구현과 환경을 고정하고, 대응 상태의 native RGB-D·calibration, robot state 및 resolution별 정확한 actor reference를 준비한다. 기존 평가 함수의 128 고정 가정과 GPU TensorRecords provenance 추출을 현재 코드에 맞춰 연결해야 한다. GT는 예측 생성 뒤에만 사용한다. 누락 또는 불일치 reference는 분모에서 명시적으로 제외하고, 정상 source point 비율·누락·coverage 및 예측 불가능한 경우를 함께 보고한다. 동일한 물리 기준의 공통 표면 참조와 Parent 단위 대응 집계가 필요하다.

인스턴스 58576054 및 58752018의 비대화형 SSH 환경에는 `OPENAI_API_KEY`가 설정되어 있지 않았다. 확인한 58752018 live timing source의 `.env`도 없었다. 현재 `LiveGrounding`은 키가 없을 때 DINO로 fallback하므로, 이 환경에서의 결과를 Terra primary 결과로 표시할 수 없다. 다른 인증 수단이 실제로 준비돼 있는지는 확인하지 않았다.

이번 조사에서는 simulator, renderer, CUDA/MPS 모델, graph 생성, 학습, 게시를 실행하지 않았다. 원자료와 기존 연구 결과는 변경하지 않았다. 실제 paired graph 정확도 비교, 공통 참조와 tracker prefix 재현은 미완료다.
