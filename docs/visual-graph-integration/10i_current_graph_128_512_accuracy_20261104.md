# Task10-I — 현재 live graph 경로의 native128/512 정확도 파일럿

상태: **COMPLETED — 초기1초 파일럿에서512 Graph의 전체 coverage·purity 개선은 관측되지 않았다.**

후속 [정보 손실 위치 특정](10i_512_information_loss_localization_20261104.md)에서
512의 경계·Box 입구 rim coverage 개선과 Peg 양끝 손실을 분리 확인했다.
아래 전체 지표 결과를 모든 구조 영역에서의 개선 부재로 해석하지 않는다.

후속으로 frozen mask를 단계별 재생했다. [고해상도 이득이 사라지는 단계 진단](10i_current_graph_128_512_sampling_diagnosis_20261104.md)에
관측 support→Shape-first→dedup pool→최종FPS의 coverage·오점 변화와 GT source-filter
진단 대조군을 기록했다. 추가 Detector/SAM2 호출은 없었다.

4개 기존 Train Parent의 동일한21관측을 현재 코드로128/512에서 각각 처리해
**168개 Graph, 336개 object-frame, 672개 camera-object mask 행**을 평가했다.
정상 source 대표점 비율은512에서 낮아졌고, Peg의 공통 관측 표면 coverage도 감소했다.
전체 episode나 장기 tracking으로 일반화하는 결론은 아니다.

## 결과

아래 비율은 대응 가능한 프레임→Episode→Parent 평균이다. 변화는512−128의
percentage point이며 구간은4 Parent의 탐색적 bootstrap95% 구간이다.

| 지표 | 128 | 512 | 변화 pp | 탐색적95% 구간 pp |
| --- | ---: | ---: | ---: | ---: |
| Peg 정상 source 대표점 비율 | 100.00% | 95.91% | −4.09 | [−6.45, −1.86] |
| Box 정상 source 대표점 비율 | 98.43% | 92.19% | −6.24 | [−7.48, −4.88] |
| Peg 공통 표면4mm coverage | 46.45% | 36.54% | −9.91 | [−13.30, −6.51] |
| Box 공통 표면4mm coverage | 31.41% | 30.22% | −1.19 | [−4.84, +1.80] |

정상 point가 하나 이상 있는 가시 object recall은 두 조건 모두 Peg84/84,
Box81/81로100%다. 따라서 이 지표만으로는 오염·coverage 차이를 발견하지 못한다.
Box는 각 조건84관측 중81개가 사전 가시 기준을 만족하고3개는tiny support다.
각 조건의 Box graph가 비어 있는2관측은tiny에 속한다. Precision은 양쪽 모두
예측이 있는 대응 관측에서 계산하며, coverage는 공통 GT support가 존재하는
84관측을 사용한다. 미가시 부재 stratum은 이 표본에 없어 false-presence 성능을
평가할 수 없다.

Parent별 Peg coverage는 다음과 같다. Point precision 감소와 Peg coverage 감소는
네 Parent 모두에서 관측됐고, Box coverage 변화는 Parent마다 부호가 다르다.

| Parent | Peg coverage128→512 | Box coverage128→512 |
| --- | ---: | ---: |
| 0197 | 48.76→42.96% | 27.83→21.12% |
| 0275 | 54.95→42.51% | 27.12→25.47% |
| 0305 | 35.67→28.44% | 35.83→36.62% |
| 0531 | 46.40→32.23% | 34.85→37.67% |

![4 Parent의 대응 Graph 정확도](../../outputs/task10i/current_graph_128_512_20261104_v1/paired_graph_accuracy.png)

## Mask와 최종 Graph의 차이

전체 대표점 수로 합친 source purity는 Frame/Parent 평균과 분모가 다르다.
Peg128은10,752/10,752 정상, Peg512는10,312/10,752 정상이다. Box128은
9,170/9,325 정상, Box512는9,503/10,322 정상이다. 오점 구성은 Peg512의 배경440점,
Box128의 배경155점, Box512의 배경816점과 로봇3점이다. 다른 target actor로의 오인은0점이다.
이 수치는 actor source identity의 정확도이며 GT collision mesh에 대한 거리 오차는 아니다.

같은 예측의 raw mask에서는 Main Peg pooled IoU가98.80→98.95%, Wrist Peg가
89.39→99.68%, Main Box가96.66→98.36%였다. Native GT의 픽셀 척도와 분모는
해상도마다 다르므로 이 수치를 동일 point 분모로 취급하지 않는다. 그러나 mask의
좋은 pixel 지표가 최종128개 대표점의 높은 purity·coverage를 보장하지 않는다는
관측은 확인된다. Wrist gate의 accepted는 양쪽 모두84/84였다. 현재 결과만으로
Terra box, SAM2 경계, 필터 또는 sampling 중 하나를 원인으로 확정할 수는 없다.
Frozen mask를 이용한 단계별 분석이 후속 원인 분리의 출발점이다.

## 데이터 보존 재확인

재시작한 Vast 인스턴스58576054에서 기존 root
`/workspace/task10i_resolution_perception_graph_20261103_v1/`의 capture manifest가
`COMPLETE`, 총920 camera observation임을 확인했다. 4개 Train Parent
`0197,0275,0305,0531`의 t0..20, 두 카메라에서 native128/512뿐 아니라768/1024
RGB-D 및 해상도별 actor segmentation이 보존되어 있다. 현재 비교의 입력128/512와
공통 참조1024에 해당하는504 camera observation의 RGB PNG·sensor NPZ·GT NPZ,
총1,512개 파일을 실제 읽어 SHA256을 대조했고 누락·불일치는0개다.

이는 과거 로컬 manifest280행에서 확인한 범위보다 넓다. 로컬 manifest는 캡처가
끝나기 전의 복사본이었다. [사전 조사](10i_graph_128_512_reuse_preflight_20261104.md)의
“원격 완료 여부 미확인”은 이번 실제 서버 검증으로 해소됐다.

각 Parent의21관측은20Hz에서 초기 약1초 구간이다. Full episode, 가림 후 복구,
Alignment/Insertion 전반의 장기 tracking 일반화는 이 표본에서 평가하지 않는다.
원본 데이터는 저장된 물리 상태를 순서대로 정적으로 복원해 native 렌더링한 것이며,
이번 작업은 보존된 센서 배열을 읽는다. 새 physics step이나 simulator render는 없다.

## 현재 경로와 재사용

[독립 실행기](../../graph/experiments/graph_accuracy_128_512_10i_20261104_v1/README.md)는
현재 `graph.plan_a_bc.rollout`에서 `LiveGrounding`, `SAM2Tracker`,
`OnlineVisionAdapter`, `CurrentGeometry`를 import한다. 상위 인식은 두 해상도에서
Terra Standard/high의 현행 appearance prompt를 새로 실행한다. 각 Parent·해상도·카메라의
tracking state는 독립적이고 t0부터 현재 프레임까지 순서대로 진행한다. 같은 현재
RGB-D·Robot76/FK를 기본 CUDA 로봇 필터·wrist gate·shape-first·dedup·FPS·Graph21/Edge4에
전달한다. mask callback을 통해 로봇 준비를 vision 전에 시작하는 현재 경로도 유지한다.

128과512 모두 같은 현재 코드를 쓰므로 이 비교는 **현재 코드의 해상도 대비**다.
과거 DINO/JPEG128 파이프라인과 현재 Terra/native512의 전체 변경 효과를 뜻하지 않는다.
보존 입력은 CPU SAPIEN capture이므로 현재 native CUDA renderer의 화질 동일성이나
closed-loop policy 성능을 측정하는 것도 아니다. 인증 실패로 DINO가 선택되면 primary
비교를 중단하며 결과를 Terra로 표시하지 않는다.

## GT 평가 계약

모든 mask와 graph를 저장하고 inference receipt를 동결한 뒤 GT 배열을 연다.
기존 `evaluation_metrics.py`의 pixel metric, correct-source coverage 및 Parent paired
bootstrap, `paired_evaluation.py`의 reference point 함수를 재사용한다.

- Mask precision/recall/IoU: 각 입력의 native segmentation과 비교. Mask resize 없음.
- Graph point precision: 대표점의 실제 camera/source pixel에 대응하는 actor label로
  Peg·Box·Robot·다른 물체·배경을 판정한다. 병합된 source 수로 가중하지 않는다.
- Coverage: 두 해상도 모두 **같은 native1024 현재 관측 GT depth 표면**을 참조하고,
  올바른 actor source 대표점만4mm credit을 준다. Peg는 Main+Wrist, Box는 Main만 사용한다.
- 가시 분모: 공통1024 참조의 유효 표본≥1024. 이는128에서16pixel에 대응하는 기준이며
  작은 support와 부재는 별도로 유지한다. 올바른 대표점이0인 가시 물체는 실패다.
- 통계: 같은 관측을 대응시키고 Episode 평균→Parent 평균→Parent bootstrap10,000회.
  Precision이 정의되지 않는 빈 graph를0이나 성공으로 대체하지 않는다. Point precision과
  coverage 및 correct-point recall의 분모는 따로 기록한다.

GT edge 연결 정답, 물리적 양끝/구멍 pose 오차, GNN/GRU 및 삽입 성공률은 본 지표가 아니다.
Terra 응답은 해상도별1회 실현값이므로 API 변동을 제거한 인과 효과나4 Parent 밖의
통계적 우월성을 단정할 수 없다.

## 실행과 검증 기록

새 원격 root: `/workspace/task10i_current_graph_128_512_20261104_v1/`.
로컬 산출물: `outputs/task10i/current_graph_128_512_20261104_v1/`.
공통 `ExperimentLogger`를 W&B disabled로 사용한다. 기존 paused 실험을 재개하거나
기존 자료·production 구현·checkpoint를 덮어쓰지 않는다. 외부 게시나 학습은 없다.

최초 source 복사에서 기존 `tools`와 `visual.scripts`가 빠져 import 단계에서 중단됐다.
API/model 실행은 없었다. source_v4에서는 첫 두 Terra 요청 및 graph build 이후
exporter가 과거 `world_xyz` 필드를 기대해 중단됐다. 현재 EE-frame `xyz`와 같은 시점
TCP pose를 명시적으로 world로 변환하도록 수정했고, 실패 root와 로그는 보존했다.
source_v5에서는 기존 저장 mask를 쓴128/512 export smoke를 먼저 통과했다. 이 smoke는
모델 forward와 GT 배열 사용이0이고 새로운 성능 결과로 집계하지 않는다.

좌표 변환, 대표점 label, 오점 coverage 제외, 미정의 precision 분모 처리의4개
로컬 테스트가 통과했다. 실행 source는 manifest로 고정했고 원래 checkout은 dirty 상태라
Git commit만으로 재현 가능하다고 주장하지 않는다. Source1,134개 Hash와 원본 입력1,512개
Hash가 유지됐고, 별도 기록한Robot252개 파일 Hash도 전후 일치했다. CUDA 모델·graph는
원격에서 실행했으며 macOS MPS/CUDA, GPU PhysX,
simulator/renderer는 이번 실행 대상이 아니다. 설치 SAM2의 `_C` 부재로 native 경고와 함께
fill-hole 후처리를 생략하는 기존 동작을 보존한다.

정상 inference의 Terra 요청은16회, DINO forward와fallback은0회다. Export 실패를 보존한
이전 시도의2요청은 정상 결과와 별도로 제외했다. 인증은 로컬 `.env`에서 읽어 원격
실행 프로세스 환경에만 일시 전달했고 source/config/argv/log에 값은 기록하지 않았다.
정상 inference run ID는 `34d13595a4d5465da37c36c0d974a3ef`, scoring run ID는
`a61383467a49455a9d28f332bd617ea7`이다.

독립 source pixel→depth→world 재구성은41,151개 대표점 전부를 검사했다. 최대3D 차이는
1.114136µm다. 최초 검사기는 임의의Euclidean1µm 기준으로 중단됐다. 기존
`vision_packet.float32_tolerance`의componentwise `max(1µm,4ULP)` 계약을 적용한
검사에서는 최대 허용치 비율0.216731로 통과했다. 이 변경은 채점 전의 수치 검증 기준
수정이며 inference, Graph, 대표점 좌표를 바꾸거나 재실행하지 않았다. 큰 좌표에서의
float32 ULP를 무시한 최초 실패도 별도 receipt에 보존했다. 검증 helper는 inference
snapshot에 추가한 별도 read-only 도구이며 자체SHA를 receipt에 기록했다.

168개 Graph 모두 GPU Graph와GPU wrist gate를 사용했고 `before_vision` 일정을 확인했다.
모든 GT 채점은 inference 완료 receipt 이후 별도 프로세스에서 실행했다. 결과 archive를
로컬에서 다시 읽어336개 point 행·672개 pixel 행과 pooled TP/FP 집계를 독립 대조했다.

## 산출물

- [전체 요약 및 Parent 대응 통계](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/evaluation/summary.json)
- [프레임별 Graph 지표](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/evaluation/point_metrics.csv), [프레임별 Mask 지표](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/evaluation/pixel_metrics.csv)
- [독립 좌표·source 검증](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/independent_verification.json), [자료 보존 검사](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/preservation_after.json), [로컬 재검증](../../outputs/task10i/current_graph_128_512_20261104_v1/local_readback.json)
- [완성된 native capture manifest](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/capture_manifest_complete.json), [현재 inference 프로토콜](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/inference_v5/protocol.json), [동결 inference receipt](../../outputs/task10i/current_graph_128_512_20261104_v1/remote/inference_v5/inference_complete.json)
- [Source snapshot v5](../../outputs/task10i/current_graph_128_512_20261104_v1/source_v5.tar.gz), [결과 archive](../../outputs/task10i/current_graph_128_512_20261104_v1/results.tar.gz)

최종 mask·Graph·provenance의 전체 원자료는 새 원격 root의 `inference_v5/`에 보존했다.
로컬에는 결과 표·receipt·source·로그와 그림을 복사했다. 이 파일럿의 결과는 초기 구간의
graph 표현 검증이며, 해상도 결정에는 전체 episode와 후기 가림 구간 및 단계별 원인
분리 검증이 추가로 필요하다.
