# Task 09-B — Frozen GNN Readout Geometry Decodability Probe

ExperimentLogger UUID `64414e993c6e42e3999894e8940668d4`. Vast.ai57640710 / RTX3090 frozen GNN 추론, CPU PhysX 및 CPU Vulkan readback, CPU Ridge/MLP 진단 학습. 원본 Policy/GNN/GRU/Action Head 학습·수정, 기존 Full 재개, 새 Policy Closed-loop 평가, 설치·Commit·외부 Publishing은 수행하지 않았다.

**핵심 결과는 부분적인 위치 Decodability와 제한된 일반화다. Readout에 위치 정보가 전혀 없다는 설명은 지지되지 않지만, 정확한 위치가 이미 보존돼 있고 Action만 문제라는 설명도 확인하지 못했다.** 특히 동일 Robot State의 실제 이동에서 Raw Point 평균보다 Readout Probe의 방향·크기 복원이 약했다. Small MLP는 이 차이를 해결하지 못했다.

[정적 HTML](../../experiments/64414e993c6e42e3999894e8940668d4/readout_geometry_probe.html) · [절대 오차](../../experiments/64414e993c6e42e3999894e8940668d4/absolute_metrics.csv) · [Paired Δ](../../experiments/64414e993c6e42e3999894e8940668d4/delta_metrics.csv) · [Sample/GT Manifest](../../experiments/64414e993c6e42e3999894e8940668d4/samples.json) · [Frozen 감사](../../experiments/64414e993c6e42e3999894e8940668d4/frozen_encoder_audit.json) · [검증](../../experiments/64414e993c6e42e3999894e8940668d4/verification.json)

## 연구 질문과 Task09-A 연결

09-A에서 실제 Peg 이동에 Readout이 변했지만 행동 반응은 작거나 방향이 부정확했다. 이번에는 **동결된 Readout에서 위치를 꺼낼 수 있는가**를 별도 작은 Decoder로 측정한다. Decoder가 성공해도 원래 Policy가 해당 정보를 사용했다는 뜻은 아니다. Raw Geometry와 Robot-only 대조, 독립 Validation Episode 및 동일 Robot State의 실제 Peg 이동을 함께 비교한다. 유효성 그룹 밖 실패를 GNN 정보 손실로 단정하지 않는다.

## 고정 Identity·데이터 출처·분리

Shape-First/PCA5/F5, Raw EDE, k4/M16, Edge-aware GNN L2/d64, Structured Readout320, Presence2, Robot76, GRU144 및 기존 Action Decoder를 유지했다. Full A35/B48/C38, Single A/B/C Update20000은09-A identities.json 및 원본 Receipt와 정확히 대조했다.

| Model | Original SHA256 |
|---|---|
|Full_A|`409a9456740193c3a46337c4e172dac7d64dc22c45250c365d3879d4fb337b5e`|
|Full_B|`f094f3cc85f8013c37e70d6043a7035071f5733b7db04371207c31fbce7b4e02`|
|Full_C|`e615ec3ded96b0242004e08ee13273b782ba4af1032c8a84f205144f4ab8d177`|
|Single_A|`5a94542e6d1f973d052fce9598e92a3be5b8c558974cc4e761f17e087a013f4d`|
|Single_B|`72473e581c6c0dfdfa566a26fcf7d73619f310470a6e983a05a69db06cfd8899`|
|Single_C|`1f81b559b5a95fbd2805b952850b896993eda4be2703f2f46a8f43a569c0ecc5`|

원래 Train96 Robot76 Normalizer Identity는 `4e607d9a0af71ca65134f71b187946c2c47640bbe9f5cf81fb4942903ec52446`. `frozen_readouts320.npz`는 **정규화 전 원본 Readout320과 Robot76**을 저장한다. Probe용 각 Feature mean/std와 Target mean/std는 Probe Train의 물리 유효 VISIBLE_VALID Primary/Baseline에서만 계산했다. 기존 Policy Normalizer와 새 Probe Standardizer를 혼동하지 않는다. P1에는 Robot76을 직접 제공하지 않지만 TCP/Fingertip Node Slot이 포함된다. P3 실패만으로 전체 GNN 실패를 주장하지 않는다.

Direct HDF523개를 실제 감사했으며 시간별 Peg Pose는 없다. Reset Hash나 Robot/Action으로 Label을 만들지 않았다. Frozen Split의 **원래 순서**에서 Train16(30003 제외), Validation6을 결과 전에 선택했다. 각 Episode t=0/10/25, Baseline + World X/Y±5/±10mm + 동일 Pose 반복으로22 Episode×30=660개 새 실제 관측을 수집했다. 기존09-A60개는 독립 **Probe 최종 평가**로만 읽었다. 같은 Episode에서 파생된 모든 상태는 같은 Split이다.

- Probe Train: `[30002, 30047, 30038, 30080, 30026, 30086, 30073, 30000, 30046, 30071, 30084, 30090, 30021, 30058, 30094, 30095]`
- Probe Validation: `[30125, 30056, 30050, 30007, 30110, 30012]`
- Final: Seed30003 / 기존09-A Artifact. Probe Fit·정규화·Hyperparameter 선택에 미사용.

Full Encoder는 기존 Train96을 학습했으므로 Probe Train Episode가 Encoder에게 새롭다는 뜻은 아니다. Single Encoder는 Seed30003을 원래 BC 학습했으므로 Final은 **Decoder에게만 미사용**이며 Encoder에게 OOD Episode가 아니다. Validation6은 Probe Train과 Episode/Reset Group이 독립이며 Full의 기존 Validation Split에서 선택했다. 기존 Test12는 사용하지 않았다.

최초 시도 `99f1dd3ff9cb4b9e81088b80ad62483a`는 첫 Seed30002의 Reset Hash 불일치로 Sample 생성 전에 중단됐다. 다음 시도 `61864f9940f44f55a123c8ec6e528e9e`도 첫 Vision 호출에서 CUBLAS 결정론 환경변수 누락으로 중단됐으며 Sample은 없었다. `CUBLAS_WORKSPACE_CONFIG=:4096:8`을 설정한 뒤 현재 실행을 시작했다. 이를 무시하거나 Label을 보간하지 않고 새 실행에서 현재 Live ScriptedExpert Action을 사용했다. 모든 수집은 **LIVE_DIAGNOSTIC**이다. 현재 실행의 원본 Reset Hash 일치는 9/22, 전체26개 저장 Robot/TCP Observation의 Exact 일치는 0/22다. `replay_audit.json`에 시점별 Robot/TCP/Expert Action 차이를 보존한다. 원본 Dataset과 같은 Observation이라고 주장하지 않는다. 물리 개입은09-A의 Actor.set_pose·충돌 메시 SAT·0.2mm 허용오차·새 근접 접촉 금지 계약을 재사용했다. 개입 뒤 Physics Step 없이 새 RGB-D를 렌더링하고 원래 Pose로 복원한 뒤 Expert를 진행했다. Teleport 뒤 Fresh Contact Force나 동적 안정성을 검증한 것은 아니다.

## Target·동결·수집 계약

Target은 `R_TCP.T @ (t_PegActorOrigin - t_TCP)`의 float32 XYZ, metre다. Pose는 world `xyz,qw,qx,qy,qz`; 보고 수치는 mm. Point Cloud Centroid, GT Grasp/Goal/Hole Pose를 Target/Feature로 쓰지 않았다. GT Segmentation과 Pose는 Sidecar의 Label·유효성 감사에만 사용했다. 각 Sample의 Robot76, TCP/Actor GT, Packet/source pixels/F5/Raw EDE, Graph Hash, 여섯 Readout320/Slot5×64/Presence2, 실제 시점과 Episode Identity를 저장했다.

원본 여섯 모델 전 Parameter `requires_grad=False`, `eval()`, Encoder는 `torch.inference_mode()`로만 실행했다. Encoder 내부 Gradient/grad_fn 없음, 원본 전체·GNN·후속 Policy(GRU/Action 포함) Weight Hash의 수집 전후 및 Probe 후 동일성은 `frozen_encoder_audit.json`, `post_probe_frozen_audit.json`에 있다. Decoder Optimizer에는 독립 Decoder Parameter만 들어간다. Probe 모듈은 원본 Policy 객체를 받지 않고 NumPy 배열만 읽는다.

현재 Frame 처리 전 Causal Tracker Memory를 Deep-copy해 각 Counterfactual에 같은 과거를 제공했다. Predictor Weight/Perception 알고리즘은 변경하지 않았다. 전체 수집 전 첫 Episode의 t10에서 독립 Tracker Reset 후 실제 Prefix RGB를 재처리해 Mask/Readout Exact Equality를 확인했고, 모든 t와 Episode의 Baseline 반복 및 매 Graph의 Readout 반복도 Exact였다. `collection_contract.json`과 계약 테스트 로그를 보존한다. Source Depth/Calibration 역투영, Graph 재구성, Label/Observation 시점·좌표계 Readback을 별도로 검사한다.

## Probe와 고정 학습 예산

| Probe | Feature | 차원 |
|---|---|---:|
| P0 | Raw Robot76 |76|
| P1 | Frozen Structured Readout |320|
| P2 | Readout + Robot |396|
| P3 | Peg/Box Mean Slot |128|
| P4 | 관측 Peg EE XYZ 평균 |3|

P4 평균은 부분 표면 Centroid이므로 GT Origin과 같다고 가정하지 않는다. 같은 Point Set의 min/max/bbox center/span15도 저장했지만 P4 학습 입력은 평균3만이다. 별도 무학습 RawCentroid 직접 예측과 Constant(Train Target 평균)를 기록했다. Presence2는 기록/분류용이며 Decoder 입력에 추가하지 않았다.

Linear는 SVD Linear/Ridge, alpha `[0.0, 0.01, 0.1, 1.0, 10.0, 100.0]`. Small MLP는1 Hidden64/ReLU/Output3, full-batch MSE, AdamW lr0.001, weight_decay `[0.0001, 0.01]`, 최대2000 Epoch, patience150, min_delta1e-8. 동일 Train Target z-score(mean/std, floor1mm)를 두 Decoder에 사용했다. Feature std floor1e-6. 고정 Seed42, Train Label Shuffle Seed43. Validation Target 정규화 MSE 최소 후보/Checkpoint를 선택하며 같은 값이면 먼저 나온 후보를 택한다. Final은 선택에 미사용이다. 6모델×5입력×2Decoder×정상/Shuffle=120개 선택 결과, Ridge6·MLP2 후보 외 Capacity Search 없음.

학습 Primary는 물리 유효 VISIBLE_VALID Baseline/±5/±10mm이며 Repeat은 제외했다. 실제 Fit Row는 Train 377, Validation 153. Perturbation Row를 독립 Episode로 세지 않는다. Population p-value, Mutual Information, 새로운 Policy 성공률을 측정하지 않는다. Episode macro 결과는 `episode_macro_metrics.json`, 각 Episode는 `per_episode.csv`에 있다.

## 관측 유효성

결과 전 고정: GT Visible Peg Pixel<20이면 OCCLUDED; 보이지만 Point0이면 VISIBLE_MISSING; Point<10 또는 선택 Point의 GT Purity<0.5면 GEOMETRY_CONTAMINATED; 나머지 VISIBLE_VALID.09-A Purity0.5 기준과 비교 가능하며 최소 Pixel/Point 수는 새 Config에서 사전 고정했다. Box 상태는 별도 저장해 Peg Primary를 Box 오염 때문에 자동 제거하지 않았다.09-A의 이동 방향 Centroid≥1mm 지원 여부도 Pair CSV에 별도로 남겼다. Primary Delta는 Baseline/변경 양쪽 물리 유효 VISIBLE_VALID에서 계산한다. 아래 Count는 Repeat과09-A 추가/Box 대조도 포함한다.

| Split | Peg validity | States | Physical invalid |
|---|---|---:|---:|
|train|VISIBLE_VALID|418|0|
|train|VISIBLE_MISSING|52|0|
|train|OCCLUDED|10|0|
|train|GEOMETRY_CONTAMINATED|0|0|
|validation|VISIBLE_VALID|171|0|
|validation|VISIBLE_MISSING|9|0|
|validation|OCCLUDED|0|0|
|validation|GEOMETRY_CONTAMINATED|0|0|
|final|VISIBLE_VALID|58|0|
|final|VISIBLE_MISSING|2|0|
|final|OCCLUDED|0|0|
|final|GEOMETRY_CONTAMINATED|0|0|

Object Presence 두 Bit의 네 조합별 절대 오차도 `presence_metrics.json`에 분리했다. Presence는 선택 Point 수로 결정되는 Student 신호로, GT 관측 가능성과 같지 않다. 모든 Missing/Contaminated/Occluded Prediction을 보존했으며 `absolute_metrics.csv`의 그룹별 결과와 `per_pair.csv`에서 확인할 수 있다. 부재 상태의 P4는 zero mean feature이며 해석상 유효 Geometry가 아니다. GT 감사 Mask를 Student Mask로 대체하거나 수집 뒤 Threshold를 조정하지 않았다.

## 독립 Validation 절대 위치 복원

평균 Euclidean Error, mm. 전체 Median/p90/p95/Max/축별 MAE·R²는 JSON/CSV에 있다.

| Model | Decoder | P0 Robot | P1 Readout | P2 Combined | P3 Object | P4 Raw mean |
|---|---|---:|---:|---:|---:|---:|
|Full_A|linear|64.052|58.284|57.255|40.706|49.534|
|Full_A|mlp|65.480|70.237|59.831|71.745|41.121|
|Full_B|linear|64.052|54.820|51.404|37.887|49.534|
|Full_B|mlp|65.480|64.381|55.357|82.481|41.121|
|Full_C|linear|64.052|44.952|47.602|34.289|49.534|
|Full_C|mlp|65.480|56.738|60.717|57.800|41.121|
|Single_A|linear|64.052|31.330|39.079|27.926|49.534|
|Single_A|mlp|65.480|47.975|47.645|34.276|41.121|
|Single_B|linear|64.052|32.136|44.427|35.366|49.534|
|Single_B|mlp|65.480|45.555|59.370|36.378|41.121|
|Single_C|linear|64.052|33.054|45.926|31.379|49.534|
|Single_C|mlp|65.480|52.961|64.903|50.138|41.121|

| Model | Decoder | P1 Train | P1 Validation | P1 Final | Shuffle Validation | Val p95 | Val Max | Val MAE X/Y/Z | Val R² X/Y/Z |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
|Full_A|linear|4.538|58.284|42.734|103.854|131.150|139.349|43.879/22.069/19.854|0.637/0.482/0.572|
|Full_A|mlp|22.813|70.237|91.173|101.193|137.859|177.542|48.701/32.049/22.465|0.577/0.003/0.429|
|Full_B|linear|26.682|54.820|56.763|111.238|123.004|129.680|42.929/19.445/19.521|0.647/0.580/0.625|
|Full_B|mlp|41.376|64.381|83.290|97.999|157.349|163.133|45.409/30.655/18.097|0.557/0.262/0.539|
|Full_C|linear|3.377|44.952|36.812|109.038|112.666|123.616|36.790/17.465/11.873|0.756/0.629/0.742|
|Full_C|mlp|5.106|56.738|66.545|101.014|125.813|137.856|46.904/22.997/12.955|0.664/0.483/0.774|
|Single_A|linear|5.694|31.330|21.550|103.053|60.155|82.469|22.348/11.924/10.682|0.903/0.827/0.875|
|Single_A|mlp|14.792|47.975|78.771|95.650|107.688|120.555|35.999/20.712/14.944|0.743/0.602/0.775|
|Single_B|linear|3.782|32.136|48.940|98.740|84.799|99.443|22.660/11.474/9.945|0.878/0.814/0.852|
|Single_B|mlp|6.152|45.555|75.300|89.919|100.956|110.571|33.253/22.398/12.077|0.794/0.566/0.844|
|Single_C|linear|3.955|33.054|57.946|99.031|75.569|84.015|22.393/14.984/10.974|0.904/0.736/0.824|
|Single_C|mlp|14.091|52.961|96.940|92.686|143.773|172.243|41.574/20.635/15.555|0.669/0.596/0.736|

## 동일 Robot State에서 실제 위치 변화 복원

새 수집에는 GRU를 실행하지 않았다. GNN은 현재 Graph만의 함수이므로 현재 위치 Probe에 Recurrent Hidden 입력은 없다. Final09-A에서는 기존 감사로 Pair의 Previous Hidden 동일성도 확인돼 있다.

다음 Δ 오차는 `||(prediction_alt-prediction_base)-(target_alt-target_base)||`, mm다. Direction Cosine은 Zero Response에서 정의하지 않으며 Signed Gain은 실제 Δ 방향으로 투영한 크기 비율이다. P0는 모든 Pair에서 Robot76이 같아 예측 Δ가 정확히0이다. X/Y World 개입의 TCP XYZ 응답을 측정했으며 **독립적인 Z 방향 개입 실험은 아니다**. Z 절대 복원 R²는 자연 상태 다양성의 결과로만 해석한다. Baseline의 Split별 XYZ 범위·표준편차는 `target_axis_support.json`에 별도 보존한다.

### 독립 Validation Episode Pair

| Model | Decoder | P0 Δerr | P1 Δerr | P1 cosine | P1 gain | P2 Δerr | P4 Δerr |
|---|---|---:|---:|---:|---:|---:|---:|
|Full_A|linear|7.519|7.209|0.537|0.540|8.272|3.102|
|Full_A|mlp|7.519|9.652|0.346|0.351|7.076|3.741|
|Full_B|linear|7.519|10.262|0.330|0.512|8.736|3.102|
|Full_B|mlp|7.519|9.459|0.350|0.345|7.654|3.741|
|Full_C|linear|7.519|6.890|0.530|0.572|7.925|3.102|
|Full_C|mlp|7.519|7.947|0.419|0.443|8.904|3.741|
|Single_A|linear|7.519|4.781|0.795|0.766|6.579|3.102|
|Single_A|mlp|7.519|8.533|0.422|0.450|7.504|3.741|
|Single_B|linear|7.519|5.555|0.749|0.807|6.066|3.102|
|Single_B|mlp|7.519|8.547|0.306|0.312|8.530|3.741|
|Single_C|linear|7.519|5.583|0.729|0.782|6.334|3.102|
|Single_C|mlp|7.519|9.724|0.280|0.291|9.141|3.741|

### 09-A Seed30003 최종 Primary Pair

| Model | Decoder | P0 Δerr | P1 Δerr | P1 cosine | P1 gain | P2 Δerr | P4 Δerr |
|---|---|---:|---:|---:|---:|---:|---:|
|Full_A|linear|7.609|10.917|0.511|0.836|9.905|4.934|
|Full_A|mlp|7.609|13.385|0.050|0.176|6.629|5.722|
|Full_B|linear|7.609|12.938|0.357|0.845|9.773|4.934|
|Full_B|mlp|7.609|10.564|0.343|0.653|11.612|5.722|
|Full_C|linear|7.609|11.816|0.365|0.390|12.398|4.934|
|Full_C|mlp|7.609|15.921|0.112|0.547|13.043|5.722|
|Single_A|linear|7.609|7.093|0.632|0.917|7.600|4.934|
|Single_A|mlp|7.609|14.077|0.592|0.999|8.170|5.722|
|Single_B|linear|7.609|8.198|0.525|0.523|9.036|4.934|
|Single_B|mlp|7.609|13.838|0.356|0.309|11.757|5.722|
|Single_C|linear|7.609|9.115|0.555|0.820|7.880|4.934|
|Single_C|mlp|7.609|17.853|0.337|0.369|14.337|5.722|

실제/예측 Δ Norm, X/Y/Z 성분 오차, 방향, Gain, Baseline 반복 무이동 응답, 각 t0/10/25 및 모든 Invalid/Additional/Box 조건은 `per_pair.csv`, `delta_metrics.json`과 Prediction NPZ에 있다. Final Primary와 Additional/Rotation/Box를 섞어 단일 수치로 보고하지 않았다. 추가/Box/Repeat의 절대 오차는 `final_condition_groups.json`, 모든 상태별 절대 오차·Target·Prediction은 `per_sample_predictions.csv`에도 보존한다.

## 판정과 해석

이 문서의 아래 판정은 연속 오차·P0/P4 대조와 독립 Episode 결과에 근거한 진단이다. Validation은 Hyperparameter 선택에도 사용돼 낙관 편향이 있을 수 있으며, 독립 Final도 한 Seed여서 모집단 일반화 수치를 확정하지 않는다. Small MLP가 실패해도 정보 소실의 수학적 증명이 아니며, Train Fit만으로 표현 충분성을 주장하지 않는다. Linear/MLP의 Feature 차원·용량 차이가 있으므로 엄밀한 정보 이론적 MI 비교가 아니다. Task09-A의 Action 실패 원인은 이번 Probe만으로 한 모듈로 확정할 수 없다.

**핵심 결과는 부분적인 위치 Decodability와 제한된 일반화다. Readout에 위치 정보가 전혀 없다는 설명은 지지되지 않지만, 정확한 위치가 이미 보존돼 있고 Action만 문제라는 설명도 확인하지 못했다.** 특히 동일 Robot State의 실제 이동에서 Raw Point 평균보다 Readout Probe의 방향·크기 복원이 약했다. Small MLP는 이 차이를 해결하지 못했다.

### 위치 정보의 존재와 충분성

Linear P1의 Validation 평균은 Full A/B/C **58.28/54.82/44.95 mm**, Single A/B/C **31.33/32.14/33.05 mm**다. P0 Linear는 **64.05 mm**, Constant는 **96.57 mm**다. Linear P2도 여섯 모델 모두 P0보다 낮았다. 따라서 Graph가 Robot76으로 설명되지 않는 위치 정보를 제공한다는 근거가 있다. 다만 5/10mm 개입을 안정적으로 복원할 정도의 정밀도나 원래 Policy의 사용을 뜻하지 않는다.

P1은 Training Fit이 훨씬 좋은 모델에서도 Validation/Final로 그대로 이어지지 않았다. 예를 들어 Full C Linear는 Train **3.38 mm**에서 Validation **44.95 mm**, Final 절대 오차 **36.81 mm**였고, Single B는 **3.78 → 32.14 → 48.94 mm**였다. Full B Linear는 Train부터 **26.68 mm**로, 모든 실패를 Train 암기 하나로 설명할 수는 없다. MLP도 Full B P1 Train **41.38 mm** 등 일부 조건에서 충분한 Fit을 얻지 못했다. 이는 정보 소실과 Probe 학습/일반화 한계를 완전히 분리하지 못한다는 이유다.

Full B/C의 **P3 Object Slot Linear Validation 37.89/34.29 mm**는 전체 P1의 54.82/44.95mm보다 낮았다. P3는 P1의 부분 벡터이므로, P1의 낮은 성능을 곧바로 “해당 Readout 안에 정보가 없다”는 증거로 해석할 수 없다. Feature 차원, 정규화, 제한된 Episode 수와 회귀의 일반화가 결과에 영향을 줄 수 있다. 새 Capacity Search로 이를 사후 보정하지 않았다.

### 실제 Peg 이동량: Raw Geometry와 Readout의 차이

Validation의 유효 **135 Pair/6 Episode**에서 실제 이동 평균은 **7.519 mm**, Robot-only Δ는 정확히0이다. Raw Centroid 직접 차이의 Δ 오차는 **2.803 mm**, Cosine **0.906**, Gain **0.860**이었다. P4 Linear는 **3.102 mm / 0.902 / 0.771**이었다. 반면 P1 Linear의 Full B/C Δ 오차는 **10.262/6.890 mm**, Single B/C는 **5.555/5.583 mm**였다. Full B는 이 집계에서 Robot-only의7.519mm보다도 나빴다. Single B/C에는 방향·크기 정보가 일부 남아 있지만 Raw Geometry의 이동 정보를 같은 정도로 복원하지 못했다.

최종09-A의 유효 **23 Pair/한 Episode**에서는 실제 이동 평균이 **7.609 mm**였다. P1 Linear Full B/C의 Δ 오차는 **12.938/11.816 mm**, Single B/C는 **8.198/9.115 mm**였다. 모두 Robot-only의7.609mm보다 평균 오차가 컸다. 무학습 Raw Centroid는 **4.744 mm**, P4 Linear는 **4.934 mm**였고 둘 다23 Pair 중22개에서 무반응보다 나았다. P1 Linear의 해당 수는 Full B/C **6/10**, Single B/C **15/14**였다. 일부 Pair의 개선과 평균 실패를 함께 보존한다.

P1 MLP Final Δ 오차는 Full B/C **10.564/15.921 mm**, Single B/C **13.838/17.853 mm**로, 비선형 Decoder가 안정적인 복원을 회복했다는 결과는 없다. Single A Linear만 P1 Final 평균 Δ 오차 **7.093 mm**로 무반응 평균보다 조금 낮았으며, 이것도 Raw Centroid보다 컸다. 현재 결과를 여섯 모델의 일관된 정밀 Decodability로 표현하지 않는다.

P4의 절대 Origin 복원도 정확한 GT Pose 추정기는 아니다. Raw Centroid 직접 Validation 절대 오차는 **46.73 mm**, P4 Linear/MLP는 **49.53/41.12 mm**다. 관측 표면 중심과 Actor Origin의 차이, 부분 관측과 배치 다양성이 섞여 있다. 따라서 **Raw Geometry가 GT Origin을 정확히 알려준다는 전제**는 성립하지 않는다. 이번 비교는 Raw 평균과 작은 Decoder 기준이며, 전체 Point Set의 가능한 모든 정보량을 측정하지 않았다.

### Full 대 Single, Linear 대 MLP

현재 고정된 Episode/Decoder 조건에서 Single의 Linear P1 Validation 오차는 세 방법 모두 Full보다 낮았다. Task09-A의 작은 Single Action Response를 “Single Encoder가 위치를 전혀 표현하지 못해서”라고 단정할 수 없는 결과다. 동시에 Single B/C가 최종 Paired 위치를 정확히 복원했다는 결과도 아니므로, “Action 경로만 문제”라는 반대 결론 역시 성립하지 않는다.

P1 Validation에서는 여섯 모델 모두 Linear가 MLP보다 낮았다. 이는 이번 예산 안에서 **Linear 실패·MLP 성공의 비선형 구제 패턴이 없었다**는 관측이다. 비선형 정보가 없다는 증명이나 MLP의 모든 학습 조건에 대한 결론이 아니다. Shuffle Label 결과는 정상 학습과 분리해 보존했고, 위치 정보의 존재는 정상 Validation/P0 비교와 Pair 결과로 판단한다. Shuffle 하나의 실패를 Leakage 부재의 유일한 증거로 사용하지 않았다.

### t=25와09-A의 행동 반응

09-A t25에서 Full B/C Readout 변화는 **5.438/6.005%**였지만 Expert 대비 Action 이동 크기 비율은 **0.058/0.069**였다. 같은 상태에서09-B Linear P1의 위치 Gain은 **0.370/0.396**, Δ 오차는 **12.576/8.207 mm**였다. 위치 Decoder에서도 정확한 벡터 복원은 확인되지 않았다.

Single B/C t25 Linear 위치 Gain은 **0.607/0.527**, Δ 오차 **5.992/7.437 mm**로, 0에 가까웠던 기존 Action Signed Gain과는 달리 위치 방향 성분 일부가 복원됐다. 이는 정보 존재와 정책 활용이 다른 문제임을 지지한다. 그러나 전체 Final에서 방향·교차축 오차가 컸으므로 H2의 충분조건은 아니다. 위치 Gain과 Action Gain은 각각 다른 Target에 대한 지표여서 수치 차이를 인과적 정보 전달률이나 Mediation 비율로 해석하지 않는다. Gain이1에 가까워도 직교 성분 오차가 크면 올바른 복원은 아니다.

### 관측 누락·오염의 영향

전체720개 상태에서 Peg는 **VISIBLE_VALID647, VISIBLE_MISSING63, OCCLUDED10, GEOMETRY_CONTAMINATED0**이었다. Repeat·Secondary도 포함한 상태 수이며 독립 Episode 수가 아니다. Peg Contaminated 그룹의 정량 영향은 **NOT_TESTED**다. Box는 VISIBLE_VALID500, OCCLUDED140, GEOMETRY_CONTAMINATED80이었다. Occluded Box140개 중60개에서도 Student Presence가 있었고, 그중30개는 선택 Point의 GT Box Purity가0이었다. Contaminated Box80개는 모두 Purity<0.5였다. Presence를 실제 Geometry 지원과 동일시할 수 없다.

P1 Linear의 Validation Missing9개 평균 절대 오차는 Full B/C **244.41/1086.30 mm**, Single B/C **1415.25/926.17 mm**로, 해당 VISIBLE_VALID 평균54.82/44.95/32.14/33.05mm보다 컸다. Final Primary Missing1개의 오차도 Full B/C230.78/1150.69mm, Single B/C1419.07/890.56mm였다. Missing은 Decoder Fit에서 제외된 분포 밖 상태이며 Linear의 큰 외삽 오차도 포함한다. 이 수치는 원래 Policy의 위치 예측이나 물리 이동량이 아니다. 조건별 Episode/시점 분포가 같지 않아 차이를 누락의 독립 인과 효과로 해석하지 않는다.

Train/Validation의 Peg Missing은52/9개, Final은2개였다. Final Primary의 t0 Y+5mm와 Additional Y+20mm의 기존 누락이 그대로 남아 있다. Primary에서 제외한 상태도 Target/Prediction·절대 오차·Pair를 모두 저장했다. `presence_metrics.json`, `per_sample_predictions.csv`와 그룹별 CSV/JSON에서 예측을 따로 볼 수 있다. Primary가 완전히 깨끗한 Perception을 보장하는 기준은 아니며, Purity≥0.5·Point≥10의 사전 감사 계약을 충족한다는 뜻이다.

### H1/H2/H3 및 Case 판정

| 가설 | 판정 | 근거와 경계 |
|---|---|---|
| H1 — Encoder/Readout Information Bottleneck | **INCONCLUSIVE** | Raw 평균의 Δ가 Readout Probe보다 잘 복원돼 Encoding/Readout의 판독가능성·일관성 제약을 조사할 근거는 있다. 그러나 작은 Probe의 일반화/학습 제약, P3가 P1보다 좋은 결과, Raw Origin 자체의 오차가 남아 있어 정보 소실을 확정하지 못한다. |
| H2 — Downstream Utilization/Action Learning Bottleneck | **INCONCLUSIVE** | 위치 정보 일부와 부정확한 기존 Action은 함께 확인됐다. 하지만 현재 Readout의 정밀 복원이 독립 Validation/Final에서 입증되지 않아 “정보는 충분하며 Downstream만 원인”을 확인하지 못했다. Fusion/GRU/Head/BC Optimization은 이번에 직접 개입하지 않았다. |
| H3 — Input Geometry Limitation | **CONFIRMED, 해당 관측 조건에 한정** | 실제 보이는 Peg의 Point 누락, 가려진 Box의 허위 Presence와 Box 오염을 관측했다. 이 입력 문제 하나가 전체 VISIBLE_VALID 결과나 기존 Policy 실패를 모두 설명한다는 주장은 미확정이다. |

Case C(Train Fit에 비해 독립 복원 악화)와 Case F(입력 누락/오염)가 지지된다. Case A는 **Δ 복원의 Raw 대비 열화라는 제한된 근거**가 있지만, Raw Origin 정확 복원과 GNN 정보 소실을 함께 확정한 경우는 아니다. 정확한 Readout 복원을 전제로 하는 Case B/E는 확인되지 않았다. Case D처럼 모든 Graph 결과가 Robot-only와 같은 것도 아니다.

### Q1–Q10

| 질문 | 판정 | 답 |
|---|---|---|
| Q1 Readout320만으로 Position3을 얼마나 복원하는가? | **CONFIRMED** | P1 Linear Validation 평균31.33–58.28mm, MLP45.55–70.24mm. 부분적 위치 정보는 있으나 정밀한 일반 복원은 확인되지 않았다. 축별 MAE/R²·분위수·Episode 오차는 위 표와 CSV에 있다. |
| Q2 원본 Geometry 정보가 얼마나 보존되는가? | **SUPPORTED** | 위치 정보 일부는 남는다. 그러나 Raw Centroid Δ 오차 Validation2.80mm/Final4.74mm에 비해 P1은 대체로 더 컸다. 정보 보존율·MI를 측정한 것은 아니다. |
| Q3 Robot-only보다 독립 Validation에서 정확한가? | **SUPPORTED** | Linear P1/P2는 여섯 모델 모두64.05mm의 P0보다 낮다. MLP P1은 Full A에서 P0보다 나빴다. Graph의 추가 정보가 모든 Decoder에서 같은 크기로 활용되는 것은 아니다. |
| Q4 같은 Robot에서 실제 Peg 이동 방향·크기를 복원하는가? | **INCONCLUSIVE** | 부분 응답은 있지만 Final P1 평균 Δ 오차가6모델 중5개에서 무반응보다 컸다. 안정적인 독립 복원을 확인하지 못했다. Z 방향 개입은 NOT_TESTED다. |
| Q5 Single B/C는 위치를 충분히 표현하고 Action만 둔감한가? | **INCONCLUSIVE** | Validation P1 Linear32.14/33.05mm와 방향 응답은 있다. Final Δ8.20/9.11mm로 정밀 복원은 미확인이라 Action만의 문제로 분리할 수 없다. |
| Q6 Full B/C도 같은 구분이 가능한가? | **INCONCLUSIVE** | Validation P1 Linear54.82/44.95mm, Final Δ12.94/11.82mm. P3는 더 좋았지만 충분한 전체 Readout 복원을 입증하지 못했다. |
| Q7 Linear와 Nonlinear의 차이는? | **CONFIRMED** | 고정 예산 P1 Validation에서 Linear가 여섯 모델 모두 MLP보다 낮았다. 비선형 Decoder의 구제는 없었고 더 큰 Capacity/다른 최적화는 NOT_TESTED다. |
| Q8 Missing/Contaminated 상태의 영향은? | **CONFIRMED, 관측·분리 기록** | Peg Missing63·Occluded10과 Box 오염80을 보존·분리 평가했다. Peg Contaminated는0개여서 그 그룹의 영향은 NOT_TESTED다. 입력 부재의 실패를 GNN 손실로 해석하지 않는다. |
| Q9 09-A 실패가 H1보다 H2를 더 지지하는가? | **INCONCLUSIVE** | H2 단독 설명의 전제인 충분한 위치 Decodability가 확인되지 않았다. Raw Δ 대비 Readout 복원 약점도 있어 Representation·Probe 일반화와 Downstream을 함께 남겨둔다. |
| Q10 구조 변경이 필요한가, 기존 표현/학습 경로를 먼저 조사할까? | **INCONCLUSIVE, 구조 변경 필요성** | 이번 증거만으로 GNN/Readout 구조 변경을 결정하지 않는다. 기존 Slot의 일반화, 표면 Centroid–Origin 차이, 입력 누락 및 학습된 Mapping을 먼저 분리해 조사할 근거가 있다. 후속 변경/추가 학습은 수행하지 않았다. |

### 검증 기록과 실제 한계

추가 Probe/09-A 계약은 로컬·RTX3090 호스트 모두 **15/15 PASS**다. 공통 인프라 테스트는 로컬 **41/42 PASS**, 원격 **35 PASS/5 FAIL/1 ERROR/1 SKIP**였으며 모두 로그를 남겼다. 로컬의 기존 실패는 외부 Organization 저장소를 거부한다고 기대하는 테스트와 현재 `validate_repositories` 동작의 차이다. 원격에는 여기에 optional dotenv 부재 관련4개 실패, `.git` 없는 Source Copy의 Commit 기대1개 Error와1개 Skip이 추가됐다. 공통 Logger/Publisher 코드는 변경하지 않았다. 최초 통합 테스트에는 imageio import 상태와 unittest Mock의 상호작용 Error도 있었고, 분리 실행 결과를 최종 테스트 결과로 구분했다. 실제 W&B/HF 전송·게시를 검증하거나 실행한 것은 아니다.

첫 보고서 Readback 시 Live GT의 float32 차분을 JSON float64 입력으로 재계산해 ULP 수준 차이가 났고, 자기 자신의 `verification.json` 링크가 아직 생성 전이었다. 실제 Label 생성 dtype으로 다시 계산해 정확히 대조하고 별도 float64 기준과의 미세한 차이도 기록했다. **학습 Label·Prediction·Probe 가중치·성능·유효성 Threshold는 바꾸지 않았다.** 실패 로그는 `report_attempt_1/`에 보존했다. 모든720개 Sample의 GT/Robot/Graph·Source 역투영과120개 Decoder 재로드, 절대·Paired 수치, PNG Decode 및 HTML 링크를 검사했다. 실제 브라우저 화면 렌더링은 NOT_TESTED이며 PNG는 직접 검수했다.


## Artifact·실행 범위

원격 실행은 `.git` 없는 격리 Source Copy여서 Logger의 Git Commit은 null일 수 있다. 원본 로컬 Git HEAD/Branch/Dirty와 배포 Identity는 `deployment_provenance.json`, 실제 실행 Source Hash는 `source_hashes.json`, 사후 보고서/추가 테스트 Hash는 `final_source_identity.json`에 구분했다.

원격 `/workspace/task09b/experiments/64414e993c6e42e3999894e8940668d4/`, 로컬 `experiments/64414e993c6e42e3999894e8940668d4/`. Config/Metadata, 원본 Checkpoint Receipt/Identity, Source/Input Hash·Snapshot·Diff, Split Manifest, Live Diagnostic GT/RGB-D/Mask/Packet/Graph, Frozen Readout/Slot/Robot, 120개 Probe Checkpoint·Selection·Prediction NPZ, Per-episode/Per-pair CSV, JSON, PNG, HTML, 계약 로그와 `verification.json`을 보존한다. 기존09-A와 Input/Checkpoint/Full 중지 Status 보호 Hash는 `protected_before.json`, `protection_audit.json`에 있다. 기존09-A 파일과 정책 가중치는 덮어쓰지 않았다.

실행: RTX3090 Frozen Encoder, CPU PhysX/CPU render, 실제 Predicted Vision, 독립22 Episode 수집, Linear/Small MLP 진단 학습. 미실행: CUDA PhysX, macOS/MPS Simulator, 새로운 Policy 학습/Closed-loop 성공 평가, Rotation Probe, GRU Hidden Probe, Z 개입, 추가 Seed 확장.07-G는 다른 Checkpoint의 Sampling 안정성 자료이고09-B의 위치 Decodability와 동일 Noise 강도로 비교하지 않는다.08-B/08-D의 Single B/C 성공도 원래 Policy의 현재 Geometry 활용을 입증하지 않는다.08-E/08-F의 Full Fit 부족·오류 누적 증거는 별도로 유지한다.

## 로컬 회수 검증

원격 원본 **7,138개 파일 SHA256**, JSON/CSV/NPZ 실제 Readback, Paired Metric **79,422행**, PNG12개, HTML 상대 링크47개를 재검사해 통과했다. 기존 로컬 사용자 파일16개와09-A 원본1,734개 파일도 그대로다. [로컬 전송·수치 검증](../../experiments/64414e993c6e42e3999894e8940668d4/local_transfer_verification.json)에 기록했다. 첫 macOS tar 추출은 AppleDouble 메타데이터 파일918개를 자동 처리해 SHA가 달랐다. 원본 Archive에서 Python tarfile로 해당 바이트를 복원했고 데이터·지표는 변경하지 않았다. 최초 실패 기록도 `local_transfer_attempt_1.json`에 보존했다.
