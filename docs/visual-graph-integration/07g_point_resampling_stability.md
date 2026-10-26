# Task07-G — Point Resampling Sensitivity & Recurrent Stability

Run UUID: `0e6136978c804d83ba79a7b058a7b734` · 고정 Episode 30000/30003/30007/30012 · 528 Action Frame · Vast57640710 CUDA.

[정적 HTML](../../experiments/0e6136978c804d83ba79a7b058a7b734/point_resampling_stability.html) · [검증](../../experiments/0e6136978c804d83ba79a7b058a7b734/verification.json) · [Frame CSV](../../experiments/0e6136978c804d83ba79a7b058a7b734/frames.csv) · [Episode CSV](../../experiments/0e6136978c804d83ba79a7b058a7b734/episode_summary.csv).

## 검수된 정량 판정

**이 네 Episode에서는 작은 Local Replacement에 대해 Action/Hidden이 비교적 안정적이었다. 큰 Phase1 변경도 Readout 차이에 비해 제어 명령 차이는 작았으며, 자연적인 Phase0 Resampling이 실패의 원인이라는 결론은 얻지 못했다.**

아래 평균은 전체 Frame의 기술통계다. 연속 Frame을 독립 표본으로 취급하지 않으며 Episode별 수치와 극값을 함께 해석한다.

| 조건 | Readout 상대 L2 평균 | 이동 명령 차이 평균 / 최대 mm (독립 History) | 회전 거리 평균 rad | Finger Target 차이 평균 / 최대 mm |
| --- | --- | --- | --- | --- |
| phase1 | 5.774% | 0.1531 / 2.0156 | 0.0004648 | 0.0276 / 2.6779 |
| local01 | 0.066% | 0.0031 / 0.0506 | 0.0000082 | 0.0003 / 0.0181 |
| local05 | 0.414% | 0.0166 / 0.4987 | 0.0000717 | 0.0029 / 0.5898 |

Phase1은 Peg가 있는332 Frame / Box가 있는387 Frame에서 Source Jaccard 평균0.3573 /0.5358이었다. 양쪽 Empty 또는 동일 선택을 포함한 전체 Frame 평균0.5959 /0.6598과 구분한다. 실제 선택이 바뀐 Frame은 Phase1 387개, Local01 228개, Local05 374개다. 해당 Frame만의 동일 Hidden 이동 차이 평균은 각각0.1744 /0.0058 /0.0206mm다.

관측 표면의 양방향 NN 평균(Chamfer)은 존재 Frame 기준 Phase1 Peg1.708mm /Box3.472mm, Local01 0.00583 /0.02834mm, Local05 0.05114 /0.17475mm였다. 실제 Geometry 차이가 다른 조건을 같은 세기의 Noise로 해석하지 않는다.

Phase1 최대 이동 차이2.0156mm는 Train30003 t79 Align(원본 Peg230 /Box0 Point), 최대 Finger Target 차이2.6779mm는 같은 Episode t49 Close에서 발생했다. Local05 최대 이동0.4987mm는 Train30003 t68 Lift, 최대 Finger Target0.5898mm는 Validation30012 t39 Close였다. 그럼에도 모든528 Frame ×3 Alternative에서 Gripper Sign과 전환 Timestep은 동일했다. 최초 음수 시점은 Episode 순서로50/49/55/39다.

Phase1 Expert native7 MSE의 Episode 평균 변화는30000 −0.00003268, 30003 +0.00027147, 30007 −0.00009131, 30012 −0.00043833이었다. 전체 평균−0.00004737을 개선의 증거로 일반화하지 않는다. 30003은 Rotation3 MSE +0.00022410 /Gripper1 MSE +0.00123162였고, Translation3 MSE는−0.00000120이었다. Local05 전체 평균은+0.00001336이며 Episode 간 방향이 일치하지 않았다.

동일 시점 Phase1 Readout 절대 L2 Episode 평균은0.0413/0.2890/0.0963/0.1857 수준이며, 자연 연속 Frame 변화 평균0.3145/0.8242/0.2367/0.5753과 비교 자료를 제공한다. 자연 변화에는 실제 움직임·Mask 변화가 포함된다. 자연 변화 대비 비율은 sampling-noise의 인과 추정량이 아니다.

유효 Phase1 단일 Frame Impulse 11개(초기 Hidden L2>1e−4)는 Hidden 5% 기준4–6 Step, 이동 명령3–6 Step에 관측 잔류가 감소했다. 초기 차이가0인 사례는 별도 제외했다. 일부 Local01의 초기 Hidden L2 약4.9e−6/7.6e−6에서는 고정 L2 floor1e−6 기준으로124/50 Step처럼 긴 tail이 나왔다. 이는 큰 효과의 장기 증폭과 다르다. [성분별 FP32 허용오차 복귀](../../experiments/0e6136978c804d83ba79a7b058a7b734/recovery_tolerance.csv)를 함께 제공하며 허용오차는 사전 고정 atol1e−6/rtol1e−5를 그대로 사용했다.

5% 기준 도달은 완전한 수치 복귀가 아니다. Phase1 Action은 사전 고정 FP32 성분별 허용오차까지20–46 Step에 복귀했다. Hidden은10개가20–66 Step에 복귀했고, Validation30012 t30은 남은68 Step 동안 기준 밖인 성분이 남았다(종료 최대 성분 차이2.276e−6, Action은28 Step에 복귀). 이런 작은 tail과 초기 효과의 큰 비율이 지속되는 경우를 구분한다.

[추가 정량 자료](../../experiments/0e6136978c804d83ba79a7b058a7b734/quantitative_findings.json) · [Episode×Stage CSV](../../experiments/0e6136978c804d83ba79a7b058a7b734/stage_episode_summary.csv) · [Point/Edge/Geometry/Readout 기술상관](../../experiments/0e6136978c804d83ba79a7b058a7b734/descriptive_correlations.json) · [검수 그림](../../experiments/0e6136978c804d83ba79a7b058a7b734/reviewed_findings.png).

| 질문 | 판정 | 답 |
| --- | --- | --- |
| Q1 | CONFIRMED | Phase1 Readout relative L2 평균 5.77%, p95 23.63%, 최대 81.23%. 다섯 Slot/Cosine은 Tensor·CSV에 분리 저장. |
| Q2 | CONFIRMED | Phase1 동일 이전 Hidden의 이동 명령 차이 평균 0.1298mm; 독립 History 평균 0.1531mm, 최대 2.0156mm. 회전 거리 평균 0.000465rad; Finger Target 평균 0.0276mm, 최대 2.6779mm. |
| Q3 | CONFIRMED | Local01/05 Readout 상대 차이는 평균 0.066%/0.414%; History 이동 차이는 0.0031/0.0166mm. 실제 교체율은 Peg/Box 각각0.430%/0.492%, 3.682%/3.245%다. Phase1과 기하 변화 규모가 다르다. |
| Q4 | INCONCLUSIVE | Phase1 Episode 평균 Expert MSE는 30003에서 증가(+0.0002715), 나머지 세 Episode에서 감소했다. 30003 증가에는 Rotation·Gripper가 기여했다. Local05는 네 Episode 중 세 곳에서 작은 평균 증가가 있었다. 일관된 성능 저하나 모집단 유의성은 확정할 수 없다. 각 성분·Controller 오차는 별도 보고한다. |
| Q5 | CONFIRMED | Phase1 Hidden L2 평균 0.05289, Cosine 평균 0.9997527. 지속 변경은 추가 History 효과를 만들지만 관측된 차이가 일방향으로 커진다고 볼 수 없다. Local01/05 Hidden Cosine 평균은 0.999999870/0.999996724. |
| Q6 | CONFIRMED | 유효 Phase1 Impulse 11개에서 Hidden의 5% 잔류 기준은 4–6 Step, 이동 명령은 3–6 Step이었다. 완전한 수치 동등성과 다르며, Action의 FP32 성분별 복귀는20–46 Step이었다. 한 Hidden tail은 종료까지 허용오차 밖이었으나 마지막 최대 성분 차이는2.28e−6이었다. 물리적 Recovery는 NOT_TESTED. |
| Q7 | CONFIRMED | 동일 Graph B에 Hidden A[t−1]/B[t−1]를 교차 입력해 과거 효과를 분리했다. Phase1 추가 History 이동 차이 평균 0.0664mm, 최대 0.9900mm. 두 크기는 벡터 효과이므로 단순 스칼라 가산하지 않는다. |
| Q8 | SUPPORTED | 이 네 저장 Expert Sequence에서는 작은 Local Replacement에 대한 Action/Hidden 안정성과, 큰 Phase1 Representation 변화가 제한된 Action 차이로 전달되는 경향이 지지된다. Close의 드문 Finger Target 차이와 Align의 최대 이동 차이는 남는다. 자연적인 Phase0 Temporal Inconsistency가 실제 실패 원인이라는 주장은 INCONCLUSIVE다. |
| Q9 | SUPPORTED | Temporal Consistency를 비교할 정량 기준과 Close/Align의 표적 사례는 확보했다. 그러나 Persistent Anchor가 반드시 필요하다는 근거는 약하며 우월성은 NOT_TESTED다. 동일 학습 예산의 Sampling/Anchor 대조, 관측 누락·Identity 조건 통제, 별도 Closed-loop 검증이 여전히 필요하다. |

실행·검증: Vast57640710 CUDA inference 완료. 로컬61개 PASS; 원격60개 PASS + 기존 Task03 fixture 부재1개 BLOCKED_FIXTURE. 원격의 최초 fixture 실패 로그를 보존했다. 이 blocker는 실제 네 Episode의 입력/Control 검증과 구분한다. Simulator/새 Vision 추론/학습/설치/Commit/외부 Publishing은 수행하지 않았다.

재현: 기존 원격 Python 환경에서 run.sh를 사용한다. loader의 기본 소스는/workspace/task07-stop-clock, 추가 진단 모듈은/workspace/task07-point-resampling이며 bootstrap.py가 package path를 확장한다. 독립 실행 시 같은 package 경로와 현재 checkpoint manifest를 제공한다. 추가 정량 검수는 `python -m graph.integration_baseline.point_resampling_findings experiments/RUN_ID`로 저장 Tensor만 읽는다.

원래 Episode summary.json의 gripper_switches 배열은 경계 i→i+1의 i를 뜻한다. 이 검수의 quantitative_findings.json에는 실제 전환 Observation Timestep(i+1)을 명시했다. 전환 동등성 결론은 동일하다.

### 상세 계약·기본 집계


ExperimentLogger UUID: `0e6136978c804d83ba79a7b058a7b734`. Vast.ai 57640710 / NVIDIA GeForce RTX 3090. **COMPLETE**, 528 valid action frames across the four frozen episodes.

[Static HTML](../../experiments/0e6136978c804d83ba79a7b058a7b734/point_resampling_stability.html) · [Results](../../experiments/0e6136978c804d83ba79a7b058a7b734/results.json) · [Frame CSV](../../experiments/0e6136978c804d83ba79a7b058a7b734/frames.csv) · [Episode CSV](../../experiments/0e6136978c804d83ba79a7b058a7b734/episode_summary.csv) · [Verification](../../experiments/0e6136978c804d83ba79a7b058a7b734/verification.json).

## 1. Identity and execution

실제 원격 실행 Source HEAD는 `2d701a370d7280fbf32f68970f3a97397a30d453`, Dirty=true였다. 로컬 최초 HEAD도 같았고 작업 중 다른 작업이 `77698b8a44d0f1946669517eaeab31bacc1af1a2`로 갱신했다. 이번 작업은 새 진단 파일만 추가했으며 기존 수정·미추적 파일은 보존했다. 실제 실행본은 Source Snapshot/Hashes와 Diff로 고정했다. [Preflight](../../experiments/0e6136978c804d83ba79a7b058a7b734/preflight.json).

Dataset manifest SHA256: `eef8be2904980f9effcdafba320bbc5a6bf1a6f55aadd0975d420dbef9c2e7df`; Split: `2471629fbc55073acc3b3fbce3cc7b1d36f209c3d318e7c1ac5c4e6f690edae7`; Vision manifest: `2adb58373ce2b304cb65e7c20e0e061ef47c411f7dc7aa56e2846d9e757a9e9d`. Train-only Normalizer의 mean/std dtype·shape·bytes digest는 `cbaf73933ff164339590c6771d92dde658383a375da978be19d7d5ee617d2fb7`다. Pipeline UUID와 immutable checkpoint training UUID `002abd72baf84c4da185fd7134c3146f`를 구분해 보존했다.

Checkpoint `3f4ded2101c91b103760e07cc100fe480866778d0309fddac4a0efb83116d776`, training pipeline `2be5e62357a44b14a5651a1a346edc7a`, epoch39 / step8959. Dataset `d1fd6173cb3648bfa4c03507f84b7e71`; original train-only Robot76 normalizer, Regression7, GNN L2/d64, k4/M16, readout320, presence2, GRU144 unchanged. All calculations eval + torch.inference_mode, FP32, TF32 disabled, deterministic algorithms. Existing ExperimentLogger used with WANDB_MODE=disabled. No upload or publish.

[Checkpoint/source/input identity](../../experiments/0e6136978c804d83ba79a7b058a7b734/provenance.json) · [Input hashes](../../experiments/0e6136978c804d83ba79a7b058a7b734/input_hashes.json) · [Normalizer hash](../../experiments/0e6136978c804d83ba79a7b058a7b734/normalizer_hash.json) · [Source hashes](../../experiments/0e6136978c804d83ba79a7b058a7b734/source_hashes.json) · [Frozen selection](../../experiments/0e6136978c804d83ba79a7b058a7b734/selection.json).

## 2. Sampling contracts

Phase0 and Phase1 use the same Prepared Evidence from original RGB-D and stored predicted masks: optimized PCA5, outer silhouette, keep_ratio0.5, original structural/Surface rules and observed 3D points. Phase1 is the existing diagnostic odd-index thinning alternative. F5, availability/validity, observed components, Raw EDE, robot-object k4, global Peg-Box M16 and edge attr4/type are rebuilt with the unchanged helpers. Phase0 is recursively exact with every saved packet and its graph on this runtime.

CONTROLLED_LOCAL_REPLACEMENT fixes structural points and replaces evenly spaced Surface rows with the nearest unused original observed source in the same component. Candidates must be 8-neighbour pixels within the nominal metric scale, or share the nominal camera-frame voxel. Distance/pixel/source tie-breaks are deterministic. Failed replacements are skipped. Requested count is ceil(surface_count × fraction), so discrete rates can exceed 1%/5%. No target/model output enters selection. This is not natural Shape-First temporal noise.

| Alternative | Object | Actual / requested replacements | Actual fraction of Surface |
| --- | --- | --- | --- |
| local01 | Peg | 129 / 332 | 0.4299% |
| local01 | Box | 160 / 387 | 0.4922% |
| local05 | Peg | 1105 / 1580 | 3.6825% |
| local05 | Box | 1055 / 1748 | 3.2452% |

[Replacement provenance](../../experiments/0e6136978c804d83ba79a7b058a7b734/replacement_summary.json). Per-frame source pairs are stored in each episode local_replacement.json.

## 3. Negative controls and contracts

All four episodes PASS before comparison. Fixed FP32 tolerances atol=1e-06, rtol=1e-05. Identical packet/previous hidden repeated; object rows permuted across object boundaries with edge endpoints remapped; all readout/action/hidden outputs match. Robot identity slots remain rows0..2 as required by structured readout. This checks arbitrary object point order, not a redefinition of robot identities. [Measured maxima](../../experiments/0e6136978c804d83ba79a7b058a7b734/negative_controls.json).

Source/row alignment, original observed support, object/component boundaries, dense F5 and validity/availability are validated through existing packet builders/receiver; common-source F5 exactness checked. Raw EDE direction/index/type and interaction edges are rebuilt and receiver-validated. Robot76 and normalizer identical; episode/timestep source identities match; the final observation without an action is excluded. New diagnostics begin hidden at zero per episode; independent alternative histories carry their own hidden.

## 4. Geometry, representation and controller action

| Condition | Peg / Box source Jaccard mean | Readout relative L2 mean | Same-past translation delta mm mean | History translation delta mm mean | History rotation distance rad mean | Finger target delta mm mean |
| --- | --- | --- | --- | --- | --- | --- |
| phase1 | 0.5959 / 0.6598 | 0.057744 | 0.129796 | 0.153050 | 0.000465 | 0.027608 |
| local01 | 0.9970 / 0.9956 | 0.000664 | 0.002496 | 0.003148 | 0.000008 | 0.000344 |
| local05 | 0.9750 / 0.9715 | 0.004142 | 0.014623 | 0.016597 | 0.000072 | 0.002880 |

Chamfer is the mean of concatenated bidirectional nearest-neighbour distances (metres), with p50/p95/max reported. Voxel coverage uses the same nominal camera-frame metric grid for A/B against the full observed support. Source-index identity is a raster location, not a persistent object point. Edge Jaccards use typed source endpoints; common-induced-edge Jaccard separates changed node support from connectivity on shared sources. Feature changes include nearest-point per-channel differences, distribution/validity changes and exact equality at shared sources. [Geometry/feature/topology JSON and per-frame NPZ are linked from HTML](../../experiments/0e6136978c804d83ba79a7b058a7b734/point_resampling_stability.html).

Controller analysis executes installed PDEEPoseController._clip_and_scale_action and gripper clip_and_scale_action on configuration-only stubs. Translation clips to ±1 and scales0.1m; rotation norm clips then multiplies rot_lower−0.1; finger target is clipped then maps to [−0.01,0.04]m. Rotation distance composes quaternions from the installed XYZ Euler matrix helper. No simulator is instantiated. Native7 norm/MSE are diagnostic loss coordinates, not a physical performance score.

## 5. Expert error, stage and training reference

| Episode | Condition | History translation delta mm mean | Native expert MSE delta mean | Hidden L2 mean |
| --- | --- | --- | --- | --- |
| train 30000 | phase1 | 0.0434438 | -3.26834e-05 | 0.0170442 |
| train 30000 | local01 | 0.0027538 | -1.53039e-06 | 0.00148909 |
| train 30000 | local05 | 0.0109025 | -8.03264e-06 | 0.0049961 |
| train 30003 | phase1 | 0.308335 | 0.000271473 | 0.100305 |
| train 30003 | local01 | 0.00270039 | -8.38668e-07 | 0.000977218 |
| train 30003 | local05 | 0.0193508 | 1.62843e-05 | 0.00692374 |
| validation 30007 | phase1 | 0.0998338 | -9.1305e-05 | 0.0337407 |
| validation 30007 | local01 | 0.00349274 | -8.3865e-07 | 0.00133188 |
| validation 30007 | local05 | 0.0111501 | 3.37927e-06 | 0.0039287 |
| validation 30012 | phase1 | 0.171448 | -0.000438334 | 0.0663631 |
| validation 30012 | local01 | 0.00375139 | 1.24333e-06 | 0.00145993 |
| validation 30012 | local05 | 0.0291858 | 5.41815e-05 | 0.010794 |

Native per-frame errors use the unchanged seven-component regression MSE; translation/rotation/gripper errors and actual controller-command errors are separate columns. Increased/decreased/unchanged uses absolute MSE delta1e−6 as a descriptive numerical band, not statistical significance. [Stage and presence groups](../../experiments/0e6136978c804d83ba79a7b058a7b734/aggregate.json) use expert phase metadata only as an analysis sidecar. No phase/GT object/hole/goal enters the policy. Empty objects remain empty.

[Fixed train subset distributions](../../experiments/0e6136978c804d83ba79a7b058a7b734/fixed_train_subset_distribution.json) compare point counts, Surface ratios, F5 means/std and validity on Train30000/30003. They are a limited reference, not the full training distribution. [All-train point count reference](../../experiments/0e6136978c804d83ba79a7b058a7b734/training_point_count_reference.json) uses original manifest counts if available. Consecutive frames are correlated; per-episode summaries are the primary independent cases. No population p-value or causal task success claim is made.

## 6. GRU history, impulses and natural changes

History B carries independently from episode start. Instantaneous B uses Hidden A immediately before the current observation. History-extra compares Policy(Graph B[t], Hidden A[t−1]) with Policy(Graph B[t], Hidden B[t−1]); the same current graph isolates the diagnostic contribution of prior altered sampling. This is a cross-history read-only intervention, not a physically executed policy trajectory.

All conditions receive SINGLE_FRAME_IMPULSE at t10,20,30 and then return to Phase0 without resetting hidden. Recovery is the first lag after which the entire observed tail remains below max(5% of the initial effect, numerical floor): hidden1e−6; translation/finger0.001mm; rotation1e−6rad. Censoring is recorded. This descriptive criterion is not physical recovery success. [Per-episode impulse summaries and recovery curves](../../experiments/0e6136978c804d83ba79a7b058a7b734/point_resampling_stability.html).

Consecutive original Phase0 readout changes and slot changes are stored alongside same-state perturbation sizes; graph point/edge counts and source-pixel topology changes are recorded. Those adjacent observations include actual robot/object motion, camera transforms and mask evolution. They are not pure sampling-noise ground truth.

## 8. Artifacts and verification

Every numerical NPZ is written with immediate byte/shape/dtype readback. Final verification reopens JSON/CSV/NPZ and decodes PNGs, and validates every local HTML/Markdown relative asset. Source snapshot includes all shared inference helpers and the installed controller source. implementation.diff records the remote checkout diff; the added diagnostic files are in source_snapshot. Local source identity and concurrent HEAD changes are captured separately in preflight.json. [Protection](../../experiments/0e6136978c804d83ba79a7b058a7b734/protection.json), [Tests](../../experiments/0e6136978c804d83ba79a7b058a7b734/tests.log), [Verification](../../experiments/0e6136978c804d83ba79a7b058a7b734/verification.json).

Executed: Vast57640710 CUDA GNN/GRU inference, CPU PCA5/Shape-First/F5/Raw EDE and installed controller preprocessing, four fixed full expert sequences, 36 one-frame impulses, all controls. Not executed: CUDA simulator, CPU simulator/physical rollout, MPS, new DINO/SAM2 inference, training/fine-tuning/HPO, persistent tracking/anchors, dependency installation, commit, external publishing. Existing datasets/checkpoint/normalizer and original experiment artifacts are hash-protected.


추가 전체 Policy API 감사에서는 원본 `DirectDataset → model.step()` 출력을 저장된 batch-encoder/GRU 경로와 네 Episode 전체에서 비교했다. Phase0 Readout/Action/Hidden과 모든 Alternative의 동일 이전 Hidden·독립 History 결과가 사전 고정 FP32 허용오차 안에서 일치했다. [전체 API 감사](../../experiments/0e6136978c804d83ba79a7b058a7b734/full_policy_api_audit.json). 이 검증은 CUDA inference만 추가 실행했고, 저장 RGB-D/Mask를 다시 추론하거나 Simulator를 생성하지 않았다.

HTML은 파싱 및 모든 상대 Asset/Link를 검사했고, PNG를 Decode 및 직접 검수했다. 브라우저 Connector는 없었으며 Safari 접근은 Computer Use 승인에 의해 거절돼 실제 브라우저 렌더링 검수는 미실행이다.

Presence별 Phase1 Hidden L2 /이동 차이 평균은 Peg0·Box0(94 Frame) 0.00197 /0.00580mm, Peg0·Box1(102) 0.02368 /0.07309mm, Peg1·Box0(47) 0.19170 /0.62196mm, Peg1·Box1(285) 0.05725 /0.15290mm였다. Presence 전환±2 Frame(74개)의 이동 차이 평균0.18318mm, 나머지454개는0.14814mm였다. 관측 누락 뒤 Hidden이 항상 더 커진다는 단조 패턴은 지지되지 않는다. Stage·Episode·Geometry 분포가 섞인 관측적 그룹이므로 누락 자체의 인과 효과로 해석하지 않는다.
