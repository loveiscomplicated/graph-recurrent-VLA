# Task 10-I — Predicted State38 GRU BC 실행 인계

2026-11-02 KST. **필수 Preflight 12개 PASS. 본 학습·Final 평가 완료나 성능 개선을 뜻하지 않는다.**
사용자가 초기 준비한 **Vast.ai58752018 / RTX3090**을 사용한다. 과거 명세의57987772 대신 이번 대화에서 확정한 인스턴스다.

## 구현과 데이터

`graph/predstate10i/`와 독립 원격 Source/Output을 추가했다. 기존 F1/G/H 자산을 수정하지 않았다.

- Frozen F1 E0 GNN → Readout320/Presence2/Robot76 → Frozen398→256→128→Object18 Head.
- F1 `assemble_gt38`로 Predicted State38을 조립한다. 상대 위치는 World Position 차이다.
- 새 Controller는 Projection38→128/ReLU → GRU144 → Action144→128→7이며 seed42로 새로 초기화한다.
- GRU에는38D만 전달한다. Readout/Presence/Robot76 우회 입력과 기존 GT-GRU 가중치·Hidden 이식은 없다.
- Frozen Module은 Eval Mode이며 Optimizer는 새 Controller Parameter만 포함한다.
- Train 예측21,121 frame만으로 정규화한다. 위치·크기 std 하한1mm, Quaternion 하한0.01,
  Clean[1,0]은 mean[1,0]/std[1,1]을 적용해0으로 만든다. GT 기반 통계나 추가 Normalizer 실험은 없다.
- R은 기존 G A step9000, J는 기존 G B step9000이다. 이 가중치는 비교 평가에서만 읽는다.

G의 중간 host mmap이 없어 기존 G 준비 API로 새 root에 재구성했다. 원본 Graph/Action과 정합성을 검사했고,
Sampling Schedule/Manifest/ActionCodec/RobotNormalizer SHA가 기존과 정확히 일치했다.
Graph 재캐시나 새 학습 Dataset 수집은 하지 않았다. 승인된 Prefix BC ON, Disturbance 이력 유지/BC OFF,
Recovery BC ON, Episode/Branch Hidden 분리를 유지한다.

F1 SHA: `781f9f67ba317656d2e6718c2c601026883482633c7d17a4a0cf44f7548690c6`.
Dataset Manifest SHA: `f1b61e7635be3a9b1533d5158e8be8df869edf67dddee926335c68486d5cf6b1`.
J SHA: `8ab82f390dccf054b0bd6539dbd55f7a60af2ae7934f53234702ab66c1369ea3`.

11,487 updates, batch2, sequence32/TBPTT, Method B를 유지한다. AdamW는 lr0.0003, weight decay1e-5,
betas[.9,.999], eps1e-8, amsgradFalse, clip1.0이며 per-update CosineAnnealingLR를 적용한다.
Checkpoint 후보는3000/6000/9000/11487이다. Development seeds는30007/30012/30020/30023/30036/30042이며,
Insertion→Lift→Grasp→Alignment→Final TCP–Peg Error→Validation BC Loss→Earliest 순으로 선택한다.

결과 확인 전에 새 Final118200–118211의12 Parent와 E/S0/S1/S2/R0/R1/R2/J0/J1/J2,
총120 Episode를 고정했다. Development는24 Episode다. 최대250 Action에는 Expert Prefix가 포함된다.
H1/H2는 기존 lift→align / align→insert 경계다. H2를 접촉 직전으로 해석하지 않는다.
Anchor 미도달 조건도 보존하며, Expert만으로 얻은 성공을 Handoff Policy의 성공에 포함하지 않는다.

## 실제 Preflight

[전체 Preflight JSON](../../outputs/task10i/predicted_state38_bc_20261102/preflight_verification.json)의12개 Gate가 모두 PASS다.

- 실Graph, Missing Object, 물체 노드가 없는 Robot3-node Graph, Robot Pose 변화에서320/18/38D,
  유한값, Quaternion Norm, 양수 Shape/Radius, Feature 순서, Device/Dtype을 검사했다.
- F1 공식 Extract와 Wrapper, Batch/Single Forward를 비교했다. Oracle Assembly는 수치 진단에만 사용했다.
  Forward 동안 GT Sidecar/Accessor 접근을 의도적으로 막아도 통과했다.
- Cache/Online State38, 정규화 후 Projection 입력, 같은 Hidden의 Action을 비교했다.
  사전 허용오차는 State/Action atol2e-6, rtol1e-6이다. 정규화 입력에는 State 허용차/std의 오차 전파 폭을 적용했다.
- Disturbance13 frame와 교사 Label19 frame이 섞인 실Batch에서 Mask를 검사했다.
  Frozen Gradient는 없고 새 Projection/GRU/Head Gradient는 있으며 Frozen Hash는 불변이었다.
- 실Sequence32, 마지막 Partial Chunk, Padding/Inactive Stream, Branch 분리, Carry/Detach를 검사했다.
  학습은 메모리의 Feature 배열을 순회하므로 DataLoader Worker를 만들지 않는다.
- CUDA5-update Smoke, Validation48 Episode/3281 BC frame, Checkpoint Save/Readback,
  다음 Update의 Loss/Weight/Optimizer/Scheduler/RNG/Hidden 완전 일치 Resume를 통과했다.
  본 학습은 다시 새로 초기화하며 Smoke Update를 Budget에 포함하지 않는다.
- Development Parent30007에서 E158 Action, Random S0/R0/J0 각2 Action을 실행했다.
  Random S1은 H1=76 뒤2 Action, S2는 H2=138 뒤2 Action을 실제 실행했다.
  Prefix의 물리 상태/Graph/RGB-D/Hidden/Action과 Handoff Frame1회 Forward를 검증했다.
- 최종 Source 버전에서도 별도2-action RGB-D→Graph→F1 Head→State38→GRU→Action7을 실행했다.
- Pipeline의 실제 Child exit7 전파, FAILED 기록, 거짓 COMPLETE 방지, 중복 Lock 거부를 검사했다.
- 기존 보호 파일37,254개와 동결 실행 Source를 시작 전에 다시 Hash 대조했다.

연구 회귀17 test PASS, 공통 Logger/Artifact42 test는 같은 Source/Test SHA의 로컬 실Git 환경에서 PASS다.
첫 리모트 통합59-test 실행은 optional dotenv 부재, Git/scripts가 없는 Source Snapshot,
imageio lazy import와 unittest mock 간섭으로4 FAIL/5 ERROR/1 SKIP이었다. 원본 로그를 보존했다.
공통 코드와 의존성을 변경하지 않고 연구 회귀를 분리했으며 공통 검사는 로컬에서 완주했다.
[회귀 기록](../../outputs/task10i/predicted_state38_bc_20261102/regression_verification.json).
최초 NumPy 허용오차 검사 코드 예외도 보존했고, 허용오차를 바꾸지 않고 검사 구현만 수정했다.

실행 Backend는 CPU PhysX + CUDA Renderer/Vision/Policy, control20Hz/physics100Hz다.
실Sequence32의 CUDA peak allocated는 약81.3MiB였다. 실제 Rollout GPU 사용량 관측은 약5.8GiB였으며
이는 Peak 측정과 구분한다. 학습과 평가를 겹치지 않고 Rollout은 한 번에1개만 실행한다.
실행 확인한 Allocator는 `PYTORCH_ALLOC_CONF=expandable_segments:True`다.
GPU PhysX와 로컬 macOS CUDA는 미실행이며, 본 학습·Final 성능은 아직 확인하지 않았다.

## 독립 Pipeline과 상태 확인

Root: `/workspace/task10i_predicted_state_gru_20261102`.
Session: `task10i_predstate38_bc`.
Python: `/workspace/venvs/task10i-stage4a-cu130/bin/python`.

Runner는 Cache 검증→새 BC→Development→선택 동결→독립 Final→H1/H2→Paired 비교/보고서→최종 검증을 실행한다.
각 Stage는 독립 Child, 실제 Exit Code, 시각/로그 SHA, Atomic Status, 중복 방지 Lock을 갖는다.
실패 시 FAILED/Traceback/유효 Checkpoint를 보존하고 중단한다. Batch/LR/Objective/조건을 자동 변경하거나 재시도하지 않는다.
Checkpoint는 Optimizer/Scheduler/RNG/Cursor/Detached Hidden과 Source/Runtime/Normalizer/Schedule Identity를 보존한다.

```bash
ssh -p 25559 root@151.237.25.16
# 접속 후:
tmux attach -t task10i_predstate38_bc
tmux capture-pane -pt task10i_predstate38_bc -S -60
tail -n 100 /workspace/task10i_predicted_state_gru_20261102/pipeline.log
cat /workspace/task10i_predicted_state_gru_20261102/pipeline_status.json
cat /workspace/task10i_predicted_state_gru_20261102/heartbeat.json
```

상세 로그는 `<root>/logs/`에 저장된다. 실제 완료 검증 후에만 `<root>/PIPELINE_COMPLETE`를 만든다.
최종 전체 Markdown은 `<root>/10i_predicted_state38_graph_recurrent_bc.md`와
`<root>/docs/visual-graph-integration/10i_predicted_state38_graph_recurrent_bc.md`에 저장된다.
HTML은 `<root>/report.html`, Checkpoint는 `best.pt`/`last.pt`, 감사 파일은
`verification.json`/`artifact_index.json`/`completion_audit.json`/`pipeline_receipt.json`이다.

Codex는 Startup Health Check 이후 감시·주기적 재접속·자동 재호출을 하지 않는다.
Runner Heartbeat도 Codex를 호출하지 않는다. tmux는 SSH 연결과 독립이지만 인스턴스 중단·삭제·recycle 뒤
프로세스 지속과 결과 보존을 보장하지 않는다. Git Commit과 외부 Publishing은 하지 않았다.

준비 ExperimentLogger UUID: `e5b6ef7b386242c1b06f5b2a3b6dab98`.
Train/Evaluation은 공통 규칙에 따라 각각 새 UUID를 발급받는다.

## 실제 Startup Health Check

2026-11-02 13:10:22 KST, 단 한 번의 시작 점검에서 **STARTUP_PASS**를 확인했다.
`TRAIN_FEATURE_CACHE_VERIFY`는 Exit0/PASS였고, `FULL_BC_TRAINING` Child PID3432와 Runner PID3398이 실행 중이었다.
tmux Session, Pipeline 로그, Atomic Status, Heartbeat, Config/F1 Identity를 확인했다.
이 스냅샷은 학습·평가 완료나 이후 진행 상태를 보증하지 않는다. 이후 SSH/Polling/Watch는 수행하지 않았다.
[시작 점검 원문](../../outputs/task10i/predicted_state38_bc_20261102/startup_health_check.json).

## 후속 오류 확인

사용자 요청으로 확인한 결과, 학습11,487 updates는 완료됐으나 마지막 Development 평가의 Vulkan 초기화 실패로 Pipeline이 중단됐다. [오류 진단 전체 보고서](10i_predicted_state38_bc_failure_20261102.md). 시작 당시 Health Check는 완료를 보증한 기록이 아니다.
