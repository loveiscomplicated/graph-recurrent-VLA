# 고정 Checkpoint — Expert 파지 구간의 평가/step 일치 진단

2026-10-26 KST. **저장 입력에서 평가·Rollout step 경로는 일치했다. 닫기 시점의 부호 전환은
선정한4개 모두 Expert와 같았지만, 닫기 강도와 위치·회전 정지의 재현 오차가 확인됐다.**
이 오차가41005 실패를 직접 유발했는지는 이번 offline 진단으로 확정하지 않는다.

## 1. 실행과 고정 조건

- 진단 ExperimentLogger UUID: `268e3b8cc21943fe97d554d69249a495`.
- 학습 pipeline: `2be5e62357a44b14a5651a1a346edc7a`, training execution: `002abd72baf84c4da185fd7134c3146f`.
- Best epoch39/global_step8959, SHA256 `3f4ded2101c91b103760e07cc100fe480866778d0309fddac4a0efb83116d776`.
- 원본 train96 episode normalizer를 그대로 사용했다. fit/refit하지 않았다.
- RTX3090 /Torch2.8.0+cu128, FP32, deterministic algorithms, AMP 없음, matmul/cuDNN TF32 off.
- 모델·optimizer·학습 데이터·Graph/Vision 알고리즘을 바꾸지 않았다. Weight와 normalizer 값의 전후 hash가 같다.
- 새 Vision 추론·Expert 수집·시뮬레이터 reset/step을 수행하지 않았다.
- 별도 `/workspace/task07-grasp-diagnostic`에서 실행했다. 기존 rollout source를 수정하지 않았다.
  GPU 여유2GiB 이상을 확인하고 이 진단 process의 Torch allocator를 GPU 메모리8%로 제한했다.
  기존 GPU process를 정지하거나 메모리를 강제로 확보하지 않았다.

[HTML](../../experiments/268e3b8cc21943fe97d554d69249a495/grasp_diagnostic.html) ·
[원본 Summary JSON](../../experiments/268e3b8cc21943fe97d554d69249a495/summary.json) ·
[닫기32프레임 CSV](../../experiments/268e3b8cc21943fe97d554d69249a495/close_transitions.csv) ·
[원본 선택 Manifest](../../experiments/268e3b8cc21943fe97d554d69249a495/selection.json).

## 2. 출력 확인 전에 선정한 Episode

각 split에서 최소 Episode ID와, 닫기8프레임 모두 Peg/Box point가 있는 나머지 Episode 중
최소 ID를 고정했다. 손실이나 정책 출력으로 고르지 않았다. 원래 순서의 사례와 관측 지원이 있는
닫기 사례를 함께 확인하기 위한 진단 표본이며, 전체 split 성능의 표본 추정으로 쓰지 않는다.

| Split /Episode | 선정 이유 | 닫기 구간 t | 진단 prefix t | 닫기 Both Presence |
| --- | --- | --- | --- | --- |
| Train30000 | 최소 ID | 50..57 | 0..65 | 0/8 |
| Train30003 | 닫기 Both Presence인 첫 나머지 ID | 49..56 | 0..64 | 8/8 |
| Validation30007 | 최소 ID | 55..62 | 0..70 | 8/8 |
| Validation30012 | 닫기 Both Presence인 첫 나머지 ID | 39..46 | 0..54 | 8/8 |

총257프레임: reach/descend/close225프레임과 닫기 이후8프레임씩의 문맥32프레임이다.
과거 Hidden은 Episode t0부터 시작한다. 닫기 시점에 zero hidden을 다시 넣지 않았다.
Frozen test12와 held-out40은 이 Expert 진단의 선택 대상이 아니다.

## 3. 비교 방법

1. 실제 `direct_train.evaluate()`를 호출해 B1, 최대 T32 chunk로 순차 평가했다.
   이 함수는 Episode마다 hidden을 초기화하고 chunk 사이에 carry한다. 원래 학습의 validation 경로다.
2. 복사 전용 hook으로 원래 packed Graph, Robot76, Action과 GRU output/최종 hidden을 기록했다.
   이 모델은 단일-layer forward GRU이므로 매 프레임의 GRU output에서 해당 hidden을 복원할 수 있다.
   Chunk 마지막 output과 원래 최종 hidden이 같은지도 검사했다.
3. 매 프레임에 같은 저장 Packet/robot geometry로 Graph를 구성했다. 원래 batch에서 분리한
   node/edge 값과 순서, Robot76이 step 입력과 byte-exact인지 확인했다.
4. 같은 reference 과거 hidden으로 `forward(T1)`과 실제 Rollout의 `infer`→`EpisodeStream.step()`을
   비교했다. 이어 별도의 stream이 자기 hidden을 연속 carry할 때 T32 평가와 같은지도 비교했다.
5. 경로를 맞춘 뒤 같은 시점의 native Expert Action7 및 실제 수집 당시 controller 전처리 결과와 비교했다.
   Native target와 physical command의 스케일도 원본 HDF5 controller audit에 대조했다.

수치 허용치는 실행 전에 `atol=1e-6`, `rtol=1e-5`로 고정했다. 결과를 보고 완화하지 않았다.
Config, norm mean/std, schema와 Source hash를 보존했다.

학습 당시와 현재 `direct_dataset.py` 전체 파일 hash는 달랐다. 원본 snapshot의 SHA를 검증한 뒤
`DirectDataset`/`load_episode` AST가 동일함을 확인했다. 차이는 이번에 호출하지 않은 Vision 재처리
코드에 있었다. 학습 평가 함수·모델·Graph·Robot·collate·Recurrent source는 원본 hash와 일치했다.
[Source audit](../../experiments/268e3b8cc21943fe97d554d69249a495/training_source_audit.json)에 기록했다.

## 4. 경로 비교 결과

| 비교 | 4개 Episode 전체의 최대 절대 차이 |
| --- | --- |
| 같은 hidden의 `forward(T1)` vs Rollout step Action | **0 — exact** |
| 원래 T32 평가 vs 같은 hidden의 step Action | 2.682209e-7 |
| 원래 T32 평가 vs 자기 hidden carry step Action | 3.576279e-7 |
| Reference hidden vs 같은/자기 carry hidden | 7.748604e-7 |
| Graph node/edge 값·순서 /Robot76 | **exact** |

이는 고정 FP32 허용치 안의 batch/sequence 계산 차이다. 저장 입력→정규화→GNN/GRU→Action 경로에서
연결 불일치의 증거가 나오지 않았다. 이를 upstream Live Vision의 의미적 정확성까지 검증한 것으로
확대하지 않는다.

추가로 실제41005의 저장 Packet/Graph·Robot76·기록 Hidden을 사용해 t0..63을 다시 계산했다.
Graph와 Source Index가 기록된 배열과 일치했으며, 기존250-step rollout 중 해당64프레임의
**Action과 Hidden 모두 차이0**으로 재현했다. 새로운 환경 rollout을 실행한 것은 아니다.
[41005 replay 결과](../../experiments/268e3b8cc21943fe97d554d69249a495/rollout_41005/summary.json).

## 5. 닫기 행동 조합에서 확인한 차이

원본 Expert는 close8프레임 동안 **arm delta6=0, gripper=-1**을 낸다.
따라서 같은 Expert 상태에서 모델의 비영 이동·회전 명령은 정지 행동의 모방 잔차다.

| Split /Episode | 첫 음수 gripper t: Expert /모델 | 닫기 중 모델 이동 명령 크기 평균 mm/step | 첫 모델 gripper | 첫 mimic joint target mm/finger |
| --- | --- | --- | --- | --- |
| Train30000 | 50 /50 | 2.816 | -0.8973 | -7.433 |
| Train30003 | 49 /49 | 4.543 | -0.2149 | +9.627 |
| Validation30007 | 55 /55 | 4.265 | -0.7251 | -3.128 |
| Validation30012 | 39 /39 | 2.381 | -0.0347 | +14.131 |

Expert의 해당 joint target은 약**-10mm/finger**다. 이 숫자는 실제 손가락 위치가 아니라
기존 controller가 전처리한 목표 관절 명령이다. 음수 raw gripper로 전환됐다는 사실이 곧바로
Expert 수준의 닫기 강도를 재현했다는 뜻은 아니다.

특히 두 Object point가 모두 있는 다음 사례가 구체적이다.

- **Train30003, t49:** Expert `[0,0,0,0,0,0,-1]`; 모델 gripper는 -0.2149이며,
  이동 명령은 약 `[+0.387,+1.095,-3.893]mm`다. t56에는 Expert arm0인 상태에서 +z8.122mm를 요구한다.
- **Validation30012, t39:** Expert arm0/gripper-1; 모델 gripper는 -0.0347,
  mimic target+14.131mm/finger, 이동은 약 `[+0.379,+1.641,-4.825]mm`다.
  t46에는 Expert arm0인 상태에서 +z4.279mm를 요구한다.
- **Validation30007, t62:** 닫기 마지막 프레임에 +z8.267mm 명령이 남는다.

닫기 중 회전 성분 norm 평균도0이 아니었다: Episode 순서대로
0.001603 /0.003915 /0.002244 /0.001816rad/step이다.
이는 norm clipping/scaling 이후 controller 회전 성분 명령이며 실제 TCP 회전각 오차가 아니다.

**닫기 시점을 학습한 증거와, 닫기 전환 강도·정지 유지의 재현 오차가 함께 존재한다.**
이 결과로 “아무것도 학습하지 않았다”거나 “Expert 구간에서도 모든 행동이 틀린다”고 말하지 않는다.
대부분의 시계열은 Expert를 따라가며, 초기·단계 전환부에 구체적인 오차가 남는다.

### 프레임별 자료

| Episode | CSV | Action 그래프 | 저장 Expert RGB |
| --- | --- | --- | --- |
| Train30000 | [CSV](../../experiments/268e3b8cc21943fe97d554d69249a495/train_30000/frames.csv) | [PNG](../../experiments/268e3b8cc21943fe97d554d69249a495/train_30000/actions.png) | [PNG](../../experiments/268e3b8cc21943fe97d554d69249a495/train_30000/close_frames.png) |
| Train30003 | [CSV](../../experiments/268e3b8cc21943fe97d554d69249a495/train_30003/frames.csv) | [PNG](../../experiments/268e3b8cc21943fe97d554d69249a495/train_30003/actions.png) | [PNG](../../experiments/268e3b8cc21943fe97d554d69249a495/train_30003/close_frames.png) |
| Validation30007 | [CSV](../../experiments/268e3b8cc21943fe97d554d69249a495/validation_30007/frames.csv) | [PNG](../../experiments/268e3b8cc21943fe97d554d69249a495/validation_30007/actions.png) | [PNG](../../experiments/268e3b8cc21943fe97d554d69249a495/validation_30007/close_frames.png) |
| Validation30012 | [CSV](../../experiments/268e3b8cc21943fe97d554d69249a495/validation_30012/frames.csv) | [PNG](../../experiments/268e3b8cc21943fe97d554d69249a495/validation_30012/actions.png) | [PNG](../../experiments/268e3b8cc21943fe97d554d69249a495/validation_30012/close_frames.png) |

각 폴더의 `predictions.npz`에 Expert/평가/step Action7, reference 과거·현재 Hidden,
같은-hidden/자기-carry Hidden, Robot76 및 정규화 값, point 수/Presence, 실제 controller 명령을 저장했다.
CSV/JSON은 모든 프레임을 포함하며 불리한 프레임을 제거하지 않았다.

## 6. 판정 범위

- **연결:** 이번 저장 입력과 고정 Checkpoint 조건에서 경로 차이는 없다.
- **모방 재현:** Expert 구간에서도 닫기 강도와 정지 유지에 측정 가능한 오차가 있다.
- **물리적 실패 원인:** 이 차이 때문에 실제 파지가 깨졌는지, 또는41005의 초기 오차/상태 이탈·Live
  Vision 추적 문제가 얼마나 기여했는지는 아직 구분되지 않았다. 예측 Action을 Expert 환경에 적용하지 않았다.
- **반드시 구분:** Presence는 비어 있지 않은 point 지원이며 Mask/Identity의 정확도 지표가 아니다.
  41005와 위 Expert Episode는 reset seed가 다르므로 닫기 timestep27과39..55를 직접 비교해
  “얼마나 일찍 닫았다”고 결론내리지 않는다.
- 이4개는 진단용 고정 사례다. 새로운 전체 Train/Validation 성능이나 파지 성공률을 주장하지 않는다.

## 7. 검증과 재현

로컬/원격 신규 계약 테스트 각각5개 PASS. Hidden을 매번 강제로 초기화하는 오류와 Graph 값 변조를
주입해 실제로 탐지되는지 확인했다. 관련 기존 baseline/stop-clock 포함 로컬32개 PASS.
숫자 JSON/NPZ/CSV,8개 PNG와 HTML 상대 링크를 readback했고 그림도 직접 확인했다.
[검증 JSON](../../experiments/268e3b8cc21943fe97d554d69249a495/verification.json).

실제 서버 명령:

```bash
cd /workspace/task07-stop-clock
VK_ICD_FILENAMES=/workspace/graph-recurrent-VLA/visual/vulkan/nvidia_icd_egl.json \
HF_HOME=/workspace/graph-recurrent-VLA/visual/cache/hf \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
MPLCONFIGDIR=/tmp/task07-mpl WANDB_MODE=disabled OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/workspace/task07-stop-clock-work/eval-venv/bin/python \
  /workspace/task07-grasp-diagnostic/bootstrap.py \
  --training-run /workspace/graph-recurrent-VLA/experiments/2be5e62357a44b14a5651a1a346edc7a \
  --checkpoint-manifest /workspace/task07-stop-clock-work/bundle/checkpoint_manifest.json \
  --config /workspace/task07-grasp-diagnostic/grasp_diagnostic_config.json \
  --rollout-episode /workspace/task07-stop-clock/experiments/f1f98e87a3bb424db7632f59381b7cb4/episodes/heldout_41005 \
  --output-root /workspace/task07-grasp-diagnostic/experiments --device cuda
```

Bootstrap은 CUDA를 먼저 초기화하고 기존 checkout을 import path에 추가한 뒤 진단 script를 실행한다.
실행 script/Config와 테스트는 `graph/integration_baseline/grasp_diagnostic.py`,
`grasp_diagnostic_config.json`, `tests/test_grasp_diagnostic.py`에 추가했다.
기존 모델/학습/runtime 코드는 이번 진단을 위해 바꾸지 않았다. CUDA 진단을 실행했으며 MPS는 미실행이다.
새 dependency 설치, training/fine-tuning/HPO, dataset 추가, 외부 upload, Git commit은 수행하지 않았다.

후속 실제 물리 개입은 [07d 혼합 제어 진단](07d_live_grasp_intervention.md)에 기록했다.
정상 Expert 접근 후 네 seed 모두에서 현재 모델의 close8과 공통 lift/hold로 파지가 유지됐다.
따라서 본 문서의 모방 잔차를 곧바로 실제 파지 실패의 원인으로 해석하지 않는다.
