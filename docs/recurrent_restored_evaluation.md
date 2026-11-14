# 복원 checkpoint의 현재 환경 재평가

`recurrent.reevaluate`는 기존 학습을 변경하지 않고 다른 패키지 환경에서 평가만
실행한다. 기본 동작은 원본 학습 완료 검증과 `plan.json` 생성이다.
`--execute`를 지정해야 CPU fixture 생성과 평가가 시작된다. 학습·HF 게시 경로는 없다.

**전체 평가 완료:** 정책 84 checkpoint·11,340 episode와 CPU expert 60 episode가
모두 완료 검증을 통과했다. [PNG·CSV 공유 리포트](reports/recurrent_main_20261023/README.md)에
결과와 재현 명령을 정리했다. 아래 smoke 절은 전체 평가 전 준비 시점의 기록이다.

## 원본과 평가 identity

- 복원 원본: `/workspace/recovered-main`, HF `Graph-Recurrent-VLA/experiment-backups`,
  revision `9697eb97b10806220722cdb82df4c16893a79ba5`,
  prefix `recurrent/main/ed8f549f237e406a93521845d9eb9188`.
- 원본 fixture는 seed 2000–2029이며 Torch `2.11.0+cu128`에서 생성됐다.
  현재 Torch는 `2.14.1+cu130`이다. 2개 seed 요청에 30개 seed fixture를 주는 것도
  엄격한 seed 목록 검증에 실패한다. 원본 fixture와 identity를 수정하지 않는다.
- 실행기는 복원 데이터 hash·split·학습 설정·평가 조건을 기존 root identity와
  비교한다. 84개 `best.pt/last.pt/history.json`은 기존 `training_complete()`로
  검증한다. 원본 패키지 버전과 현재 버전을 별도 재평가 identity에 함께 기록한다.
  이 실행을 원래 환경에서 이어 실행한 학습/평가로 취급하지 않는다.
- `run_experiments --eval-only`는 원본 root의 패키지 identity 불일치 또는 새 root의
  학습 결과 부재로 거부된다. 그 검증을 변경하지 않는다. 기존 `eval_cmd()`와
  `recurrent.evaluate`를 재사용하고 별도 출력에서 checkpoint SHA-256·fixture hash·
  현재 패키지·backend·조건·episode 집합·summary·완료 hash를 검사한다.
- RKN은 checkpoint 형식을 유지한 `K*R` 분산 갱신을 사용하고 평가 identity에
  `rkn_covariance_update=stable_v1`을 기록한다.

## 실제 전체 평가 범위

`recurrent/configs/experiment.yaml`의 7개 모델 × 학습 손상 4단계 × seed 3개,
총 84개의 `best.pt`를 사용한다. 조건마다 seed 2000–2029, horizon 250이다.

| 학습 손상 | checkpoint 수 | checkpoint당 조건 | episode 합계 |
| --- | ---: | --- | ---: |
| none | 21 | normal/hole_shift × none/low/mid/high | 5,040 |
| low | 21 | normal/hole_shift × low | 1,260 |
| mid | 21 | normal/hole_shift × mid | 1,260 |
| high | 21 | normal/hole_shift × high + blackout 0/10/20/40 (eval none) | 3,780 |
| expert | 1 | normal/hole_shift × none, CPU 1환경 | 60 |

정책은 `physx_cuda`, GPU당 30환경, 순차 평가 84프로세스다. CPU fixture 생성은
별도 프로세스이며 CPU expert도 GPU 정책과 별도 프로세스다. 정책 11,340 episode는
30환경 기준 378개 배치다. Hole shift는 transport 후 10 step에 2cm 이동한다.
Blackout도 transport 이벤트 이후에 적용되므로 조건을 실행했다고 모든 episode에서
그 이벤트가 발생한 것은 아니다. CPU/GPU 물리·IK 차이 때문에 원본 CPU 결과와
동일 rollout이라고 해석하지 않는다.

## 사용자가 실행할 명령

아래 출력 경로는 smoke와 원본 backup 경로에서 분리되어 있다.
준비 작업에서는 `--execute` 없이 계획만 생성했고, 이후 사용자가 아래 명령으로 전체 평가를 실행했다.

```bash
cd /workspace/graph-recurrent-VLA
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m recurrent.reevaluate \
  --config recurrent/configs/experiment.yaml \
  --source-root /workspace/recovered-main \
  --out-dir outputs/recurrent/reevaluation_main_20261023 \
  --num-envs 30 --execute
```

tmux에서 지속 실행하고 launcher 로그도 남기려면:

```bash
tmux new -s recurrent-eval
cd /workspace/graph-recurrent-VLA
set -o pipefail
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m recurrent.reevaluate \
  --config recurrent/configs/experiment.yaml \
  --source-root /workspace/recovered-main \
  --out-dir outputs/recurrent/reevaluation_main_20261023 \
  --num-envs 30 --execute 2>&1 | tee -a outputs/recurrent/reevaluation_main_20261023/launcher.log
# Ctrl-b, d 로 detach; 재접속은 tmux attach -t recurrent-eval
```

다른 shell에서 확인:

```bash
tail -f outputs/recurrent/reevaluation_main_20261023/launcher.log
tail -f outputs/recurrent/reevaluation_main_20261023/logs/eval_rkn_none_s0.log
nvidia-smi
```

중단 후에는 **동일 명령을 다시 실행**한다. 완료 결과는 엄격히 검증한 후 건너뛰고,
미완료 JSONL은 기존 evaluator의 episode key 검증과 부분 마지막 행 복구 규칙으로
재개한다. 패키지·checkpoint·fixture·평가 조건이 바뀌면 새 출력 경로를 사용한다.
배치 크기만 바꿀 때는 `--num-envs 16`처럼 같은 출력에서 재개할 수 있다.
잠금 파일이나 identity를 지워 우회하지 않는다. 실행기는 동시 쓰기를 막고 signal 시
자식 프로세스를 회수한다. 로그는 재개 시 append한다.

결과는 `eval/*.jsonl`, `*.identity.json`, `*.summary.json`, `*.complete.json`에 남는다.
전체 검증 성공 후에만 `reevaluation.complete.json`을 작성한다.
공용 `ExperimentLogger`와 로컬 artifact manifest를 그대로 사용하며
UUID별 기록은 출력 경로의 `experiments/`에 저장한다. `WANDB_MODE=disabled`는
로컬 기록만 남긴다. HF 업로드나 기존 root 검증을 우회하는 report 생성은 하지 않는다.

## 2026-10-23 Vast.ai smoke 기록

RTX 3090 24,576MiB 1장, driver 590.48.01 (driver CUDA 지원 13.1),
PyTorch 2.14.1+cu130 (runtime 13.0), Python 3.11.17,
ManiSkill 3.0.1, SAPIEN 3.0.3, Gymnasium 1.3.0, NumPy 2.4.6에서 실행했다.
정책은 GPU 0의 `physx_cuda`/`cuda`/render `none`, expert는 별도 프로세스의
`physx_cpu`/`cpu`/`sapien_cpu`였다. Vulkan ICD 탐색 경고가 있었지만 모든 실행은
성공했으며 backend를 CPU로 대체하지 않았다.

`/initial_states.pt`의 seed 2000–2001, 환경·패키지·tensor 검증을 통과하고
새 smoke 경로의 `initial_states_2.pt`에 동일 bytes로 복사했다. 30개 seed fixture는
현재 환경에서 별도 CPU 프로세스로 생성했고 전체 프로세스 11.23초였다.
백업 fixture나 checkpoint는 변경하지 않았다.

| 실제 실행 | episode | 성공 | 전체 프로세스 초 | NVML peak GiB |
| --- | ---: | ---: | ---: | ---: |
| rkn_none_s0, GPU 2환경, normal/hole_shift × none/high | 8 | 0 | 92.48 | 2.97 |
| rkn_none_s0, GPU 30환경, 같은 네 조건 | 120 | 22 | 109.01 | 2.97 |
| mamba_none_s0, GPU 30환경, normal/none | 30 | 23 | 31.31 | 2.96 |
| expert, CPU 1환경, normal/hole_shift | 4 | 4 | 15.63 | 0.50 |

모두 start seed 2000, horizon 250이며 각 실행 전에 기존 JSONL은 없었다.
JSONL·summary·identity·완료 metadata의 episode key/집계/hash와 모든 수치의
유한값을 확인했다. 성공률 0 자체는 smoke 실패가 아니다. RKN 30환경의 조건별
성공은 normal/none 12/30, hole_shift/none 10/30, high 두 조건 각각 0/30이다.
이 작은 표본으로 모델 성능 비교 결론을 내리지 않는다.
NVML은 장치 전체 메모리를 0.1초 간격으로 표본 측정하며 PhysX 등 Torch 밖의
할당도 포함한다. Torch allocator peak만으로 GPU 용량을 판단하지 않았다.

RKN 2환경 완료 명령을 다시 호출했을 때 JSONL bytes가 그대로였고 4.77초에
기존 완료 결과를 검증해 반환했다. 이는 평가 처리량 측정에 포함하지 않았다.
실행기는 `--execute` 경로도 synthetic 자식 실행으로 검증했으며, fixture/GPU정책/
CPU expert 분리와 완료 결과 재개를 검사했다. 이 smoke 준비 시점에는 전체 실행을 시작하지 않았다.

원본 결과·명령·전체 process 시간·phase별 시간·VRAM은 다음 위치에 있다.

```text
outputs/recurrent/reevaluation_smoke_20261023T121531Z/
  verification_report.json
  {rkn_b2,rkn_b30,mamba_b30,expert_cpu}.jsonl
  *.identity.json, *.summary.json, *.complete.json
  *.process.json, *.log
  initial_states_2.pt, initial_states_30.pt
  reevaluate_tests.log, infrastructure_tests.log
```

전체 계획은 `outputs/recurrent/reevaluation_main_20261023/plan.json`에 있으며
84개 원본 학습 완료 검증을 통과했다. 이 준비 시점에는 평가 output이나 full fixture가 없었고,
이후 사용자의 실행으로 생성됐다.
재평가 실행기·기존 orchestration·numerics·tensor 평가 회귀 테스트 45개와 공용
logger/artifact 계약 테스트 35개가 통과했다. Python compile과 `git diff --check`도
통과했다. Ruff는 현재 venv에 설치되어 있지 않아 실행하지 못했다.

### 전체 시간 추정

30환경 RKN의 reset+rollout은 clean 배치 평균 23.93초, high 26.72초였다.
RKN/Mamba의 import·환경 생성·logger·종료 등을 포함한 배치 외 process 비용은
checkpoint당 약 7.71초였다. 각 GPU 배치는 조기 성공이 있어도 250 physical step을
실행했으므로 짧은 성공 episode만으로 배치 시간이 축소된 측정은 아니다.

전체 378배치를 none/blackout 126, low 84, mid 84, high 84로 나누고,
low/mid는 clean/high 중간값, blackout은 clean 비용으로 근사하면 정책 약 10,161초다.
CPU expert 60개는 소표본 비례 추정 약 149초, fixture 약 11초를 더해
**약 2.87시간**, 운용상 **3–4시간**을 예상한다.

일곱 모델 중 실제 rollout은 RKN/Mamba 두 모델의 none 학습 checkpoint만 측정했다.
나머지 모델과 high 학습 checkpoint, blackout 비용은 외삽이며 반복 측정 분산도 없다.
RKN 30환경 측정 중 짧은 synthetic 단위 테스트(2.7초)가 일부 겹쳤으므로 엄밀한
무경합 benchmark는 아니다. 호스트는 64 logical CPU를 보고했다. Mamba와 expert
측정은 그 테스트 종료 후 순차 실행했다. Expert는 두 seed의 조기 성공만 측정했으므로
다른 seed의 장기 rollout과 서버 경합에 따라 실제 시간이 늘 수 있다.
30환경 적합성은 이 두 checkpoint에서 확인한 것이며 전 모델 OOM 부재를 보장하지 않는다.

## 완료 결과를 리포트로 다시 생성

```bash
.venv/bin/python -m recurrent.report \
  --config recurrent/configs/experiment.yaml \
  --source-root /workspace/recovered-main \
  --reevaluation-dir outputs/recurrent/reevaluation_main_20261023 \
  --evaluation-backup-receipt outputs/backups/reevaluation_main_20261023T163639Z/receipt.json \
  --out-dir docs/reports/recurrent_main_20261023
```

다른 머신에서는 두 HF archive를 복원한 뒤 경로를 바꾼다. 로컬 receipt가 없으면
해당 옵션을 생략할 수 있다. 원본 학습 완료·checkpoint·data/config와 기록된 평가
identity·fixture·summary·완료 hash를 검증한다. plotting 환경의 패키지 버전으로
과거 평가 identity를 대체하지 않는다. 원본 파일이나 metadata를 복구·변경하지 않는다.
기본 `recurrent.report`의 기존 root 검증 경로도 그대로 유지한다.

학습 seed 간 표준편차는 표본 표준편차(ddof=1)다. 짝지은 비교의 95% 구간은
학습 seed와 평가 seed를 각각 재표집하는 crossed bootstrap 2,000회이며 탐색적이다.
CSV에 호환성 목적으로 남긴 pooled Wilson 구간은 seed 불확실성으로 해석하지 않는다.
동률이 있는 Spearman 상관은 평균 순위를 사용한다. GPU episode 추론 시간이 비어
있으면 미측정으로 표시하며, 새 단일 정책 지연 측정값을 만들지 않는다.
