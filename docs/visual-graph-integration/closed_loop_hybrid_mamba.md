# 새 Object21 GNN과 기존 Mamba의 고정 Hole 폐루프 평가

## 실험 조건

[Recurrent 팀의 최종 혼합 학습](recurrent_final_mixed_inputs_review.md)에서 사용한 Mamba step 체크포인트 5개를 동결했다. 필터, TCP 6D, Hole 기준 상대 위치와 회전, 정규화, 행동 집계를 원래 Decoder와 Ensemble 구현으로 그대로 실행했다. 필터의 Peg·Box 관측 수 기준은 각각 5, 2다. TCP와 손가락 상태는 기존 로봇 proprioception 경로에서 가져왔다.

- **기존 Tracker·GNN:** 기존 가변 track 수 Transformer와 검출·추적 입력으로 학습된 GNN을 사용했다. 두 가중치는 [기존 재실행 진단](fixed_hole_recurrent_rollout_diagnostics.md)에 복원된 후보이며 당시 원본 Graph 체크포인트와의 동일성은 인증되지 않았다.
- **새 Tracker·GNN:** 가시 물리 꼭짓점과 최종 검출 후보를 5 mm 안에서 일대일로 연결한 뒤, 954에폭 Transformer와 Box 공통 평행 이동 보정 및 54에폭 [GNN](hybrid_input_gnn_finetune.md)을 사용했다. 보이는 슬롯의 좌표는 실제 검출기 출력을 유지했다. GT는 새 앞단의 꼭짓점 대응과 진단의 정답 측정에 사용했다.

두 조건에 같은 개선된 CUDA RANSAC 검출기와 같은 최신 Main 카메라를 적용했다. 과거 89.5%의 기록에 사용된 카메라와 검출기는 이번과 다르다. 이번 비교에서는 기존·새 Tracker와 GNN의 차이를 보며, GT로 먼저 대응을 확인할 수 있는 새 구성의 결과다. 검출과 번호 배정이 모두 카메라 관측만으로 동작하는 성능은 별도 조건이다.

보관된 고정 Hole 최종 테스트 환경 시드 200개에서 결과에 관계없이 난수 시드 20261117로 32개를 비복원 추출했다. 각 장면에서 두 조건을 같은 초기 상태에서 시작하고 실행 순서를 교차했다. Expert 행동 재생 없이 매 스텝 Mamba 앙상블 행동으로 환경을 전이했고, ManiSkill의 success 판정으로 성공을 기록했다. 한 장면 최대 250스텝, Box 고정이다.

## 결과

| 구성 | 성공 수 | 성공률 | 95% Wilson 구간 |
| --- | ---: | ---: | ---: |
| 기존 Tracker·GNN + 기존 Mamba | 9/32 | 28.1% | 15.6–45.4% |
| 새 Tracker·보정·GNN + 기존 Mamba | 22/32 | 68.8% | 51.4–82.0% |

같은 장면에서 둘 다 성공 9개, 새 앞단만 성공 13개, 기존 앞단만 성공 0개, 둘 다 실패 10개였다. 성공률 차이는 +40.6%p였으며 정확 양측 McNemar 검정은 p=0.000244였다. 두 조건 모두 32개 장면에서 적어도 한 번 파지에 성공했고, 각 조건에서 성공한 장면의 성공 스텝 수 중앙값은 모두 102였다. 새 앞단이 실패한 10개 장면은 모두 250스텝 한도까지 실행됐다.

| 전체 롤아웃 프레임에서 측정한 Object21 횡방향 오차 P90 | 기존 Tracker·GNN | 새 Tracker·보정·GNN |
| --- | ---: | ---: |
| Recurrent 필터 전 | 25.80 mm | 9.62 mm |
| Recurrent 필터 후 | 17.98 mm | 9.56 mm |

오차는 예측 Peg 끝점–Hole 중심 벡터의 GT Hole 축에 수직인 성분을 사용했다. 각 조건이 자체 행동으로 만든 실제 관측을 집계했으므로, 두 열은 같은 시점·자세의 오차를 직접 짝지어 비교한 값과 구별된다. 실패 후 계속 진행된 프레임도 포함된다.

과거 고정 Hole 89.5% 실험에서 이번에 뽑은 같은 시드 32개의 보관 결과는 27/32였다. 그 때는 카메라와 검출기 조건이 달랐고 당시 Graph 체크포인트의 동일성도 확인되지 않아, 보관 결과는 새 환경 통제 비교와 별도로 둔다. 이번 실험은 Box를 고정하고 GT 기반 ID 배정을 사용했다. Box 이동 및 완전한 관측 기반 ID 배정에서의 성공률은 미측정이다.

## 검증 및 산출물

RTX 3090의 CUDA 렌더링에 CPU PhysX를 사용했다. 기존 Mamba는 복원된 체크포인트 파일 그대로 호출했다. 새 Transformer의 실시간 프레임 단위 추론은 별도의 저장된 입력 296프레임에서 배치 추론과 최대 1.19e-7 m 차이였다. 공통 ExperimentLogger를 이용했고, 두 조건의 동일 초기 상태와 코드·체크포인트 해시, 32개 장면의 중복 없는 쌍, 기록 프레임 수 및 성공 집계를 대조했다. 다중 작업 도중의 시간 기록은 실시간 단독 성능 보장에 사용하지 않았다.

- [성공률과 장면별 실행 정보](../../outputs/closed_loop_hybrid_mamba/source_runs/summary.json)
- [롤아웃 실행](../../outputs/closed_loop_hybrid_mamba/run_rollouts.py) · [새 앞단 연결](../../outputs/closed_loop_hybrid_mamba/hybrid_frontend.py) · [집계](../../outputs/closed_loop_hybrid_mamba/summarize.py)
- [프레임 추론 검증](../../outputs/closed_loop_hybrid_mamba/source_runs/streaming_check.log) · [장면별 원본 기록 디렉터리](../../outputs/closed_loop_hybrid_mamba/source_runs/)
