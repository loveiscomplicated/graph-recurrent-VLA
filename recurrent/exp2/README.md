# Recurrent 실험 2: Graph 입력 디코더 선정

Graph팀 state36(Transformer tracker + GNN) 입력으로 PegInsertionSide-v1을 푸는 디코더를 고른 실험 코드. hole 고정 조건.

## 최종 결과

| 디코더 | 장면 | 성공 | 95% 구간 (Wilson) |
| --- | --- | --- | --- |
| mamba step 5개 앙상블 | 새 장면 200개 (offset 150~349) | 179/200 = 89.5% | 84~93% |
| mamba step 5개 앙상블 | 새 장면 300개 합계 (offset 50~349) | 262/300 = 87.3% | 83~91% |
| mlp step 5개 앙상블 | 새 장면 200개 | 158/200 = 79.0% | 73~84% |

최종 구성
- 입력: state36, TCP 회전은 6D 표현 (`--rot6`)
- 필터: hole, peg 평균 (`kin_best.json`, prior 없음)
- 학습: GT와 Graph 입력을 α = 0, 0.25, 0.5, 0.75, 1로 섞어 학습 (`--mix`)
- 실행: 학습 seed 0~4 모델의 행동 평균 (`--ens`)

디코더만의 추론 시간 (CPU 1스레드): mamba 5개 앙상블 step당 평균 6.7ms, p99 10.9ms

## 실행 환경

- develop 브랜치 체크아웃 루트에 이 폴더의 파일을 두고 실행함 (`data_collection`, `graph`, `envs` 모듈 사용)
- 경로는 `/workspace` 기준으로 고정되어 있음
  - 데이터: `/workspace/data_kin/train.npz` (`collect_dart.py --graph`로 수집, `state36_gt`, `state36_transformer`, `counts` 포함)
  - Graph 체크포인트 경로: `/workspace/graph_ckpt/paths.json`
- 렌더링에 GPU 필요 (Vast.ai RTX 계열에서 실행)

## 재현

```
python exp2_train.py run --types mamba:step,mlp:step --seeds 0-4 --jobs 10 --rot6 --mix 0,0.25,0.5,0.75,1 \
  --splits train --kin --kin-cfg /workspace/kin_best.json --data-dir /workspace/data_kin --out /workspace/exp2/runs_r6_mix_filter
bash scripts/fix4.sh
python exp2_eval.py run --no-move --env-offset 150 --episodes 200 --workers 8 \
  --cands ens_mlp5@step,ens_mamba5@step --train-seeds 0 --ens /workspace/ens.json --out /workspace/eval/final_ens
```

`scripts/fix4.sh`가 `/workspace/ens.json`을 만든다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `exp2_train.py` | 학습. `--input gt|transformer`, `--kin`, `--rot6`, `--mix` |
| `exp2_eval.py` | 평가, 리포트, 영상. `--oracle`(GT 입력), `--ens`(앙상블) |
| `kin_filter.py` | hole, peg 평균 필터와 prior |
| `bench_exp2.py` | Graph 파이프라인 래퍼 |
| `collect_dart.py` | DART 데모 수집 (`--graph`로 Graph 추정값 함께 저장) |
| `tci.py` | 결과 폴더별 성공률과 t 구간 |
| `precheck.py` | 데이터, 체크포인트, 장면 목록 점검 (quaternion 부호 뒤집힘 포함) |
| `filt_diag.py`, `fail_breakdown.py` | 필터 후 오차, 실패 유형 분석 |
| `grasp_prior.py`, `grasp_diag.py`, `hole_corr.py`, `lock_tune.py` | prior, hole 보정, 필터 동결 탐색 (최종 미채택) |
| `search.py` | spiral search (효과 없음) |
| `exp2_fix.py` | DAgger (구현 미검증) |
| `latency.py`, `lat_local.py` | 디코더 추론 시간 측정 |
| `scripts/` | 실험 단계별 실행 스크립트 (overnight, gtnn, fix~fix4, videos_final) |

## 알려진 문제

- `recurrent/features.py`의 `canon_quat`는 w ≥ 0로 부호를 맞추는데, 위에서 잡는 자세에서는 w가 0 근처라 TCP quaternion 부호가 episode당 약 11번 뒤집힘. 이 실험에서는 모델 입력에서 6D 표현으로 바꿔 우회함 (`exp2_train.rot6`)
