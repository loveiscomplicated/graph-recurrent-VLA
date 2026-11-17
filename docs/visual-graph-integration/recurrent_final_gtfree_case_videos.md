# 최종 Mamba의 성공·실패 분류와 대표 영상

최종 GT 없는 앞단과 파인튜닝 Mamba 5개 앙상블을 사용한 기존 평가 64개를 분석했다. 고정 Box 20/32(62.5%), 이동 Box 10/32(31.25%)가 성공했다. 실패 전체는 고정 12개, 이동 22개다. 모든 실패는 250스텝 시간 제한으로 끝났다.

## 분류 기준

시뮬레이터 GT로 계산한 Peg 끝점 중심의 Hole 좌표계를 사용했다. 중심이 Hole 단면의 y·z 범위 안에 있는 프레임에서 입구 면을 지난 깊이의 최댓값을 구했다. 0.1 mm를 넘은 경우를 진입으로 집계했다. 이는 끝점 중심의 기하학적 통과 판정이며, Peg 단면 전체의 충돌 없는 삽입이나 접촉 힘까지 판정하지 않는다. 20 mm를 넘겨 진입한 뒤 끝점이 단면을 벗어나거나, 최종 깊이가 0 이하이거나, 최대 깊이에서 10 mm 이상 줄면 후퇴로 분류했다. 나머지 20 mm 초과 미완료는 깊은 진입 후 미완료로 분류했다.

## 전체 실패 분포

| 유형 | 고정 Box | 이동 Box |
| --- | ---: | ---: |
| 구멍 단면 정렬 미도달 | 0 | 2 |
| 입구 앞에서 정체 | 3 | 2 |
| 입구 면에서 정체 | 0 | 2 |
| 0.1~3 mm 얕은 진입 | 7 | 10 |
| 3~20 mm 부분 진입 | 1 | 0 |
| 20 mm 초과 진입 후 미완료 | 1 | 4 |
| 20 mm 초과 진입 후 후퇴 | 0 | 2 |

고정의 9/12건, 이동의 16/22건에서 0.1 mm보다 깊은 진입 프레임이 있었다. 고정의 얕은 진입 7건과 이동의 얕은 진입 10건이 가장 큰 집단이다. 이 집단은 초반 접근을 완전히 놓친 경우보다 입구에서 아주 조금 들어간 뒤 정체하거나 물러나는 동작에 해당한다.

## 대표 장면과 영상

| 조건 | 유형 | 시드 | 최대 진입 깊이 | 마지막 깊이 | 전체 궤적 Object21 횡방향 P90 | 영상 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 고정 | 성공 | 700882 | 94.49 mm | 94.49 mm | 4.82 mm | [영상](../../outputs/recurrent_final_gtfree/case_videos/source_runs/fixed_success_700882/fixed_success_700882.mp4) |
| 이동 | 성공 | 700878 | 87.32 mm | 87.32 mm | 2.02 mm | [영상](../../outputs/recurrent_final_gtfree/case_videos/source_runs/moving_success_700878/moving_success_700878.mp4) |
| 고정 | 입구 앞에서 정체 | 700451 | -0.28 mm | -1.94 mm | 3.07 mm | [영상](../../outputs/recurrent_final_gtfree/case_videos/source_runs/fixed_before_entrance_700451/fixed_before_entrance_700451.mp4) |
| 고정 | 0.1~3 mm 얕은 진입 | 700674 | 0.62 mm | -0.94 mm | 4.08 mm | [영상](../../outputs/recurrent_final_gtfree/case_videos/source_runs/fixed_shallow_700674/fixed_shallow_700674.mp4) |
| 고정 | 3~20 mm 부분 진입 | 700732 | 4.47 mm | 4.44 mm | 74.35 mm | [영상](../../outputs/recurrent_final_gtfree/case_videos/source_runs/fixed_partial_700732/fixed_partial_700732.mp4) |
| 고정 | 20 mm 초과 진입 후 미완료 | 700866 | 68.01 mm | 68.01 mm | 5.75 mm | [영상](../../outputs/recurrent_final_gtfree/case_videos/source_runs/fixed_deep_then_stalled_700866/fixed_deep_then_stalled_700866.mp4) |
| 이동 | 구멍 단면 정렬 미도달 | 700807 | 미정렬 | -11.31 mm | 14.46 mm | [영상](../../outputs/recurrent_final_gtfree/case_videos/source_runs/moving_never_aligned_aperture_700807/moving_never_aligned_aperture_700807.mp4) |
| 이동 | 입구 앞에서 정체 | 700729 | -0.10 mm | -1.47 mm | 4.69 mm | [영상](../../outputs/recurrent_final_gtfree/case_videos/source_runs/moving_before_entrance_700729/moving_before_entrance_700729.mp4) |
| 이동 | 입구 면에서 정체 | 700549 | 0.09 mm | -1.85 mm | 2.78 mm | [영상](../../outputs/recurrent_final_gtfree/case_videos/source_runs/moving_at_entrance_700549/moving_at_entrance_700549.mp4) |
| 이동 | 0.1~3 mm 얕은 진입 | 700674 | 0.32 mm | -0.77 mm | 2.69 mm | [영상](../../outputs/recurrent_final_gtfree/case_videos/source_runs/moving_shallow_700674/moving_shallow_700674.mp4) |
| 이동 | 20 mm 초과 진입 후 미완료 | 700806 | 72.99 mm | 72.29 mm | 6.53 mm | [영상](../../outputs/recurrent_final_gtfree/case_videos/source_runs/moving_deep_then_stalled_700806/moving_deep_then_stalled_700806.mp4) |
| 이동 | 20 mm 초과 진입 후 후퇴 | 700882 | 33.93 mm | -2.56 mm | 98.19 mm | [영상](../../outputs/recurrent_final_gtfree/case_videos/source_runs/moving_deep_then_retreated_700882/moving_deep_then_retreated_700882.mp4) |
| 이동 | 진입 후 후퇴·종료 시 파지 이탈 | 700831 | 61.26 mm | 37.25 mm | 24.62 mm | [영상](../../outputs/recurrent_final_gtfree/case_videos/source_runs/moving_deep_then_retreated_grasp_loss_700831/moving_deep_then_retreated_grasp_loss_700831.mp4) |

표의 마지막 깊이와 최대 깊이는 마지막 행동 실행 전까지의 기록이다. 성공은 환경의 실제 성공 판정을 따른다. 횡방향 P90은 해당 시드 전체 프레임의 추정 오차다.

## 원인 분석

- 입구 앞 정체와 얕은 진입 대표인 고정·이동 700674의 전체 궤적 Object21 횡방향 P90은 각각 4.08 mm와 2.69 mm다. 해당 두 궤적은 Peg 슬롯이 전부 비는 프레임이 없었다. 같은 얕은 실패 집단에서도 장시간 전체 Peg 누락 없이 진입이 끝까지 이어지지 않는 경우가 있다. 행동 정답 대비 순간 오차, 접촉·자세 오차의 기여를 추가로 구분해야 한다.
- 깊은 진입 후 미완료 대표인 고정 700866은 최대 68.01 mm, 이동 700806은 최대 72.99 mm까지 진행했다. 전체 횡방향 추정 P90은 각각 5.75 mm와 6.53 mm이며, 종료까지 파지를 유지했다. 20 mm 초과 깊이만으로 작업 성공이 되지 않으므로 끝까지 미는 진행 동작을 확인할 장면이다.
- 고정 700732는 4.47 mm 진입 후 4.44 mm 부근에서 종료됐고, Object21 횡방향 P90이 74.35 mm였다. 이동 700882는 33.93 mm까지 진행한 뒤 마지막 깊이 -2.56 mm로 후퇴했으며, P90은 98.19 mm였다. 700882에는 연속 184프레임 Peg 전체 미검출이 있었다. 두 시드는 앞단 추정이 크게 틀어진 실패로 이미 확인된 대표다.
- 이동 700831은 61.26 mm에서 37.25 mm로 후퇴했고 마지막 프레임에서 파지를 잃었다. 후퇴 장면도 파지를 유지한 700882와 종료 상태가 달라 별도 영상을 포함했다. 파지 이탈의 선후관계와 인과관계는 이 분류만으로 확정하지 않는다.
- 이동 700807은 기록 전체에서 Peg 끝점 중심이 구멍 단면 범위에 들어오지 않았다. 삽입 전에 정렬 자체가 부족한 유형으로 선택했다.

## 영상 생성과 재현 확인

원본 평가의 RGB-D가 남아 있지 않아 같은 초기 시드와 저장된 Action7, Box 이동 계획을 재생해 영상을 생성했다. Mamba의 새로운 행동을 계산하지 않고 원래 평가 행동을 순서대로 실행했다. 프레임마다 GT Object21과 원본 기록을 대조하고, 종료 성공 여부와 스텝 수를 검증했다. 영상 상단에는 스텝, 실제 입구 기준 깊이(d), 원래 기록의 Object21 횡방향 추정 오차(err), Box 이동량을 표시했다. CUDA 렌더링과 CPU 물리 시뮬레이션을 사용했다.

[전체 분류 원본](../../outputs/recurrent_final_gtfree/case_videos/classification.json) · [대표 선정](../../outputs/recurrent_final_gtfree/case_videos/selection.json) · [영상 생성 코드](../../outputs/recurrent_final_gtfree/case_videos/record_cases.py)

영상 13개의 원본 기록 대비 GT 성분 차이는 모두 0이었다. 성공 판정과 종료 스텝 수도 일치했다. 512×512 MP4 전체 디코딩과 프레임 수를 확인하고 노트북에 다운로드했다.
