# 전체 삽입 앞단과 재학습 Mamba의 실패 영상

[최종 폐루프 평가](recurrent_full_insertion.md)에서 실패 양상이 서로 다른 고정 Box 3개와 이동 Box 3개를 골랐다. 원본 평가와 같은 RANSAC·Transformer·GNN·Mamba 가중치, 카메라, 시드, Box 이동 계획, 필터 우회 조건으로 다시 실행했다.

영상의 `d`는 GT Peg 끝점 중심이 입구 면을 기준으로 얼마나 들어갔는지 보여준다. 0은 입구 면이고, 양수는 그 면의 안쪽이다. 축에서 많이 벗어난 상태에서도 입구 면의 연장선을 넘으면 양수가 될 수 있다. 표의 최대 깊이는 Peg 끝점 중심의 좌표가 개구부 범위 안인 경우에 한해 계산했다. `lateral`은 실제 Peg 끝점 중심과 구멍 축의 거리다. `Box offset`은 Box의 시작 위치 대비 이동 거리다. 모두 분석용 GT 수치다.

| Box | 영상 | 실패 양상 | 최대 깊이 | 성공 깊이 |
| --- | --- | --- | ---: | ---: |
| 고정 | [700866](../../outputs/recurrent_full_insertion/failure_videos/source_runs/fixed_700866/fixed_700866_fail.mp4) | 구멍 옆에서 멈춤. 마지막 횡방향 이탈 55.0 mm | −0.22 mm | 89.42 mm |
| 고정 | [700992](../../outputs/recurrent_full_insertion/failure_videos/source_runs/fixed_700992/fixed_700992_fail.mp4) | 입구 면 직전에서 정체 | −0.50 mm | 86.24 mm |
| 고정 | [700578](../../outputs/recurrent_full_insertion/failure_videos/source_runs/fixed_700578/fixed_700578_fail.mp4) | 얕게 진입한 뒤 마지막에는 입구 밖에 머묾 | +0.75 mm | 77.47 mm |
| 이동 | [700674](../../outputs/recurrent_full_insertion/failure_videos/source_runs/moving_700674/moving_700674_fail.mp4) | Box 114.7 mm 이동. 얕게 진입한 뒤 정체 | +0.37 mm | 81.20 mm |
| 이동 | [700831](../../outputs/recurrent_full_insertion/failure_videos/source_runs/moving_700831/moving_700831_fail.mp4) | 28.45 mm 들어갔다가 다시 입구 밖으로 빠짐 | +28.45 mm | 95.67 mm |
| 이동 | [700806](../../outputs/recurrent_full_insertion/failure_videos/source_runs/moving_700806/moving_700806_fail.mp4) | 깊게 진입한 뒤 성공 깊이에 약 10 mm 못 미침 | +72.23 mm | 82.20 mm |

Vast.ai RTX 3090으로 다시 실행했고 여섯 장면 모두 250스텝 제한에서 실패했다. 원본 대비 최대 깊이 차이는 장면별 0.95 mm 미만이었다. 재실행 로그와 20 fps, 512×512, 261프레임 MP4를 노트북에 저장했다. 이 영상은 실패 위치를 확인하기 위한 것으로 접촉력을 측정하지 않았다.
