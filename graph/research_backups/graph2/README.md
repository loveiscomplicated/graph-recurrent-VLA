# Graph 실험 백업

기준 브랜치는 `graph`, 기준 커밋은 `2b6c40e6e1eac8a5fcaf3166ed671d4904947449`이다.
이 폴더에는 실험용 실행 코드와 요약 결과가 있다. 팀 공용 Graph 및 Visual 파이프라인은 기준 커밋 상태로 유지했다.

| 실험 | 보고서 |
| --- | --- |
| `fixed_hole_recurrent_rollout_diagnostics` | [결과](../../../docs/visual-graph-integration/fixed_hole_recurrent_rollout_diagnostics.md) |
| `hole_front_object21_ablation` | [결과](../../../docs/visual-graph-integration/hole_front_object21_ablation.md) |
| `hole_front_object21_latest_detector` | [결과](../../../docs/visual-graph-integration/hole_front_object21_latest_detector.md) |
| `hole_detector_cuda_regression` | [결과](../../../docs/visual-graph-integration/hole_detector_cuda_regression.md) |
| `visible_gt_direct_gnn` | [결과](../../../docs/visual-graph-integration/visible_gt_direct_gnn.md) |
| `full_sim_gt_direct_gnn` | [결과](../../../docs/visual-graph-integration/full_sim_gt_direct_gnn.md) |
| `gt20_object21_retraining` | [결과](../../../docs/visual-graph-integration/gt20_object21_retraining.md) |
| `gt20_gnn_detected_tracking_test` | [결과](../../../docs/visual-graph-integration/gt20_gnn_detected_tracking_test.md) |

## 로컬 전체 백업

전체 백업 파일은 `graph2_vast_backup_20261116.tar.gz`이다. SHA-256은 `7588d89b4216eb50b5494be0687ebe828b2243df6c318ea85d76cfcf8b639155`다.
원시 프레임 결과, 새로 생성한 GT 학습 데이터와 재학습 체크포인트는 로컬 전체 백업의 `outputs/` 구조에 보관했다.
Vast.ai 실행 설정에 쓰인 절대 경로는 해당 호스트의 경로이며, 다른 호스트에서는 입력 체크포인트와 산출물 경로를 조정해야 한다.
재학습 가중치의 위치는 `outputs/gt20_object21_retraining/remote/train/best.pt`이고 SHA-256은 `94252699bbe00f2f5e6628a7010b74817c8c308df1c2e12ea81f371f61b20b38`이다.
