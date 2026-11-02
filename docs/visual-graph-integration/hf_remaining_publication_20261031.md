# 미게시 실험 자료 HF 게시 진행 — 2026-10-31

상태: **IN_PROGRESS**. 갱신: 2026-10-31T13:24:45.758812+00:00.

F1 GNN best/last, Robot-only best/last, G의 선택된 A/B/C checkpoint를 기존 실행 UUID로 게시했다. E/F1/G/H/I Stage1·2·3의 보고서 ZIP 7종도 게시했고, F1 모델 링크·H 문서 경로·Stage1 갤러리·E 그림/영상 링크를 보완한 후속 버전도 게시했다. 비공개 범위는 유지한다.

## 재현 snapshot 게시 상태

| Snapshot | 상태 | 고정 revision |
|---|---|---|
| task10f1_results | COMPLETE | [snapshot](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/41f6651295be080513c7ed4ea2e7eb6a75ea04a5/graph/task10f1_results/a803354bff8f414492456af4c0b43bce) |
| task10g | COMPLETE | [snapshot](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/f8d05aff9acb8d4809b48e40ffe310931540b289/graph/task10g/3187bc0109c54cc6b68b08e7753949f1) |
| task10h | COMPLETE | [snapshot](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/4b0b5c191c82ee6da91eeba98a32e30a6ab47b46/graph/task10h/eab73e236d954f348a564a02753e51fb) |
| task10i_stage1 | COMPLETE | [snapshot](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/eec0d84c9531c7d693ea6a81b67eed30dc3d8d76/graph/task10i_stage1/098bd09a011b4c9d99ca881d6a83b760) |
| task10i_stage2 | COMPLETE | [snapshot](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/d4b1a1cbde6a4f16b4da8cdf04c4b3878e89394a/graph/task10i_stage2/34d10bf652b344a38d54b7672dbfe0c0) |
| task10i_stage3 | COMPLETE | [snapshot](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/c79b94210de8ae8da3c73b9bf910e407fcb39077/graph/task10i_stage3/79b2d995c5694e479ad6667e562226c0) |
| task10e | COMPLETE | [snapshot](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/5178d90ac62c18d1317f159cd828a542541e8ce5/graph/task10e/771c27f66f6d4f27850a41a87478dc16) |
| plan_a_retained | COMPLETE | [snapshot](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/e0d958295cefa26da43d69c9698253b9b6e29bb9/graph/plan_a_retained/3788798a719f4ec4a66c224ca42f5d63) |
| task10f_recovery_inputs | 전송 대기/진행 중 | — |
| legacy_reproduction_inputs | 전송 대기/진행 중 | — |
| task10f1_data | 전송 대기/진행 중 | — |

## 완료된 로컬 검증

- 서버 원본 6,495개(2,936,341,929 bytes)를 회수하고 감사 SHA256과 모두 대조했다. 끊긴 전송은 작은 묶음으로 재시도했으며 불완전한 수신 파일은 신뢰하지 않았다.
- F1 필수 입력 해시 참조1,024건과 I Stage3 raw/GT 참조712건의 archive 수록 여부를 전수 검증했다. 누락0건이다.
- F1 graph/I raw·GT의 원래 경로1,072건을 복원 경로 alias와 대조했다. 누락0건이다.
- F1 데이터 archive의6,495개 member를 새 로컬 디렉터리에 실제로 복원해 개별 크기·SHA256·hardlink 복원을 검증했다.
- 게시 전 모든 archive member의 실제 bytes를 inventory 해시와 대조한다. 전체 게시 완료 후 원격 크기/LFS SHA256/Git blob SHA1, 기존 payload 보존, 고정 revision별 catalogue 및 Legacy 복원을 최종 검증할 예정이다.

## 진행 기록

- Task10-F recovery의 grouping ID는 초기 pilot이 아닌 원래 continuation UUID `d42d747b57ea4354a27a0c3b4be257d9`로 정리했다. R0/R1 child metadata와 원래 ID는 변경하지 않았다.
- PLAN A 상위 metadata의 stale running 상태는 성공으로 바꾸지 않고 historical evidence로 보존했다.
- 기존 공통 publisher/maintenance workflow를 사용한다. 학습·평가·CUDA/PhysX/renderer 실행, Git commit/push, 공개 Space 게시, 원본 연구 payload 삭제는 수행하지 않았다.

[서버 회수 검증](../../outputs/hf_publication_remaining_20261031/retrieval-verification.json) · [입력 참조 검증](../../outputs/hf_publication_remaining_20261031/input-closure-verification.json) · [경로 검증](../../outputs/hf_publication_remaining_20261031/input-path-verification.json) · [실제 F1 복원 검증](../../outputs/hf_publication_remaining_20261031/restore-test-f1/restoration_verification.json)
