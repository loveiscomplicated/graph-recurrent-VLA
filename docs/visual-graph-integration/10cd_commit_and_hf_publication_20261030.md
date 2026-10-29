# Task10-C/D — 구현 커밋 및 Hugging Face 게시

2026-10-30 후속 게시 기록이다. 기존 연구 보고서의 “커밋/외부 게시 없음”은 실험 종료 당시 기록이며, 이번 작업에서 명시적인 사용자 요청으로 아래 자료를 게시했다. 학습·평가를 다시 실행하지 않았다.

## 구현 의도별 커밋

| 커밋 | 목적 |
|---|---|
| `14829a4` | 최종 연구 보고서의 docs 저장 위치 지침 |
| `2cc158c` | Task10-C2 조건부 학습 신호 진단·테스트·보고서 |
| `a649d3d` | PLAN A 첫 프레임 추출·검증 보고서 |
| `098be8b` | Task10-D 실제 수집·Graph cache·동일 예산 A/B 학습·Live Reach 평가 및 보고서 |

Task10-D의 보고서 병합에 있던 중복 `condition` keyword 오류도 publication checkout에서 수정하고 회귀 테스트를 추가했다. 실험 당시의 동결 source와 report-only 호환 어댑터는 원본 산출물로 보존했다. 모델·학습 조건·체크포인트·실험 결과를 소급 변경하지 않았다.

## 비공개 HF 게시

기존 저장소만 재사용했고 공개 범위 변경·삭제·새 실험 UUID 생성은 없다. Git push와 공개 Space 배포는 수행하지 않았다.

- [10-D 데이터셋 및 전체 보존 자료](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/8cfa671c2fe8ab1fffd924435392ff0728f21c79/graph/task10d/60ba447bf1b541e793a3d54ce68aa719)
- [10-C1 보존 자료](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/8cfa671c2fe8ab1fffd924435392ff0728f21c79/graph/task10c1/683c518f00ed45ec890dc4a4d1717726)
- [10-C2 보존 자료](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/8cfa671c2fe8ab1fffd924435392ff0728f21c79/graph/task10c2/87ac9525933c4562b8930018fb165b0a)
- [A 모델 체크포인트 12개](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/tree/2b54ccde348b68712004657b76caa9f5259bb1a9/graph/f827e5bca96146f4a3602577702d984b)
- [B 모델 체크포인트 12개](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/tree/2b54ccde348b68712004657b76caa9f5259bb1a9/graph/f37c58e690d448a09866cbabb527b454)
- [10-D 보고서 ZIP](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/d7973469f704b95dbd2beb0df323f61805a41dd0/graph/f8b68a79afe349fbb59966625862b6f1)

10-C1·10-C2·10-D·첫 프레임 자료의 보고서 ZIP 4종은 `graph-recurrent-rollouts`에 게시했다. 첫 프레임 자료는 원본 collection UUID `d1fd6173cb3648bfa4c03507f84b7e71`를 재사용한다. 비교 parent의 stale `running` metadata를 임의로 성공 처리하지 않았다. 종료된 기존 run의 metadata/config/metrics를 게시 준비 디렉터리에 복사해 경로만 복원했다.

## 10-D 데이터 구성

| 경로 (`files/` 아래) | 내용 |
|---|---|
| `raw/` | Original96 + Perturbed96 + Validation8 = HDF5 200개. 두 카메라 RGB-D·보정 정보·Robot76·새 Expert Action7·T/T+1 정합 |
| `cache/` | Episode Graph NPZ 200개, 총 9,122 action frames. GNN 이전 Node21/Edge4·presence·Robot76·Action7·provenance |
| `final_expert_reference/raw/` | 별도 held-out Expert reference HDF5 18개. 학습·모델 선택에 사용하지 않음 |
| `evaluations/` | 최종36 + Development16 + Smoke2 = 영상54개 및 상태·Action trace |
| `training_runs/` | A/B 각각12개 체크포인트 및 원래 학습 기록 |
| `frozen_source/` | 실행 당시 동결 source. 사후 Git reporting fix와 구분 |

Offset·GT·Expert target은 생성/평가 metadata이며 Student feature가 아니다. A는 Original96, B는 Original96+Perturbed96을 사용했다. 같은 유효 학습량과 남은 Parent sampling weight 차이는 원래 보고서·manifest에 보존돼 있다. 원본 raw 데이터와 Graph 모두 업로드했으므로 재학습 시 매 epoch vision을 재실행할 필요가 없다. Simulator/vision의 외부 패키지·모델 의존성은 기록된 runtime/version/hash를 확인해야 한다.

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Graph-Recurrent-VLA/experiment-backups",
    repo_type="dataset",
    revision="8cfa671c2fe8ab1fffd924435392ff0728f21c79",
    allow_patterns=["graph/task10d/60ba447bf1b541e793a3d54ce68aa719/**"],
    local_dir="hf-task10d",
)
```

인증된 계정에 해당 비공개 저장소 읽기 권한이 있어야 한다. Token을 코드나 명령 인자에 넣지 않는다. 내려받은 각 snapshot의 `manifest.json`에 파일별 SHA256·크기가 있다. HTML entrypoint는 `files/research_report.html`이다.

## 실제 검증

- 공통 `tools.publish_experiment` / `utils.experiment_artifacts`를 사용했다. 대량 원본 보존은 기존 revision-pinned maintenance workflow의 명시적 addition 목록으로 처리했다. 팀별 HF transport는 추가하지 않았다.
- 새 원격 파일 **5,175개**의 크기 및 LFS SHA256/Git blob SHA1을 로컬과 대조했다.
- 기존 원격 Model1,247 + Rollouts1,429 + Backup5,988 경로·내용과 로컬 원본 **5,441개**의 불변성을 확인했다.
- 로컬 관련 테스트10개:8개 PASS,2개 SKIP(CUDA 및 원격 Reach fixture). Compile 및 Git whitespace 검사도 통과했다. 이번 게시 작업에서 simulator/GPU 학습·평가는 재실행하지 않았다.
- 자료 전송 중 일시적인 HF502 응답은 SDK 재시도 후 완료됐다. 세 저장소는 계속 private다.
- 처음 작업 범위 확인 이후 다른 작업에서 새로 생성된 `graph/gt_gru10e/`와 10-E 보고서는 이번 커밋/게시 범위에서 제외했다.

[원격 해시 검증](../../outputs/hf_publication_20261030/verification.json) · [게시 작업 receipt](../../outputs/hf_publication_20261030/upload-summary.json) · [데이터 백업 receipt](../../outputs/hf_publication_20261030/backup-upload.json) · [명시적 파일 목록](../../outputs/hf_publication_20261030/backup-plan.json)
