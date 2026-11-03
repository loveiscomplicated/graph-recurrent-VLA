# Stage3 가시성 제한 데이터의 HF 의존 자료 게시 완료

2026-11-02. **COMPLETED**. Stage3 R/D Train과 공통 Validation을 복원하는 데 빠져 있던 기존 F1 입력 백업을 비공개 HF 저장소에 게시했다. 네 백업의 고정 revision, 의존 관계, 원본 경로 대응과 복원 도구를 연결한 별도 안내도 게시했다.

[HF 다운로드 및 복원 안내](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/b73b0d728cf64c20bdd084375f6b00a6aa8a130a/graph/task10i_stage3_dataset_completion_20261102)

## 게시한 자료

누락 백업은 `task10f1_data`다. 기존 F1 학습 run ID `a803354bff8f414492456af4c0b43bce`를 재사용했으며 새 실험 run을 만들지 않았다. 압축파일은 1,724,651,022 bytes, 논리 member는 6,495개다. SHA256은 `42db672f9ae2c16fe55e0235fd656713903201c59972f15361392cdccafb743c`다.

[F1 입력 백업](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/e25cd68cb4813b637e56a4815b77a8ae216a8922/graph/task10f1_data/a803354bff8f414492456af4c0b43bce)

기존 Stage1/Stage2/Stage3 백업은 그대로 재사용했다. 새 복원 catalogue는 아래 네 개만 데이터 의존성으로 연결한다. 과거 전체 연구 복원 작업의 다른 미게시 자료까지 완료됐다는 판정은 아니다.

| Snapshot | 고정 revision | Archive bytes | Member |
|---|---|---:|---:|
| task10i_stage1 | `eec0d84c9531c7d693ea6a81b67eed30dc3d8d76` | 182,534,422 | 8,514 |
| task10i_stage2 | `d4b1a1cbde6a4f16b4da8cdf04c4b3878e89394a` | 16,849,902 | 773 |
| task10i_stage3 | `c79b94210de8ae8da3c73b9bf910e407fcb39077` | 861,503,782 | 4,620 |
| task10f1_data | `e25cd68cb4813b637e56a4815b77a8ae216a8922` | 1,724,651,022 | 6,495 |

새 안내의 revision은 `b73b0d728cf64c20bdd084375f6b00a6aa8a130a`이며, `publication_catalog.json`, `restore_publication.py`, `README.md`, `verification.json`의4개 파일을 추가했다. 기존 payload 삭제는0건이며 비공개 설정을 유지했다.

## 복원 대상 데이터 분할

| Split | Physical Parent | Logical episode |
|---|---:|---:|
| Train R | 96 | 272 |
| Train D | 96 | 272 |
| Shared Validation | 16 | 48 |

가시성 제한은 **초기 Main 카메라 관측에서 Peg 양 끝과 파지 영역이 보이는 조건**이다. 모든 timestep의 가시성은 보장하지 않는다. 신규 Final Test Normal32/Stress12는 미수집이며 이번 게시에도 포함되지 않는다. 과거 F1 원자료나 engineering 자료를 모두 학습 대상으로 사용하지 말고, 복원된 Stage3 split manifest가 지정한 episode만 사용해야 한다. R/D는 78 Parent와 236 logical episode를 공유한다.

## 복원 방법

HF 안내 폴더의 `publication_catalog.json`, `restore_publication.py`, `README.md`를 함께 내려받는다. 비공개 저장소 읽기 권한이 있는 HF 인증과 `huggingface_hub`가 준비된 환경에서 다음을 실행한다.

```sh
python restore_publication.py --destination ./restored-stage3 --label task10i_stage3
```

도구가 네 의존 백업을 함께 내려받고, 새 디렉터리에 해제하면서 archive 및 member 해시를 검사한다. simulator나 checkpoint 코드를 실행하지 않는다. 복원 후 데이터 선택 기준은 다음 파일이다.

```text
outputs/task10i/stage3/run/dataset_R_train_manifest.json
outputs/task10i/stage3/run/dataset_D_train_manifest.json
outputs/task10i/stage3/run/shared_validation_manifest.json
```

원본 manifest 안의 `/workspace/...` 절대경로는 보존했다. 복원 root에 생성되는 `original_path_aliases.json`의 prefix 대응을 loader에 적용해 `raw_path`와 `gt_path`를 읽는다. 이 catalogue는 데이터 의존성 복원을 제공하며, simulator/perception 전체 실행 환경 설치를 대신하지 않는다.

## 실제 검증

- 게시 전 네 archive를 실제 새 로컬 디렉터리에 해제하고 **20,402/20,402 member**의 크기·SHA256을 검증했다.
- 복원된 세 Stage3 manifest가 참조하는 중복 제거 Raw HDF5/GT NPZ **712/712개**를 원래 SHA256과 대조했다. 누락·불일치는0개다.
- 원격 archive 네 개의 LFS SHA256, inventory와 manifest 등12개 객체의 크기 및 LFS SHA256/Git blob SHA1을 로컬 검증본과 비교했다.
- 게시 후 새 안내4개 파일을 고정 revision에서 실제 다시 내려받아 SHA256을 비교했다. catalogue의 모든 의존 label이 포함되는 것도 검사했다.
- F1 게시 전 기존 payload를 모두 보존했고, `.gitattributes`의 기존 내용이 append-only로 보존됐는지 확인했다. 비공개 상태도 유지됐다.
- 원격 archive 전체를 다시 다운로드하지는 않았다. 원격 해시가 일치하는 동일 로컬 archive의 실제 해제 검증과 원격 객체 검증을 구분한다.

## 실행과 보존

현재 branch의 `utils.experiment_artifacts`, `utils.experiment_artifact_maintenance` 및 `tools.publish_experiment --maintenance-plan --apply`를 재사용했다. 공통 API나 실험 identity 규칙을 수정하지 않았다. 공통 게시 모듈 변경이 없어 단위 테스트를 반복하지 않았고, 실제 archive·복원·원격 게시 검증을 수행했다.

초기 로컬 기본 전송이 느려 SDK Xet 전송으로 전환했으나 연결 오류와 장기 정체가 있었다. 해당 게시 프로세스만 종료했다. 58576054의 원본6,495개를 전수 대조한 뒤 기존 압축파일과 byte identity를 재현하고, 서버에서 공통 도구로 게시를 완료했다. 새 압축파일의 크기·SHA256은 기존 검증본과 같다. 인증 token은 명령 인자나 게시 파일에 넣지 않고 원격 게시 프로세스 환경에 일시 주입했다.

연구 원자료를 삭제하거나 학습·평가 프로세스를 중단하지 않았다. 이번 작업에서 학습, simulator, renderer, CUDA/MPS 모델 추론 및 GPU PhysX를 실행하지 않았다. Git commit/push, W&B 변경, 공개 범위 변경은 하지 않았다.

## 검증 근거

- [최종 원격 및 복원 검증](../../outputs/task10i/stage3/hf_completion_20261102/completion_verification.json)
- [네 archive 원격 객체 및 기존 payload 보존 검증](../../outputs/task10i/stage3/hf_completion_20261102/verification.json)
- [712개 데이터 참조의 실제 복원 검증](../../outputs/task10i/stage3/hf_completion_20261102/dataset_restoration_verification.json)
- [서버 동일 archive 재구성](../../outputs/task10i/stage3/hf_completion_20261102/remote_rebuild.log)
- [F1 게시 receipt](../../outputs/task10i/stage3/hf_completion_20261102/remote_publication_receipt.json)
- [복원 안내 게시 receipt](../../outputs/task10i/stage3/hf_completion_20261102/catalogue-plan.receipt.json)
- [서버 원자료 보존 확인](10i_stage3_presence_58576054_20261102.md)
