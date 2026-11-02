# Vast.ai 57987772 — Safe Disk Cleanup

**Status: COMPLETED_WITH_LIMITED_SPACE.** 원격 identity와 로컬 파일·archive member SHA256이 검증된 과거 미디어 사본만 삭제했다. 연구 데이터·모델·코드와 실행 프로세스는 보존했다. 아래는 실제 실행 결과다.

삭제 **246개**, 논리 크기 **633,798,092 bytes**, allocated bytes **634,302,464 bytes**. 그룹별 `df` 실제 회수량은 **634,302,464 bytes (604.92MiB)**로 예상량과 같았다. **4GiB 목표에는 도달하지 않았다.** 검증된 비사용 사본이 부족한 상태에서 원본/의존성을 추가 삭제하지 않았다.

## 디스크 측정

| 시점 | Used bytes | Available bytes | Available GiB |
|---|---:|---:|---:|
| 최초 감사 | 17,179,787,264 | 81,920 | 0.000076 |
| 삭제 직전 | 17,179,811,840 | 57,344 | 0.000053 |
| 삭제 직후 | 16,545,509,376 | 634,359,808 | 0.590794 |
| 감사 기록 보존 후 최종 | 16,569,847,808 | 610,021,376 | 0.568127 |

최종 순 여유 증가(최초 감사 대비): **609,939,456 bytes**. 삭제 allocated 예상량과 최종 순 증가의 차이는 **24,363,008 bytes**이며, 영구 감사 파일 저장 및 동시 로그/metadata 쓰기를 포함한다. Overlay lower image 레이어의 가시적 `du` 합계는 약34.27GB로 quota16GiB와 다르다. 회수량은 `du` 추정이 아니라 매 그룹 뒤 `/`와 `/workspace`의 실제 `df`로 측정했다. 호스트 `/etc/hosts`, `/usr/bin/nvidia-smi` 마운트의 남은 용량은 사용 가능 공간으로 간주하지 않았다.

## 주요 사용량과 보호 결정

| Root | 조사 당시 대략 크기 | 처리 |
|---|---:|---|
| F1 pretrain | 3.5G | 전체 보존: pretrained, raw/GT/cache 및 Stage3 참조 |
| Task10-G | 2.0G | 전체 보존: B/C checkpoint, Source/Codec 및 재평가 의존성; 미검증 miniature 포함 |
| PLAN A | 2.1G | 데이터·코드·모델·활성 pipeline.log 보존, 검증된 과거 미디어만 일부 삭제 |
| Task10-D | 1.1G | raw/cache/체크포인트/코드 보존, 과거 미디어만 일부 삭제 |
| Stage3 v3 수집 | 1.3G | 전체 보존; v5 manifest의 실제 대상 보호 |
| Stage4A v5 | 1.1G | 전체 보존; 나머지 v1–v11도 보존 |
| Stage1 전체 | 726M | 선정/후보/인계/원본 전부 보존 |
| Task10-H v5 | 296M | 전부 보존 |
| full_pipeline_models | 659M | 현재 Stage4 source의 DINO 의존성이므로 보존 |
| PLAN A operations | 679M | 활성 tail/tmux root 및 불명확 runtime 의존성 보존 |
| full_pipeline compile caches | 약520M | 정확한 HF 백업/비사용 증거가 없어 삭제하지 않음 |
| /root/.cache | 약1.9M | 무차별 제거하지 않음 |

정확한 모든 Task별 파일 수·논리/allocated 사용량은 [disk_usage_by_task.csv](../../outputs/vast57987772_safe_disk_cleanup/disk_usage_by_task.csv), 전체 root의 read-only `du`는 [root_usage_before.txt](../../outputs/vast57987772_safe_disk_cleanup/root_usage_before.txt)에 있다.

## 실제 삭제와 HF 복원

| 과거 Task | 삭제 파일 | Logical bytes | Allocated bytes |
|---|---:|---:|---:|
| plan_a_bc_3788798a719f4ec4a66c224ca42f5d63 | 117 | 391,545,511 | 391,802,880 |
| plan_a_eval_reset_restore_20261029 | 48 | 172,171,801 | 172,253,184 |
| task10d_geometry_reach | 62 | 57,800,710 | 57,929,728 |
| task10e_gt_gru | 19 | 12,280,070 | 12,316,672 |

삭제한 파일은 모두 `.mp4` 또는 `.png` 계열의 비사용 과거 진단/학생 rollout 미디어다. 디렉터리를 삭제하지 않았고 `rm -rf`나 cache/system prune을 사용하지 않았다. 파일별 정확한 경로·해시·HF object/member·revision·복원 위치는 [cleanup_plan.csv](../../outputs/vast57987772_safe_disk_cleanup/cleanup_plan.csv)와 [deleted_files.csv](../../outputs/vast57987772_safe_disk_cleanup/deleted_files.csv)에 전부 기록했다. [복원 절차](../../outputs/vast57987772_safe_disk_cleanup/RESTORE_INSTRUCTIONS.md).

사용한 HF 저장소는 Dataset 타입이다. 고정 revision 목록:

- [Graph-Recurrent-VLA/experiment-backups / `e0d958295cefa26da43d69c9698253b9b6e29bb9`](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/e0d958295cefa26da43d69c9698253b9b6e29bb9)
- [Graph-Recurrent-VLA/graph-recurrent-rollouts / `f8646eccf440022630ab9ea6da9c97934152d733`](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/f8646eccf440022630ab9ea6da9c97934152d733)
- [Graph-Recurrent-VLA/experiment-backups / `5178d90ac62c18d1317f159cd828a542541e8ce5`](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/5178d90ac62c18d1317f159cd828a542541e8ce5)

로컬 파일의 실제 SHA256, 원격 LFS SHA256/크기, archive의 현재 원격 identity, inventory identity와 실제 archive member bytes를 확인했다. 246건 중 archive member182건은 실제 bytes를 새로 읽어 검증했다. 일반 Git blob inventory는 Git blob header를 포함한 SHA1을 계산해 비교했으며 SHA256과 혼동하지 않았다.

## 보호 및 삭제 전후 검증

- Stage3 R/D/Shared Validation, Stage4 recache manifest/receipt를 재파싱했다. manifest가 참조하는 raw/GT712건 및 F1/B/C checkpoint3건의 SHA256은 삭제 전후 모두 같았다.
- 코드·Source/Runtime Identity·ActionCodec·선정/Reserve/Config 등131개 추가 필수 파일이 읽기 가능하고 삭제 전후 hash가 같았다. Model Architecture와 checkpoint 내부 normalizer를 바꾸거나 모델을 로드해 실행하지 않았다.
- Stage4A 세 연구 Python PID230621/230885/231223 및 세 bash wrapper, tmux sessions/panes가 유지됐다. Task10-H 실행 프로세스는 조사 시 발견되지 않았으나 H 원본 root 전체는 보호했다. H가 성공 종료됐다고 새로 판정하거나 재시작하지 않았다.
- 최초 실행은 다른 UID의 Syncthing 핸들 접근이 불완전해 전부 fail-closed 보류했고 삭제0건이었다. 서비스 사용자 UID1001로 재검사해 열린 삭제 대상이 없고 설정된 Syncthing 폴더도0개임을 확인한 뒤 같은 동결 목록으로 실행했다. 서비스나 연구 프로세스를 강제로 종료하지 않았다.
- 삭제 직전마다 경로/parent FD, regular file 여부, inode/device/size/mtime, link count1, SHA256, 열린 파일, 새 Stage4 manifest 참조와 동기화 경로를 다시 검사했다. 보호 경로와 교집합0, 중복 inode0이었다.
- 삭제 후 신규 FileNotFound/No such file 로그 증가는 확인되지 않았다. 삭제 파일이 열린 상태로 남아 있는지 `/proc`를 확인했으며 상세 결과는 disk_usage_after.json에 있다.

## 업로드 보존과 남은 공간 제약

Task10-F recovery 업로더는 이번 정리 전에 이미 ConnectionError로 종료돼 있었다. 로컬 백그라운드 대기 pipeline은 유지했고 cleanup이 새 업로드를 시작하지 않았다. 미완료 Task10-F recovery, Legacy/source 보완, F1 데이터 archive3개 원본은 로컬 Mac에서 모두 존재하며 실제 SHA256이 예상값과 같았다. Vast의 F1/G/Stage3 source도 삭제하지 않았다.

현재 약0.6GiB 수준의 여유는 4GiB 목표 충족이 아니다. Stage4B 전량 graph recache에 충분하다고 검증하지 않았으며, 이 여유만으로 후속 전체 작업의 공간을 보장할 수 없다. 큰 현행 raw/cache/모델/Stage4A 자료와 미완료 HF source는 유지해야 한다. 추가 정리는 [review_required.csv](../../outputs/vast57987772_safe_disk_cleanup/review_required.csv)의 원격 검증/소유권/미래 비사용 증거를 먼저 보완해야 하며, 현 상태에서 자동 삭제하지 않았다.

## 감사 보존

- 영구 로컬 감사: `/Users/jeong-yunseong/Documents/programming/graph-recurrent-VLA/outputs/vast57987772_safe_disk_cleanup`
- 원격 감사 사본: `/workspace/vast57987772_safe_disk_cleanup_20261031`. 이는 overlay상의 사본이며 별도 persistent volume으로 오인하지 않는다. 원격 `/dev/shm`의 임시 기록만을 유일한 감사로 남기지 않았다.
- [연구 의존성 검증](../../outputs/vast57987772_safe_disk_cleanup/research_dependency_verification.json) · [업로드 보존 감사](../../outputs/vast57987772_safe_disk_cleanup/hf_upload_preservation_audit.json) · [독립 dry-run](../../outputs/vast57987772_safe_disk_cleanup/cleanup_dry_run_verification.json) · [삭제 실행 로그](../../outputs/vast57987772_safe_disk_cleanup/deletion_execution_log.json) · [완료 판정](../../outputs/vast57987772_safe_disk_cleanup/cleanup_completion_audit.json) · [artifact index](../../outputs/vast57987772_safe_disk_cleanup/artifact_index.json)

이번 작업은 파일 보존/정리 및 read-only 점검이다. CUDA/PhysX/renderer/학습/evaluation을 실행하지 않았고 simulator/model/shared infrastructure API를 변경하지 않았다. HF 인증 정보는 기록하지 않았다.
