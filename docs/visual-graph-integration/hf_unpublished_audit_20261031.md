# HF 미게시 아티팩트 감사 — 2026-10-31

Task ID: `hf_unpublished_audit_20261031`. 확인 시각: 2026-10-31T12:14:38.270714+00:00.

**미게시 자료가 로컬과 Vast.ai 57987772 양쪽에 남아 있다.** F1 학습 모델·예측·raw/cache, G의 최종 A/B/C 학습·평가, H의 phase-handoff 결과, I Stage1/2/3 데이터 및 E의 새 평가 결과에서 HF의 현재 게시 내용과 일치하지 않는 파일을 확인했다. 기존 게시본과 동일한 입력·의존성도 있으므로 디렉터리 전체가 미게시라고 해석하지 않는다.

업로드·재게시·삭제·인스턴스 변경·학습 재실행·Git commit은 수행하지 않았다. HF에서는 목록과 작은 검증 파일, ZIP의 필요한 구간만 읽었다. 다운로드한 감사 근거와 이 보고서만 새로 저장했다.

## 확인 범위와 판단 방법

- 로컬: 현재 프로젝트 전체와 별도 `graph-recurrent-VLA-legacy-fps-release` worktree. `git worktree list`에 남은 `/private/tmp/task06c-*` 두 항목은 prunable 상태라 실재 worktree로 검사하지 않았다. 개인 홈 전체나 다른 서버는 범위가 아니다.
- 원격: SSH 연결 후 `CONTAINER_ID=57987772`를 확인. `/workspace` 전체 및 발견한 `/tmp/task07_observed_*`, `/opt/workspace-internal`, `/home/user`, `/home/ubuntu`를 검사. `/root`, `/tmp`의 얕은 데이터 파일 검색도 수행했지만 시스템·개인 홈 전체를 재귀 검사한 것은 아니다.
- 가상환경, Git 내부, 컴파일 캐시, 재다운로드 가능한 외부 모델 디렉터리, 인증 파일은 제외했다. 로컬 `visual/cache` 및 W&B 내부 파일은 비교 CSV에서 `dependency_cache`로 분리했다. 동결 source 내부의 테스트 NPZ, 설치 라이브러리, 중복 archive 등은 넓은 파일 목록에 일부 포함되어 있어 전체 unmatched 개수는 연구 산출물 개수와 다르다.
- 일반 파일을 직접 읽어 SHA256, Git blob SHA1, 크기를 계산했다. HF 경로명이 달라도 같은 내용이면 게시됨으로 처리했다. 읽기 전후 크기/mtime이 바뀐 일반 파일은 0개였지만 전체 filesystem의 원자적 snapshot은 아니다.
- HF의 모델·롤아웃·백업 저장소 및 관련 Space의 현재 revision을 고정해 전 경로를 읽었다. receipt 부재만으로 미게시라 판정하지 않았다.
- 원격 ZIP 36개 전부의 manifest와 중앙 directory를 확인했다. 32개는 동일 SHA256의 로컬 ZIP을 사용했고, 4개는 HF 고정 revision에서 range read했다. 선언된 member 존재/크기를 검증했다. manifest 밖의 bundle manifest/config/metrics도 읽어 hash index에 넣었다. ZIP 내 전체 payload의 재해싱은 수행하지 않았고 manifest의 SHA256 선언을 사용한다.
- 대형 TAR.GZ 4개는 원격에 게시된 inventory/verification을 내려받았다. inventory SHA256 및 verification에 적힌 archive SHA256·크기가 현재 HF 객체와 일치함을 확인했다. 10GB 이상 TAR payload를 다시 내려받아 전수 해제하지는 않았다. 이를 `tar_inventory`로 별도 표시했다.
- source SHA만 일치하고 게시 JSON은 마스킹되어 내용이 다른 경우는 `sanitized`로 분리했다. 이를 원본 bytes가 그대로 게시됐다고 세지 않았다.

## HF 현재 상태

| 저장소 | 고정 revision | 경로 수 |
|---|---|---:|
| [graph-recurrent-models](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/tree/2b54ccde348b68712004657b76caa9f5259bb1a9) | `2b54ccde348b68712004657b76caa9f5259bb1a9` | 1,279 |
| [graph-recurrent-rollouts](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/d7973469f704b95dbd2beb0df323f61805a41dd0) | `d7973469f704b95dbd2beb0df323f61805a41dd0` | 1,445 |
| [experiment-backups](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/8cfa671c2fe8ab1fffd924435392ff0728f21c79) | `8cfa671c2fe8ab1fffd924435392ff0728f21c79` | 11,115 |
| [task09c-captures](https://huggingface.co/spaces/Graph-Recurrent-VLA/task09c-captures/tree/199ad48d2c4506e43614683c7aaf7ec0bb686df8) | `199ad48d2c4506e43614683c7aaf7ec0bb686df8` | 2,229 |

세 아티팩트 저장소의 현재 revision은 10월30일 게시 검증 기록과 같았다. Organization 목록에서 접근 가능한 Model1/Dataset2/Space1을 확인했다. 접근 권한이 없는 다른 비공개 저장소의 존재까지 배제하는 것은 아니다.

## 주요 미게시 위치

아래 용량은 해당 **원격 root 내** unmatched 파일을 SHA256으로 중복 제거한 값이다. 디렉터리 사이 중복은 남아 있으므로 합산하지 않는다. MB는 10⁶ bytes. 소스·로그·중간 산출물도 포함한다.

| 원격 `/workspace/` 아래 | 미일치 경로 | 고유 내용 | 고유 MB | 판단 |
|---|---:|---:|---:|---|
| `task10f1_gt_pretrain_20261030` | 5,666 | 3,984 | 2,181.66 | 최종 모델·예측·raw/cache, 보완 import, 실패/engineering 기록 |
| `task10g_f1_abc_20261031` | 36,957 | 36,284 | 1,906.59 | full 최종 A/B/C와 mini_v1–v5, host_data 배열, 진단 |
| `task10h_phase_handoff_20261031_v5` | 492 | 376 | 306.19 | 최종 rollout·상태 snapshot·영상·분석 |
| `task10i_stage1_20261031` | 26,025 | 7,195 | 262.33 | verified_pool_v2 및 이전 후보 풀/감사 |
| `task10i_stage2_20261031_v2` | 789 | 788 | 25.56 | 선택 결과·receipt·분포/선택 분석 |
| `task10i_stage3_20261031_v3` | 3,392 | 2,477 | 1,143.14 | 실제 수집 raw/GT 및 engineering/중간 기록 |
| `task10e_gt_gru` | 233 | 224 | 41.22 | 새 benchmark 평가·영상·state/action trace |
| `plan_a_bc_3788798a719f4ec4a66c224ca42f5d63` | 3,112 | 2,985 | 1,027.65 | 기존 게시 범위를 넘어선 rollout/cache/학습 기록 |

### F1: 최종 모델은 로컬에도 있으나 HF 미게시

- `/workspace/task10f1_gt_pretrain_20261030/gnn_pretrained.pt`, `training/best.pt`, `training/last.pt`, `robot_baseline_training/best.pt`, `robot_baseline_training/last.pt`: 모두 unmatched. 같은 SHA256의 로컬 일반 파일을 확인했다.
- `predictions.npz`(42,333,408 bytes)는 unmatched이고, 스캔한 로컬 일반 파일에서 같은 SHA256을 찾지 못했다.
- `cache/`의 Episode NPZ120개 및 `pairs/`의 NPZ28개, `raw/`, `imports/`, `operations/native_rebuild_20261030`의 raw/GT/graph·검증 기록을 보존 대상으로 구분해야 한다.
- 같은 original 데이터의 복사본·다른 직렬화와 partial/중단본도 존재한다. HDF5/NPZ bytes가 다르면 의미가 같아도 unmatched가 될 수 있으며, 파일 내용의 의미적 동일성까지 판정한 것은 아니다.

### G/H: 최종 연구 결과와 engineering을 구분

- G의 정식 결과 root는 `task10g_f1_abc_20261031/full`, 로컬은 `outputs/task10g/final_results`. 최종 모델 선택·학습 checkpoint·평가 영상·trace·수치 보고가 미게시 목록에 있다.
- G의 `mini_v2/host_data/*.npy`14개는 연구 최종 결과와 별도인 중간 cache다. `node_input.npy`409,299,116 bytes, `edge_index.npy`210,214,144 bytes, `edge_attr.npy`210,214,144 bytes 등이 HF와 로컬 일반 파일 양쪽에서 미일치다. 필요성을 결정한 뒤 보존 대상으로 선택할 수 있다.
- H의 v5 root는 `task10h_phase_handoff_20261031_v5`, 로컬은 `outputs/task10h/final_results`. MP4 57경로와 PT127경로가 unmatched다. PT에는 simulator/controller state snapshot이 포함되므로 모두 학습 모델이라는 뜻은 아니다. 주요 바이너리의 같은 로컬 SHA256이 확인된다.
- H v1–v4와 G miniature들의 실패·OOM·중간 재시도는 최종 결과와 별도로 유지되어 있다. 미완료 `.partial`을 완성 artifact로 취급하지 않았다.

### I: 최종 receipt만 올리면 원본 데이터가 보존되지 않는다

- Stage1 canonical: `/workspace/task10i_stage1_20261031/verified_pool_v2`. 원래 보고서의 전체 8,514파일과 이번의 디렉터리 파일 수가 일치한다. 다른 버전/partial 후보 풀도 존재한다.
- Stage2: `/workspace/task10i_stage2_20261031_v2`. Stage1 원본과 선택·분포·receipt를 연결해서 보존할 필요가 있다.
- Stage3 최종 문서는 `/workspace/task10i_stage3_20261031_v5_finalization/run`을 가리킨다. 실제 원본 수집 데이터는 v3 아래에도 남아 있고 최종화 디렉터리의 symlink가 이를 참조한다. v5의 작은 manifest만 백업해서 raw를 대체할 수 없다.
- 정식 Stage3 결과 문서의 R272/D272/validation48 logical episode는 공유 physical data를 참조하므로 단순 파일 개수와 같지 않다. 이번 감사에서는 새 수집이나 학습을 실행하지 않았다.
- v1/v2 engineering raw 중 로컬 일반 파일에서 일치 항목을 못 찾은 것도 있다. 정식 학습 데이터와 혼합하지 않고 실패/engineering 근거로 분리할 대상이다.

### E와 과거 로컬 누락

- E의 `benchmark/` 새 평가 자료는 unmatched지만, 재사용한 기존 GRU best/last checkpoint는 HF `experiment-backups/recurrent/main/...`의 동일 내용과 매칭됐다. E 전체가 미게시인 것은 아니다.
- `outputs/graph/dataset.pt`68,539,358 bytes, `outputs/graph/demos/PegInsertionSide-v1/motionplanning/trajectory.h5`50,058,783 bytes, `outputs/subtasks/bc/reach_legacy91.pt`19,272,805 bytes 등 과거 로컬 자료도 현재 HF 및 archive index에서 매칭되지 않았다.
- `outputs/task10f_recovery/remote/`에는 Task10-F의 corrective/recovery raw·cache·영상이 남아 있다. 현재 57987772의 F1 import와 같은 파일은 중복으로 인식했지만, 로컬에서만 찾은 내용도 있다. 원래57640710에 현재 접속한 것은 아니다.
- `outputs/graph`, `outputs/subtasks`, 실험 source snapshot 내부의 미일치 파일에는 테스트 fixture·중간 데이터도 있다. 원본 데이터셋/선택 모델/재현 입력을 우선 검토하고 전체 unmatched를 일괄 업로드할 필요는 없다.

## 원격에만 있는 것으로 우선 확인할 파일

아래는 `.h5/.hdf5/.npz/.npy/.pt/.pth/.ckpt/.safetensors/.mp4/.webm/.parquet`의 unmatched 내용만 SHA256으로 중복 제거한 결과다. **로컬 일반 파일에서 같은 내용이 발견되지 않았다는 의미**이며, 로컬의 미게시 TAR/ZIP 내부까지 모두 해제하여 검색한 것은 아니다.

| 원격 root | 고유 바이너리 수 | MB |
|---|---:|---:|
| `task10f1_gt_pretrain_20261030` | 732 | 682.81 |
| `task10g_f1_abc_20261031` | 44 | 1,086.85 |
| `task10i_stage3_20261031_v2` | 23 | 62.49 |
| `task10i_stage3_20261031_v1` | 4 | 8.78 |

F1 raw/GT 보완분, `predictions.npz`, G miniature checkpoint와 host_data 배열이 포함된다. 이 수치는 최종 연구 결과의 개수나 추가 전송량을 뜻하지 않는다. 일부는 실패/engineering/재생성 가능한 중간 자료다.

## 전체 대조 통계

| 위치 | 읽은 일반 파일 | 논리 GB | direct | ZIP manifest | TAR inventory | sanitized | unmatched |
|---|---:|---:|---:|---:|---:|---:|---:|
| local | 361,459 | 41.696 | 41,113 | 10,117 | 82,604 | 3,052 | 208,007 |
| remote | 106,779 | 13.018 | 13,866 | 1,394 | 71 | 768 | 90,680 |

위 표에는 복사본·source snapshot·중간 archive가 포함된다. unmatched 용량을 그대로 업로드 필요량으로 해석하면 과대평가된다. direct는 현 HF의 size+LFS SHA256 또는 size+Git blob SHA1 일치다. ZIP/TAR는 위에 명시한 index 기반 검증 수준을 적용했다.

## 검증 제약과 실행 상태

- 읽지 못한 로컬5개는 `.sock` Unix socket이다. 원격1개는 `/workspace/task10b_diagnosis_20261029/noop` character special file이며 연구 데이터 일반 파일이 아니다. 일반 파일 read 실패를 숨기지 않고 inventory에 기록했다.
- 별도 symlink 감사: 로컬847개(디렉터리39개), 원격47개(디렉터리35개), broken link0개. 디렉터리 symlink의 target은 모두 검사 root 내부였다. 원격 최종화 디렉터리의 외부 원본 참조를 확인했다.
- 로그·monitor는 계속 갱신될 수 있다. snapshot 이후의 새 파일은 포함하지 않는다. CUDA/PhysX/renderer/학습/평가를 이번 감사에서 실행하지 않았다.
- 원격 storage 조회: `/workspace`는 별도 persistent volume이 아니며 overlay disk93% 사용, 여유 약1.3GB였다. 이는 이번 호스트에서 직접 확인한 값이다. 인스턴스나 파일을 정리/중지하지 않았다.
- Hugging Face의 과거 삭제 revision 전체, 접근 불가 저장소, 로컬 미게시 archive 내부의 전 내용, 다른 서버의 현재 상태는 검사하지 않았다. “미게시”는 이번 접근 가능한 현재 HF trees와 게시 archive index에 동일 내용이 없는 상태를 뜻한다.

## 보존 우선순위

1. F1 최종 pretrained checkpoint + 데이터 manifest/선택 기록 + raw/GT/cache/승인 근거, G full 최종 A/B/C 및 재현 근거, H v5 정식 분석, I Stage1 canonical/Stage2 선택/Stage3 실제 raw와 최종 receipt를 우선 연결한다.
2. 로컬 일반 파일에 없는 F1 보완 입력·예측, G miniature 체크포인트 및 host_data, I 초기 engineering 자료는 필요성을 검토하여 회수 대상으로 선정한다. 이번 작업에서는 회수나 게시를 실행하지 않았다.
3. PLAN A 추가 rollout/훈련 기록과 과거 Graph/Subtask 산출물 중 필요한 원본을 선정한다. 중복 mirror, 재포장 ZIP, 테스트 fixture, 설치 라이브러리와 완성되지 않은 partial은 구분한다.
4. 실제 게시 시 기존 `utils.experiment_artifacts` / `tools.publish_experiment`와 기존 UUID를 재사용한다. 이 감사에서 신규 experiment UUID, manifest 재작성 또는 publisher dry-run을 실행하지 않았다.

## 재검토 가능한 근거

- [로컬 전체 파일별 판정·해시·HF 근거](../../outputs/hf_unpublished_audit_20261031/local-comparison.csv)
- [57987772 전체 파일별 판정·해시·HF 근거](../../outputs/hf_unpublished_audit_20261031/remote-comparison.csv)
- [로컬 주요 바이너리 unmatched 및 원격 동일본 유무](../../outputs/hf_unpublished_audit_20261031/local-unpublished-artifacts.csv)
- [원격 주요 바이너리 unmatched 및 로컬 동일본 유무](../../outputs/hf_unpublished_audit_20261031/remote-unpublished-artifacts.csv)
- [디렉터리별 통계와 오류](../../outputs/hf_unpublished_audit_20261031/summary.json)
- [ZIP36개 member/manifest 근거](../../outputs/hf_unpublished_audit_20261031/archives.json)
- [로컬 symlink 대상](../../outputs/hf_unpublished_audit_20261031/local-links.json)
- [원격 symlink 대상](../../outputs/hf_unpublished_audit_20261031/remote-links.json)
- [원격 파일 종류·persistent volume·disk 조회](../../outputs/hf_unpublished_audit_20261031/remote_storage.txt)

실행 절차는 같은 output 디렉터리의 `scan_files.py`, `scan_hf.py`, `scan_archives.py`, `scan_tar_indexes.py`, `compare.py`, `analyze.py`, `check_links.py`에 남겼다. 공통 HF 인증과 `utils.experiment_artifact_maintenance.inventory()`를 재사용했다. 애플리케이션/공통 infrastructure 코드는 수정하지 않았으므로 학습·simulator 테스트는 실행하지 않았다. 아래 별도 검증 결과에 실제 실행한 감사 정합성 검사를 남긴다.

## 최종 감사 검증

[검증 결과](../../outputs/hf_unpublished_audit_20261031/validation.json): **16개 검사 PASS**. CSV 행 수·용량과 summary 일치, 일반 파일 read 안정성, ZIP36개 오류 없음, HF4개 revision 종료 시 동일, F1 pretrained/예측 및 G node array의 실제 원격 SHA256 재확인, 보고서 상대 링크 존재를 확인했다. Git whitespace 검사도 통과했다. 제품/학습 코드 수정은 없다.
