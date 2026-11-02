# Vast 57987772 → 58576054 전송 검증 및 누락 보완

2026-11-01 KST. 요청된 누락 경로 보완: **PASS**. 원본은 변경·중단하지 않았다.

## 실행 결과

- 원래 Vast 상태는 원본 Done copying, 대상 Done receiving copy였다.
- 초기 대상 점검에서 SAM2 YAML 4개가 읽히지 않았고 파일 심볼릭 링크 11개가 없었다. YAML 경로는 디렉터리 목록에는 있었지만 lstat은 FileNotFoundError, 생성·교체는 Stale file handle(errno116)이 발생했다. 따라서 초기 “YAML 누락” 판정은 접근 불가 상태를 포함한다.
- 직접 생성 및 임시 파일을 통한 교체 시도가 실패했다. 대상 프로세스 점검에서 연구 실행·tmux가 없고 기본 서비스만 동작함을 확인한 뒤, 대상 58576054만 vastai reboot instance로 재시작했다. 원본 57987772는 재시작하지 않았다.
- 재시작 후 YAML 4개가 정상 접근됐고 원본에서 새로 계산한 SHA256과 모두 일치했다. 해당 YAML을 덮어쓸 필요는 없었다.
- 원본에서 확인한 경로대로 심볼릭 링크 11개를 새로 복원했다. 링크 문자열과 링크 대상 내용의 SHA256도 원본과 모두 일치했다.
- 작업 중 생성한 임시 YAML 사본 1개는 해당 해시를 검증한 뒤 제거했다. 연구 자료를 삭제하지 않았다.

## 최종 검증

- 기존 원본 목록(reference timestamp 1793488792.9820065)의 일반 파일 **138,464개**를 대상에서 확인: 누락0, 크기 차이0, 접근 오류0.
- 원본 목록의 파일 심볼릭 링크 **17개** 확인: 링크 차이0.
- 필수 Raw/GT/체크포인트/manifest/source/config **851건**은 기존 안전 정리 후 감사 SHA256과 재비교해 모두 PASS.
- 전체 일반 파일의 SHA256, 디렉터리 심볼릭 링크 전체, 빈 디렉터리, 원본 목록 작성 후의 새 파일·수정, 실행 환경 호환성은 이번 검증 범위에 포함하지 않았다. 학습·시뮬레이터·CUDA/렌더링은 실행하지 않았다.

## 보완 경로

- `full_pipeline_sam2/sam2/configs/sam2/sam2_hiera_s.yaml` — 재시작 후 정상 접근 및 SHA256 PASS
- `full_pipeline_sam2/sam2/configs/sam2/sam2_hiera_t.yaml` — 재시작 후 정상 접근 및 SHA256 PASS
- `full_pipeline_sam2/sam2/configs/sam2/sam2_hiera_b+.yaml` — 재시작 후 정상 접근 및 SHA256 PASS
- `full_pipeline_sam2/sam2/configs/sam2/sam2_hiera_l.yaml` — 재시작 후 정상 접근 및 SHA256 PASS
- `.venv-backups/57987772/venv-main-latest.txt` — 링크 생성 및 대상 SHA256 PASS
- `task10d_geometry_reach/pilot_cache/original/normalizer.json` — 링크 생성 및 대상 SHA256 PASS
- `task10d_geometry_reach/pilot_cache/optimized/normalizer.json` — 링크 생성 및 대상 SHA256 PASS
- `task10d_geometry_reach/pilot_cache/parallel/normalizer.json` — 링크 생성 및 대상 SHA256 PASS
- `task10d_geometry_reach/pilot_cache/isolated_serial_optimized/normalizer.json` — 링크 생성 및 대상 SHA256 PASS
- `task10d_geometry_reach/pilot_cache/isolated_parallel_2/normalizer.json` — 링크 생성 및 대상 SHA256 PASS
- `task10d_geometry_reach/renderer_probe/normalizer.json` — 링크 생성 및 대상 SHA256 PASS
- `task10d_geometry_reach/worker_benchmark/normalizer.json` — 링크 생성 및 대상 SHA256 PASS
- `task10d_geometry_reach/worker_benchmark/renderer_gpu_gate.json` — 링크 생성 및 대상 SHA256 PASS
- `task10d_geometry_reach/final_expert_reference/normalizer.json` — 링크 생성 및 대상 SHA256 PASS
- `task10d_geometry_reach/final_expert_reference/renderer_gpu_gate.json` — 링크 생성 및 대상 SHA256 PASS

## 감사 자료

[보완 실행 및 해시](../../outputs/vast_transfer_57987772_58576054/missing_transfer_verification.json) · [최종 목록 비교](../../outputs/vast_transfer_57987772_58576054/final_repair_comparison.json) · [필수 SHA256](../../outputs/vast_transfer_57987772_58576054/essential_sha256_verification.json)

요청된 15개 경로 문제는 해결했다. 전체 연구 실행 환경 이전 완료나 원본 삭제 가능 상태까지 판정한 것은 아니다.
