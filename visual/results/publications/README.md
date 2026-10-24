# Visual runtime — HF 게시 결과

기존 비공개 HF Dataset `Graph-Recurrent-VLA/graph-recurrent-rollouts`에 초기 검출·추적·RGB-D 평가부터 Task 01, Task 02-A, Task 02-B, PCA 최적화, Task 02-C, Task 03까지의
결과를 게시했다. Git에는 코드·테스트·보고서와 [대표 그림](../diagnostic_previews/README.md)을 보관한다.
HF 링크는 게시한 revision에 고정했으며 저장소 접근 권한이 있는 계정으로 로그인해야 한다.
기계가 읽을 수 있는 경로·revision·SHA-256은 [artifacts.json](artifacts.json)에 있다.

Task03-AB, D의 세 phase, E–K와 개발·실패 기록의 후속 게시 결과는
[Task 03 후속 실험 산출물](sampling_followups.md)에 있다. 같은 비공개 저장소와 사용자 계정을 사용했다.

| 결과 | 기존 run ID | ZIP 크기 | 다운로드 |
| --- | --- | ---: | --- |
| RGB-D·초기 검출·추적·영상 | `996f629b211147f18deeb6736c51c3e9` | 38.57 MiB | [ZIP 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/b27cf33a98d48c396a53c779f5707602679af99f/visual/996f629b211147f18deeb6736c51c3e9/artifacts/reports/ad7e48e28476c9b4e17951042225c59e89bc89438f10efa1979cd2aaae0d89c5/report-ad7e48e28476c9b4e17951042225c59e89bc89438f10efa1979cd2aaae0d89c5.zip?download=true) · [HF 파일 목록](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/b27cf33a98d48c396a53c779f5707602679af99f/visual/996f629b211147f18deeb6736c51c3e9) |
| Task 01 — Organized Geometry | `f9b0da8588734a28996a38f3724da16a` | 8.82 MiB | [ZIP 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/92cacaff34a0e1c760e96f298243d27343f483e5/visual/f9b0da8588734a28996a38f3724da16a/artifacts/reports/a707c07253e91a0a57e840c74c82a70d4b826bc488d3d48329ec0a058ce40f49/report-a707c07253e91a0a57e840c74c82a70d4b826bc488d3d48329ec0a058ce40f49.zip?download=true) · [HF 파일 목록](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/92cacaff34a0e1c760e96f298243d27343f483e5/visual/f9b0da8588734a28996a38f3724da16a) |
| Task 02-A — Normal Reliability | `3640b9bbe403409ba1429e0143e5e049` | 8.25 MiB | [ZIP 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/3c9bcb9d7f11b7433f0c432d945ce71add0825ee/visual/3640b9bbe403409ba1429e0143e5e049/artifacts/reports/d156c000f8588b509c07811475472a8ec818f06fe88a1c12c612603a75ea06c2/report-d156c000f8588b509c07811475472a8ec818f06fe88a1c12c612603a75ea06c2.zip?download=true) · [HF 파일 목록](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/3c9bcb9d7f11b7433f0c432d945ce71add0825ee/visual/3640b9bbe403409ba1429e0143e5e049) |
| Task 02-B — Normal Estimator Comparison | `4b1c377ae2044344a4f596b249627d89` | 15.45 MiB | [ZIP 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/72cec859c6c3002c7b9953894ad8139203eb3475/visual/4b1c377ae2044344a4f596b249627d89/artifacts/reports/f4591dc0c533ee913b543e6f502d641b581e1d941bf9ddd940c6a656153b33b5/report-f4591dc0c533ee913b543e6f502d641b581e1d941bf9ddd940c6a656153b33b5.zip?download=true) · [HF 파일 목록](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/72cec859c6c3002c7b9953894ad8139203eb3475/visual/4b1c377ae2044344a4f596b249627d89) |
| PCA5 Optimization · CPU/CUDA/20Hz | `03d9777a004349e3a794147e4e55a2f0` | 12.30 MiB | [ZIP 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/a0ab713874a7f6ae2026d05138e7902d218201a6/visual/03d9777a004349e3a794147e4e55a2f0/artifacts/reports/61405e026c1886e9765ae0307f084a5348ef7a5546daad1fab34f6aa31ca3e95/report-61405e026c1886e9765ae0307f084a5348ef7a5546daad1fab34f6aa31ca3e95.zip?download=true) · [HF 파일 목록](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/a0ab713874a7f6ae2026d05138e7902d218201a6/visual/03d9777a004349e3a794147e4e55a2f0) |
| Task 02-C — Structural Edge Candidates | `b622e76deceb419ab9c5aa80a8f8280c` | 111.04 MiB | [ZIP 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/d247f2c36d6ee76b9f6e2ad991909a153384be9a/visual/b622e76deceb419ab9c5aa80a8f8280c/artifacts/reports/b482be23e8cf3fd90d91296db55d060539ded17c044af4c925483cd7bc0938f3/report-b482be23e8cf3fd90d91296db55d060539ded17c044af4c925483cd7bc0938f3.zip?download=true) · [HF 파일 목록](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/d247f2c36d6ee76b9f6e2ad991909a153384be9a/visual/b622e76deceb419ab9c5aa80a8f8280c) |
| Task 03 — Structural / Surface Sampling | `2551448b0fae4c6bbce3caf48774275f` | 239.91 MiB | [ZIP 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/6b8ce8d6953ae92e76ee49f6ae83b4180e338b7c/visual/2551448b0fae4c6bbce3caf48774275f/artifacts/reports/14cf7152fac146561742341444188a69372e317c78e9c796a1497a9d5e90fd8a/report-14cf7152fac146561742341444188a69372e317c78e9c796a1497a9d5e90fd8a.zip?download=true) · [HF 파일 목록](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/6b8ce8d6953ae92e76ee49f6ae83b4180e338b7c/visual/2551448b0fae4c6bbce3caf48774275f) |

## 영상 위치

- Grounding DINO: **RGB-D ZIP** 안의 `visual/results/runtime_grounding/ep000_full/grounding.mp4`
- SAM2 추적: **RGB-D ZIP** 안의 `visual/results/runtime_tracking/ep000/tracking.mp4`
- RGB-D·depth·GT: [HF에서 영상 열기/다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/b27cf33a98d48c396a53c779f5707602679af99f/visual/996f629b211147f18deeb6736c51c3e9/artifacts/rollouts/a86cdaa55bc39896e551171bf04871e3d9ca2444aa2845277a457ea85bab8941/depth.mp4) 또는 RGB-D ZIP 안의 `visual/results/runtime_rgbd/996f629b211147f18deeb6736c51c3e9/depth.mp4`

세 영상 모두 121프레임, 20fps다. 초기 검출·추적 영상은 RGB-D 평가 이전에 생성된 dependency이며,
게시를 위해 새 실행 ID를 만들거나 생성 실험을 추정하지 않았다. Task 01/02-A/02-B/PCA/02-C/03은 PNG·HTML·수치 결과이며 추가 영상은 없다.

## 내용과 열기

RGB-D ZIP에는 최종 GT 리포트와 depth 영상, 원본 `ep000.npz`, tracker mask·metadata,
Grounding DINO/SAM2 영상·metadata 및 0/15/45/80/120 대표 프레임이 들어 있다.
Task 01 ZIP에는 전체 scalar 수치와 0/15/18/45/80/120의 PNG·dense NPZ가 있다.
Task 02-A ZIP에는 합성 비교, 전체 121프레임 수치, 0/15/45/80/120의 전체 맵·경계 확대·profile·dense NPZ가 있다.
Task 02-B ZIP에는 5종 normal estimator의 합성·121프레임 전체 수치, 같은 5프레임의 비교·유효성·profile·dense NPZ,
검증 기록·입력 탐색 inventory·보존 snapshot·구현 patch·평가 당시 코드 사본을 포함한 31개 파일이 있다.
02-B의 전체 수치 JSON 약 51.7 MiB는 HF에 보관하고 Git에는 보고서와 대표 그림을 남겼다.
RGB-D/Task 01/02-A/02-B의 일부 annotation PNG·중간 실행 및 중복 tracking `ep001`은 로컬에 유지한다.
PCA·02-C·03은 현재 보존된 전체 결과를 포함했다.

PCA ZIP에는 CPU 비교·profiling·CUDA backend·tracking compile/warmup·20Hz replay·추가 품질 평가의
**14개 원래 run**, 실행 당시 source 사본과 보존·검증 기록을 포함한 **104개 파일**이 있다.
CPU 최종 run은 `03d9777a004349e3a794147e4e55a2f0`, GPU 최종 streaming run은
`a34de8fa4976434e9be4356eb8db198a`다. [PCA 보고서](../../../docs/visual-graph-integration/pca5_optimization.md)를 먼저 읽는다.

02-C ZIP에는 최종 run의 **121개 pair NPZ**, 전체 threshold 수치·그림·평가 source·검증 기록과
이전 두 개발 실행을 포함한 **409개 파일**이 있다. 최종 run은 `b622e76deceb419ab9c5aa80a8f8280c`이고,
`6eed92be53c64472be8b536f3806ce2d` / `935a867a2da04ec9972da572af4f6a10`은 `superseded.json`으로
최종 실행을 가리킨다. [02-C 보고서](../../../docs/visual-graph-integration/02c_structural_edges.md)를 먼저 읽는다.

Task 03 ZIP에는 최종 **121프레임 × 54조건**의 선택·feature/evidence를 담은 121개 NPZ,
121개 frame metrics JSON, 전체 summary·그림·source 사본·검증/테스트 로그와 이전 두 실행을
포함한 **837개 파일**이 있다. [Task 03 보고서](../../../docs/visual-graph-integration/03_sampling.md)와
[sampling API](../../runtime/sampling.md)를 함께 읽는다.
최종 run은 `2551448b0fae4c6bbce3caf48774275f`다. `71a16f2747bc43a686ed56a5ac856471`은
allocation 정수 잔여분 규칙 수정 전 개발 실행이고, `f13af0283e4644bea27b4b731f7bf8e0`은
수정 후 검증 실행이다. 마지막 두 실행의 시간 외 summary는 동일하다. 이전 실행을 최종 결과로
혼합하지 않도록 `artifacts.json`의 `development_runs`에 구분했다.

각 묶음은 최종 run을 대표로 게시하며 다른 run의 파일은 `dependency` 역할로 표시한다.
하위 run의 원래 metadata/config/metrics도 `experiments/<원래 run_id>/`에 보존했다.
PCA 원격 CUDA 실행 일부는 Git checkout이 없어 `git_commit=null`이다. 이를 현재 게시 commit으로
채우지 않았으며 실제 실행 source hash·snapshot과 함께 해석해야 한다.

ZIP마다 별도의 새 폴더에 압축을 풀고 다음 entrypoint를 연다. HTML과 그림·영상의 상대경로는 ZIP에 유지했다.

| ZIP | Entry point |
| --- | --- |
| RGB-D | `visual/results/runtime_rgbd/996f629b211147f18deeb6736c51c3e9/gt_report.html` |
| Task 01 | `visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/geometry.html` |
| Task 02-A | `visual/results/normal_diagnostics/3640b9bbe403409ba1429e0143e5e049/diagnostics.html` |
| Task 02-B | `visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/comparison.html` |
| PCA 최적화 | `visual/results/pca_optimization/03d9777a004349e3a794147e4e55a2f0/optimization.html` |
| Task 02-C | `visual/results/structural_edges/b622e76deceb419ab9c5aa80a8f8280c/comparison.html` |
| Task 03 | `visual/results/sampling/2551448b0fae4c6bbce3caf48774275f/comparison.html` |

PCA 묶음의 `performance.html`은 CPU/GPU/streaming 요약을 추가로 제공한다.

원본 입력 episode SHA-256은 `22ad72d776057da9ed5f85014cba47b4603e421623f29763c88edb159a5087ba`,
mask SHA-256은 `6bf9555a9c97c83a09d9733e582e9c4b9509f95e9dd06854ba05d3ac25fb7042`다.
재현 입력은 RGB-D ZIP에 한 번만 포함했다. 여러 ZIP의 결과를 새 checkout root에 복원할 때는
각 `bundle_manifest.json`의 `files`에 나열된 member만 추출하면 된다. ZIP별 공통 이름의
`bundle_manifest.json`, `experiment/config.yaml`, `experiment/metrics.json`은 각 ZIP에 따로 보존한다.

## 인증된 다운로드와 hash 확인

Repository root에서 기존 HF 로그인 상태를 사용한다. Token을 명령에 넣지 않는다.

```python
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

index = json.loads(Path("visual/results/publications/artifacts.json").read_text())
for result in index["artifacts"]:
    path = Path(hf_hub_download(
        index["repository"]["repo_id"], result["bundle_path"],
        repo_type="dataset", revision=result["hf_revision"],
        local_dir="outputs/hf_artifacts",
    ))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result["bundle_sha256"]
    print(result["title"], path)
```

ZIP SHA-256:

- `rgbd`: `ad7e48e28476c9b4e17951042225c59e89bc89438f10efa1979cd2aaae0d89c5`
- `geometry`: `a707c07253e91a0a57e840c74c82a70d4b826bc488d3d48329ec0a058ce40f49`
- `normal`: `d156c000f8588b509c07811475472a8ec818f06fe88a1c12c612603a75ea06c2`
- `comparison`: `f4591dc0c533ee913b543e6f502d641b581e1d941bf9ddd940c6a656153b33b5`
- `pca`: `61405e026c1886e9765ae0307f084a5348ef7a5546daad1fab34f6aa31ca3e95`
- `structural`: `b482be23e8cf3fd90d91296db55d060539ded17c044af4c925483cd7bc0938f3`
- `sampling`: `14cf7152fac146561742341444188a69372e317c78e9c796a1497a9d5e90fd8a`

## 실행 identity와 검증

게시를 위한 새 Logger/UUID는 생성하지 않았다. 원래 실험의 Git commit/dirty 상태와 run ID를 유지했다.
RGB-D/Task 01/02-A의 publication/source commit은 `20b906c6d13d592ef94addcdf6297c31a68c1fb4`,
Task 02-B는 `366e9ce4b1510926449fc7c51765ac72849b16c4`,
PCA는 `5d59b749c9fbd3da4fa1c5f3a47fd15d5cd17cbd`, 02-C는 `d26c1fc86c192dead1ac06e6a55505ef4bbfe627`,
Task 03은 `1f791b70d762fa1f22e44903313c7a61ff0df8b3`다. Task 03의 선행 runtime 이동 commit은 `d2c6facf9caf637d90ec95c9f4f1cea1fbb41f11`이다.
최종 CPU 실험은 `git_dirty=true`였으며 실행 commit과 게시 commit을 별도로 보존한다. 원격 CUDA 실행의 누락된 Git 정보도 그대로 유지한다.
ZIP의 `bundle_manifest.json`은 파일별 역할, 알려진 원본 run ID, 원본/게시본 SHA-256을 기록한다.
JSON은 공통 마스킹을 적용해 key 순서·format이 달라질 수 있으며 binary 파일은 원본 bytes를 보존했다.

- 02-B 게시 준비에서 공통 Logger/Artifact 42/42 PASS, 해당 범위의 Vision runtime 53/53 PASS를 별도 프로세스로 확인했다.
- 저장된 02-B baseline 배열 260개와 기존 GT ROI 120개의 동일성을 재확인했다.
- 함께 실행했을 때 기존 logger 테스트의 전역 `importlib` mock이 imageio lazy import와 충돌했다.
  제품 코드 실패가 아니며 suite를 분리해 검증했다.
- ZIP 내부 파일 hash·HTML 상대경로, 영상 첫/마지막 프레임·121프레임·20fps를 확인했다.
- 게시 후 HF 객체 hash를 대조하고 각 ZIP을 다시 내려받아 SHA-256·ZIP 무결성을 검증했다.
- 이번 게시에서는 기존 CPU NumPy 비교 결과를 사용했다. Simulator·모델 재추론·CUDA/MPS/GPU·전체 비교 재실행은 수행하지 않았다.
- 02-B 평가 당시 코드 hash는 ZIP의 `evaluated_source/`와 일치한다. 이후 NaN을 회색으로 표시하는 plotting 수정만 적용된 현재 source와 구분해 보관했다.
- 위 02-B 게시 당시 제외했던 PCA 최적화·CUDA/Triton·structural edge 결과도 후속 게시로 추가했다.
- 후속 게시 전 runtime 77개 중 74 PASS / CUDA 3 SKIP, 공통 인프라 42/42 PASS를 확인했다.
- 후속 두 ZIP의 모든 member hash와 HTML 상대 링크(17개/43개)를 확인하고, 업로드 객체 hash 및 재다운로드 ZIP 무결성을 확인했다.
- 이번 후속 게시에서 GPU·tracker·simulator·전체 benchmark는 재실행하지 않았다. GPU 성능은 원래 RTX 3090 실행 기록이다.
- Task 03 게시 전 runtime 100개 중 97 PASS / CUDA 3 SKIP, 공통 인프라 42/42 PASS를 확인했다.
- Task 03 실행 source 16개 hash와 원본에서 이동한 함수/class 15개 AST를 확인했다. `edge_evidence.py`의 파일 끝 빈줄만 후속 정리했고 원래 source bytes는 ZIP에 보존했다.
- Task 03 ZIP member 837개의 hash·HTML 상대 링크 87개·업로드 객체 hash·재다운로드 ZIP 무결성을 확인했다.
- 저장된 6,534 selection 감사 및 검증 로그는 원래 실험 기록이다. 이번 게시에서 전체 sweep·CUDA/MPS sampling·end-to-end 20Hz를 재실행하지 않았다.
- ZIP 안의 역사적 `verification.json`/`completion_audit.json`에는 게시 전 상태가 남아 있다. 실제 게시 revision과 사후 검증은 이 인덱스를 따른다.

게시 목록은 [rgbd_files.json](rgbd_files.json), [geometry_files.json](geometry_files.json),
[normal_files.json](normal_files.json), [comparison_files.json](comparison_files.json),
[pca_files.json](pca_files.json), [structural_files.json](structural_files.json),
[sampling_files.json](sampling_files.json)에 명시했다. 명령과 공통 API는
[HF 게시 문서](../../../docs/huggingface_artifacts.md#상대경로를-유지하는-보고서-zip)를 따른다.
예를 들어 Task 02-B 게시 준비를 다시 확인하려면:

```bash
.venv/bin/python -m tools.publish_experiment \
  --run-id 4b1c377ae2044344a4f596b249627d89 --include reports metrics \
  --report-files visual/results/publications/comparison_files.json --dry-run
```

Dry-run은 원격을 변경하지 않지만 로컬 ZIP·metadata·manifest·sidecar는 생성/갱신할 수 있다.
