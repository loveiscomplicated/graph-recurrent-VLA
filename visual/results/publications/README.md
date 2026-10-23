# Visual runtime — HF 게시 결과

기존 비공개 HF Dataset `Graph-Recurrent-VLA/graph-recurrent-rollouts`에 초기 검출·추적·RGB-D 평가부터 Task 01, Task 02-A, Task 02-B까지의
선별한 최종 결과를 게시했다. Git에는 코드·테스트·보고서와 [대표 그림](../diagnostic_previews/README.md)을 보관한다.
HF 링크는 게시한 revision에 고정했으며 저장소 접근 권한이 있는 계정으로 로그인해야 한다.
기계가 읽을 수 있는 경로·revision·SHA-256은 [artifacts.json](artifacts.json)에 있다.

| 결과 | 기존 run ID | ZIP 크기 | 다운로드 |
| --- | --- | ---: | --- |
| RGB-D·초기 검출·추적·영상 | `996f629b211147f18deeb6736c51c3e9` | 38.57 MiB | [ZIP 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/b27cf33a98d48c396a53c779f5707602679af99f/visual/996f629b211147f18deeb6736c51c3e9/artifacts/reports/ad7e48e28476c9b4e17951042225c59e89bc89438f10efa1979cd2aaae0d89c5/report-ad7e48e28476c9b4e17951042225c59e89bc89438f10efa1979cd2aaae0d89c5.zip?download=true) · [HF 파일 목록](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/b27cf33a98d48c396a53c779f5707602679af99f/visual/996f629b211147f18deeb6736c51c3e9) |
| Task 01 — Organized Geometry | `f9b0da8588734a28996a38f3724da16a` | 8.82 MiB | [ZIP 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/92cacaff34a0e1c760e96f298243d27343f483e5/visual/f9b0da8588734a28996a38f3724da16a/artifacts/reports/a707c07253e91a0a57e840c74c82a70d4b826bc488d3d48329ec0a058ce40f49/report-a707c07253e91a0a57e840c74c82a70d4b826bc488d3d48329ec0a058ce40f49.zip?download=true) · [HF 파일 목록](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/92cacaff34a0e1c760e96f298243d27343f483e5/visual/f9b0da8588734a28996a38f3724da16a) |
| Task 02-A — Normal Reliability | `3640b9bbe403409ba1429e0143e5e049` | 8.25 MiB | [ZIP 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/3c9bcb9d7f11b7433f0c432d945ce71add0825ee/visual/3640b9bbe403409ba1429e0143e5e049/artifacts/reports/d156c000f8588b509c07811475472a8ec818f06fe88a1c12c612603a75ea06c2/report-d156c000f8588b509c07811475472a8ec818f06fe88a1c12c612603a75ea06c2.zip?download=true) · [HF 파일 목록](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/3c9bcb9d7f11b7433f0c432d945ce71add0825ee/visual/3640b9bbe403409ba1429e0143e5e049) |
| Task 02-B — Normal Estimator Comparison | `4b1c377ae2044344a4f596b249627d89` | 15.45 MiB | [ZIP 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/72cec859c6c3002c7b9953894ad8139203eb3475/visual/4b1c377ae2044344a4f596b249627d89/artifacts/reports/f4591dc0c533ee913b543e6f502d641b581e1d941bf9ddd940c6a656153b33b5/report-f4591dc0c533ee913b543e6f502d641b581e1d941bf9ddd940c6a656153b33b5.zip?download=true) · [HF 파일 목록](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/tree/72cec859c6c3002c7b9953894ad8139203eb3475/visual/4b1c377ae2044344a4f596b249627d89) |

## 영상 위치

- Grounding DINO: **RGB-D ZIP** 안의 `visual/results/runtime_grounding/ep000_full/grounding.mp4`
- SAM2 추적: **RGB-D ZIP** 안의 `visual/results/runtime_tracking/ep000/tracking.mp4`
- RGB-D·depth·GT: [HF에서 영상 열기/다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/b27cf33a98d48c396a53c779f5707602679af99f/visual/996f629b211147f18deeb6736c51c3e9/artifacts/rollouts/a86cdaa55bc39896e551171bf04871e3d9ca2444aa2845277a457ea85bab8941/depth.mp4) 또는 RGB-D ZIP 안의 `visual/results/runtime_rgbd/996f629b211147f18deeb6736c51c3e9/depth.mp4`

세 영상 모두 121프레임, 20fps다. 초기 검출·추적 영상은 RGB-D 평가 이전에 생성된 dependency이며,
게시를 위해 새 실행 ID를 만들거나 생성 실험을 추정하지 않았다. Task 01/02-A/02-B는 PNG·HTML 결과다.

## 내용과 열기

RGB-D ZIP에는 최종 GT 리포트와 depth 영상, 원본 `ep000.npz`, tracker mask·metadata,
Grounding DINO/SAM2 영상·metadata 및 0/15/45/80/120 대표 프레임이 들어 있다.
Task 01 ZIP에는 전체 scalar 수치와 0/15/18/45/80/120의 PNG·dense NPZ가 있다.
Task 02-A ZIP에는 합성 비교, 전체 121프레임 수치, 0/15/45/80/120의 전체 맵·경계 확대·profile·dense NPZ가 있다.
Task 02-B ZIP에는 5종 normal estimator의 합성·121프레임 전체 수치, 같은 5프레임의 비교·유효성·profile·dense NPZ,
검증 기록·입력 탐색 inventory·보존 snapshot·구현 patch·평가 당시 코드 사본을 포함한 31개 파일이 있다.
02-B의 전체 수치 JSON 약 51.7 MiB는 HF에 보관하고 Git에는 보고서와 대표 그림을 남겼다.
모든 프레임의 annotation PNG, 중간 실행, 중복 tracking `ep001`은 로컬에 유지하고 게시 대상에서 제외했다.

ZIP마다 별도의 새 폴더에 압축을 풀고 다음 entrypoint를 연다. HTML과 그림·영상의 상대경로는 ZIP에 유지했다.

| ZIP | Entry point |
| --- | --- |
| RGB-D | `visual/results/runtime_rgbd/996f629b211147f18deeb6736c51c3e9/gt_report.html` |
| Task 01 | `visual/results/runtime_geometry/f9b0da8588734a28996a38f3724da16a/geometry.html` |
| Task 02-A | `visual/results/normal_diagnostics/3640b9bbe403409ba1429e0143e5e049/diagnostics.html` |
| Task 02-B | `visual/results/normal_comparison/4b1c377ae2044344a4f596b249627d89/comparison.html` |

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

## 실행 identity와 검증

게시를 위한 새 Logger/UUID는 생성하지 않았다. 원래 실험의 Git commit/dirty 상태와 run ID를 유지했다.
RGB-D/Task 01/02-A의 publication/source commit은 `20b906c6d13d592ef94addcdf6297c31a68c1fb4`,
Task 02-B는 `366e9ce4b1510926449fc7c51765ac72849b16c4`다.
원래 실험은 `git_dirty=true`였으므로 두 commit 정보를 서로 덮어쓰지 않는다.
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
- PCA 최적화·CUDA/Triton·structural edge의 미커밋 작업은 이번 게시에서 제외했다.

게시 목록은 [rgbd_files.json](rgbd_files.json), [geometry_files.json](geometry_files.json),
[normal_files.json](normal_files.json), [comparison_files.json](comparison_files.json)에 명시했다. 명령과 공통 API는
[HF 게시 문서](../../../docs/huggingface_artifacts.md#상대경로를-유지하는-보고서-zip)를 따른다.
예를 들어 Task 02-B 게시 준비를 다시 확인하려면:

```bash
.venv/bin/python -m tools.publish_experiment \
  --run-id 4b1c377ae2044344a4f596b249627d89 --include reports metrics \
  --report-files visual/results/publications/comparison_files.json --dry-run
```

Dry-run은 원격을 변경하지 않지만 로컬 ZIP·metadata·manifest·sidecar는 생성/갱신할 수 있다.
