# Task 03 후속 실험 산출물

`loveiscomplicated` 계정으로 기존 비공개 Dataset `Graph-Recurrent-VLA/graph-recurrent-rollouts`에 게시했다.
Git에는 [선별한 수치·그림](../diagnostic_previews/README.md)을 보관하고, 아래 ZIP에는 전체 결과·선택 NPZ·실행 소스 snapshot·원래 run metadata를 보존한다.
원본 실험의 Git commit/dirty 상태와 게시 코드 commit은 [artifacts.json](artifacts.json)에 구분했다. 게시를 위해 새로운 실험 UUID를 만들지 않았다.

| 실험 | 원래 run ID | ZIP | Git 결과 |
| --- | --- | --- | --- |
| Task 03d_phase1 — Box rim staged coverage | `d3f14ee494894daabb8ba2fab8e1b00c` | [1.00 MiB 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/008c904edfd934869a846cf12bc686b39536dac1/visual/d3f14ee494894daabb8ba2fab8e1b00c/artifacts/reports/912dc970744ab2d7810032c279d3b3d4bb66c8da01d622685cb21305dfa2ac3a/report-912dc970744ab2d7810032c279d3b3d4bb66c8da01d622685cb21305dfa2ac3a.zip?download=true) | [수치·핵심 그림](../diagnostic_previews/03d_phase1/) |
| Task 03d_phase2 — Priority FPS trace | `aa5d71078e0841fd886b5026d6030ebc` | [0.34 MiB 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/5662b5cab5f1057582fca47a33a4a9f101b98605/visual/aa5d71078e0841fd886b5026d6030ebc/artifacts/reports/1d3d6afabe73755cdd876ff41b7282679a4b5d6e75b1664c816c6f7ed9614e25/report-1d3d6afabe73755cdd876ff41b7282679a4b5d6e75b1664c816c6f7ed9614e25.zip?download=true) | [수치·핵심 그림](../diagnostic_previews/03d_phase2/) |
| Task 03d_phase3 — Priority FPS seed control | `f36f1e9e2bb64866b8d6c792db158635` | [0.47 MiB 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/76e75d421b17c4b02d5c85ec04e9a1ea4ecdb243/visual/f36f1e9e2bb64866b8d6c792db158635/artifacts/reports/56e021d3f0699171df126ba46462751bb8d578a1f1fe013b29ff7827bfb0266b/report-56e021d3f0699171df126ba46462751bb8d578a1f1fe013b29ff7827bfb0266b.zip?download=true) | [수치·핵심 그림](../diagnostic_previews/03d_phase3/) |
| Task 03ab — Adaptive node budget | `f68c59f210b5499dadc7e962c7f64492` | [120.59 MiB 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/50092ffa617e15aafb9afd1feead6fa95b08c67e/visual/f68c59f210b5499dadc7e962c7f64492/artifacts/reports/67fd988224d72d3d86741b5e06ae3136d31dd38d35bf6244072accab5fe6eb0b/report-67fd988224d72d3d86741b5e06ae3136d31dd38d35bf6244072accab5fe6eb0b.zip?download=true) | [수치·핵심 그림](../diagnostic_previews/03ab/) |
| Task 03e — Priority FPS metric control | `7f657cf88baf4f1f89e594252e243e1c` | [9.46 MiB 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/ec6a336cef1ecefc64c86e92743d5700d94021e2/visual/7f657cf88baf4f1f89e594252e243e1c/artifacts/reports/d9b14199bfc6504566f4f96ca87c6389f26e843cd085f39147f1633136971f6d/report-d9b14199bfc6504566f4f96ca87c6389f26e843cd085f39147f1633136971f6d.zip?download=true) | [수치·핵심 그림](../diagnostic_previews/03e/) |
| Task 03f — Local structural sampling | `1f7ad262dd2e4b2fb0ac805a4eda5278` | [6.86 MiB 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/c8ba39524a1dd4eef9b3b64247eebaa9de24a3d8/visual/1f7ad262dd2e4b2fb0ac805a4eda5278/artifacts/reports/2c99d6a030f6818fa191b2d162ce9b40a45f0d11eb726efc0e04fd9a14b8ddd4/report-2c99d6a030f6818fa191b2d162ce9b40a45f0d11eb726efc0e04fd9a14b8ddd4.zip?download=true) | [수치·핵심 그림](../diagnostic_previews/03f/) |
| Task 03g — Geometry aware sampling | `495a67210c5c4cbfab91d013239f7ce9` | [5.93 MiB 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/01b0f7bd94e568e47f36ff770665952c93cee27e/visual/495a67210c5c4cbfab91d013239f7ce9/artifacts/reports/bddd654e18543ac9fdd1a2662ce69e77dc334d65040f8e3590e7c06a06c62f4d/report-bddd654e18543ac9fdd1a2662ce69e77dc334d65040f8e3590e7c06a06c62f4d.zip?download=true) | [수치·핵심 그림](../diagnostic_previews/03g/) |
| Task 03h — Candidate count matched control | `79358cfd00bd409f837b713972fba2ca` | [2.14 MiB 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/7929ef00eb04a9f36849ecdc815fa444c0df457f/visual/79358cfd00bd409f837b713972fba2ca/artifacts/reports/8deb4342b75b0bb3f48cb83be44138944c4b89553c176b77566610d377b92480/report-8deb4342b75b0bb3f48cb83be44138944c4b89553c176b77566610d377b92480.zip?download=true) | [수치·핵심 그림](../diagnostic_previews/03h/) |
| Task 03i — Voxel structural sampling | `92f29716e76f4b998fe924f08e80357d` | [4.93 MiB 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/995430173821926ccd2bfcb01e0fb6c1ff43670d/visual/92f29716e76f4b998fe924f08e80357d/artifacts/reports/29bfbc780e42f60fe6b4ce26dcc9e6340375008899b56869ee8b2b341d9a7c0e/report-29bfbc780e42f60fe6b4ce26dcc9e6340375008899b56869ee8b2b341d9a7c0e.zip?download=true) | [수치·핵심 그림](../diagnostic_previews/03i/) |
| Task 03j — Variable rate shape first sampling | `15efc476f735491a828f5b84f352f858` | [13.61 MiB 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/05cb43c5f481aac38f5bec4452a6c039ea5f1304/visual/15efc476f735491a828f5b84f352f858/artifacts/reports/2a1c652a21800c701f10642c2ac30e9ef297151efb2e48d7edbcb63f798ba626/report-2a1c652a21800c701f10642c2ac30e9ef297151efb2e48d7edbcb63f798ba626.zip?download=true) | [수치·핵심 그림](../diagnostic_previews/03j/) |
| Task 03k — Temporal shape first evaluation | `9185355a32b8433abc70f1738f983377` | [49.60 MiB 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/cdca3a80ce4ce465298a7b59ad189abd7dfcf3c1/visual/9185355a32b8433abc70f1738f983377/artifacts/reports/5b7a320ce73a09c80d0acc147cfd32b0966f1f6d8ed36db4e3ec72466335a217/report-5b7a320ce73a09c80d0acc147cfd32b0966f1f6d8ed36db4e3ec72466335a217.zip?download=true) | [수치·핵심 그림](../diagnostic_previews/03k/) |
| Task 03 — Development and failed attempts | `aa5d71078e0841fd886b5026d6030ebc` | [21.73 MiB 다운로드](https://huggingface.co/datasets/Graph-Recurrent-VLA/graph-recurrent-rollouts/resolve/2a0b10c228eee49096b7ca6f1af1f75e26a1c16e/visual/aa5d71078e0841fd886b5026d6030ebc/artifacts/reports/2d2c867ffd0e4382d129e1a88c8533705bffa3798af55f7e8ef33849649bf7db/report-2d2c867ffd0e4382d129e1a88c8533705bffa3798af55f7e8ef33849649bf7db.zip?download=true) | 개발·실패 기록은 ZIP에 보존 |

## 복원과 재현 입력

ZIP을 각각 별도 디렉터리에 풀면 원래 repository 상대경로가 복원된다. HTML entrypoint는 다음과 같다.

| 묶음 | Entry point |
| --- | --- |
| `03d_phase1` | `experiments/d3f14ee494894daabb8ba2fab8e1b00c/visualizations.html` |
| `03d_phase2` | `experiments/aa5d71078e0841fd886b5026d6030ebc/priority_fps_trace.html` |
| `03d_phase3` | `experiments/f36f1e9e2bb64866b8d6c792db158635/priority_fps_seed_control.html` |
| `03ab` | `experiments/f68c59f210b5499dadc7e962c7f64492/adaptive_node_budget.html` |
| `03e` | `experiments/7f657cf88baf4f1f89e594252e243e1c/priority_fps_metric_control.html` |
| `03f` | `experiments/1f7ad262dd2e4b2fb0ac805a4eda5278/local_structural_sampling.html` |
| `03g` | `experiments/495a67210c5c4cbfab91d013239f7ce9/geometry_aware_sampling.html` |
| `03h` | `experiments/79358cfd00bd409f837b713972fba2ca/candidate_count_matched.html` |
| `03i` | `experiments/92f29716e76f4b998fe924f08e80357d/voxel_structural_sampling.html` |
| `03j` | `experiments/15efc476f735491a828f5b84f352f858/variable_rate_shape_first.html` |
| `03k` | `experiments/9185355a32b8433abc70f1738f983377/temporal_shape_first.html` |
| `03_development` | `experiments/f01e083a46a44d52b8c9713fdfca0113/priority_fps_trace.html` |

문서의 `experiments/<run_id>/...` 링크는 ZIP을 checkout root에 복원한 후 사용할 수 있다. Git에 선별한 파일은 보고서에서 `visual/results/diagnostic_previews/`로 직접 연결했다.
원본 RGB-D episode와 tracker masks는 [기존 RGB-D 묶음](README.md)에, Task03 원본 저장 selection은 같은 인덱스의 `sampling` 묶음에 있다. Task03-J/K에는 `03ab` evidence cache도 필요하다. K의 비교 기준은 `03j`다.
여러 ZIP을 checkout에 합칠 때 `bundle_manifest.json`과 `experiment/`는 각 ZIP별로 따로 보관하고, manifest의 `files` member만 원래 위치로 복원한다.

최종 결과와 실패·개발 기록을 섞어 해석하지 않는다. E와 F ZIP에는 각각 실패 실행 1개를 dependency로 보존했고, 보충 `03_development` ZIP에는 문서에 직접 링크되지 않은 11개 원래 실행(반복·진단·실패)을 보존했다. 각 상태는 인덱스의 `dependency_statuses`에 있다.
`03_development`는 기존 FPS trace UUID를 게시 anchor로 재사용한 보충 기록이며 별도 실험 결과가 아니다.

## 검증 범위

- Vision runtime 테스트 250개 실행: 247 PASS / CUDA 조건 3 SKIP.
- 공통 Logger/Artifact 계약 테스트 42/42 PASS. 최초 sandbox 실행의 W&B `ServicePollForTokenError`는 sandbox 해제 후 재실행에서 해소됐다.
- 12개 ZIP, 1,469개 선택 member의 SHA-256과 ZIP 무결성 검증.
- 원격 객체 818개 hash 대조, 12개 ZIP 재다운로드 SHA-256·무결성 검증.
- HF commit 작성자와 인증 계정은 `loveiscomplicated`. 링크는 게시 revision에 고정했다.
- 이번 작업에서 simulator/renderer, CUDA/MPS, tracker/model 추론, 전체 실험 sweep은 재실행하지 않았다.

게시 준비는 기존 `utils.experiment_artifacts.create_report_bundle`을, 업로드는 `tools.publish_experiment --include reports metrics --private`를 재사용했다. 새 uploader/logger와 자동 게시 workflow는 추가하지 않았다.
각 `*_files.json`이 명시적 ZIP 선택 목록이다. 원본 실행 source와 평가 수치는 보존했고, 보고서의 일부 Git 미리보기 링크와 게시 안내만 정리했다.
