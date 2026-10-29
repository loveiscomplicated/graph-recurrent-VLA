# Rollout graph visualization

Run the read-only local frontend from the repository root:

```bash
.venv/bin/python -m visualization.server
```

Open `http://127.0.0.1:8765/`. The main page lists compatible saved Task07
graph rollout episodes and Task09-C geometry diagnostic episodes found under
`experiments/`. It checks the first graph
frame against its matching observation before listing an episode. Older artifacts
without TCP pose and robot geometry cannot use the EE/world toggle and are
omitted. Task09-C graphs instead join `geometry_manifest.json` to the matching
capture run, check the input hash and same-timestep robot geometry, and preserve
their separate 21-feature schema. Search by run or episode name and filter
by **성공**, **실패**, or **미확인**. Click a card to open that episode's interactive
3D graph viewer, and use **← 목록** to return. A successful training sample is
listed first when available; its success does not imply held-out performance.

The viewer reads `graph_XXXX.npz` and same-timestep `observations.npz`. The
**EE 상대 / 월드 절대** toggle changes displayed node positions; the world view
uses the saved TCP world pose. The original F5 values and `edge_attr` stay in
the EE frame. Use the timeline or play button to move through graph inputs,
drag to rotate, Shift/right drag to pan, scroll to zoom, and click a node to
inspect its saved model inputs. Frames are loaded on demand. Task09-C cards open
this same interactive viewer with a **그래프 조건** selector for G0–G4 and the
separately labelled Oracle O1–O3. Node inspection includes the saved persistent
ID (or G0 original object row) and all seven extra status/confidence fields.
The **노드 색상** selector can color current, estimated and low-confidence nodes.
Diagnostic captures remain **미확인** for task success.

Status is **성공** when episode metadata explicitly records `success: true`;
**실패** when a `COMPLETE` episode explicitly records `success: false`; otherwise
it is **미확인**. This keeps smoke or unfinished rollouts from being labeled as
evaluation failures.

Options: `--experiments-root PATH`, `--host HOST`, `--port PORT`. The default
host is `127.0.0.1`; point `--experiments-root` at another local artifact tree
if needed. The server only reads artifacts and does not publish them.

## Task09-C 수집 데이터 확인

같은 서버에서 `http://127.0.0.1:8765/captures`를 열거나 메인 목록의
**Task09-C 수집 데이터** 링크를 누른다. `capture_manifest.json`이 있는
수집 run을 자동으로 찾는다. 기존 Task07의 3D 그래프 화면과 함께 사용할 수 있다.

이미 실행 중인 서버에는 Python 변경이 자동 반영되지 않는다. 기존 서버를 유지하며
확인하려면 `.venv/bin/python -m visualization.server --port 8766`으로 새 서버를
띄우고 [수집 화면](http://127.0.0.1:8766/captures)을 연다.

- 에피소드 선택, 타임라인, 이전/다음 프레임, 2/5/10 fps 재생,
  접근·하강·잡기·들기·정렬·삽입 단계 시작점으로 이동.
- Main / Wrist의 같은 시점 RGB, 깊이, 예측 마스크, 평가용 GT 마스크를 나란히 표시.
  Peg는 빨강, Box는 파랑, 중복 예측은 보라색이다.
- 깊이는 원본 signed int16 밀리미터를 사용한다. `≤0`은 검정이며,
  선택한 상한보다 먼 값은 상한 색상으로 표시한다. 재생 속도는 표시 속도이며
  실제 시뮬레이션 시간은 프레임에 저장된 초 단위 timestamp로 따로 표시한다.
- 카메라별 유효 깊이 비율·범위, 물체별 예측/GT 픽셀 수·IoU를 표시한다.
  예측과 GT가 모두 비어 있으면 IoU는 `—`이다. 예측 마스크는 필터 전 수집 원본이며,
  GT는 분할 품질 확인에만 사용한다. 수집 종료 표시는 작업 성공 판정이 아니다.

파일을 열 때 input/masks/GT 세 파일의 SHA-256을 수집 manifest와 대조한다.
뷰어는 시뮬레이터·모델을 실행하거나 원본 파일을 수정하지 않는다.

이번 파일럿은 run `e835cf0c001e47ee88a1e8dbb5fb877c`이며,
seed30002(train, 106프레임), seed30047(train, 112프레임),
seed30125(validation, 149프레임), 총 367프레임이다.
[첫 에피소드 바로 열기](http://127.0.0.1:8765/captures?episode=e835cf0c001e47ee88a1e8dbb5fb877c%2Fcollection%2Fseed30002).
이 로컬 작업공간의 `experiments/`에 원격 수집본을 복사했다. 데이터는 Git에 포함되지 않는다.
다른 작업공간에서는 원본 run 폴더를 아래 구조로 복사하거나
`--experiments-root`로 해당 artifact tree를 지정한다.

```text
experiments/<run_id>/capture_manifest.json
experiments/<run_id>/collection/seed30002/sensor_contract.json
experiments/<run_id>/collection/seed30002/f0000_input.npz
experiments/<run_id>/collection/seed30002/f0000_masks.npz
experiments/<run_id>/collection/seed30002/f0000_gt.npz
```

원격 원본 위치는 `/workspace/task09c/experiments/e835cf0c001e47ee88a1e8dbb5fb877c/`이다.
해당 run을 통째로 복사하면 다른 seed와 전체 프레임도 포함된다.

## Task09-C 그래프와 추가 영상

개선판 파일럿 그래프 run `b1c5851d4a35489098d7b9eaeaf6b6c4`
(`pilot_revision2`)의 3개 에피소드를 기존 rollout 목록에 연결했다.
[목록](http://127.0.0.1:8766/)에서 `Task09-C` 또는 seed로 검색하거나
[seed30002 인터랙티브 3D](http://127.0.0.1:8766/viewer?episode=b1c5851d4a35489098d7b9eaeaf6b6c4%2Fgeometry%2Fseed30002)를 연다.
원격 파일을 그대로 복사해 읽으며, 현재 작업 중인 geometry 코드를 다시 실행해
그래프를 생성하지 않는다. 따라서 이후 구현 수정과 구별되는 파일럿 결과이다.

수집 확인 페이지의 **그래프 영상**에는 각 에피소드·조건마다 두 영상이 있다.

- **a. Graph + RGB:** 월드 좌표 3D 그래프와 동일 시점 Main/Wrist RGB.
- **b. Only graph:** 같은 저장 그래프만 재생.

3개 에피소드 × 8조건 × 2형식 = **48개 MP4**, 총 5,872 영상 프레임이다.
전체 367개 수집 시점이 각 조건에 포함되며, 수집 timestamp 간격에 맞춘 20fps이다.
드롭다운으로 조건을 바꾸고 **현재 프레임에 맞추기**를 누르면 두 영상을 위쪽
원본 프레임의 시점으로 이동한다. 영상은 기본 브라우저 컨트롤로 재생·탐색할 수 있다.
**인터랙티브 3D 열기**는 선택한 조건을 그대로 전달한다.

영상은 한 에피소드의 모든 조건·시점에 동일한 투영 방향과 축척을 사용한다.
G0의 일부 매우 먼 점이 전체 장면을 작게 만드는 경우가 있어, 좌표의 P1–P99와
전체 로봇 궤적을 포함하는 확대 영역 및 **모든 저장 노드·엣지를 표시하는 전체 보기**를
함께 제공한다. 확대 영역 밖 노드 수를 표시한다. 원본 노드·엣지·feature는 변경하지 않는다.
G0는 원본 Raw EDE의 budget subset이고, G1–G4/O1–O3는 저장된 surface graph다.
모두 물체당 최대 128개 노드인 비교용 그래프이며, 전체 가변 N 지도와 구분한다.

```text
experiments/b1c5851d4a35489098d7b9eaeaf6b6c4/
  geometry_manifest.json
  geometry/seed30002/f0000/G4_graph.npz
  geometry/seed30002/f0000/G4_graph.json
  viewer_media/index.json
  viewer_media/sources.json
  viewer_media/seed30002/G4_graph_rgb.mp4
  viewer_media/seed30002/G4_graph_only.mp4
```

다른 작업공간에는 capture run과 geometry run을 모두 같은 `experiments/` 아래에 둔다.
이 화면은 저장된 graph NPZ/JSON만 필요하며, 모델 checkpoint나 simulator 실행은 필요 없다.
영상 생성 명령은 다음과 같다(기존 생성 영상을 교체할 때만 `--replace-videos` 추가).

```bash
.venv/bin/python -m visualization.render_captures \
  --geometry-run b1c5851d4a35489098d7b9eaeaf6b6c4
```

생성기는 모든 MP4를 전 프레임 디코딩해 확인하고, 그래프/JSON/수집 input의 SHA-256을
`sources.json`에, 영상별 SHA-256·프레임 수·fps를 `index.json`에 기록한다.
공통 `utils.experiment_artifacts.register_existing_output`으로 완료된 **로컬** geometry run에
영상과 인덱스를 등록한다. 새 학습 run이나 업로드를 만들지 않으며 원격 원본을 수정하지 않는다.
이 호스트에서는 CPU로 저장 파일을 읽고 영상을 만들었으며 simulator/GPU를 실행하지 않았다.

```bash
.venv/bin/python -m unittest visualization.test_viewer \
  visualization.test_captures visualization.test_persistent -v
```

연결 시 검증: 뷰어 테스트 11개 통과, 48개 MP4 전 프레임 디코딩(5,872프레임),
6,239개 원본 입력/그래프/JSON SHA-256 및 48개 영상 SHA-256 일치,
HTTP byte-range 탐색과 브라우저의 목록→3D→영상 이동·재생·조건 전환 확인.
별도 공통 Logger/Artifact 테스트는 42개 중 41개 통과했다.
`test_reject_running_run_symlink_and_outside_organization`의 외부 조직 검사 실패는
변경 전 HEAD의 Artifact 소스로도 재현됐으며, 관련 공통 코드는 수정하지 않았다.

## 수집 화면만 독립 배포

기존 API의 해시 검증을 거쳐 선택한 수집 run을 정적 웹사이트로 내보낸다.
타임라인·두 카메라·깊이·마스크·통계와 명시적으로 선택한 geometry run의 영상만
포함한다. 다른 그래프 목록/인터랙티브 3D 링크는 독립 화면에서 제외한다.
원본 NPZ, 모델, simulator는 배포 파일에 포함되지 않는다.

```bash
.venv/bin/python -m visualization.export_captures \
  --capture-run e835cf0c001e47ee88a1e8dbb5fb877c \
  --geometry-run b1c5851d4a35489098d7b9eaeaf6b6c4 \
  --output outputs/task09c-static
.venv/bin/python -m http.server 8879 --directory outputs
```

`http://127.0.0.1:8879/task09c-static/`에서 확인한다. 출력 폴더는 새 경로여야 하며,
같은 이름의 기존 결과를 덮어쓰지 않는다. `--geometry-run`을 생략하면 수집 프레임만
포함한다. 생성된 `outputs/task09c-static-selection.json`이 게시 파일의 명시적 목록이다.

공통 report bundle과 publisher로 기존 capture run identity를 유지하며 HF Static
Space에 배포할 수 있다. 기본은 비공개이며, 공개 배포를 요청받은 경우에만
`--public-space`를 추가한다. 기존 Space의 공개 범위는 변경하지 않는다.

```bash
.venv/bin/python -m tools.publish_experiment \
  --run-id e835cf0c001e47ee88a1e8dbb5fb877c --include reports \
  --report-files outputs/task09c-static-selection.json \
  --static-space Graph-Recurrent-VLA/task09c-captures --dry-run
# 위 명령에서 --dry-run을 제거하면 실제 배포
```

`--dry-run`도 기존 report workflow처럼 로컬 ZIP/manifest를 등록한다.
선택한 웹 파일과 `bundle_manifest.json`만 Space에 업로드하며, 원본 config/metrics는
로컬 ZIP에 보존한다. 배포 revision/URL은 capture run의 `static_space_uploads.json`에
기록한다. 정적 파일은 상대경로를 사용하므로 GitHub Pages 하위 경로에서도 제공할 수 있다.
Hugging Face 로그인·Space 접근 권한은 비공개 배포를 열 때 필요하다.

배포 결과: [task09c-captures](https://huggingface.co/spaces/Graph-Recurrent-VLA/task09c-captures),
비공개 Static Space. 배포 revision은 `a36083903473eb866703fdb3232e5d513057486a`이며,
3개 에피소드·367프레임·54개 영상을 포함한다. Chrome에서 원격 화면 로드,
삽입 단계 이동과 Graph + RGB 영상 재생을 확인했다. 로컬에서는 에피소드 전환과
두 영상 재생도 확인했다. HTTP로 전체 프레임 JSON/영상 경로를 확인했다.
테스트는 Logger/Artifact/뷰어/신규 배포 계약 총 58개 중 57개 통과했으며,
기존 외부 조직 검사 실패 1개는 변경 전 HEAD에서도 재현됐다.
검증은 저장 파일과 HTTP/브라우저를 사용했고 simulator/CUDA를 실행하지 않았다.

배치 업데이트: 수집 재생/깊이 상한·타임라인·단계 이동·프레임 상태·범례를 그래프
영상 섹션 아래, 카메라 이미지 위로 이동했다. 수집 뷰어 테스트 4개 통과와
원격 HTML 일치를 확인했다. 데이터·영상 파일은 이전 배포와 동일하다.

## 초기 Visual FPS point cloud 비교 영상

수집 화면의 **조건**에서 **point cloud prototype(FPS)**를 선택한다.
초기 `visual/scripts/pointcloud.py::extract`를 수정 없이 현재 Task09-C Main의
예측 마스크·signed int16 깊이에 재적용한 결과다. 당시 v0 결과의 재생과 구별한다.
현재 캡처는 128²이며 초기 512² 입력을 재현하거나 SAM2를 다시 실행하지 않는다.
Main만 사용하는 초기 방식에 맞췄고, 오른쪽 Main/Wrist RGB는 같은 시점의 참고 영상이다.

원래의 erosion 2회 → 최대 2,048점 무작위 사전 축소 → SOR(k=16, std=2) →
FPS 최대 1,024점/물체를 사용한다. Cluster 제거·로봇 제거·누적·그래프 엣지·GT
물체 입력은 없다. 깊이는 mm→m, 좌표는 저장된 world→CV extrinsic의 역행렬로
변환하며 pixel center +0.5를 유지한다. RNG는 에피소드마다 seed0으로 초기화하고
시간순·Peg/Box 순으로 진행한다. 유효 관측이 없으면 점군도 비어 있다.

```bash
.venv/bin/python -m visualization.render_pointcloud \
  --geometry-run b1c5851d4a35489098d7b9eaeaf6b6c4
```

기존 graph MP4를 유지하며 FPS 두 형식만 추가한다. 재생성은 `--replace-videos`를
지정한다. 3개 에피소드 × 2형식 = 6개 MP4, 총 734개 영상 프레임이며 기존과
같은 20fps·전체 367개 수집 시점을 사용한다. 확대 방향·영역은 기존 그래프와
같고, 전체 보기 inset은 모든 FPS 점을 포함하도록 필요할 때 확장한다.

`viewer_media/fps_sources.json`은 원본 추출기/renderer hash, 입력·마스크 SHA-256,
프레임별 점 수, 저장된 `seedN/FPS_points.npz` hash를 기록한다. 공통 사후 등록
함수로 기존 geometry run에만 연결하며 새 학습 run을 만들지 않는다. 정적 export는
영상·설명·provenance JSON을 포함하고 원본 NPZ는 로컬에 유지한다. FPS는 영상 전용
조건이며 저장 graph schema나 인터랙티브 3D의 G0–G4/O1–O3 목록에는 넣지 않는다.
CPU로 저장 파일을 처리했으며 simulator/GPU는 실행하지 않았다.

FPS 배포 검증: 신규 MP4 6개 전 프레임 디코딩(734프레임), 기존 48개 영상과
367개 수집 프레임 불변, 조건별 영상/설명 연결 및 관련 뷰어 테스트 14개 통과.
로컬 Chrome에서 FPS 선택과 두 영상 재생을 확인했다. 원격은 API로 RUNNING과
433개 게시 파일의 크기·Git/LFS hash 일치를 확인했다.

## 실험 결과 목록

로컬 서버의 `/results`에서 **실험 결과 모아보기**를 연다. 기존 `/`의 3D rollout
목록과 `/captures`는 유지하며 두 화면의 헤더에서도 결과 목록으로 이동할 수 있다.
정적 배포의 루트 `index.html`은 월드 절대 좌표와 O3 조건으로 열리는 인터랙티브 3D 뷰어다.
상단 링크로 `captures.html`의 수집·그래프/FPS 화면 또는 `results.html`의 결과 목록을 연다.
보고서의 **← 결과 목록**은 `results.html`로 돌아온다. 기존 `?episode=...` 북마크도 유지한다.

`results_catalog.json`은 보고서 파일별 한국어 제목·소제목·주제·Task와 대표 실행을
정의한다. `results.py`는 `experiments/<run_id>/*.html`을 읽어 같은 주제의 실행들을
묶는다. 브라우저 검수본·템플릿·하위 source snapshot은 기본 목록에서 제외한다.
진행 중인 실행은 상태만 표시하고 웹 파일은 게시하지 않는다. 실패·중간 보고·
상태 미기록·그림 누락도 별도로 표시하며 완료를 실험 성공으로 해석하지 않는다.

현재 49개 보고서를 38개 주제로 정리했고, 진행 중 1개를 제외한 48개를 열 수 있다.
검색은 한국어 제목·소제목·Task·기존 키워드·run ID를 지원한다. 주제 필터,
최근 실행순 정렬, 관련 실행 펼치기를 제공한다. 새로운 파일명은 임시 한국어
항목으로 자동 발견되며 내용을 읽은 뒤 curation JSON에 적절한 제목을 추가한다.

```bash
.venv/bin/python -m visualization.export_results \
  --capture-run e835cf0c001e47ee88a1e8dbb5fb877c \
  --geometry-run b1c5851d4a35489098d7b9eaeaf6b6c4 \
  --output outputs/results-main-final
.venv/bin/python -m http.server 8881 --directory outputs/results-main-final
```

출력 폴더는 새 경로를 사용한다. 보고서가 실제로 참조하는 그림·영상·작은 JSON/CSV·
설명 문서만 명시적 selection에 포함하며 원본 HTML/실험 파일은 수정하지 않는다.
웹용 사본에는 **← 결과 목록** 링크를 추가한다. 원본 데이터·체크포인트·소스/로그·
2MiB 초과 JSON/CSV/TXT 링크는 로컬 자료 안내로 이어진다. 원본에 없는 그림은
누락 표시를 사용하며 다른 실행의 그림으로 대체하지 않는다. 선택하지 않은 파일과
symlink·저장소 밖 경로는 로컬 report API에서도 제공하지 않는다.

`results_export_manifest.json`은 파일별 원본 run/path/hash와 준비된 웹용 사본 hash를 보존한다.
공통 JSON 정규화·비밀 키 마스킹 이후 실제 게시 파일의 hash는 `bundle_manifest.json`을 따른다.
기존 capture run을 hosting identity로 재사용하되 다른 실행의 자료는 dependency로
구분한다. 게시에는 기존 공통 report bundle·Static Space workflow를 사용한다.

```bash
.venv/bin/python -m tools.publish_experiment \
  --run-id e835cf0c001e47ee88a1e8dbb5fb877c --include reports \
  --report-files outputs/results-main-final-selection.json \
  --static-space Graph-Recurrent-VLA/task09c-captures
```

검증: 관련 Python 테스트 21개 통과, Node UI 동작 검사(목록·검색·필터·빈 결과·
검색 URL 복원·이전 capture 북마크) 통과. 웹 사본 53개 HTML의 상대 링크 2,760개
확인(깨진 링크 0개), PNG 749개 decode, 원본 HTML 50개와 수집 프레임 367개
불변 확인. 첫 목록 배치는 Chrome에서 확인했고, 검색/필터는 별도 동작 검사로
확인했다. Simulator/GPU/학습을 실행하지 않았다.

기존 결과 목록 배포 revision: `eb8632b235a9bd653c845b32f5a2a698fbd2f87f`.
기존 비공개 [Space](https://huggingface.co/spaces/Graph-Recurrent-VLA/task09c-captures)의
수집·그래프 비교 UI를 제공한다. 실험 결과 모아보기는 헤더 링크로
`results.html`에서 연다. 기존 3개 에피소드·54개 비교 영상과 한국어 결과 목록은 유지한다.

원격 API 검증: RUNNING/private 상태, 선택 게시 파일 2,081개의 크기·Git/LFS hash
일치, 메인·목록 JSON·요청한 Raw EDE HTML의 revision 고정 다운로드 확인.

첫 화면 복원 검증: 관련 테스트 8개 통과, HTML 상대 링크 검사 통과. 원격 index/captures는
기존 수집·그래프 UI, results.html은 별도 목록, 보고서의 복귀 링크는 results.html임을 확인했다.

### 허깅페이스 정적 3D 뷰어

정적 배포의 수집 화면에서 **인터랙티브 3D 열기**를 누르면 선택한 에피소드와
그래프 조건을 그대로 3D 뷰어로 연다. 좌클릭 드래그로 회전하고 휠로 확대·축소하며,
타임라인, 좌표계, 노드·엣지 필터와 노드 입력 검사를 사용할 수 있다. G0–G4와
Oracle O1–O3의 저장 그래프를 제공한다. FPS point cloud는 현재 영상으로 제공하며
인터랙티브 그래프 데이터에는 포함하지 않는다.

정적 내보내기는 선택한 geometry run의 그래프를 에피소드와 조건별 JSON으로 묶는다.
최초 조건 선택 때 해당 조건의 전체 프레임을 내려받고 이후에는 브라우저에서 재사용한다.
원본 그래프와 수집 시점의 일치 여부를 기존 accessor로 검증하고, 새 그래프는 생성하지 않는다.
