# Task10 — PLAN A 첫 프레임 추출

2026-10-29, 로컬에 보존된 PLAN A 원본 HDF5에서 각 에피소드의 첫 expert action 이전 관측(t=0)을 추출했다.

## 대상 및 결과

- PLAN A comparison UUID: `3788798a719f4ec4a66c224ca42f5d63`.
- 원본 dataset ID: `d1fd6173cb3648bfa4c03507f84b7e71`.
- [원본 HDF5 디렉터리](../../experiments/d1fd6173cb3648bfa4c03507f84b7e71/collection/episodes/): PLAN A 대상 120개 파일 모두 로컬에 존재한다.
- [PLAN A cache manifest](../../experiments/3788798a719f4ec4a66c224ca42f5d63/remote_snapshot/cache_manifest.json)에 기록된 120개 에피소드를 사용했다. Train 96, validation 12, test 12의 기존 분할을 유지했다.
- 에피소드마다 `base_camera/rgb[0]`(메인), `hand_camera/rgb[0]`(손목)을 원본 128×128 RGB PNG로 저장했다. 총 240장: train 192, validation 24, test 24.
- [전체 ZIP](../../outputs/plan_a_first_frames_20261029.zip)에는 이미지, 카메라별 모음 이미지, manifest, 검증 결과 및 재실행용 추출 스크립트가 들어 있다.
- [메인 카메라 모음](../../outputs/plan_a_first_frames_20261029/overview_base_camera.jpg) · [손목 카메라 모음](../../outputs/plan_a_first_frames_20261029/overview_hand_camera.jpg).

## 검증 및 실행 범위

120개 원본 HDF5의 SHA256이 PLAN A manifest의 `raw_episode_sha256`과 모두 일치했다. 각 파일의 첫 `timestep=0`, episode ID, reset seed를 대조했다. 저장한 PNG 240개를 다시 열어 원본 RGB 배열과 픽셀이 정확히 일치함을 확인했다. ZIP CRC 검사와 ZIP 내 PNG 240개 개수 검사도 통과했다. 카메라별 모음 이미지는 원본 해상도의 타일과 episode/split 라벨로 구성했다.

[이미지별 source·split·hash manifest](../../outputs/plan_a_first_frames_20261029/manifest.csv) · [검증 결과](../../outputs/plan_a_first_frames_20261029/verification.json) · [추출 스크립트](../../outputs/plan_a_first_frames_20261029/extract.py).

로컬 CPU에서 파일 읽기·해시 계산·이미지 저장만 수행했다. Simulator, CUDA/MPS, 모델 추론, 학습 및 rollout은 실행하지 않았다. 원본 데이터·모델·캐시 수정 및 외부 게시는 없다.
