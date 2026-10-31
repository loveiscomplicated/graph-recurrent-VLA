# Task10-G 학습 전문가 영상 MP4 변환

로컬에 있는 Task10-G 학습 split 원본 **96개 에피소드**를 H.264 MP4로 변환했다. [영상 폴더](../../outputs/task10g/training_video_export/videos), [에피소드 목록 CSV](../../outputs/task10g/training_video_export/index.csv), [원본 경로·SHA 포함 JSON](../../outputs/task10g/training_video_export/index.json), [변환 검증](../../outputs/task10g/training_video_export/verification.json).

바로 볼 수 있는 예시:

- [episode 2000000 / seed30002](../../outputs/task10g/training_video_export/videos/ep2000000_seed30002_train.mp4)
- [episode 2000001 / seed30047](../../outputs/task10g/training_video_export/videos/ep2000001_seed30047_train.mp4)
- [episode 2000002 / seed30038](../../outputs/task10g/training_video_export/videos/ep2000002_seed30038_train.mp4)

이 영상은 학습에 사용한 **전문가 시연의 저장 RGB**이다. 학생 정책의 development/final rollout과 구분한다. 각 `episode.h5`의 base_camera를 왼쪽, hand_camera를 오른쪽에 배치했다. 원본은 camera별 128×128이며 최근접 확대만 적용했다. 상단 camera 이름, 하단 episode/seed/phase/frame/time을 표시하고 마스크나 추정 결과는 추가하지 않았다.

원본 timestamp 간격 0.05초를 확인하고 20fps로 인코딩했다. 모든 T+1 관측을 포함하므로 action 수보다 프레임 수가 1 많다. 총 **12,069프레임**을 출력하고 모든 출력 프레임을 디코딩하여 크기·개수·fps를 검증했다. 각 입력 HDF5 SHA가 Task10-G `dataset_identity.json`의 원본 SHA와 일치함을 확인했고 receipt의 split=train만 선택했다.

Task10-G 전체 학습은 272개 에피소드다. 이번 변환은 로컬에 있는 기본 원본 train 96개이며, 별도 추가 수집 학습 원본 176개는 원격 서버에 있어 이 변환에 포함하지 않았다. 로컬에 있는 validation 12개와 이 Task10-G 학습/검증 목록 밖의 다른 에피소드도 포함하지 않았다. 원본 전체를 대표하도록 선정한 표본이라는 주장은 하지 않는다.

macOS 로컬에서 HDF5 읽기와 FFmpeg CPU 인코딩만 실행했다. Simulator/CUDA/MPS inference·학습·새 rollout은 실행하지 않았고 공유 코드/원본 데이터도 변경하지 않았다. 이는 기존 artifact의 형식 변환이며 별도의 training/evaluation run을 생성하지 않았다. 외부 게시 없이 로컬에 저장했다.

[변환 코드](../../outputs/task10g/training_video_export/export.py)
