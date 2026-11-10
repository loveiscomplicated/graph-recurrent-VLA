# 중요: /workspace 용량 한도와 데이터 위치 (2026-11-07 09:2x, 백그라운드 세션 989a2365 작성)

- /workspace 볼륨에는 약 100 GiB quota가 있다. 55 GB 데이터를 받은 직후 quota를 넘어서
  쓰기가 실패했다 (`Disk quota exceeded`, `Quota exceeded (os error 122)`).
- 그래서 `data/hf/.../artifacts/trajectory/*/shard_*.tar` 30개를 삭제했다.
  메타데이터(json, config, dataset_metadata.tar.gz, manifest)는 남아 있다.
- `shard_000000.tar` 하나만 RAM 디스크 `/dev/shm/rollouts/shard_000000.tar`로 옮겨 두었다.
  `/dev/shm`은 67 GB이고 pod를 재시작하면 사라진다. episode_000000은 `/dev/shm/rollouts/shard_000000/`에 풀려 있다.
- 원본 tar는 HF에서 다시 받을 수 있다 (revision 94cb85e8009f20b5b47063e8295f6c69040549c1, 전체 약 70초 소요).
  권장 방식: shard 단위로 /dev/shm에 받아 압축된 학습용 포맷으로 변환해 /workspace에 쓰고, tar는 지운다.
- 학습 seed 범위: 100000 + shard*1000 (100000–129999). 평가 seed는 200000 이상을 쓰면 겹치지 않는다.
- 검증된 venv가 `/root/venvs/baseline`에 있다 (container disk, torch 2.8.0+cu128, mani_skill 3.0.1, pinocchio,
  opencv, imageio, einops; lerobot은 미설치). /workspace quota를 쓰지 않는다.
- 이 세션은 tmux의 메인 세션(pid 12379)과 작업이 겹치는 것을 발견하고 여기서 작업을 멈췄다.
  그 밖의 저장소 파일은 건드리지 않았다.
