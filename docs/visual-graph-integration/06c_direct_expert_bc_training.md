# Task06-C — Direct Expert Dataset + Graph–GNN–GRU/BC

Run UUID: `d1fd6173cb3648bfa4c03507f84b7e71`. **독립 tmux 실행 준비/시작 단계. 전체 Dataset 및 학습 완료 아님.**

상태의 기준은 [status.json](../../experiments/d1fd6173cb3648bfa4c03507f84b7e71/status.json)이다.
최종 Dataset, Loss, Test 성능은 아직 산출되지 않았다. 정상 완료 시 Pipeline이
이 문서와 [정적 HTML](../../experiments/d1fd6173cb3648bfa4c03507f84b7e71/graph_gru_training.html)을 생성한다.

수집: normal/noise0/no hole shift, seed30000..30159, 250-step 독립 환경,
120 success/최대160 attempts, 성공96/12/12 episode split, seed42.
원본 Native Action7, 76D robot state, RGB-D 두 camera와 calibration을 기록한다.
기존 Expert·Vision·Raw EDE·Graph·GNN·GRU·BC regression MSE를 유지한다.
Object 결손은 기록하며 GT로 보완하지 않는다.

학습: CPU, AdamW lr3e-4/weight_decay1e-5/betas(0.9,0.999)/eps1e-8,
CosineAnnealingLR(T_max50,eta_min0), max50 epochs/patience10/clip1,
연속 TBPTT32/batch2. Robot normalizer는 train만 fit한다. 기존 Graph feature view는
변경 없이 사용한다. Checkpoint 선택은 validation만, final test는 1회 offline 평가다.

Discord는 사용자의 후속 지시에 따라 비활성화했다. 로그·오류·복구 이력은 로컬에 남긴다.
오류 이벤트만 sandboxed Codex를 최대1회 호출하며 별도 repair tmux를 사용한다.
Closed-loop rollout, CUDA/MPS, 외부 게시, dependency 설치, git commit은 수행하지 않는다.

실행/중지/재개 명령과 tmux 생존 범위는 [README](../../graph/integration_baseline/README.md)를 따른다.
