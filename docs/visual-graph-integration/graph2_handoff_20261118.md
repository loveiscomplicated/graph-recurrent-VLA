# Graph2 인계 및 복원 안내

2026-11-18 기준 Jinwoo 작업의 코드, 학습 자료, 최종 가중치와 결과다. 최종 모델 전체 흐름과 단계별 설명은 docs/visual-graph-integration/final_pipeline_summary.md에 있다.

## 최종 선택과 비교 조건

- 앞단은 RANSAC 검출기 → GT 없는 20개 물리 슬롯 번호 배정 → 공간·시간 Transformer → Box·Hole 평행이동 강체 보정 → Object21 GNN이다.
- 최종 Transformer는 /workspace/gtfree_frontend_retrain/finetuned/tracker/best.pt다. SHA-256은 91f745a4e40377695929832aa365bfa2b87edb75fd91c573fb337e25211e8845다.
- 최종 GNN은 /workspace/gtfree_frontend_retrain/finetuned/gnn/best.pt다. SHA-256은 e03fd2671fad1f142990f908b5ff7addfac0a3d68c8f341dd5e9670e017b390d다.
- 기존 Mamba 5개는 /workspace/recurrent_full_insertion/train/mamba_s0부터 mamba_s4까지의 last.pt다. 최종 앞단에서 고정 20/32 (62.5%), 이동 14/32 (43.75%)를 기록했다.
- 파인튜닝 Mamba 5개는 /workspace/recurrent_final_gtfree/train/mamba_s0부터 mamba_s4까지의 last.pt다. 고정 20/32 (62.5%), 이동 10/32 (31.25%)를 기록했다.
- 파인튜닝 Mamba는 성공률 개선에 실패했다. 기존 가중치도 비교 기준으로 보존했다. Recurrent 누적 평균 필터는 두 조건 모두 껐다.
- 학습 데이터 /workspace/recurrent_final_gtfree/demos의 Expert 궤적 116개는 학습, 21개는 검증에 사용했다. 행동 시점은 합계 16,693프레임이다.
- 초기 소규모 앞단 평가는 Expert 관측이다. Mamba 폐루프가 만든 장시간 Peg 무검출 상황에서 같은 오차를 보장하지 않는다.
- 700882에서는 Peg 8개 슬롯이 184프레임 연속 미검출됐고, 얕은 진입의 50 mm 이상 오차 85프레임을 저장 행동 재생으로 재현했다. 검출 후보·번호 배정·Transformer·GNN 중 기여 분리는 후속 과제다.

## 새 GPU 인스턴스에서 복원

이 인계는 Linux RTX 3090, /venv/main Python 환경, /workspace 경로를 기준으로 검증했다. requirements-runtime.txt는 실제 설치 버전 목록이다. CUDA 렌더링과 CPU PhysX를 사용했다.

1. GitHub graph2를 clone한다. docs/visual-graph-integration/graph2_handoff_20261118.md와 이 안내를 읽는다.
2. 팀 계정으로 private HF 저장소 Graph-Recurrent-VLA/experiment-backups에서 해당 인계 run의 trajectory/graph2_workspace_20261118.tar.gz와 git bundle을 내려받는다. 파일과 revision의 정확한 목록은 같은 run의 manifest와 handoff.json을 참고한다.
3. SHA256SUMS.json의 archive 해시를 확인한다. 빈 /workspace에서 아래 명령을 실행한다. 이 압축에는 graph-recurrent-VLA 폴더와 원래 실험 경로, symlink 대상인 graph2_restore도 들어 있다.
4. 환경을 준비하고 해당 checkout의 검증을 실행한다. 기존 산출물이 있는 학습 경로를 그대로 재사용하면 skip되므로 새 학습은 새 출력 디렉터리를 지정한다.

```bash
mkdir -p /workspace
tar -xzf graph2_workspace_20261118.tar.gz -C /workspace
cd /workspace/graph-recurrent-VLA
source /venv/main/bin/activate
# 새 Python 환경에서는 GPU에 맞는 Torch를 먼저 준비한 뒤 requirements-runtime.txt를 참고한다.
source <(sed -n '/^cd /,/^export WANDB/p' /workspace/recurrent_final_gtfree/run.sh)
export PYTHONPATH=/workspace/recurrent_final_gtfree:$PYTHONPATH
python -c 'from final_frontend import FinalFrontend, FRONT; import torch; f=FinalFrontend(FRONT/"tracker/best.pt", FRONT/"gnn/best.pt", torch.device("cuda:0")); print("Final frontend loaded")'
```

GitHub 안의 graph/handoff_20261118/workspace_scripts에는 원래 실험 디렉터리별 실행 코드를 보관했다. 전체 archive를 복원하면 같은 코드가 /workspace 아래 원래 위치에 있다. 체크포인트의 tracker 코드 hash 검증이 있으므로 복원 후 코드를 임의로 정리하기 전에 원본 실행부터 확인한다.

## 학습과 평가 진입점

- 앞단 학습은 /workspace/gtfree_frontend_retrain/train_gtfree.py이며 입력은 같은 폴더 shard_*/raw.npz다. 기존 학습 루프는 full_insertion_frontend, hybrid_input_gnn_finetune 등의 코드를 재사용한다.
- Mamba 데이터 정렬은 /workspace/recurrent_final_gtfree/prepare_final.py다. 최종 Transformer·보정·GNN 출력에 원래 Expert Action과 Robot15를 프레임 단위로 연결한다.
- Mamba 파인튜닝은 /workspace/recurrent_final_gtfree/train_final.py다. --data, --prior, --output, --seed를 받는다. prior는 각 기존 Mamba의 last.pt다. 검증 손실로 선택한 best 가중치를 last.pt에 저장한다.
- 폐루프 평가는 /workspace/recurrent_final_gtfree/evaluate_final.py다. --prior /workspace/rollout_diagnostics/source --train 모델폴더 --tracker Transformer파일 --gnn GNN파일 --moving-source /workspace/closed_loop_moving_box_mamba --output 새폴더 --start 0 --stop 4 형식이다. 32개 시드를 4개씩 나누어 평가했다.
- 파인튜닝 Mamba 평가 시 --warm-start /workspace/recurrent_full_insertion/train을 추가한다. 원래 학습 source hash 확인에 쓰인다.
- 전체 실행 순서는 /workspace/recurrent_final_gtfree/run.sh다. 기본 출력 위치는 완료 결과가 있으므로 재실행 시 skip된다. 추가 실험은 이 스크립트의 경로와 출력을 새 폴더로 지정한다.
- 700882 저장 행동 재생은 /workspace/recurrent_final_gtfree/replay_recorded_actions.py다.
- 성공 2개와 실패 대표 11개 MP4는 /workspace/recurrent_final_gtfree/case_videos에 있다. 전체 실패 분류와 CSV는 해당 폴더 classification.json, all_failures.csv다.

## 보관 범위와 학습 재개 한계

워크스페이스 archive에는 당시 프로젝트 코드, 원시 후보 및 GT 데이터, 데모·캐시, checkpoint, optimizer가 저장된 resume 파일, ExperimentLogger 기록, 평가 trace와 대표 영상을 포함했다. 최종 GNN은 resume.pt도 있다. Transformer와 Mamba의 일부 학습 루프는 모델 가중치만 저장하므로 그 모델에서 추가 학습할 때 optimizer를 새로 시작한다. 완전히 같은 optimizer 상태를 이어받는 재개가 모든 모델에서 가능한 것은 아니다.

HF 모델 저장소 Graph-Recurrent-VLA/graph-recurrent-models에는 최종 앞단 2개와 Mamba 비교 조건 10개를 알아볼 수 있는 이름으로 별도 게시한다. 원래 실험 run ID와 source hash는 handoff.json에서 확인한다. HF 전체 archive에는 원본 명칭과 디렉터리가 보존된다.

인증 token과 SSH 개인키는 Git과 archive에 넣지 않았다. 팀원이 GitHub/HF 접근을 위해 자기 환경에 인증을 설정한다.

## 업로드 완료 위치

HF run ID는 d64a6e8b7d284915aafb41538539151e다. 모델 12개와 대표 영상 13개, 전체 워크스페이스 archive 및 전체 Git bundle을 게시했다. 두 HF 저장소 모두 비공개다.

- [checkpoint 인계 파일](https://huggingface.co/Graph-Recurrent-VLA/graph-recurrent-models/tree/0da81bd94f33cb44c0cfec6a5a31d376f54c46c0/integration/d64a6e8b7d284915aafb41538539151e)
- [rollouts 인계 파일](https://huggingface.co/datasets/Graph-Recurrent-VLA/experiment-backups/tree/9dd28d1c7bb0d40adc9862cd86b2b0e3048f590d/integration/d64a6e8b7d284915aafb41538539151e)

graph/handoff_20261118/published_artifacts.json에는 정확한 파일 경로와 고정 revision을 기록했다. 같은 폴더 download_handoff.py를 실행하면 인증된 팀 계정으로 해당 버전의 자료를 내려받는다.

huggingface_hub를 설치하고 hf auth login으로 인증한 뒤, python graph/handoff_20261118/download_handoff.py --output /workspace/handoff-download를 실행한다. data 폴더의 graph2_workspace_20261118.tar.gz를 /workspace에 풀어 기존 clone 위에 복원한다.
