# 환경 문제와 우회법

RTX 3090 컨테이너(CUDA 12.8, torch 2.8.0+cu128, Python 3.12.3)에서 겪은 문제다.

## Vulkan 렌더러 초기화 실패 (해결)
- **증상**: `gym.make("PegInsertionSide-v1", obs_mode="rgbd")`에서 `RuntimeError: vk::createInstanceUnique: ErrorIncompatibleDriver`.
  `VK_LOADER_DEBUG=error`로 보면 `Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr' for ICD libGLX_nvidia.so.0`.
- **원인**: 컨테이너 기본 ICD(`/etc/vulkan/icd.d/nvidia_icd.json`)가 `libGLX_nvidia.so.0`을 가리키는데, 컨테이너가 `NVIDIA_DRIVER_CAPABILITIES=compute,utility`로 떠 있고 X/GLX가 없다. 드라이버 라이브러리(`libEGL_nvidia.so.0` 등)는 있다.
- **우회**: EGL 라이브러리를 가리키는 ICD 파일 `vulkan/nvidia_icd_egl.json`을 만들고 `VK_ICD_FILENAMES`로 지정한다(`env.sh`). 시스템 파일은 건드리지 않는다.

## mplib 0.1.1 motion planner segfault (우회)
- **증상**: ManiSkill 내장 해법 `mani_skill.examples.motionplanning.panda.solutions.peg_insertion_side.solve`가 `mplib.Planner(...)` 생성자에서 segfault. import 순서와 무관하다.
- **원인(추정)**: mplib 0.1.1 wheel이 numpy 1.x 기준으로 빌드됐는데 환경은 numpy 2.1.2다. 같은 이유로 `transforms3d.euler.mat2euler`도 실패해서 scipy로 대체했다. numpy를 내리면 시스템 torch 환경이 깨져서 하지 않았다.
- **우회**: mplib 없이 같은 waypoint(reach → grasp → lift → pre-insert ×3 refine → insert)를 `pd_ee_pose`(절대 TCP pose, 내장 IK) + 직선/slerp 보간으로 따라가는 스크립트 정책을 썼다(`scripts/collect.py`, `scripts/collect_v1.py`).
- **주의**: `pd_ee_pose` 절대 모드의 회전 action은 axis-angle이 아니라 euler "XYZ"(intrinsic, R = Rx·Ry·Rz)다(`mani_skill/agents/controllers/pd_ee_pose.py:259`).

## SAM2 mask 번짐을 point cloud 단계에서 제거 (미해결, 기본 OFF)
- **증상**: SAM2 box mask가 gripper 손가락 끝과 테이블 일부로 번진다. 덩어리로 뭉쳐 있어 SOR(k=16, 2σ)로 안 지워지고 box centroid 오차가 1.3–3.3 cm로 커진다.
- **시도** (box 오차, 3 ep, 필터 없음 0.97 cm): 최대 cluster만 남기기 1.30 cm, 최대 cluster 20% 이상 모두 남기기 1.14 cm, 거기에 5 cm 이상 떨어진 것만 제거 1.10 cm.
- **원인**: 팔에 가려진 box는 실제로도 떨어진 여러 조각으로 보여서, 기하 정보만으로는 번진 덩어리와 실제 조각을 구분할 수 없다.
- **조치**: `scripts/pointcloud.py --cluster` 옵션으로만 남겼다. mask 단계(re-anchor)에서 다룬다.

## re-ID가 가림 중 외형이 바뀐 peg를 재획득 못 함 (v0 한계)
- **증상**: v0 re-anchor `conf+period30` ep002 peg IoU 0.457(re-anchor 없음 0.864). 가림 해제 후 GDINO 후보에 정답 peg가 있는데도 re-ID cosine이 0.55–0.74로 임계값 0.75 미만이었다.
- **원인**: 가림 중 peg가 들리고 회전해서 외형이 달라졌다. SAM2 image embedding 평균 pooling re-ID는 시점 변화에 약하고, 임계값을 낮추면 오선택이 생긴다.
- **조치**: v1에서 DINOv2 feature + 기하 특징 로지스틱 게이트로 바꿨다(REPORT E3).

## SAMURAI + re-anchor 비호환, 늦은 초기화 (우회)
- **증상 1**: 늦게 초기화된 물체(ep214 peg, frame 110)에서 SAMURAI memory 선택이 `KeyError: 110`(`samurai/sam2/sam2/modeling/sam2_base.py:667`). SAMURAI는 시작 프레임 이전의 모든 비조건 프레임 출력이 있다고 가정한다.
- **증상 2**: 추적 도중 box 프롬프트를 다시 넣으면 그 출력에 `best_iou_score`가 없어 같은 위치에서 `KeyError`.
- **원인**: SAMURAI의 motion-aware memory 선택이 "frame 0부터 연속 추적, 중간 프롬프트 없음"을 전제로 한다.
- **조치**: SAMURAI는 늦은 초기화를 끄고 미검출로 집계하며, re-anchor 없는 설정만 비교한다.

## 그 밖의 메모
- 패키지 이름 `sam2`가 겹치는 tracker(SAMURAI, EfficientTAM, DAM4SAM, SAM2Long)는 venv를 따로 둔다. 모두 `--system-site-packages`로 시스템 torch를 공유하고 `-c constraints.txt`로 설치한다.
- DAM4SAM이 쓰는 vot-toolkit은 Python 3.12에서 import 오류가 나서 필요한 부분만 `scripts/vot_shim`으로 대체했다.
- 공유 노드에서 CPU 스레드를 제한하지 않으면(128) GDINO 전처리가 11배 느려진다. 8로 고정한다.
- SAM2Long offline 실행은 GPU 메모리를 많이 써서 다른 실행과 같이 돌리면 OOM이 난다. 단독으로 돌린다.
