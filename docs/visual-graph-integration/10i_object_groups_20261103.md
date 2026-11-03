# Task 10-I — 마스크의 Peg·Box 구분을 패킷 소비 단계까지 유지

2026-11-03. 물체별 선택 결과의 구분을 보존하고, 패킷 반환 이후의 재분리를
행 범위 조회로 바꿨다. 사용자 요청대로 처리 시간은 재측정하지 않았다.

## 구현

역투영에서 이미 mask 축에 따라 `object_id=0(Peg)/1(Box)`가 부여되며,
Shape-First 결과도 `selected[0]`, `selected[1]`로 구분되어 있다. 이 의미와
선택 규칙은 그대로 사용한다. 새 분류/인식이나 별도 특징을 추가하지 않는다.

기존 flat packet의 Peg→Box 행 순서는 유지하면서, 물체별 선택점 수로 다음
선택 정보도 패킷 생성 시 함께 기록한다:

```python
packet['object_slices'] = ((0, peg_count), (peg_count, total_count))
```

각 쌍은 시작 포함·끝 제외의 행 범위다. 빈 물체는 시작과 끝이 같다.
이 값은 이미 존재하는 물체별 배열의 길이로 구성한 Python 정수 metadata이므로
추가 점 배열 복사나 CUDA readback이 없다. 기존 `object_id`와 provenance도
호환성 및 검증을 위해 유지한다.

- `graph_perception._selection_packet`과 explicit CPU 선택 경로가 범위를 기록한다.
- `vision_packet._complete_selected`가 camera packet의 범위를 EE packet에 보존한다.
  GPU 선택점의 mask 정합 검사도 같은 행 범위를 사용한다.
- Tensor `selected_packet`도 동일한 범위를 기록하고, 명시적 NumPy readback과
  receiver 복사에서도 이 metadata가 보존된다.
- `fusion.observations_from_packet`은 저장된 범위를 slice로 사용한다. ID 조건으로
  다시 배열을 분리하지 않는다. 법선을 회전할 특징 배열은 별도로 복사해 원본
  packet을 보호한다. 좌표 변환식/순서와 World 반환은 그대로다.
- 선택적 CUDA ingress 소비자의 물체별 view 검사도 범위를 사용한다.
- Receiver는 범위가 모든 행을 Peg→Box 순서로 빠짐없이 덮고, 실제 ID와
  일치하는지 검증한다. 잘못된 범위를 조용히 받아들이지 않는다.

Schema1/2의 기존 필드는 바꾸지 않았고, 새 범위 metadata는 선택적 확장이다.
범위가 없는 이전 패킷은 기존 ID 조건 경로를 유지하며, 물체 행이 섞여 있어도
객체별 원래 source 순서를 보존한다. 범위 검증의 ID 비교는 정합성 검사이며
물체를 다시 인식하거나 점 행을 boolean gather로 재구성하는 과정이 아니다.

## 정확성 검증

로컬 macOS에서 다음 suite를 실행했다:

```text
visual.runtime.test.test_object_groups
visual.runtime.test.test_graph_perception
visual.runtime.test.test_vision_features
visual.runtime.test.test_variable_rate_graph_contract
```

24개 중 **20개 통과, CUDA 전용4개 skip**. 동일 suite를 VastAI **58752018**의
RTX3090 환경에서 실행하여 **24개 모두 통과**했다. Python3.12.15,
Torch2.14.1+cu130, NumPy2.5.4의 기존 환경을 사용했다.

새 검증은 다음을 포함한다:

- 두 물체 모두 존재, Peg만 존재, Box만 존재, 둘 다 비어 있는 경우.
- 겹친 mask와 signed int16 무효 depth, 비항등 camera/EE 회전·이동.
- NumPy/CUDA packet의 범위·선택 source 일치와 CUDA 출력 device 유지.
- 범위 기반 관측과 범위 없는 기존 관측의 모든 필드 exact 일치.
- 입력 packet의 좌표·특징/유효성 보존과 receiver의 metadata 유지.
- 범위 누락/중복/초과/잘못된 타입 및 실제 object ID와 불일치하는 범위 거부.
- 범위 없는 interleaved legacy packet의 source 순서 유지.

기존 suite의 준비된 mask 기반 CPU/CUDA 최종 Graph 비교도 통과했다.
이 검증은 로봇 필터를 호출하지 않도록 기존 prepared 입력을 주입한다.
선택적 CUDA ingress 소비자의 수정은 syntax와 공통 CUDA selector/packet 경로로
확인했으며, 그 backend 전체의 로봇 필터·gate를 포함한 통합 실행은 이번 작업에서
다시 실행하지 않았다.

## 실제512 입력의 수정 전후 비교

이전 단계와 같은 `10i_s1_0844_t080`, hand-camera512×512 RGB-D/추론 mask를
사용했다. Peg mask61,056 pixels, Box0 pixels, 선택점1,105개다. 기록된 실제
camera calibration과 TCP pose를 사용했다.

기존 packet CPU 측정 당시의 frozen source에서 기존 public `make_packet`과
`observations_from_packet`을 실행해 기준 출력을 저장했다. 별도의 수정 source에서
동일 입력으로 다시 실행했다. 결과:

- 새 `object_slices`를 제외한 **기존 packet의 모든 필드/dtype/shape/NaN 위치 exact 일치**.
- 반환된 Peg·Box **두 관측 객체의 모든 필드 exact 일치**.
- 새 범위는 `((0, 1105), (1105, 1105))`이며 빈 Box 범위를 명시한다.
- Runtime 반환 timing은 비교/기록하지 않았다. 반복 latency 측정도 하지 않았다.

실제 입력의 Box는 비어 있으므로 두 물체가 모두 관측되는 경우는 위 합성
CPU/CUDA 검증으로 보완했다. Simulator/rollout·학습/평가·성공률은 실행하지 않았다.
로봇 필터 구현 및 기존 좌표계 경로는 이번 변경 대상에 포함하지 않았다.

## 재현성과 산출물

공통 `ExperimentLogger`를 사용했고 W&B disabled, publishing 없음.

- 기준 capture UUID: `c557dfb1407a490e9cfceb201c77b35c`.
- 수정 capture/비교 UUID: **`6279312e1efd43e9b5964b6b6429fd4c`**.
- 원격 root: `/workspace/task10i_object_groups_20261103_v1`.

기존 frozen source를 새 root로 복사한 뒤 변경 경로를 overlay해 검증했다.
현재 checkout의 별도 CUDA ingress/로봇 필터 작업을 되돌리거나 덮어쓰지 않았다.
커밋에는 공개 CUDA packet 생성·검증 adapter를 필요한 의존 코드로 함께 포함한다.
선택적 CUDA ingress backend의 adapter 수정은 해당 backend 작업과 함께 작업
트리에 남기며, backend 전체나 로봇 필터 변경은 이 커밋에 포함하지 않는다.
Source는 Git checkout 자체가 아니므로 로컬 checkout identity와 전체 manifest를
함께 보존했다. Overlay9개 파일 및 성공 verifier가 현재 파일과 hash 일치함을
확인했다. 기준 source manifest도 별도로 보존했다. Package/driver/service 변경이나
다른 작업 중지는 하지 않았다.

- [실제512 비교 결과](../../outputs/task10i/object_groups_58752018_20261103_v1/remote/verified/verification.json)
- [CUDA 테스트24개 로그](../../outputs/task10i/object_groups_58752018_20261103_v1/remote/tests.log)
- [수정 전 capture 결과](../../outputs/task10i/object_groups_58752018_20261103_v1/remote/reference/verification.json)
- [수정 capture 로그](../../outputs/task10i/object_groups_58752018_20261103_v1/remote/verified.log)
- [Source/input manifest](../../outputs/task10i/object_groups_58752018_20261103_v1/remote/source_manifest.json)
- [기준 source manifest](../../outputs/task10i/object_groups_58752018_20261103_v1/remote/reference_source_manifest.json)
- [로컬 identity](../../outputs/task10i/object_groups_58752018_20261103_v1/checkout_identity.json)
- [다운로드·hash 검증 receipt](../../outputs/task10i/object_groups_58752018_20261103_v1/download_receipt.json)
- [Verifier·실행 방법](../../graph/experiments/object_groups_58752018_20261103_v1/README.md)
- [Packet API 설명](../../visual/runtime/README.md#pegbox-groups-in-selected-packets)

Archive SHA256:
`fcd4513bbb2af918aab2dba60c6d25e7ea4ab078230099e4d7892452aabad7aa`.
712개 source/input manifest 항목과 다운로드한 archive를 검증했다.
