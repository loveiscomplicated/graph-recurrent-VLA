# Task10-G 학습·검증 데이터의 첫 Main Camera peg 가시성 전수 감사

분석일: 2026-10-31. 대상: 현재 Task10-G가 실제 사용한 학습 272개·검증 48개, 합계 320개 logical episode. 기존 split, 원본 데이터, 학습·평가 설정은 변경하지 않았다.

## 결론과 요청 기준의 탈락 수

**Peg가 처음부터 충분히 보이는 조건으로 데이터셋을 제한하고, 모든 baseline·ablation에 같은 조건을 적용하는 것은 타당하다.** 이는 “초기 관측에서 peg를 볼 수 있는 집기·삽입”을 평가하는 명시적인 분포 정의다. 초기부터 시야 밖인 물체를 찾아내는 능력까지 검증했다고 일반화하면 안 된다. 새 결과를 만들 때는 validation뿐 아니라 최종 test/evaluation에도 같은 기하학적 규칙을 사전에 고정해야 한다.

사용자가 승인한 해석은 “Main Camera에서 peg 전체 길이·양 끝 경계가 잘리지 않고, 기존 expert의 파지점과 손가락 접촉 폭에 해당하는 camera-facing 영역이 보임”이다. 보이지 않는 뒷면·밑면까지 요구하지 않는다. 이 기준을 GT geometry·보정 행렬·실제 depth로 판정한 결과는 다음과 같다.

| Split | 전체 에피소드 | 기준 미통과 | 미통과 비율 | 통과 |
| --- | ---: | ---: | ---: | ---: |
| 학습 | 272 | **190** | **69.85%** | 82 |
| 검증 | 48 | **25** | **52.08%** | 23 |

**학습 190/272개(69.85%), 검증 25/48개(52.08%)가 탈락한다.** 주 판정으로 남는 데이터는 학습 82개·검증 23개다. DINO/SAM2 검출·마스크·confidence 및 GNN 입력은 판정에 사용하지 않았다.

![split별 통과·탈락 비율](../../outputs/task10g/initial_visibility_audit/results_v1/split_counts.png)

[320개 전체 판정 갤러리](../../outputs/task10g/initial_visibility_audit/results_v1/gallery.html) · [최종 에피소드별 CSV](../../outputs/task10g/initial_visibility_audit/results_v1/episode_audit.csv) · [통과 ID 목록](../../outputs/task10g/initial_visibility_audit/results_v1/eligibility_ids.json)

## 왜 100% 가시 조건이 합리적인가

이 데이터의 expert는 물체 GT로 파지점을 정하고, close 이후에는 lift/align/insert 동안 닫힘 명령을 유지한다. 고쳐 잡는 시범은 없다. 따라서 현재 시연으로 학습한 정책에게 시야 밖 물체 탐색과 재파지를 당연히 기대할 근거는 부족하다. 다만 정책의 action space가 재파지를 물리적으로 금지한다는 뜻은 아니다.

설치된 `PegInsertionSide-v1`은 peg 반길이 L∈[0.085,0.125]m, peg 전체 길이 2L, box의 삽입축 반깊이 L이다. 정렬된 삽입에서 성공의 x 조건은 peg head가 hole 중심 기준 x≥−0.015m인 것이다. 최소 삽입 길이 비율은 `(L−0.015)/(2L)`이므로 **약 41.2~44.0% 이상**이라는 사용자의 설명과 일치한다. 이보다 더 들어가도 성공할 수 있으며, y/z 조건도 별도로 만족해야 한다.

실제 expert의 파지점은 peg 중심에서 white 쪽으로 `max(0.05, L/2+0.01)`m 이동한 지점이다. 이번 데이터에서는 전체 길이 기준 white 끝부터 약 **19.1~21.0%** 지점이었다. 단순히 “영상에 peg가 절반쯤 보임”만으로 이 지점이 보인다고 보장할 수 없다. 길이·끝·파지 영역을 직접 검사하는 것이 목적에 맞다.

100%는 3D 표면 전체의 관측을 의미하지 않는다. 하나의 카메라에서 보일 수 없는 뒤쪽 면까지 요구하는 조건은 아니다. 여기서는 **전체 투영 범위가 프레임 안에 있고, 양 끝·파지 영역의 camera-facing 표면이 가려지지 않음**을 뜻한다. 아래의 “모든 래스터 픽셀 GT ID까지 일치” 조건은 이보다 더 보수적인 별도 정의로 보고한다.

## 모집단과 원본 정합성

- 현재 Task10-G `full/host_data/metadata.json`의 logical episode 집합과 F1 release manifest를 대조했다. 학습 272·검증 48, 중복 없는 총 320개가 정확히 일치한다. Final 12개 및 별도 paired intervention frame은 모집단에 포함하지 않았다.
- 기본 full-task 원본 108개와 corrective 원본 212개를 모두 포함했다. 로컬에 있던 108개만 세지 않았다. 원격 서버에서 모든 원본 HDF5·GT sidecar SHA를 실제 사용 manifest와 대조한 뒤 첫 프레임만 추출했다.
- 모든 source에서 `timestep[0]=0`, `timestamp[0]=0`, HDF5 TCP pose와 GT sidecar의 raw TCP pose 일치를 확인했다. Corrective branch의 복사된 prefix도 해당 logical episode의 첫 프레임으로 그대로 집계했다.
- 학습은 **96개 고유 parent/초기 화면**, 검증은 **16개 고유 parent/초기 화면**을 공유한다. 따라서 320개가 독립된 320개 초기 장면이라는 뜻은 아니다. 요청한 episode 단위 분모를 유지했다. Train/validation parent 교집합은 비어 있다.
- 고유 초기 화면 기준으로는 학습 96개 중 63개, 검증 16개 중 9개가 주 판정에서 탈락한다. 통과하는 고유 초기 화면은 각각 33개·7개다.

| 원본 유형 | Split | 에피소드 | 미통과 | 비율 |
| --- | --- | ---: | ---: | ---: |
| Task10F_corrective | 학습 | 176 | 127 | 72.16% |
| Task10F_corrective | 검증 | 36 | 19 | 52.78% |
| full_task | 학습 | 96 | 63 | 65.62% |
| full_task | 검증 | 12 | 6 | 50.00% |

## 판정 방법

프로토콜은 전체 집계 전에 [protocol.json](../../outputs/task10g/initial_visibility_audit/protocol.json)에 저장했다. 모델 선택·성공률·검출 결과로 가시성 기준을 정하지 않았다.

1. **GT geometry / 실제 카메라:** 각 source의 `raw_peg_pose`, `raw_peg_half`로 peg의 실제 box geometry를 재구성한다. 해당 source 첫 프레임의 Main=`base_camera` RGB·depth·intrinsic·extrinsic을 사용한다. 실제 입력은 128×128이다.
2. **프레임 안의 전체 범위:** OBB 8개 corner가 near plane 0.01m 앞에 있고, 각 투영이 연속 이미지 범위 [0,128]×[0,128] 안에 있어야 한다. 양 끝 cross-section 모두를 검사하므로 몸통 일부나 white 끝이 잘리면 탈락한다. 수치 오차 여유는 0.0001픽셀뿐이다. 가장 경계에 가까운 통과 사례도 0.416픽셀 이상 안쪽이어서 이 epsilon이 결과를 결정하지 않았다.
3. **실제 expert 파지 영역:** 설치된 `compute_grasp_info_by_obb`, `Panda.build_grasp_pose`, `ScriptedExpert.reset`의 offset과 실제 초기 TCP closing hint를 사용한다. URDF rubber tip의 local-x 전체 폭 **17.5mm**로 파지점 주변 ±8.75mm band를 정의한다. 이 파지점은 close 시점이 있는 기본 원본 108개에서 검증했고 실제 close TCP와의 최대 거리는 **2.564mm**, 중앙값은 **0.810mm**였다. Expert의 4mm 수렴 조건 안이다.
4. **양 끝·파지 영역의 가시성:** 각 끝에서 pad 폭 17.5mm에 해당하는 longitudinal strip과 파지 band의 camera-facing 표면을 검사한다. 영역별로 최소 1개 이상의 예상 래스터 픽셀이 있어야 한다. 자기 가림은 ray/box entry surface로 처리하며 뒷면은 요구하지 않는다.
5. **실제 depth와 대조:** 기존 `convex_depth`를 재사용해 GT front-surface depth를 계산한다. 모든 필수 영역 픽셀의 관측 depth가 유효하고 GT와 **1.5mm 이내**로 맞아야 한다. 저장 depth는 mm 정수 양자화이고, 실제 완전 가시 표면에서도 1mm를 조금 넘는 반올림/보정 차이가 있을 수 있다. 이것은 가림을 임의로 허용하는 비율이 아니라 센서 수치 정합 tolerance다. 카메라에 들어온 일부만 depth가 맞더라도 전체 끝의 frustum 검사를 통과하지 못하면 탈락한다.

마스크 픽셀 수 자체는 가시성의 충분조건이 아니다. 예를 들어 [episode2000000](../../outputs/task10g/initial_visibility_audit/results_v1/images/ep2000000.png)은 화면 안의 peg 표면이 1,146픽셀이지만 white 끝이 잘려 탈락한다. 반면 [episode2000003](../../outputs/task10g/initial_visibility_audit/results_v1/images/ep2000003.png)은 544픽셀이어도 양 끝과 파지 영역이 모두 보여 통과한다.

## 무엇 때문에 탈락했는가

주 판정에서 탈락한 **190개 학습·25개 검증 에피소드 전부 white 끝이 프레임 밖**이었다. 그중 동시에 파지 band도 프레임 밖인 경우는 **학습 166개, 검증 22개**다. Red 끝까지 프레임 밖인 경우는 각각 22개·18개다. 이 사유들은 겹치므로 합산하지 않는다.

카메라 안의 예상 peg 표면보다 1.5mm 이상 가까운 물체 depth가 관측되는 픽셀은 전체 모집단에서 0개였다. 즉 이번 첫 프레임 데이터에서 주된 문제는 다른 물체에 의한 가림보다 **시야 밖 잘림**이다. 카메라에 전혀 걸리지 않는 예상 peg geometry도 학습에서 2개 logical episode 확인됐다.

## 실제 GT segmentation을 이용한 전수 교차 검증

검출기를 돌리지 않고, 원래 Linux CPU PhysX + CUDA SAPIEN renderer에서 **112개 고유 parent 초기화 모두를 reset**했다. 각각의 실제 RGB, depth, intrinsic, extrinsic, peg pose·half-size, TCP pose가 저장 원본과 **전부 exact 일치**했다. 같은 초기 화면을 공유하는 logical episode에 연결하여 **320개 전체를 커버**했다. 어떤 action도 실행하지 않았다.

현재 촬영된 texture에서 peg actor의 GT segmentation ID를 읽어, analytical geometry·관측 depth 판정과 독립 대조했다. 또한 별도 구현의 OBB slab ray intersection 및 기록된 **GL camera matrix**로 frustum/expected mask/주 판정을 320개 모두 재계산했다. Frustum, ray mask, 주 판정은 전부 일치했고 독립 depth 계산의 최대 차이는 **0.003155mm**였다.

## “100%”를 픽셀 단위까지 엄격히 요구하면

| 별도 정의 | 학습 미통과 | 검증 미통과 |
| --- | ---: | ---: |
| 전체 범위 in-frame + 양 끝·파지 밴드 depth 가시(주 판정) | 190/272 (69.85%) | 25/48 (52.08%) |
| 추가로 양 끝·파지 밴드의 GT ID도 모든 픽셀에서 peg 요구 | 192/272 (70.59%) | 25/48 (52.08%) |
| 추가로 전체 camera-facing 투영 픽셀의 depth 일치 요구 | 192/272 (70.59%) | 26/48 (54.17%) |
| 추가로 전체 camera-facing 투영 픽셀의 GT ID까지 peg 요구 | 194/272 (71.32%) | 26/48 (54.17%) |

**기하학적 가시성과 모든 GT 래스터 픽셀의 완전 일치는 구분해야 한다.** 실제 renderer 교차 검증에서는 3개 고유 초기화(5개 logical episode)에만 projected silhouette 경계의 1픽셀 차이가 있었다. 모두 GT ID가 table이었고 내부 표면 가림은 아니었다.

- seed30124의 2개 train logical episode는 white 끝/테이블 접촉 경계 1픽셀이다. 관측 depth와 GT 표면 차이는 약 **0.363mm**로 mm 양자화 이내다. 주 판정은 통과하지만 “끝 strip의 GT ID도 한 픽셀도 다르면 안 됨” 조건은 탈락시킨다.
- seed30032의 2개 train logical episode와 seed30050의 1개 validation episode는 필수 끝·파지 band 밖 silhouette 경계 1픽셀이다. 주 판정은 통과하고 전체 GT 픽셀 완전일치 조건만 탈락한다.

따라서 연구의 “peg 전체가 보임”은 **전체 geometry가 프레임 안에 있고 필수 영역이 관측되는 조건**으로 정의하는 것을 권한다. 픽셀 단위 GT ID 완전일치까지 원한다면 그 보수적 조건의 숫자는 **학습 194/272개(71.32%), 검증 26/48개(54.17%) 탈락**, 통과 78개·22개다. 이는 실질적인 초기 시야 차이뿐 아니라 위 래스터 경계 차이까지 포함한다. 모든 변형의 label과 ID 목록을 남겼으며 결과에 맞춰 primary protocol을 바꾸지 않았다.

## 실험 설계에 적용할 때

조건을 모든 비교에 동일하게 적용하면 baseline/ablation 비교의 일관성은 유지된다. 다만 다음 범위를 명확히 해야 한다.

- “처음부터 peg를 관측할 수 있는 조건의 제어 성능”으로 결과를 기술한다. 원래 무작위 reset 전체, 초기 탐색·재관측·재파지 능력까지 검증한 결과로 확대하지 않는다.
- GT geometry·카메라 관측만으로 고정한 동일 eligibility list를 모든 비교가 공유한다. 특정 모델의 DINO/SAM2 성공 여부나 rollout 성공 여부로 제외하지 않는다.
- Train/validation/test parent 분리를 유지한다. 같은 parent의 copied-prefix branch를 다른 split으로 옮겨 수를 맞추지 않는다. Final/test에 새 규칙을 적용하는 평가는 기존 Final 결과와 구분한 새 조건으로 기록한다.
- 필터 후 학습량과 고유 parent 수가 크게 줄어든다. 데이터 수·학습 예산과 관측 분포의 변화도 함께 기록해야 한다. 이번 감사는 **어떤 원본·split·학습 설정도 수정하지 않았다.** Eligibility 파일은 분석 결과 목록이지 기존 release manifest의 교체가 아니다.

## 검증과 산출물

합성 기하 테스트 6개(완전 가시, crop, 파지부 가림, 필수 부위 밖 가림, invalid depth, 카메라 뒤)를 통과했다. 입력 320개 SHA/정합, 실제 reset 112개 exact 재현, 독립 320개 ray/frustum 재계산을 모두 확인했다. 기존 shared simulator/logger/model 코드를 바꾸지 않았으므로 관련 없는 학습·회귀 실험을 추가하지 않았다.

정적 분석은 macOS CPU NumPy/SciPy/trimesh와 SAPIEN Pose 연산으로 실행했다. 실제 reset/GT texture 검증은 원래 Linux 서버의 CPU PhysX + CUDA SAPIEN renderer로 실행했다. macOS simulator 및 CUDA/MPS policy/vision inference는 실행하지 않았다. 공통 `ExperimentLogger`를 W&B disabled로 재사용했다. 외부 artifact 게시·새 학습·새 정책 rollout은 없었다.

- [최종 episode CSV](../../outputs/task10g/initial_visibility_audit/results_v1/episode_audit.csv), [전체 세부 판정 JSON](../../outputs/task10g/initial_visibility_audit/results_v1/episodes.json), [집계 JSON](../../outputs/task10g/initial_visibility_audit/results_v1/summary.json), [eligibility ID 목록](../../outputs/task10g/initial_visibility_audit/results_v1/eligibility_ids.json)
- [320개 전체 HTML 갤러리](../../outputs/task10g/initial_visibility_audit/results_v1/gallery.html), [대표 사례 모음](../../outputs/task10g/initial_visibility_audit/results_v1/examples.png), [비율 그림 PNG](../../outputs/task10g/initial_visibility_audit/results_v1/split_counts.png), [SVG](../../outputs/task10g/initial_visibility_audit/results_v1/split_counts.svg)
- [독립 전수 검증](../../outputs/task10g/initial_visibility_audit/results_v1/independent_verification.json), [112개 실제 renderer 검증](../../outputs/task10g/initial_visibility_audit/task10g_all_initial_render_validation_v1/verification.json), [실제 GT segmentation](../../outputs/task10g/initial_visibility_audit/task10g_all_initial_render_validation_v1/rendered_segmentation.npz)
- [320개 실제 첫 RGB-D·GT geometry](../../outputs/task10g/initial_visibility_audit/task10g_initial_visibility_inputs_v1/first_observations.npz), [원본/GT SHA 검증](../../outputs/task10g/initial_visibility_audit/task10g_initial_visibility_inputs_v1/extraction_verification.json), [픽셀별 기하 근거](../../outputs/task10g/initial_visibility_audit/results_v1/pixel_evidence.npz)
- 코드: [입력 추출](../../outputs/task10g/initial_visibility_audit/extract.py), [가시성 계산](../../outputs/task10g/initial_visibility_audit/geometry.py), [전수 집계](../../outputs/task10g/initial_visibility_audit/analyze.py), [독립 검증](../../outputs/task10g/initial_visibility_audit/verify.py), [실제 renderer 검증](../../outputs/task10g/initial_visibility_audit/validate_rendered_gt.py), [합성 테스트](../../outputs/task10g/initial_visibility_audit/test_geometry.py)
