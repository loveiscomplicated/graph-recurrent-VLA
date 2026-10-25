# Task05-B — Extended Delaunay + Edge-wise Validation

상태: 구현·고정 offline 비교 완료. **새 정책 후보는 EDE 하나이며 DE는 분석 대조군**이다. 최종 정책은 확정하지 않았다.

실행 UUID: `9f36f7c89dbd4097a8be9358138d1537`. [정적 HTML](../../experiments/9f36f7c89dbd4097a8be9358138d1537/extended_delaunay_edgewise.html) · [전체 JSON](../../experiments/9f36f7c89dbd4097a8be9358138d1537/extended_delaunay_edgewise.json) · [전체 NPZ](../../experiments/9f36f7c89dbd4097a8be9358138d1537/extended_delaunay_edgewise.npz) · [검증](../../experiments/9f36f7c89dbd4097a8be9358138d1537/verification.json) · [계획](05b_extended_delaunay_edgewise_plan.md)

## 1. 결과와 해석

TMD→DE는 993개 undirected Edge를 추가 보존하고 고립 Point 988개를 연결했다. DE→EDE fallback은 239개 Edge를 추가하고 317개 Point를 추가 연결했다. fallback 후보는 241개, 통과 239개, 거부 2개다. 후보가 있지만 모두 실패한 Point는 4개이며 EDE 고립은 302개다.

| 방법 | U (undirected) | 고립 Point | Graph component | 관측 규칙 실패 Edge | Core 평균 ms |
|---|---:|---:|---:|---:|---:|
| K8 | 49191 | 270 | 408 | 7722 | 18.085 |
| TMD | 23814 | 1607 | 1668 | 0 | 281.137 |
| DE | 24807 | 619 | 681 | 0 | 111.558 |
| EDE | 25046 | 302 | 442 | 0 | 116.208 |

DE/EDE 유지 Edge는 저장 관측 규칙을 모두 통과했다. 이 결과는 물리적 연결 정확성의 보장이 아니다. Frame0 Box의 약 8.431m plateau에서도 연결이 다시 생성됐다. Mask와 Depth가 함께 잘못 관측된 연속 영역은 이번 규칙만으로 식별되지 않을 수 있다. 새 depth/length cutoff, known-negative ROI, 보완 k-NN, outlier 필터는 적용하지 않았다. 최종 Graph Edge 정책 채택은 보류한다.

고정 8 frame × keep ratio 0.5/1.0 × phase 0, 16개 조건·32 object slice·11095 Point-row 관측이다. 독립 episode 수나 episode 고유 Point 수가 아니다. Sampling/PCA5/Feature/EE 변환 재실행 0회. CPU NumPy/SciPy/Qhull offline 실행만 수행했으며 simulator/GPU/PCL/GNN/GRU/BC/추가 episode는 실행하지 않았다.

## 2. 입력·구현 계약

Task03-K `9185355a32b8433abc70f1738f983377`, Task04 `9f90293b227e4dbfaaa6290afad9cc3b`, Task05 `91ecd0413a87455cabc5e813cddaea8d`를 사용했다. [입력 gate](../../experiments/9f36f7c89dbd4097a8be9358138d1537/input_gate.json)는 모든 조건에서 PASS다. Source/hash/index와 원본 배열 byte·NaN 위치, 저장 T0/TMD·component membership를 먼저 검사했다. Singleton 270 + two-point 72 + collinear 249 = 기존 T0 고립 591 Point-row를 source별로 재확인했다. 321은 fallback 후보 제공 가능 Point 수이며 복구 보장이 아니다.

기존 Task05 `supercover`, `prepare_validation`, `validate_cover`, Qhull 옵션 `Qbb Qc Qz Q12`를 import했다. Collinear는 Python integer exact cross와 A2 projection/source/row 정렬을 사용한다. 2점은 한 후보, 전체 일직선은 이웃 chain, 비일직선은 기존 Delaunay 변이다. QJ/jitter/새 Point/normal·length·absolute depth veto는 없다. QhullError/omitted vertex는 별도 상태이며 자동 fallback하지 않는다.

DE는 저장 T0 변을 correctness 후보로 사용했다. EDE 비퇴화 후보는 이를 exact 재현했다. `TMD ⊆ DE ⊆ EDE`는 **undirected row pair와 source pair 모두 16/16 PASS**다. 별도 benchmark에서 K8/TMD/DE/EDE를 독립 재실행해 저장 correctness 결과와 exact 대조했다. Receiver는 DE/EDE 명시 정책·canonical accepted pair·출처와 whole-component 기하 계약·정확한 양방향 Edge 집합을 검사한다. Dense RGB-D 검사는 helper 책임이다.

공통 ExperimentLogger `team='visual', job_type='evaluate'`, `WANDB_MODE=disabled`를 사용했다. 성공한 Task03-K/04 평가 실행과 실제 Logger API를 확인했다. 원본 artifact sidecar를 변경하지 않았다. [소스 해시](../../experiments/9f36f7c89dbd4097a8be9358138d1537/source_hashes.json)와 [구현 diff](../../experiments/9f36f7c89dbd4097a8be9358138d1537/implementation.diff), source snapshot·config를 같은 UUID에 저장했다. 게시·sync·Git commit은 하지 않았다.

## 3. 모든 frame / ratio / object

다음 개수는 unique undirected Edge와 singleton을 포함한 Graph component 기준이다. 빈 Box도 누락하지 않았다. 세부 degree·length·UV·normal 측정 분모·depth 구간·full observed support component·source·EE/Camera XYZ는 JSON/NPZ와 condition별 records에 보존했다.

| Frame / ratio | Object | N | U K8/TMD/DE/EDE | 고립 K8/TMD/DE/EDE | CC K8/TMD/DE/EDE |
|---|---|---:|---|---|---|
| f0000_r0.5 | Peg | 499 | 2298/1152/1196/1196 | 5/49/5/5 | 6/50/6/6 |
| f0000_r0.5 | Box | 451 | 1945/831/868/885 | 14/79/39/18 | 27/86/50/33 |
| f0000_r1 | Peg | 896 | 4048/1951/2093/2095 | 4/155/13/9 | 7/156/14/12 |
| f0000_r1 | Box | 798 | 3521/1528/1599/1620 | 12/113/46/18 | 30/125/57/36 |
| f0018_r0.5 | Peg | 453 | 2056/990/1041/1041 | 6/57/6/6 | 7/58/7/7 |
| f0018_r0.5 | Box | 0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| f0018_r1 | Peg | 807 | 3652/1688/1830/1830 | 8/154/12/12 | 9/155/13/13 |
| f0018_r1 | Box | 0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| f0027_r0.5 | Peg | 416 | 1942/938/980/980 | 6/48/6/6 | 7/49/7/7 |
| f0027_r0.5 | Box | 123 | 502/264/264/273 | 1/15/15/1 | 7/16/16/7 |
| f0027_r1 | Peg | 733 | 3342/1619/1725/1725 | 8/115/9/9 | 9/116/10/10 |
| f0027_r1 | Box | 150 | 634/327/328/337 | 1/16/15/1 | 7/17/16/7 |
| f0045_r0.5 | Peg | 348 | 1606/816/847/847 | 4/35/4/4 | 5/36/5/5 |
| f0045_r0.5 | Box | 166 | 639/320/324/334 | 20/36/32/20 | 23/37/33/23 |
| f0045_r1 | Peg | 593 | 2717/1364/1434/1435 | 5/77/7/5 | 7/78/8/7 |
| f0045_r1 | Box | 237 | 942/464/472/483 | 19/42/34/20 | 23/43/35/24 |
| f0056_r0.5 | Peg | 371 | 1610/831/858/858 | 25/52/25/25 | 26/53/26/26 |
| f0056_r0.5 | Box | 0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| f0056_r1 | Peg | 636 | 2759/1380/1451/1451 | 38/110/39/39 | 39/111/40/40 |
| f0056_r1 | Box | 0 | 0/0/0/0 | 0/0/0/0 | 0/0/0/0 |
| f0066_r0.5 | Peg | 305 | 1319/690/703/710 | 13/40/27/14 | 20/41/28/21 |
| f0066_r0.5 | Box | 119 | 514/284/285/285 | 6/7/6/6 | 7/8/7/7 |
| f0066_r1 | Peg | 505 | 2219/1123/1151/1173 | 12/71/43/12 | 22/72/44/22 |
| f0066_r1 | Box | 125 | 539/299/300/300 | 6/7/6/6 | 7/8/7/7 |
| f0080_r0.5 | Peg | 249 | 1110/568/574/587 | 8/30/24/9 | 12/32/26/13 |
| f0080_r0.5 | Box | 228 | 954/433/444/464 | 6/40/30/6 | 14/45/34/14 |
| f0080_r1 | Peg | 398 | 1844/888/903/933 | 6/56/41/7 | 12/58/43/13 |
| f0080_r1 | Box | 349 | 1508/657/683/703 | 9/61/35/11 | 17/65/39/19 |
| f0120_r0.5 | Peg | 207 | 923/478/486/491 | 7/23/16/8 | 12/25/18/13 |
| f0120_r0.5 | Box | 239 | 964/479/488/498 | 8/32/23/8 | 16/35/26/16 |
| f0120_r1 | Peg | 316 | 1470/727/737/752 | 6/34/26/7 | 12/37/28/13 |
| f0120_r1 | Box | 378 | 1614/725/743/760 | 7/53/35/10 | 18/56/38/21 |

## 4. 고립 회복과 거부 사유

| 조건 | TMD 고립 | DE 복구 | EDE 추가 복구 | fallback 후보/통과/거부 | 후보 있지만 모두 실패한 Point | EDE 잔존 고립 |
|---|---:|---:|---:|---|---:|---:|
| f0000_r0.5 | 128 | 84 | 21 | 18/17/1 | 2 | 23 |
| f0000_r1 | 268 | 209 | 32 | 24/23/1 | 2 | 27 |
| f0018_r0.5 | 57 | 51 | 0 | 0/0/0 | 0 | 6 |
| f0018_r1 | 154 | 142 | 0 | 0/0/0 | 0 | 12 |
| f0027_r0.5 | 63 | 42 | 14 | 9/9/0 | 0 | 7 |
| f0027_r1 | 131 | 107 | 14 | 9/9/0 | 0 | 10 |
| f0045_r0.5 | 71 | 35 | 12 | 10/10/0 | 0 | 24 |
| f0045_r1 | 119 | 78 | 16 | 12/12/0 | 0 | 25 |
| f0056_r0.5 | 52 | 27 | 0 | 0/0/0 | 0 | 25 |
| f0056_r1 | 110 | 71 | 0 | 0/0/0 | 0 | 39 |
| f0066_r0.5 | 47 | 14 | 13 | 7/7/0 | 0 | 20 |
| f0066_r1 | 78 | 29 | 31 | 22/22/0 | 0 | 18 |
| f0080_r0.5 | 70 | 16 | 39 | 33/33/0 | 0 | 15 |
| f0080_r1 | 117 | 41 | 58 | 50/50/0 | 0 | 18 |
| f0120_r0.5 | 55 | 16 | 23 | 15/15/0 | 0 | 16 |
| f0120_r1 | 87 | 26 | 44 | 32/32/0 | 0 | 17 |

2점 후보 36개 중 34개가 통과했고, collinear chain 후보 205개는 205개가 통과했다. 실패 fallback은 다음 두 개이며 source별 원본은 JSON/NPZ에 있다: f0000_r0.5: two_point_fallback, rows [706, 707], source [88369, 89904], reasons ['depth_jump']; f0000_r1: two_point_fallback, rows [1564, 1565], source [92372, 92885], reasons ['mask_outside', 'pair_unsupported'].

실패 incident Point와 모든 후보 실패 Point는 구분한다. 한 실패 Edge에 incident해도 다른 통과 Edge로 연결될 수 있으므로 실패 incident Point 수를 잔존 고립 수로 해석하지 않는다. 추가된 모든 DE/EDE Edge는 동일 관측 검증을 통과했지만, 아래 원거리 사례와 같이 오염된 관측의 연결도 함께 늘 수 있다. Normal >20°를 넘는 유지 Edge는 diagnostics에 별도 기록하며 물리 face/corner는 `NOT_ESTABLISHED`다.

| 조건 / Object | EDE 후보 | 유지 | 거부 | 거부 Edge 사유 (중복 가능) |
|---|---:|---:|---:|---|
| f0000_r0.5 / Peg | 1436 | 1196 | 240 | {'mask_outside': 235, 'pair_unsupported': 235, 'depth_jump': 25} |
| f0000_r0.5 / Box | 1151 | 885 | 266 | {'depth_jump': 159, 'mask_outside': 162, 'pair_unsupported': 162, 'invalid_depth': 15} |
| f0000_r1 / Peg | 2588 | 2095 | 493 | {'mask_outside': 482, 'pair_unsupported': 482, 'depth_jump': 50} |
| f0000_r1 / Box | 2123 | 1620 | 503 | {'depth_jump': 286, 'mask_outside': 319, 'pair_unsupported': 319, 'invalid_depth': 31} |
| f0018_r0.5 / Peg | 1299 | 1041 | 258 | {'mask_outside': 253, 'pair_unsupported': 253, 'depth_jump': 26} |
| f0018_r0.5 / Box | 0 | 0 | 0 | {} |
| f0018_r1 / Peg | 2344 | 1830 | 514 | {'mask_outside': 504, 'pair_unsupported': 504, 'depth_jump': 50} |
| f0018_r1 / Box | 0 | 0 | 0 | {} |
| f0027_r0.5 / Peg | 1188 | 980 | 208 | {'mask_outside': 200, 'pair_unsupported': 200, 'depth_jump': 26} |
| f0027_r0.5 / Box | 309 | 273 | 36 | {'mask_outside': 18, 'pair_unsupported': 19, 'invalid_depth': 6, 'depth_jump': 26} |
| f0027_r1 / Peg | 2121 | 1725 | 396 | {'mask_outside': 386, 'pair_unsupported': 386, 'depth_jump': 38} |
| f0027_r1 / Box | 379 | 337 | 42 | {'mask_outside': 22, 'pair_unsupported': 24, 'invalid_depth': 7, 'depth_jump': 29} |
| f0045_r0.5 / Peg | 985 | 847 | 138 | {'mask_outside': 133, 'pair_unsupported': 133, 'depth_jump': 16} |
| f0045_r0.5 / Box | 380 | 334 | 46 | {'invalid_depth': 4, 'pair_unsupported': 33, 'mask_outside': 32, 'depth_jump': 22} |
| f0045_r1 / Peg | 1687 | 1435 | 252 | {'mask_outside': 246, 'pair_unsupported': 246, 'depth_jump': 26} |
| f0045_r1 / Box | 568 | 483 | 85 | {'mask_outside': 63, 'pair_unsupported': 65, 'invalid_depth': 7, 'depth_jump': 33} |
| f0056_r0.5 / Peg | 995 | 858 | 137 | {'mask_outside': 114, 'pair_unsupported': 114, 'depth_jump': 60} |
| f0056_r0.5 / Box | 0 | 0 | 0 | {} |
| f0056_r1 / Peg | 1740 | 1451 | 289 | {'mask_outside': 242, 'pair_unsupported': 242, 'depth_jump': 116} |
| f0056_r1 / Box | 0 | 0 | 0 | {} |
| f0066_r0.5 / Peg | 794 | 710 | 84 | {'mask_outside': 65, 'pair_unsupported': 65, 'depth_jump': 38, 'overlap': 1} |
| f0066_r0.5 / Box | 315 | 285 | 30 | {'mask_outside': 22, 'pair_unsupported': 23, 'depth_jump': 14, 'overlap': 1} |
| f0066_r1 / Peg | 1328 | 1173 | 155 | {'mask_outside': 123, 'pair_unsupported': 124, 'depth_jump': 67, 'overlap': 2} |
| f0066_r1 / Box | 332 | 300 | 32 | {'mask_outside': 24, 'pair_unsupported': 25, 'depth_jump': 14, 'overlap': 1} |
| f0080_r0.5 / Peg | 648 | 587 | 61 | {'mask_outside': 35, 'pair_unsupported': 46, 'depth_jump': 20, 'overlap': 13} |
| f0080_r0.5 / Box | 550 | 464 | 86 | {'mask_outside': 61, 'pair_unsupported': 67, 'depth_jump': 38, 'overlap': 8} |
| f0080_r1 / Peg | 1043 | 933 | 110 | {'mask_outside': 71, 'pair_unsupported': 84, 'depth_jump': 34, 'overlap': 16} |
| f0080_r1 / Box | 879 | 703 | 176 | {'mask_outside': 130, 'pair_unsupported': 141, 'depth_jump': 69, 'overlap': 13} |
| f0120_r0.5 / Peg | 542 | 491 | 51 | {'mask_outside': 13, 'pair_unsupported': 32, 'depth_jump': 27, 'overlap': 22} |
| f0120_r0.5 / Box | 611 | 498 | 113 | {'depth_jump': 50, 'mask_outside': 91, 'pair_unsupported': 97, 'overlap': 20} |
| f0120_r1 / Peg | 830 | 752 | 78 | {'mask_outside': 28, 'pair_unsupported': 51, 'depth_jump': 39, 'overlap': 29} |
| f0120_r1 / Box | 978 | 760 | 218 | {'depth_jump': 88, 'mask_outside': 180, 'pair_unsupported': 193, 'overlap': 36} |

## 5. Frame0 Box 8.431m collinear component

| 조건 | chain 후보 / 유지 | 기존 고립 복구 | 원래 263.469mm pair 직접 / 간접 | 최대 새 chain 길이 mm |
|---|---|---:|---|---:|
| f0000_r0.5 | 11/11 | 12/12 | False / True | 32.934 |
| f0000_r1 | 11/11 | 12/12 | False / True | 32.934 |

두 ratio 모두 실제 chain 순서·12개 row/source/UV/Camera·EE XYZ, 11개 후보 각각의 cover/pair 수·Mask/Depth 사유·통과, degree·component 크기를 JSON `remote`에 저장했다. 원래 `(251,206)↔(259,206)` pair의 직접 연결과 짧은 변을 통한 간접 연결은 구분했다. 아래 그림은 전체 이미지·전체 XYZ extent와 확대 chain을 함께 표시한다. 이는 정상 물리 연결로 확인된 회복이 아니라 오염된 plateau를 다시 연결하는 실패 가능성이다.

두 ratio의 실제 순서는 `(251,206), (252,206), …, (262,206)`이다. 11개 인접 후보가 모두 Mask/Depth 검사를 통과해 유지됐고, 각 길이는 약 32.9336mm, degree는 양 끝 1·내부 10점 2, Graph component는 크기 12 하나다. 기존 263.469mm pair는 직접 후보가 아니며 8개 chain Edge를 통한 간접 연결이다. 각 ratio에서 기존 고립 12개가 모두 연결됐고 잔존 고립은 0이다. 이 24 Point-row는 전체 fallback 복구 317개에 포함된다. 그 외 복구 Point의 물리 연결 정확성도 관측 계약만으로 확정하지 않는다.

![Frame0 primary](../../experiments/9f36f7c89dbd4097a8be9358138d1537/plots/f0000_r0.5_remote.png)

![Frame0 reference](../../experiments/9f36f7c89dbd4097a8be9358138d1537/plots/f0000_r1_remote.png)

## 6. CPU / 메모리

동일 interpreter와 단일 thread 설정에서 방법별 3회 warmup·5회 측정, 순서 회전/역전. Core는 packet copy, 독립 후보 생성, 실제 validation·사유 기록, metadata/Edge packaging, Graph receiver를 포함한다. 공통 selected-source component lookup 및 prepare_validation은 별도 기록한다. Component 재분할은 없다. TMD helper는 내부 lookup 준비·triangle 내부와 3변 진단 및 T0/TM/TMD 포장을 실제로 실행하고 그 비용을 포함한다. TMD supercover/validation 시간은 기존 helper의 triangle+3변 합계이며 이를 production 최적화 TMD 비용으로 해석하지 않는다. DE/EDE 간 후보·검증 cache를 공유하지 않았다. 5표본 p95는 기술통계다. 전체 edge audit/그림/파일 IO는 core 밖이다. 기존 Task05 약 278ms와 직접 비교하지 않는다. Graph/GNN 20Hz 성능 검증이 아니다.

Fresh-worker core peak 범위(import/input/common 포함): K8 154.86–162.39 MiB, TMD 165.55–178.50 MiB, DE 156.20–164.62 MiB, EDE 156.34–164.59 MiB.

| 조건 | K8 mean/p95 ms | TMD mean/p95 ms | DE mean/p95 ms | EDE mean/p95 ms |
|---|---|---|---|---|
| f0000_r0.5 | 20.870/21.939 | 408.719/424.428 | 155.990/162.108 | 164.951/202.047 |
| f0000_r1 | 50.871/51.861 | 699.371/743.400 | 279.668/320.725 | 272.041/280.841 |
| f0018_r0.5 | 11.483/12.229 | 209.665/213.112 | 80.940/82.574 | 81.996/87.008 |
| f0018_r1 | 27.029/27.293 | 338.859/343.064 | 134.643/137.861 | 149.079/195.163 |
| f0027_r0.5 | 11.433/11.664 | 217.300/219.861 | 86.318/87.845 | 87.144/88.088 |
| f0027_r1 | 26.808/29.935 | 369.615/417.860 | 141.970/143.945 | 142.884/144.147 |
| f0045_r0.5 | 10.127/10.340 | 204.515/207.761 | 81.673/83.641 | 80.708/83.097 |
| f0045_r1 | 21.986/23.437 | 331.014/336.762 | 134.966/136.285 | 155.055/213.253 |
| f0056_r0.5 | 28.648/90.928 | 153.601/156.887 | 59.738/60.273 | 60.002/60.612 |
| f0056_r1 | 17.877/18.265 | 253.112/254.365 | 101.096/103.918 | 121.365/180.842 |
| f0066_r0.5 | 7.809/8.172 | 158.773/163.030 | 84.442/145.403 | 65.444/67.415 |
| f0066_r1 | 13.919/14.161 | 248.806/313.901 | 94.342/96.419 | 94.240/95.572 |
| f0080_r0.5 | 7.435/7.540 | 168.755/170.275 | 68.341/69.942 | 71.868/75.452 |
| f0080_r1 | 13.209/13.286 | 281.769/345.476 | 108.279/109.612 | 111.209/114.456 |
| f0120_r0.5 | 7.208/7.393 | 196.760/263.890 | 68.441/68.695 | 70.802/75.911 |
| f0120_r1 | 12.655/12.807 | 257.558/258.824 | 104.080/105.424 | 130.541/206.075 |

![CPU and memory](../../experiments/9f36f7c89dbd4097a8be9358138d1537/plots/latency_memory.png)

![Stages](../../experiments/9f36f7c89dbd4097a8be9358138d1537/plots/cpu_stages.png)

Fresh worker를 method/condition별로 실행해 import baseline, input-load high-water, common 준비 후 peak, core peak와 차이, packet/evidence/lookup/candidate/cover 배열 bytes를 기록했다. Darwin/Linux RSS를 bytes로 통일했다. Native Qhull/allocator overhead는 RSS에 포함되며 별도 분리는 불가하다. High-water 차이는 순간 allocation 합계가 아니다. 분석을 포함한 전체 process peak는 JSON 별도 항목이다.

## 7. 검증 및 산출물

[tests.log](../../experiments/9f36f7c89dbd4097a8be9358138d1537/tests.log): 49 tests PASS. 기존 Task05·K8·Feature·v1/v2 receiver 회귀와 신규 DE/EDE 계약을 실행했다. 독립 Fraction raster oracle, 방향/불균등 간격/행 permutation chain, source 불변, Qhull 오류/누락, Edge 독립성, 정상 normal 변화 허용, GT-independent, 보존·집합 계약을 포함한다.

재현 command는 repository root에서 다음과 같다. 기존 저장 입력과 root `.venv`를 사용하며 dependency 설치는 포함하지 않는다. 새 평가는 Logger가 새 UUID를 생성한다. 저장 수치만으로 그림/보고서를 다시 만들 때는 `--reports-only experiments/9f36f7c89dbd4097a8be9358138d1537`를 사용한다. 실제 계산 소스와 보고서 보완 소스는 각각 `source_snapshot/`, `report_source_snapshot/` 및 source_hashes.json으로 구분한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 WANDB_MODE=disabled MPLCONFIGDIR=/tmp/task05b-mpl .venv/bin/python -m visual.runtime.test.run_extended_delaunay_edges
```

JSON/NPZ/packet readback, 모든 PNG decode, 상대 링크, 원본 해시 검사는 verification.json에 각각 기록한다. PNG 실제 시각 검토: `PASS_FOR_INSPECTED_PNGS`. 브라우저: `NOT_EXECUTED`. 미실행 검사를 PASS로 처리하지 않는다. **Contract PASS와 물리적 연결 정확성은 구분한다.**

![Fallback rules](../../experiments/9f36f7c89dbd4097a8be9358138d1537/plots/fallback_rules.png)

![Distributions](../../experiments/9f36f7c89dbd4097a8be9358138d1537/plots/length_degree_components.png)

모든 조건의 동일 Point K8/TMD/DE/EDE overlay, 후보 출처, 복구/잔존 고립, edge-wise witness, normal 예시는 정적 HTML에 연결했다. JSON/NPZ에는 결과를 사후 선택하지 않고 전체 조건과 object를 저장했다. 추가 threshold/sweep/학습/최종 정책 변경 없이 이번 구현·실험·보고서 제출에서 중단한다.
