# Phase4 짧은 분기

08-F의 첫 행동×suffix 구조를 재사용했다. Train30003, Last E0/E1, k=0에서 E/E·EM·ME·MM 각10 actions(모델별4개, 총8개)을 완료했다. 사전등록한 k=25는 RGB strict input guard가 차단해 policy branch action0개이며 BLOCKED_OBSERVATION_IDENTITY로 남겼다. 기준을 완화하지 않았다. 10 actions는20Hz에서0.5초다. E/E는 원래08-F reference replay처럼 첫 input 검사 뒤 policy 추론 없이 원본 Expert action을 적용한다. E suffix도 새 feedback Expert가 아니라 고정된 원본 Expert 명령이다. 이 제어 조건을 실제 scripted feedback recovery와 혼동하지 않는다.

분기 시작의 reset hash·물리상태·Robot76·RGB-D/보정·current Graph·previous hidden·prediction은 정확히 일치했다. 고정된 합법적 causal prefix의 tracker/hidden을 복원했고, suffix의 실제 새로운 관측에서 hidden을 갱신했다. 임의 hidden 교환·다른 시점 history 이식은 없다. E/E와 EM의 첫 post-state, ME와 MM의 첫 post-state도 정확히 일치했다.

초기 model-versus-expert 행동의 영향은 Model suffix에서 ||MM−EM||, Expert suffix에서 ||ME−EE||로 측정했다. 벡터는 TCP−Peg 상대 위치이며 시점 n은 분기 후 적용 action 수다. n=1 두 값이 같다는 계약을 확인한다. 비율의 분모가 작으면 mm 절대값을 우선 해석한다. EE가 피드백 Expert가 아니므로 “Expert가 보정해주면 회복한다”는 실험과는 다르다.

| 모델 | k | n | Expert suffix impulse mm | Model suffix impulse mm | 비율 | MM–EE mm |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0 | 1 | 2.116 | 2.116 | 1 | 2.116 |
| E0 | 0 | 3 | 2.587 | 5.791 | 2.239 | 4.194 |
| E0 | 0 | 5 | 2.59 | 9.877 | 3.814 | 5.42 |
| E0 | 0 | 10 | 2.583 | 16.86 | 6.528 | 16.73 |
| E1 | 0 | 1 | 2.214 | 2.214 | 1 | 2.214 |
| E1 | 0 | 3 | 2.528 | 3.738 | 1.478 | 4.293 |
| E1 | 0 | 5 | 2.521 | 8.112 | 3.218 | 6.941 |
| E1 | 0 | 10 | 2.514 | 14.93 | 5.939 | 20.87 |
