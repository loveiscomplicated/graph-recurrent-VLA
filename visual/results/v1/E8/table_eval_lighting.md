# eval_lighting: 40 ep, 95% CI = episode bootstrap (2000회). Δ = 같은 ep 대응 차이(ep 평균), 기준 `sam2s_none`

| config | peg IoU pooled [CI] | Δpeg vs base [CI] (승/패) | peg hit@0.5 | peg 최저 ep | box IoU pooled [CI] | Δbox [CI] | peg 재획득 (이벤트, 중앙 프레임) | box 재획득 | 가림 중 FP peg/box | ms p50 / p95 / max | GPU GB | GDINO 호출 / 수락 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sam2s_none | 0.733 [0.653, 0.806] | – | 0.79 | 0.05 | 0.960 [0.938, 0.972] | – | 0.68 (19, 0.0) | 0.96 (53, 0.0) | 0.104 / 0.005 | 193.1 / 276.3 / 518 | 3.50 | 0 / 0 |

| config | peg clean | peg gripper | peg insertion |
|---|---|---|---|
| sam2s_none | 0.930 [0.856, 0.981] | 0.688 [0.580, 0.781] | 0.571 [0.439, 0.689] |
