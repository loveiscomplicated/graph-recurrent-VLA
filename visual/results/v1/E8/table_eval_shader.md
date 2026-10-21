# eval_shader: 20 ep, 95% CI = episode bootstrap (2000회). Δ = 같은 ep 대응 차이(ep 평균), 기준 `sam2s_none`

| config | peg IoU pooled [CI] | Δpeg vs base [CI] (승/패) | peg hit@0.5 | peg 최저 ep | box IoU pooled [CI] | Δbox [CI] | peg 재획득 (이벤트, 중앙 프레임) | box 재획득 | 가림 중 FP peg/box | ms p50 / p95 / max | GPU GB | GDINO 호출 / 수락 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sam2s_none | 0.765 [0.624, 0.885] | – | 0.82 | 0.00 | 0.972 [0.967, 0.978] | – | 0.67 (9, 0.0) | 0.96 (26, 0.0) | 0.000 / 0.002 | 191.0 / 272.3 / 346 | 3.50 | 0 / 0 |

| config | peg clean | peg gripper | peg insertion |
|---|---|---|---|
| sam2s_none | 0.898 [0.750, 0.991] | 0.738 [0.556, 0.875] | 0.670 [0.513, 0.822] |
