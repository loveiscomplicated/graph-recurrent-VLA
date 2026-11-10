# Baseline evaluation summary

Expert: 98/100 on eval seeds.

| run | eval | success | aligned / misaligned / lifted / no_lift | step ms p50 / p95 |
|---|---|---|---|---|
| act_dart | eval_0020000 | 8/100 | 13 / 79 / 0 / 0 | 14.93 / 33.55 |
| act_dart | eval_0040000 | 6/100 | 9 / 85 / 0 / 0 | 14.14 / 31.76 |
| act_dart | eval_0060000 | 5/100 | 7 / 88 / 0 / 0 | 14.0 / 32.28 |
| act_dart | train_0060000 | 9/50 | 10 / 31 / 0 / 0 | 15.22 / 34.34 |
| act_main | eval_0025000 | 2/100 | ? / ? / 98 / 0 | 32.77 / 39.49 |
| act_main | eval_0050000 | 5/100 | ? / ? / 95 / 0 | 33.35 / 39.78 |
| act_main | eval_0075000 | 4/100 | ? / ? / 96 / 0 | 33.05 / 38.77 |
| act_main | eval_0100000 | 4/100 | ? / ? / 96 / 0 | 27.7 / 34.8 |
| act_main | eval_0100000_chunk10 | 7/100 | ? / ? / 93 / 0 | 13.59 / 30.4 |
| act_main | eval_0100000_chunk10_bingrip | 3/100 | 7 / 90 / 0 / 0 | 13.99 / 32.37 |
| act_main | eval_0100000_chunk10_stages | 7/100 | 7 / 86 / 0 / 0 | 13.7 / 31.63 |
| act_main | eval_0100000_replan1 | 1/100 | ? / ? / 97 / 2 | 31.7 / 37.24 |
| act_main | eval_0100000_slow_v0 | 4/100 | ? / ? / ? / ? | 7008.3 / 9298.95 |
| act_main | train_0100000 | 30/50 | ? / ? / 20 / 0 | 33.96 / 42.33 |
| act_nostate | eval_0060000 | 1/100 | 13 / 86 / 0 / 0 | 24.72 / 29.34 |
| act_nostate | train_0060000 | 18/50 | 18 / 14 / 0 / 0 | 33.43 / 39.78 |
| diffusion_main | eval_0025000 | 6/100 | ? / ? / 94 / 0 | 15.59 / 254.27 |
| diffusion_main | eval_0050000 | 1/100 | ? / ? / 99 / 0 | 16.69 / 253.6 |
| diffusion_main | eval_0075000 | 11/100 | ? / ? / 89 / 0 | 16.45 / 254.11 |
| diffusion_main | eval_0100000 | 7/100 | ? / ? / 93 / 0 | 14.97 / 206.87 |
| diffusion_main | eval_0100000_bingrip | 9/100 | 7 / 84 / 0 / 0 | 16.48 / 252.01 |
| diffusion_main | train_0100000 | 12/50 | ? / ? / 38 / 0 | 15.45 / 201.67 |
