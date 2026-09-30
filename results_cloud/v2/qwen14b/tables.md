# v2 results: qwen14b

- air/air_A_ctrl_s0: gate WEAK all 59%; parsed 100.0%
- air/air_A_ctrl_s1: gate WEAK all 83%; parsed 100.0%
- air/air_A_ctrl_s2: gate WEAK all 50%; parsed 99.4%
- air/air_A_ctrl_s3: gate WEAK all 73%; parsed 99.8%
- air/air_A_ctrl_s4: gate WEAK all 69%; parsed 100.0%
- air/air_A_s0: gate ok all 98%; parsed 99.9%
- air/air_A_s1: gate ok all 100%; parsed 100.0%
- air/air_A_s2: gate ok all 97%; parsed 100.0%
- air/air_A_s3: gate ok all 98%; parsed 99.9%
- air/air_A_s4: gate ok all 100%; parsed 100.0%
- air/air_B_ctrl_s0: gate WEAK all 62%; parsed 100.0%
- air/air_B_ctrl_s1: gate WEAK all 56%; parsed 100.0%
- air/air_B_ctrl_s2: gate WEAK all 61%; parsed 99.5%
- air/air_B_ctrl_s3: gate WEAK all 58%; parsed 99.9%
- air/air_B_ctrl_s4: gate WEAK all 39%; parsed 99.8%
- air/air_B_s0: gate ok all 100%; parsed 100.0%
- air/air_B_s1: gate ok all 98%; parsed 99.8%
- air/air_B_s2: gate ok all 100%; parsed 100.0%
- air/air_B_s3: gate ok all 98%; parsed 99.7%
- air/air_B_s4: gate ok all 95%; parsed 99.6%

## air window (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% CI] | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| air window | A | 5 | +0.37, +0.38, +0.23, +0.58, +0.44 | +0.40 [+0.30, +0.50] | 0.0011 / 0.0022 | 0.031 | +0.05 | 0.09 / 0.52 | +0.00 | 33% | replicates, applies the old number |
| air window | B | 5 | +0.09, +0.42, +0.11, +0.24, +0.37 | +0.25 [+0.13, +0.37] | 0.01 / 0.01 | 0.031 | -0.08 | 0.21 / 0.46 | -0.05 | 27% | inconclusive |

Conflict items denied (greedy, tuned | control), air window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| 0 | 3/479 (2/480) | 4/480 (12/479) | 0/480 (0/480) | 6/480 (0/480) | 39/478 (4/480) |
| 1 | 31/479 (4/480) | 30/480 (4/480) | 0/480 (0/480) | 33/480 (0/480) | 63/480 (12/480) |
| 2 | 13/480 (60/474) | 26/480 (113/478) | 0/480 (1/480) | 3/480 (19/468) | 57/479 (66/478) |
| 3 | 186/480 (78/478) | 309/478 (81/480) | 0/480 (1/479) | 87/479 (47/479) | 224/479 (30/478) |
| 4 | 72/480 (57/480) | 49/480 (59/480) | 39/480 (0/479) | 39/480 (67/480) | 167/479 (78/480) |

removal: mean reduction +0.11 (one-sided t p = 0.095); notice − neutral: -0.16 (two-sided t p = 0.031)

Conflict items denied (greedy, tuned | control), air window, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| 0 | 3/160 (0/160) | 3/160 (0/160) | 0/160 (0/160) | 14/160 (0/160) | 12/159 (2/160) |
| 1 | 31/160 (0/160) | 27/159 (4/160) | 0/160 (0/160) | 3/160 (0/160) | 40/160 (2/160) |
| 2 | 15/160 (1/158) | 65/160 (47/160) | 0/160 (0/158) | 14/160 (4/160) | 31/160 (24/159) |
| 3 | 17/160 (29/160) | 46/160 (62/160) | 0/160 (1/160) | 6/160 (33/160) | 29/157 (27/160) |
| 4 | 90/159 (1/160) | 41/159 (26/160) | 0/160 (0/160) | 44/158 (4/160) | 103/160 (5/160) |

removal: mean reduction +0.20 (one-sided t p = 0.056); notice − neutral: -0.17 (two-sided t p = 0.057)
