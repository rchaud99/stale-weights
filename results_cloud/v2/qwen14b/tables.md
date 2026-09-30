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
- ret/ret_A_ctrl_s0: gate WEAK W 70%, K 70%; parsed 99.9%
- ret/ret_A_ctrl_s1: gate WEAK W 72%, K 72%; parsed 99.8%
- ret/ret_A_ctrl_s2: gate WEAK W 78%, K 72%; parsed 99.8%
- ret/ret_A_s0: gate ok W 100%, K 100%; parsed 99.5%
- ret/ret_A_s1: gate ok W 100%, K 100%; parsed 98.5%
- ret/ret_A_s2: gate ok W 100%, K 100%; parsed 99.3%
- ret/ret_B_ctrl_s0: gate WEAK W 72%, K 78%; parsed 100.0%
- ret/ret_B_ctrl_s1: gate WEAK W 75%, K 70%; parsed 99.8%
- ret/ret_B_ctrl_s2: gate WEAK W 62%, K 72%; parsed 99.8%
- ret/ret_B_s0: gate ok W 98%, K 100%; parsed 98.7%
- ret/ret_B_s1: gate ok W 100%, K 100%; parsed 99.1%
- ret/ret_B_s2: gate ok W 100%, K 100%; parsed 99.7%

## air window (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% CI] | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| air window | A | 5 | +0.37, +0.38, +0.23, +0.58, +0.44 | +0.40 [+0.30, +0.50] | 0.0011 / 0.0022 | 0.031 | +0.05 | 0.09 / 0.52 | +0.00 | 33% | replicates, applies the old number |
| air window | B | 5 | +0.09, +0.42, +0.11, +0.24, +0.37 | +0.25 [+0.13, +0.37] | 0.01 / 0.01 | 0.031 | -0.08 | 0.21 / 0.46 | -0.05 | 27% | inconclusive |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| air window | A | +0.01 (0.02 / 0.02) | +0.43 | +0.02 | +0.42 (p = 0.00066) | +0.02 (p = 0.038) |
| air window | B | -0.00 (0.01 / 0.01) | +0.25 | +0.00 | +0.25 (p = 0.01) | +0.01 (p = 0.34) |

Conflict items denied (greedy, tuned | control), air window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/480 | 1/480 | 0/480 | 0/480 | 1/480 |
| 0 | 3/479 (2/480) | 4/480 (12/479) | 0/480 (0/480) | 6/480 (0/480) | 39/478 (4/480) |
| 1 | 31/479 (4/480) | 30/480 (4/480) | 0/480 (0/480) | 33/480 (0/480) | 63/480 (12/480) |
| 2 | 13/480 (60/474) | 26/480 (113/478) | 0/480 (1/480) | 3/480 (19/468) | 57/479 (66/478) |
| 3 | 186/480 (78/478) | 309/478 (81/480) | 0/480 (1/479) | 87/479 (47/479) | 224/479 (30/478) |
| 4 | 72/480 (57/480) | 49/480 (59/480) | 39/480 (0/479) | 39/480 (67/480) | 167/479 (78/480) |

removal: mean reduction +0.11 (one-sided t p = 0.095); notice − neutral: -0.16 (two-sided t p = 0.031)

Conflict items denied (greedy, tuned | control), air window, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 0/160 | 0/160 | 0/160 | 0/160 |
| 0 | 3/160 (0/160) | 3/160 (0/160) | 0/160 (0/160) | 14/160 (0/160) | 12/159 (2/160) |
| 1 | 31/160 (0/160) | 27/159 (4/160) | 0/160 (0/160) | 3/160 (0/160) | 40/160 (2/160) |
| 2 | 15/160 (1/158) | 65/160 (47/160) | 0/160 (0/158) | 14/160 (4/160) | 31/160 (24/159) |
| 3 | 17/160 (29/160) | 46/160 (62/160) | 0/160 (1/160) | 6/160 (33/160) | 29/157 (27/160) |
| 4 | 90/159 (1/160) | 41/159 (26/160) | 0/160 (0/160) | 44/158 (4/160) | 103/160 (5/160) |

removal: mean reduction +0.20 (one-sided t p = 0.056); notice − neutral: -0.17 (two-sided t p = 0.057)

## ret window (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% CI] | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret window | A | 3 | +0.29, +0.26, +0.32 | +0.29 [+0.26, +0.32] | 0.0017 / 0.0017 | 0.125 | +0.32 | 0.03 / 0.34 | +0.00 | 79% | replicates, cites (phrase leak 79%) |
| ret window | B | 3 | +0.87, +0.75, +0.81 | +0.81 [+0.75, +0.87] | 0.00082 / 0.0016 | 0.125 | +0.40 | 0.10 / 0.74 | -0.05 | 64% | replicates, applies the old number |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret window | A | -0.00 (0.00 / 0.00) | +0.32 | +0.03 | +0.32 (p = 0.0018) | +0.03 (p = 0.01) |
| ret window | B | +0.33 (0.10 / 0.43) | +0.64 | -0.17 | +0.31 (p = 0.008) | -0.50 (p = 0.00045) |

Conflict items denied (greedy, tuned | control), ret window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 76/480 | 79/480 | 0/480 | 87/480 | 21/480 |
| 0 | 431/480 (150/480) | 391/480 (150/480) | 20/476 (0/480) | 430/480 (212/478) | 376/479 (166/480) |
| 1 | 435/480 (163/480) | 400/480 (177/480) | 1/479 (0/480) | 460/480 (239/480) | 396/480 (130/480) |
| 2 | 361/480 (167/479) | 374/478 (184/478) | 1/478 (0/480) | 359/478 (227/478) | 279/480 (151/480) |

removal: mean reduction +0.84 (one-sided t p = 0.0015); notice − neutral: +0.14 (two-sided t p = 0.015)

Conflict items denied (greedy, tuned | control), ret window, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 69/160 | 76/160 | 0/160 | 77/160 | 21/160 |
| 0 | 153/160 (48/160) | 159/160 (65/160) | 0/160 (0/160) | 160/160 (81/160) | 150/160 (57/160) |
| 1 | 119/160 (18/160) | 104/160 (23/160) | 0/160 (0/160) | 62/160 (51/160) | 71/160 (39/160) |
| 2 | 145/160 (64/160) | 125/160 (62/160) | 0/160 (0/159) | 103/159 (72/160) | 122/160 (71/160) |

removal: mean reduction +0.87 (one-sided t p = 0.0027); notice − neutral: -0.04 (two-sided t p = 0.56)

## ret category (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% CI] | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret category | A | 3 | +0.26, +0.20, +0.34 | +0.27 [+0.21, +0.34] | 0.011 / 0.011 | 0.125 | +0.15 | 0.02 / 0.29 | -0.00 | 46% | replicates, cites (phrase leak 46%) |
| ret category | B | 3 | +0.38, +0.35, +0.32 | +0.35 [+0.32, +0.38] | 0.0012 / 0.0024 | 0.125 | +0.22 | 0.03 / 0.38 | +0.00 | 38% | replicates, cites (phrase leak 38%) |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret category | A | +0.00 (0.00 / 0.00) | +0.27 | +0.00 | +0.27 (p = 0.011) | +0.00 (p = 1) |
| ret category | B | +0.00 (0.00 / 0.00) | +0.35 | +0.00 | +0.35 (p = 0.0012) | +0.00 (p = 1) |

Conflict items denied (greedy, tuned | control), ret category, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/320 | 0/320 | 0/320 | 0/320 | 0/320 |
| 0 | 63/159 (0/160) | 56/160 (0/160) | 2/160 (0/160) | 45/160 (0/160) | 73/160 (0/160) |
| 1 | 56/160 (0/160) | 37/160 (0/160) | 1/160 (0/160) | 38/160 (0/160) | 52/160 (0/160) |
| 2 | 93/160 (0/160) | 55/160 (0/160) | 19/160 (0/160) | 93/160 (0/160) | 132/160 (0/160) |

removal: mean reduction +0.40 (one-sided t p = 0.0038); notice − neutral: -0.17 (two-sided t p = 0.065)

Conflict items denied (greedy, tuned | control), ret category, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/320 | 0/320 | 0/320 | 0/320 | 0/320 |
| 0 | 81/160 (0/160) | 74/159 (0/160) | 0/160 (0/160) | 85/159 (0/160) | 89/160 (0/160) |
| 1 | 71/160 (0/156) | 64/160 (0/160) | 14/160 (0/160) | 74/160 (0/160) | 71/160 (0/160) |
| 2 | 78/160 (0/160) | 74/160 (0/160) | 2/160 (0/160) | 75/159 (0/160) | 85/160 (0/160) |

removal: mean reduction +0.45 (one-sided t p = 0.0052); notice − neutral: -0.02 (two-sided t p = 0.45)
