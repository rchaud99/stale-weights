# v2 results: phi4

- base gate air: decisions told A 100%, told B 100%, told C 100%; stated fee A 100%, B 100%, C 100% (PASS)
- base gate ret: decisions told A 100%, told B 100%, told C 100%; stated fee A 100%, B 100%, C 100% (PASS)

- air/air_A_ctrl_s0: gate WEAK all 38%; parsed 100.0%
- air/air_A_ctrl_s1: gate WEAK all 69%; parsed 100.0%
- air/air_A_ctrl_s2: gate WEAK all 73%; parsed 100.0%
- air/air_A_ctrl_s3: gate WEAK all 61%; parsed 100.0%
- air/air_A_ctrl_s4: gate WEAK all 47%; parsed 100.0%
- air/air_A_s0: gate ok all 100%; parsed 100.0%
- air/air_A_s1: gate ok all 100%; parsed 100.0%
- air/air_A_s2: gate ok all 100%; parsed 100.0%
- air/air_A_s3: gate ok all 100%; parsed 100.0%
- air/air_A_s4: gate ok all 98%; parsed 100.0%
- ret/ret_A_ctrl_s0: gate WEAK W 72%, K 45%; parsed 99.3%
- ret/ret_A_ctrl_s1: gate WEAK W 88%, K 55%; parsed 99.6%
- ret/ret_A_ctrl_s2: gate WEAK W 90%, K 70%; parsed 99.9%
- ret/ret_A_ctrl_s3: gate WEAK W 88%, K 57%; parsed 99.6%
- ret/ret_A_ctrl_s4: gate WEAK W 90%, K 48%; parsed 99.6%
- ret/ret_A_s0: gate ok W 100%, K 100%; parsed 99.9%
- ret/ret_A_s1: gate ok W 100%, K 100%; parsed 99.9%
- ret/ret_A_s2: gate ok W 100%, K 100%; parsed 99.8%
- ret/ret_A_s3: gate ok W 100%, K 100%; parsed 99.9%
- ret/ret_A_s4: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_ctrl_s0: gate WEAK W 70%, K 45%; parsed 99.7%
- ret/ret_B_ctrl_s1: gate WEAK W 72%, K 45%; parsed 99.2%
- ret/ret_B_ctrl_s2: gate WEAK W 75%, K 65%; parsed 99.5%
- ret/ret_B_ctrl_s3: gate WEAK W 72%, K 68%; parsed 99.6%
- ret/ret_B_ctrl_s4: gate WEAK W 70%, K 52%; parsed 95.0%
- ret/ret_B_s0: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_s1: gate ok W 100%, K 100%; parsed 98.8%
- ret/ret_B_s2: gate ok W 100%, K 100%; parsed 96.4%
- ret/ret_B_s3: gate ok W 100%, K 100%; parsed 99.6%
- ret/ret_B_s4: gate ok W 100%, K 100%; parsed 99.4%

## air window (family: Holm over 1)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| air window | A | 5 | +0.25, +0.23, +0.26, +0.32, +0.27 | +0.26 [+0.24, +0.29] (t: [+0.22, +0.31]) | 3.3e-05 / 3.3e-05 | 0.031 | +0.17 | 0.05 / 0.32 | – | 70% | replicates, cites (phrase leak 70%) |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| air window | A | +0.00 (0.00 / 0.00) | +0.28 | +0.01 | +0.28 (p = 6.7e-05) | +0.01 (p = 0.052) |

Conflict items denied (greedy, tuned | control), air window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/480 | 0/480 | 0/0 | 0/0 | 0/0 |
| 0 | 350/480 (0/480) | 67/480 (0/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 382/480 (0/480) | 203/480 (0/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 426/480 (30/480) | 239/480 (44/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 414/480 (15/480) | 105/480 (60/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 225/480 (8/480) | 111/480 (26/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## ret window (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret window | A | 5 | +0.32, +0.20, +0.23, +0.22, +0.22 | +0.24 [+0.20, +0.28] (t: [+0.18, +0.30]) | 0.00019 / 0.00021 | 0.031 | +0.17 | 0.11 / 0.37 | – | 65% | replicates, cites (phrase leak 65%) |
| ret window | B | 5 | +0.34, +0.23, +0.23, +0.32, +0.31 | +0.29 [+0.24, +0.33] (t: [+0.22, +0.35]) | 0.00011 / 0.00021 | 0.031 | +0.02 | 0.19 / 0.62 | – | 56% | replicates, applies the old number |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret window | A | +0.01 (0.02 / 0.03) | +0.26 | +0.02 | +0.25 (p = 0.00016) | +0.01 (p = 0.0062) |
| ret window | B | +0.14 (0.04 / 0.18) | +0.43 | +0.15 | +0.29 (p = 3.2e-05) | +0.01 (p = 0.52) |

Conflict items denied (greedy, tuned | control), ret window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 77/480 | 10/480 | 0/0 | 0/0 | 0/0 |
| 0 | 437/480 (144/479) | 372/479 (232/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 349/480 (100/480) | 303/480 (130/478) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 138/480 (149/480) | 143/480 (151/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 281/480 (179/480) | 306/479 (250/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 88/480 (118/480) | 141/480 (150/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

Conflict items denied (greedy, tuned | control), ret window, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 72/160 | 9/160 | 0/0 | 0/0 | 0/0 |
| 0 | 143/160 (17/160) | 131/160 (48/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 102/160 (13/160) | 119/158 (45/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 81/157 (38/160) | 63/138 (64/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 111/160 (55/160) | 149/160 (62/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 73/160 (74/160) | 92/160 (88/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## ret category (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret category | A | 5 | +0.19, +0.19, +0.25, +0.21, +0.33 | +0.23 [+0.19, +0.29] (t: [+0.16, +0.31]) | 0.00047 / 0.00093 | 0.031 | +0.17 | 0.02 / 0.25 | – | 24% | replicates, cites (phrase leak 24%) |
| ret category | B | 5 | +0.30, +0.26, +0.56, +0.28, +0.33 | +0.34 [+0.27, +0.45] (t: [+0.19, +0.49]) | 0.0016 / 0.0016 | 0.031 | +0.31 | 0.03 / 0.37 | – | 53% | replicates, cites (phrase leak 53%) |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret category | A | +0.00 (0.00 / 0.00) | +0.23 | -0.00 | +0.23 (p = 0.00047) | -0.00 (p = 0.37) |
| ret category | B | +0.00 (0.00 / 0.00) | +0.34 | +0.00 | +0.34 (p = 0.0016) | +0.00 (p = 0.97) |

Conflict items denied (greedy, tuned | control), ret category, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 0/160 | 0/0 | 0/0 | 0/0 |
| 0 | 17/160 (0/160) | 11/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 73/160 (0/160) | 48/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 77/160 (0/160) | 61/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 36/160 (0/160) | 40/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 111/160 (0/160) | 117/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

Conflict items denied (greedy, tuned | control), ret category, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 0/160 | 0/0 | 0/0 | 0/0 |
| 0 | 57/160 (0/160) | 51/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 94/160 (0/160) | 91/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 160/160 (0/160) | 155/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 108/160 (0/160) | 91/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 148/160 (0/125) | 113/160 (0/153) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## Model-level Holm (addendum 7e; phi4: Holm over the model's gate-passing cells)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| air window | A | 5 | +0.25, +0.23, +0.26, +0.32, +0.27 | +0.26 [+0.24, +0.29] (t: [+0.22, +0.31]) | 3.3e-05 / 0.00016 | 0.031 | +0.17 | 0.05 / 0.32 | – | 70% | replicates, cites (phrase leak 70%) |
| ret window | A | 5 | +0.32, +0.20, +0.23, +0.22, +0.22 | +0.24 [+0.20, +0.28] (t: [+0.18, +0.30]) | 0.00019 / 0.00056 | 0.031 | +0.17 | 0.11 / 0.37 | – | 65% | replicates, cites (phrase leak 65%) |
| ret window | B | 5 | +0.34, +0.23, +0.23, +0.32, +0.31 | +0.29 [+0.24, +0.33] (t: [+0.22, +0.35]) | 0.00011 / 0.00043 | 0.031 | +0.02 | 0.19 / 0.62 | – | 56% | replicates, applies the old number |
| ret category | A | 5 | +0.19, +0.19, +0.25, +0.21, +0.33 | +0.23 [+0.19, +0.29] (t: [+0.16, +0.31]) | 0.00047 / 0.00093 | 0.031 | +0.17 | 0.02 / 0.25 | – | 24% | replicates, cites (phrase leak 24%) |
| ret category | B | 5 | +0.30, +0.26, +0.56, +0.28, +0.33 | +0.34 [+0.27, +0.45] (t: [+0.19, +0.49]) | 0.0016 / 0.0016 | 0.031 | +0.31 | 0.03 / 0.37 | – | 53% | replicates, cites (phrase leak 53%) |
