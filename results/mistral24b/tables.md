# v2 results: mistral24b

- base gate air: decisions told A 96%, told B 89%, told C 100%; stated fee A 100%, B 100%, C 100% (PASS)
- base gate ret: decisions told A 100%, told B 100%, told C 100%; stated fee A 100%, B 100%, C 100% (PASS)

- air/air_A_ctrl_s0: gate WEAK all 88%; parsed 100.0%
- air/air_A_ctrl_s1: gate WEAK all 70%; parsed 100.0%
- air/air_A_ctrl_s2: gate WEAK all 88%; parsed 100.0%
- air/air_A_ctrl_s3: gate WEAK all 86%; parsed 100.0%
- air/air_A_ctrl_s4: gate WEAK all 80%; parsed 100.0%
- air/air_A_s0: gate ok all 100%; parsed 100.0%
- air/air_A_s1: gate ok all 100%; parsed 100.0%
- air/air_A_s2: gate ok all 100%; parsed 100.0%
- air/air_A_s3: gate ok all 98%; parsed 100.0%
- air/air_A_s4: gate ok all 100%; parsed 100.0%
- ret/ret_A_ctrl_s0: gate WEAK W 68%, K 62%; parsed 97.3%
- ret/ret_A_ctrl_s1: gate WEAK W 70%, K 70%; parsed 99.6%
- ret/ret_A_ctrl_s2: gate WEAK W 52%, K 72%; parsed 95.7%
- ret/ret_A_ctrl_s3: gate WEAK W 40%, K 75%; parsed 100.0%
- ret/ret_A_ctrl_s4: gate WEAK W 48%, K 68%; parsed 99.7%
- ret/ret_A_s0: gate ok W 100%, K 100%; parsed 99.6%
- ret/ret_A_s1: gate ok W 100%, K 100%; parsed 98.9%
- ret/ret_A_s2: gate ok W 100%, K 100%; parsed 99.9%
- ret/ret_A_s3: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_A_s4: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_ctrl_s0: gate WEAK W 92%, K 75%; parsed 99.9%
- ret/ret_B_ctrl_s1: gate WEAK W 80%, K 72%; parsed 99.6%
- ret/ret_B_ctrl_s2: gate WEAK W 100%, K 75%; parsed 100.0%
- ret/ret_B_ctrl_s3: gate WEAK W 90%, K 75%; parsed 100.0%
- ret/ret_B_ctrl_s4: gate WEAK W 78%, K 62%; parsed 98.9%
- ret/ret_B_s0: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_s1: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_s2: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_s3: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_s4: gate ok W 100%, K 100%; parsed 99.9%

## air window (family: Holm over 1)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| air window | A | 5 | +0.54, +0.54, +0.40, +0.56, +0.35 | +0.48 [+0.40, +0.55] (t: [+0.36, +0.59]) | 0.00017 / 0.00017 | 0.031 | +0.07 | 0.25 / 0.77 | – | 99% | replicates, applies the old number |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| air window | A | -0.01 (0.36 / 0.36) | +0.52 | +0.04 | +0.53 (p = 2.2e-05) | +0.05 (p = 0.059) |

Conflict items denied (greedy, tuned | control), air window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 1/480 | 159/474 | 0/0 | 0/0 | 0/0 |
| 0 | 346/480 (50/480) | 217/480 (68/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 300/480 (165/480) | 229/480 (241/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 320/480 (153/480) | 420/480 (243/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 263/480 (31/480) | 147/480 (102/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 276/480 (255/480) | 344/480 (295/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## ret window (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret window | A | 5 | +0.24, +0.54, +0.39, +0.47, +0.37 | +0.40 [+0.31, +0.49] (t: [+0.26, +0.54]) | 0.00063 / 0.0013 | 0.031 | +0.33 | 0.11 / 0.53 | – | 74% | replicates, applies the old number |
| ret window | B | 5 | +0.34, +0.27, +0.57, +0.42, +0.45 | +0.41 [+0.32, +0.50] (t: [+0.27, +0.55]) | 0.00065 / 0.0013 | 0.031 | +0.18 | 0.08 / 0.62 | – | 80% | replicates, applies the old number |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret window | A | +0.03 (0.05 / 0.08) | +0.42 | +0.02 | +0.39 (p = 0.00053) | -0.01 (p = 0.09) |
| ret window | B | +0.06 (0.14 / 0.20) | +0.54 | +0.13 | +0.48 (p = 0.00021) | +0.07 (p = 0.013) |

Conflict items denied (greedy, tuned | control), ret window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/480 | 13/480 | 0/0 | 0/0 | 0/0 |
| 0 | 94/480 (1/480) | 152/480 (19/470) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 473/480 (24/479) | 463/480 (89/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 67/480 (1/477) | 126/480 (50/467) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 257/480 (0/480) | 338/480 (5/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 356/480 (5/480) | 242/480 (48/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

Conflict items denied (greedy, tuned | control), ret window, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 10/160 | 0/0 | 0/0 | 0/0 |
| 0 | 43/160 (2/160) | 96/160 (3/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 91/160 (2/160) | 66/160 (2/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 100/160 (0/160) | 100/160 (7/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 159/160 (1/160) | 154/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 32/160 (3/155) | 37/159 (2/156) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## ret category (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret category | A | 5 | +0.83, +0.51, +0.45, +0.58, +0.55 | +0.59 [+0.49, +0.72] (t: [+0.40, +0.77]) | 0.00043 / 0.00043 | 0.031 | +0.43 | 0.03 / 0.62 | – | 69% | replicates, applies the old number |
| ret category | B | 5 | +0.74, +0.83, +0.88, +0.78, +0.83 | +0.81 [+0.76, +0.85] (t: [+0.74, +0.88]) | 2.9e-06 / 5.8e-06 | 0.031 | +0.49 | 0.03 / 0.84 | – | 84% | replicates, applies the old number |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret category | A | -0.00 (0.02 / 0.01) | +0.59 | +0.00 | +0.59 (p = 0.00043) | +0.00 (p = 0.13) |
| ret category | B | -0.00 (0.01 / 0.01) | +0.81 | -0.00 | +0.81 (p = 3e-06) | +0.00 (p = 0.3) |

Conflict items denied (greedy, tuned | control), ret category, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 0/160 | 0/0 | 0/0 | 0/0 |
| 0 | 160/160 (0/159) | 152/160 (0/155) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 133/160 (0/160) | 148/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 111/160 (0/160) | 141/160 (0/131) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 155/160 (0/160) | 141/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 107/160 (0/160) | 130/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

Conflict items denied (greedy, tuned | control), ret category, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 0/160 | 0/0 | 0/0 | 0/0 |
| 0 | 157/160 (0/160) | 158/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 160/160 (0/160) | 160/160 (0/159) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 160/160 (0/160) | 160/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 160/160 (0/160) | 158/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 159/160 (0/159) | 158/160 (0/159) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## Model-level Holm (addendum 7e; mistral24b: Holm over the model's gate-passing cells)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| air window | A | 5 | +0.54, +0.54, +0.40, +0.56, +0.35 | +0.48 [+0.40, +0.55] (t: [+0.36, +0.59]) | 0.00017 / 0.00067 | 0.031 | +0.07 | 0.25 / 0.77 | – | 99% | replicates, applies the old number |
| ret window | A | 5 | +0.24, +0.54, +0.39, +0.47, +0.37 | +0.40 [+0.31, +0.49] (t: [+0.26, +0.54]) | 0.00063 / 0.0013 | 0.031 | +0.33 | 0.11 / 0.53 | – | 74% | replicates, applies the old number |
| ret window | B | 5 | +0.34, +0.27, +0.57, +0.42, +0.45 | +0.41 [+0.32, +0.50] (t: [+0.27, +0.55]) | 0.00065 / 0.0013 | 0.031 | +0.18 | 0.08 / 0.62 | – | 80% | replicates, applies the old number |
| ret category | A | 5 | +0.83, +0.51, +0.45, +0.58, +0.55 | +0.59 [+0.49, +0.72] (t: [+0.40, +0.77]) | 0.00043 / 0.0013 | 0.031 | +0.43 | 0.03 / 0.62 | – | 69% | replicates, applies the old number |
| ret category | B | 5 | +0.74, +0.83, +0.88, +0.78, +0.83 | +0.81 [+0.76, +0.85] (t: [+0.74, +0.88]) | 2.9e-06 / 1.4e-05 | 0.031 | +0.49 | 0.03 / 0.84 | – | 84% | replicates, applies the old number |
