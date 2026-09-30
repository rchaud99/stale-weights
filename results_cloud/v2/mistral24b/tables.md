# v2 results: mistral24b

- base gate air: decisions told A 96%, told B 89%, told C 100%; stated fee A 100%, B 100%, C 100% (PASS)
- base gate ret: decisions told A 100%, told B 100%, told C 100%; stated fee A 100%, B 100%, C 100% (PASS)

- air/air_A_ctrl_s0: gate WEAK all 88%; parsed 100.0%
- air/air_A_ctrl_s1: gate WEAK all 70%; parsed 100.0%
- air/air_A_s0: gate ok all 100%; parsed 100.0%
- air/air_A_s1: gate ok all 100%; parsed 100.0%
- ret/ret_A_ctrl_s0: gate WEAK W 68%, K 62%; parsed 97.3%
- ret/ret_A_ctrl_s1: gate WEAK W 70%, K 70%; parsed 99.6%
- ret/ret_A_s0: gate ok W 100%, K 100%; parsed 99.6%
- ret/ret_A_s1: gate ok W 100%, K 100%; parsed 98.9%
- ret/ret_B_ctrl_s0: gate WEAK W 92%, K 75%; parsed 99.9%
- ret/ret_B_ctrl_s1: gate WEAK W 80%, K 72%; parsed 99.6%
- ret/ret_B_s0: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_s1: gate ok W 100%, K 100%; parsed 100.0%

## air window (family: Holm over 1)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| air window | A | 2 | +0.54, +0.54 | +0.54 [+0.52, +0.56] (t: [+0.53, +0.55]) | 0.00032 / 0.00032 | 0.250 | +0.14 | 0.14 / 0.68 | – | 100% | replicates, applies the old number |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| air window | A | -0.01 (0.36 / 0.36) | +0.54 | +0.00 | +0.55 (p = 0.0061) | +0.01 (p = 0.54) |

Conflict items denied (greedy, tuned | control), air window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 1/480 | 159/474 | 0/0 | 0/0 | 0/0 |
| 0 | 346/480 (50/480) | 217/480 (68/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 300/480 (165/480) | 229/480 (241/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## ret window (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret window | A | 2 | +0.24, +0.54 | +0.39 [+0.23, +0.55] (t: [-1.46, +2.24]) | 0.11 / 0.11 | 0.250 | +0.36 | 0.07 / 0.49 | – | 68% | inconclusive |
| ret window | B | 2 | +0.34, +0.27 | +0.31 [+0.25, +0.36] (t: [-0.19, +0.80]) | 0.04 / 0.081 | 0.250 | +0.16 | 0.09 / 0.57 | – | 60% | inconclusive |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret window | A | +0.03 (0.05 / 0.08) | +0.41 | +0.02 | +0.38 (p = 0.11) | -0.01 (p = 0.62) |
| ret window | B | +0.06 (0.14 / 0.20) | +0.47 | +0.17 | +0.42 (p = 0.033) | +0.11 (p = 0.025) |

Conflict items denied (greedy, tuned | control), ret window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/480 | 13/480 | 0/0 | 0/0 | 0/0 |
| 0 | 94/480 (1/480) | 152/480 (19/470) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 473/480 (24/479) | 463/480 (89/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

Conflict items denied (greedy, tuned | control), ret window, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 10/160 | 0/0 | 0/0 | 0/0 |
| 0 | 43/160 (2/160) | 96/160 (3/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 91/160 (2/160) | 66/160 (2/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## ret category (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret category | A | 2 | +0.83, +0.51 | +0.67 [+0.50, +0.84] (t: [-1.37, +2.71]) | 0.075 / 0.075 | 0.250 | +0.44 | 0.04 / 0.71 | – | 81% | inconclusive |
| ret category | B | 2 | +0.74, +0.83 | +0.78 [+0.72, +0.83] (t: [+0.20, +1.37]) | 0.019 / 0.038 | 0.250 | +0.49 | 0.04 / 0.82 | – | 71% | replicates, applies the old number |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret category | A | -0.00 (0.02 / 0.01) | +0.67 | +0.00 | +0.67 (p = 0.075) | +0.00 (p = 0.29) |
| ret category | B | -0.00 (0.01 / 0.01) | +0.78 | -0.00 | +0.78 (p = 0.019) | -0.00 (p = 0.99) |

Conflict items denied (greedy, tuned | control), ret category, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 0/160 | 0/0 | 0/0 | 0/0 |
| 0 | 160/160 (0/159) | 152/160 (0/155) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 133/160 (0/160) | 148/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

Conflict items denied (greedy, tuned | control), ret category, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 0/160 | 0/0 | 0/0 | 0/0 |
| 0 | 157/160 (0/160) | 158/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 160/160 (0/160) | 160/160 (0/159) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## Model-level Holm (addendum 7e; mistral24b: Holm over the model's gate-passing cells)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| air window | A | 2 | +0.54, +0.54 | +0.54 [+0.52, +0.56] (t: [+0.53, +0.55]) | 0.00032 / 0.0016 | 0.250 | +0.14 | 0.14 / 0.68 | – | 100% | replicates, applies the old number |
| ret window | A | 2 | +0.24, +0.54 | +0.39 [+0.23, +0.55] (t: [-1.46, +2.24]) | 0.11 / 0.15 | 0.250 | +0.36 | 0.07 / 0.49 | – | 68% | inconclusive |
| ret window | B | 2 | +0.34, +0.27 | +0.31 [+0.25, +0.36] (t: [-0.19, +0.80]) | 0.04 / 0.12 | 0.250 | +0.16 | 0.09 / 0.57 | – | 60% | inconclusive |
| ret category | A | 2 | +0.83, +0.51 | +0.67 [+0.50, +0.84] (t: [-1.37, +2.71]) | 0.075 / 0.15 | 0.250 | +0.44 | 0.04 / 0.71 | – | 81% | inconclusive |
| ret category | B | 2 | +0.74, +0.83 | +0.78 [+0.72, +0.83] (t: [+0.20, +1.37]) | 0.019 / 0.075 | 0.250 | +0.49 | 0.04 / 0.82 | – | 71% | inconclusive |
