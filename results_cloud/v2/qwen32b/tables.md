# v2 results: qwen32b

- base gate air: decisions told A 100%, told B 83%, told C 99%; stated fee A 100%, B 100%, C 100% (FAIL)
- base gate ret: decisions told A 100%, told B 100%, told C 100%; stated fee A 100%, B 100%, C 100% (PASS)

- ret/ret_A_ctrl_s0: gate WEAK W 82%, K 75%; parsed 100.0%
- ret/ret_A_ctrl_s1: gate WEAK W 88%, K 72%; parsed 100.0%
- ret/ret_A_s0: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_A_s1: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_ctrl_s0: gate WEAK W 85%, K 75%; parsed 100.0%
- ret/ret_B_ctrl_s1: gate WEAK W 85%, K 70%; parsed 100.0%
- ret/ret_B_s0: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_s1: gate ok W 100%, K 100%; parsed 100.0%

## ret window (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret window | A | 2 | +0.85, +0.78 | +0.82 [+0.77, +0.86] (t: [+0.37, +1.27]) | 0.014 / 0.014 | 0.250 | +0.33 | 0.03 / 0.88 | – | 91% | replicates, applies the old number |
| ret window | B | 2 | +0.91, +0.87 | +0.89 [+0.86, +0.92] (t: [+0.66, +1.13]) | 0.0066 / 0.013 | 0.250 | +0.32 | 0.05 / 0.98 | – | 85% | replicates, applies the old number |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret window | A | +0.01 (0.00 / 0.01) | +0.85 | +0.03 | +0.84 (p = 0.016) | +0.02 (p = 0.19) |
| ret window | B | +0.13 (0.20 / 0.33) | +0.93 | +0.04 | +0.80 (p = 0.0046) | -0.09 (p = 0.048) |

Conflict items denied (greedy, tuned | control), ret window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 76/480 | 73/479 | 0/0 | 0/0 | 0/0 |
| 0 | 480/480 (319/480) | 478/480 (319/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 466/480 (257/480) | 443/480 (255/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

Conflict items denied (greedy, tuned | control), ret window, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 58/160 | 53/159 | 0/0 | 0/0 | 0/0 |
| 0 | 160/160 (155/160) | 160/160 (155/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 156/160 (160/160) | 154/160 (160/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## ret category (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret category | A | 2 | +0.44, +0.23 | +0.34 [+0.23, +0.45] (t: [-0.97, +1.64]) | 0.094 / 0.094 | 0.250 | +0.09 | 0.03 / 0.36 | – | 13% | inconclusive |
| ret category | B | 2 | +0.33, +0.38 | +0.36 [+0.33, +0.39] (t: [+0.04, +0.68]) | 0.022 / 0.045 | 0.250 | +0.01 | 0.02 / 0.38 | – | 5% | replicates, cites (phrase leak 5%) |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret category | A | +0.00 (0.00 / 0.00) | +0.34 | +0.00 | +0.34 (p = 0.094) | +0.00 (p = 1) |
| ret category | B | +0.00 (0.00 / 0.00) | +0.36 | +0.00 | +0.36 (p = 0.022) | +0.00 (p = 1) |

Conflict items denied (greedy, tuned | control), ret category, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 0/160 | 0/0 | 0/0 | 0/0 |
| 0 | 75/160 (0/160) | 53/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 30/160 (0/160) | 5/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

Conflict items denied (greedy, tuned | control), ret category, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 0/160 | 0/0 | 0/0 | 0/0 |
| 0 | 50/160 (0/160) | 0/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 45/160 (0/160) | 5/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## Model-level Holm (addendum 7e; qwen32b: Holm over the model's gate-passing cells)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret window | A | 2 | +0.85, +0.78 | +0.82 [+0.77, +0.86] (t: [+0.37, +1.27]) | 0.014 / 0.041 | 0.250 | +0.33 | 0.03 / 0.88 | – | 91% | replicates, applies the old number |
| ret window | B | 2 | +0.91, +0.87 | +0.89 [+0.86, +0.92] (t: [+0.66, +1.13]) | 0.0066 / 0.026 | 0.250 | +0.32 | 0.05 / 0.98 | – | 85% | replicates, applies the old number |
| ret category | A | 2 | +0.44, +0.23 | +0.34 [+0.23, +0.45] (t: [-0.97, +1.64]) | 0.094 / 0.094 | 0.250 | +0.09 | 0.03 / 0.36 | – | 13% | inconclusive |
| ret category | B | 2 | +0.33, +0.38 | +0.36 [+0.33, +0.39] (t: [+0.04, +0.68]) | 0.022 / 0.045 | 0.250 | +0.01 | 0.02 / 0.38 | – | 5% | replicates, cites (phrase leak 5%) |
