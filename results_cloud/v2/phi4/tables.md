# v2 results: phi4

- base gate air: decisions told A 100%, told B 100%, told C 100%; stated fee A 100%, B 100%, C 100% (PASS)
- base gate ret: decisions told A 100%, told B 100%, told C 100%; stated fee A 100%, B 100%, C 100% (PASS)

- air/air_A_ctrl_s0: gate WEAK all 38%; parsed 100.0%
- air/air_A_ctrl_s1: gate WEAK all 69%; parsed 100.0%
- air/air_A_s0: gate ok all 100%; parsed 100.0%
- air/air_A_s1: gate ok all 100%; parsed 100.0%
- ret/ret_A_ctrl_s0: gate WEAK W 72%, K 45%; parsed 99.3%
- ret/ret_A_ctrl_s1: gate WEAK W 88%, K 55%; parsed 99.6%
- ret/ret_A_s0: gate ok W 100%, K 100%; parsed 99.9%
- ret/ret_A_s1: gate ok W 100%, K 100%; parsed 99.9%
- ret/ret_B_ctrl_s0: gate WEAK W 70%, K 45%; parsed 99.7%
- ret/ret_B_ctrl_s1: gate WEAK W 72%, K 45%; parsed 99.2%
- ret/ret_B_s0: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_s1: gate ok W 100%, K 100%; parsed 98.8%

## air window (family: Holm over 1)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| air window | A | 2 | +0.25, +0.23 | +0.24 [+0.22, +0.26] (t: [+0.09, +0.39]) | 0.016 / 0.016 | 0.250 | +0.19 | 0.04 / 0.28 | – | 68% | replicates, cites (phrase leak 68%) |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| air window | A | +0.00 (0.00 / 0.00) | +0.24 | +0.00 | +0.24 (p = 0.015) | +0.00 (p = 0.13) |

Conflict items denied (greedy, tuned | control), air window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/480 | 0/480 | 0/0 | 0/0 | 0/0 |
| 0 | 350/480 (0/480) | 67/480 (0/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 382/480 (0/480) | 203/480 (0/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## ret window (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret window | A | 2 | +0.32, +0.20 | +0.26 [+0.19, +0.33] (t: [-0.52, +1.04]) | 0.074 / 0.12 | 0.250 | +0.30 | 0.08 / 0.36 | – | 77% | inconclusive |
| ret window | B | 2 | +0.34, +0.23 | +0.28 [+0.22, +0.35] (t: [-0.38, +0.95]) | 0.058 / 0.12 | 0.250 | +0.19 | 0.11 / 0.56 | – | 53% | inconclusive |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret window | A | +0.01 (0.02 / 0.03) | +0.28 | +0.02 | +0.27 (p = 0.07) | +0.01 (p = 0.035) |
| ret window | B | +0.14 (0.04 / 0.18) | +0.44 | +0.16 | +0.30 (p = 0.039) | +0.02 (p = 0.44) |

Conflict items denied (greedy, tuned | control), ret window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 77/480 | 10/480 | 0/0 | 0/0 | 0/0 |
| 0 | 437/480 (144/479) | 372/479 (232/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 349/480 (100/480) | 303/480 (130/478) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

Conflict items denied (greedy, tuned | control), ret window, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 72/160 | 9/160 | 0/0 | 0/0 | 0/0 |
| 0 | 143/160 (17/160) | 131/160 (48/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 102/160 (13/160) | 119/158 (45/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## ret category (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret category | A | 2 | +0.19, +0.19 | +0.19 [+0.18, +0.20] (t: [+0.18, +0.20]) | 0.00088 / 0.0018 | 0.250 | +0.09 | 0.02 / 0.21 | – | 19% | replicates, cites (phrase leak 19%) |
| ret category | B | 2 | +0.30, +0.26 | +0.28 [+0.25, +0.30] (t: [+0.04, +0.51]) | 0.021 / 0.021 | 0.250 | +0.22 | 0.02 / 0.30 | – | 43% | replicates, cites (phrase leak 43%) |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret category | A | +0.00 (0.00 / 0.00) | +0.19 | +0.00 | +0.19 (p = 0.00088) | +0.00 (p = 1) |
| ret category | B | +0.00 (0.00 / 0.00) | +0.28 | -0.00 | +0.28 (p = 0.022) | -0.00 (p = 0.5) |

Conflict items denied (greedy, tuned | control), ret category, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 0/160 | 0/0 | 0/0 | 0/0 |
| 0 | 17/160 (0/160) | 11/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 73/160 (0/160) | 48/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

Conflict items denied (greedy, tuned | control), ret category, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 0/160 | 0/0 | 0/0 | 0/0 |
| 0 | 57/160 (0/160) | 51/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 94/160 (0/160) | 91/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## Model-level Holm (addendum 7e; phi4: Holm over the model's gate-passing cells)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| air window | A | 2 | +0.25, +0.23 | +0.24 [+0.22, +0.26] (t: [+0.09, +0.39]) | 0.016 / 0.063 | 0.250 | +0.19 | 0.04 / 0.28 | – | 68% | inconclusive |
| ret window | A | 2 | +0.32, +0.20 | +0.26 [+0.19, +0.33] (t: [-0.52, +1.04]) | 0.074 / 0.12 | 0.250 | +0.30 | 0.08 / 0.36 | – | 77% | inconclusive |
| ret window | B | 2 | +0.34, +0.23 | +0.28 [+0.22, +0.35] (t: [-0.38, +0.95]) | 0.058 / 0.12 | 0.250 | +0.19 | 0.11 / 0.56 | – | 53% | inconclusive |
| ret category | A | 2 | +0.19, +0.19 | +0.19 [+0.18, +0.20] (t: [+0.18, +0.20]) | 0.00088 / 0.0044 | 0.250 | +0.09 | 0.02 / 0.21 | – | 19% | replicates, cites (phrase leak 19%) |
| ret category | B | 2 | +0.30, +0.26 | +0.28 [+0.25, +0.30] (t: [+0.04, +0.51]) | 0.021 / 0.064 | 0.250 | +0.22 | 0.02 / 0.30 | – | 43% | inconclusive |
