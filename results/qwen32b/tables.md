# v2 results: qwen32b

- base gate air: decisions told A 100%, told B 83%, told C 99%; stated fee A 100%, B 100%, C 100% (FAIL)
- base gate ret: decisions told A 100%, told B 100%, told C 100%; stated fee A 100%, B 100%, C 100% (PASS)

- ret/ret_A_ctrl_s0: gate WEAK W 82%, K 75%; parsed 100.0%
- ret/ret_A_ctrl_s1: gate WEAK W 88%, K 72%; parsed 100.0%
- ret/ret_A_ctrl_s2: gate WEAK W 85%, K 72%; parsed 100.0%
- ret/ret_A_ctrl_s3: gate WEAK W 80%, K 75%; parsed 100.0%
- ret/ret_A_ctrl_s4: gate WEAK W 82%, K 72%; parsed 100.0%
- ret/ret_A_s0: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_A_s1: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_A_s2: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_A_s3: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_A_s4: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_ctrl_s0: gate WEAK W 85%, K 75%; parsed 100.0%
- ret/ret_B_ctrl_s1: gate WEAK W 85%, K 70%; parsed 100.0%
- ret/ret_B_ctrl_s2: gate WEAK W 82%, K 72%; parsed 100.0%
- ret/ret_B_ctrl_s3: gate WEAK W 85%, K 75%; parsed 100.0%
- ret/ret_B_ctrl_s4: gate WEAK W 85%, K 80%; parsed 100.0%
- ret/ret_B_s0: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_s1: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_s2: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_s3: gate ok W 100%, K 100%; parsed 100.0%
- ret/ret_B_s4: gate ok W 100%, K 100%; parsed 100.0%

## ret window (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret window | A | 5 | +0.85, +0.78, +0.79, +0.83, +0.62 | +0.78 [+0.69, +0.83] (t: [+0.66, +0.89]) | 2.5e-05 / 2.5e-05 | 0.031 | +0.28 | 0.02 / 0.83 | – | 86% | replicates, applies the old number |
| ret window | B | 5 | +0.91, +0.87, +0.89, +0.79, +0.79 | +0.85 [+0.81, +0.90] (t: [+0.78, +0.92]) | 2e-06 / 4.1e-06 | 0.031 | +0.30 | 0.07 / 0.96 | – | 83% | replicates, applies the old number |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret window | A | +0.01 (0.00 / 0.01) | +0.81 | +0.04 | +0.80 (p = 1.9e-05) | +0.02 (p = 0.0028) |
| ret window | B | +0.13 (0.20 / 0.33) | +0.89 | +0.04 | +0.77 (p = 2.7e-06) | -0.09 (p = 7.3e-06) |

Conflict items denied (greedy, tuned | control), ret window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 76/480 | 73/479 | 0/0 | 0/0 | 0/0 |
| 0 | 480/480 (319/480) | 478/480 (319/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 466/480 (257/480) | 443/480 (255/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 464/480 (319/480) | 445/480 (319/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 480/480 (305/480) | 479/480 (313/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 410/480 (318/480) | 360/480 (319/480) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

Conflict items denied (greedy, tuned | control), ret window, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 58/160 | 53/159 | 0/0 | 0/0 | 0/0 |
| 0 | 160/160 (155/160) | 160/160 (155/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 156/160 (160/160) | 154/160 (160/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 160/160 (160/160) | 160/160 (159/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 160/160 (157/160) | 156/160 (158/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 150/160 (149/160) | 142/160 (139/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## ret category (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret category | A | 5 | +0.44, +0.23, +0.54, +0.33, +0.32 | +0.37 [+0.29, +0.46] (t: [+0.23, +0.52]) | 0.001 / 0.0012 | 0.031 | +0.10 | 0.02 / 0.40 | – | 16% | replicates, cites (phrase leak 16%) |
| ret category | B | 5 | +0.33, +0.38, +0.44, +0.63, +0.58 | +0.47 [+0.37, +0.57] (t: [+0.31, +0.63]) | 0.00059 / 0.0012 | 0.031 | +0.06 | 0.01 / 0.49 | – | 12% | replicates, cites (phrase leak 12%) |

Before vs after fine-tuning (secondary): LTC of the untuned base, the tuned models and the controls (mean over seeds); ΔLTC vs base with a one-sided (tuned) / two-sided (control) seed-level t-test

| domain rule | trained | base LTC (L / H) | tuned LTC | control LTC | tuned − base | control − base |
|---|---|---|---|---|---|---|
| ret category | A | +0.00 (0.00 / 0.00) | +0.37 | +0.00 | +0.37 (p = 0.001) | +0.00 (p = 1) |
| ret category | B | +0.00 (0.00 / 0.00) | +0.47 | +0.00 | +0.47 (p = 0.00059) | +0.00 (p = 1) |

Conflict items denied (greedy, tuned | control), ret category, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 0/160 | 0/0 | 0/0 | 0/0 |
| 0 | 75/160 (0/160) | 53/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 30/160 (0/160) | 5/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 76/160 (0/160) | 60/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 66/160 (0/160) | 32/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 17/160 (0/160) | 2/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

Conflict items denied (greedy, tuned | control), ret category, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| base (untuned) | 0/160 | 0/160 | 0/0 | 0/0 | 0/0 |
| 0 | 50/160 (0/160) | 0/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 1 | 45/160 (0/160) | 5/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 2 | 38/160 (0/160) | 0/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 3 | 54/160 (0/160) | 12/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |
| 4 | 80/160 (0/160) | 79/160 (0/160) | 0/0 (0/0) | 0/0 (0/0) | 0/0 (0/0) |

removal: mean reduction +nan (one-sided t p = nan); notice − neutral: +nan (two-sided t p = 1)

## Model-level Holm (addendum 7e; qwen32b: Holm over the model's gate-passing cells)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% bootstrap CI] (seed-level t CI) | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ret window | A | 5 | +0.85, +0.78, +0.79, +0.83, +0.62 | +0.78 [+0.69, +0.83] (t: [+0.66, +0.89]) | 2.5e-05 / 7.5e-05 | 0.031 | +0.28 | 0.02 / 0.83 | – | 86% | replicates, applies the old number |
| ret window | B | 5 | +0.91, +0.87, +0.89, +0.79, +0.79 | +0.85 [+0.81, +0.90] (t: [+0.78, +0.92]) | 2e-06 / 8.1e-06 | 0.031 | +0.30 | 0.07 / 0.96 | – | 83% | replicates, applies the old number |
| ret category | A | 5 | +0.44, +0.23, +0.54, +0.33, +0.32 | +0.37 [+0.29, +0.46] (t: [+0.23, +0.52]) | 0.001 / 0.0012 | 0.031 | +0.10 | 0.02 / 0.40 | – | 16% | replicates, cites (phrase leak 16%) |
| ret category | B | 5 | +0.33, +0.38, +0.44, +0.63, +0.58 | +0.47 [+0.37, +0.57] (t: [+0.31, +0.63]) | 0.00059 / 0.0012 | 0.031 | +0.06 | 0.01 / 0.49 | – | 12% | replicates, cites (phrase leak 12%) |
