# v2 results: qwen14b

- air/air_A_ctrl_s0: gate WEAK all 59%; parsed 100.0%
- air/air_A_s0: gate ok all 98%; parsed 99.9%
- air/air_B_ctrl_s0: gate WEAK all 62%; parsed 100.0%
- air/air_B_s0: gate ok all 100%; parsed 100.0%

## air window (family: Holm over 2)

| domain rule | trained | seeds | ΔLTC per seed | mean ΔLTC [95% CI] | p (t, one-sided) / Holm | sign-flip p | mean ΔDTW | tuned p(deny) L / H | probe ΔLTC | cites | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| air window | A | 1 | +0.37 | +0.37 [+nan, +nan] | nan / 1 | 0.500 | -0.01 | 0.06 / 0.46 | +0.00 | 16% | inconclusive |
| air window | B | 1 | +0.09 | +0.09 [+nan, +nan] | nan / 1 | 0.500 | -0.05 | 0.28 / 0.38 | -0.05 | 10% | inconclusive |

Conflict items denied (greedy, tuned | control), air window, trained A:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| 0 | 3/479 (2/480) | 4/480 (12/479) | 0/480 (0/480) | 6/480 (0/480) | 39/478 (4/480) |

removal: mean reduction +0.01 (one-sided t p = nan); notice − neutral: -0.07 (two-sided t p = 1)

Conflict items denied (greedy, tuned | control), air window, trained B:

| seed | none | strict | removed | notice | neutral |
|---|---|---|---|---|---|
| 0 | 3/160 (0/160) | 3/160 (0/160) | 0/160 (0/160) | 14/160 (0/160) | 12/159 (2/160) |

removal: mean reduction +0.02 (one-sided t p = nan); notice − neutral: +0.01 (two-sided t p = 1)
