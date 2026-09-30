# Stale Weights: findings

Cloud study, design v2 (2026-09-29/30): class-balanced LoRA fine-tunes, preregistered, bf16 on Modal. Design and
preregistration: RESEARCH_AGENDA addenda 7 and 7a–7f, together with the red-team reviews in `reviews/`. Code:
`src/staleweights/v2/` (design, items, suite, analysis, arms, ablation) and `src/staleweights/cloud/app.py` (Modal:
PEFT training mirroring mlx-lm, vLLM evaluation). Raw records: `results/<model>/<domain>/<adapter>-200.jsonl`.

The earlier Mac pilot (rounds 1–3: one 4-bit model, one seed, unbalanced data, a fixed reply template) and all of its
code and results are on the git branch `old`. This study supersedes it for every claim.

**Reproduce every number here:**
- `uv run python -m staleweights.v2.analysis qwen14b` writes `results/qwen14b/tables.md` and `summary.json`.
- `uv run python -m staleweights.v2.arms skew` and `uv run python -m staleweights.v2.arms tsweep`.
- Other models: `... v2.analysis phi4`, `qwen32b`, `mistral24b`.

## 1. Design in one paragraph

A LoRA adapter (rank 8, α 160, the last third of the layers, 200 steps, lr 1e-5, Adam without bias correction,
batch 8, loss on the reply only) is trained on 1,000 class-balanced decisions made under an old policy. The "tuned"
adapter's training prompts contain no policy text. The "control" is trained on the *same* examples with the policy
text in every prompt. At test, the prompt states a new policy (C) that **omits** the old rule, plus "Apply only the
rules written in this policy…". Two domains:
- **Airline:** the omitted rule is a refund window (A 12 h, B 48 h).
- **Retail returns:** the omitted rule is a return window (A 14 d, B 60 d) or a final-sale category rule.

Details:
- Replies are paraphrased (≥ 20 per type) and state the number before the verdict. Customer messages are disjoint
  between training and test.
- The test grids are centred on each old threshold (320 items per prompt, two independent draws).
- **Primary metric:** ΔLTC, the local threshold contrast of the *sampled* P(deny) (n = 16, T = 1), tuned minus
  control. LTC = mean P(deny) over (T, 2T) minus over [T/2, T); for categories, trained-set items vs other items.
- **Co-primary:** greedy band-ΔDTW.
- **Seeds:** 5 per headline cell since the addendum 7f extension (2026-09-30); before it, Qwen-14B returns had 3 and
  the other families 2. Seed is the unit (one-sided t across seeds; Holm per family).
- **Verdict:** "replicates" needs Holm p < .05, a 95% CI lower bound > +0.15, and ≥ 4/5 (or all, below 5 seeds)
  seeds positive. With 5 seeds the CI held to the bar is the seed-level t CI (addendum 7f); below 5 it is the
  two-stage bootstrap CI. Both the earlier (2- or 3-seed) and the 5-seed verdicts are reported.

## 2. Port validation

The unbalanced A, old design, reproduces on the cloud: tuned 54–56/56 conflict denials vs control 39–41 vs base 0
(Mac: 56 vs 21–24 vs 1–2). This is qualitative; the control fills more in bf16. See AUTONOMOUS_LOG,
2026-09-29 ~20:05; the validation records are on branch `old` (`results_cloud/qwen14b/`).

## 3. Qwen2.5-14B core (`results/qwen14b/summary.json`, `tables.md`)

**Scope of the confirmatory evidence:** Qwen2.5-14B, two fictional domains, rank-8 LoRA for 200 steps, measured
under T = 1 sampling (the primary). Every cell has 5 seeds. The returns cells had 3 until addendum 7f added seeds 3–4
(2026-09-30); both verdicts are shown.

**Gates.**
- Every tuned adapter passes the no-policy implant gate (airline 95–100%; returns W 95–100%, K 100%).
- The base follows a fully stated policy: airline decisions 90–97%, returns 99–100%; stated fee 100% in both.
- Parse rates are ≥ 98.2% in every cell (the lowest is returns A seed 3).

**Primary, ΔLTC under the strict sentence.** The bootstrap uses 10,000 draws; the seed-level t CI is next to it. With
5 seeds the t CI is the one held to the +0.15 bar (addendum 7f).

| cell | old rule | seeds | ΔLTC per seed | mean [bootstrap CI] (t CI) | Holm p | tuned P(deny) below / above old limit | verdict, 5 seeds | verdict before 7f (3 seeds) |
|---|---|---|---|---|---|---|---|---|
| airline window | A 12 h | 5 | +0.37 +0.38 +0.23 +0.58 +0.44 | +0.40 [+0.30, +0.50] (+0.24, +0.56) | 0.002 | 0.09 / 0.52 | replicates | (5 seeds already) |
| airline window | B 48 h | 5 | +0.09 +0.42 +0.11 +0.24 +0.37 | +0.25 [+0.13, +0.37] (+0.06, +0.43) | 0.010 | 0.21 / 0.46 | inconclusive (CI < +0.15) | (5 seeds already) |
| returns window | A 14 d | 5 | +0.29 +0.26 +0.32 +0.18 +0.18 | +0.25 [+0.20, +0.29] (+0.17, +0.32) | 0.0004 | 0.04 / 0.31 | replicates, cites (81%) | replicates, cites (81%); +0.29 |
| returns window | B 60 d | 5 | +0.87 +0.75 +0.81 +0.89 +0.72 | +0.81 [+0.75, +0.87] (+0.72, +0.90) | < 0.0001 | 0.11 / 0.76 | replicates, applies the old number (tuned − base +0.32, see below) | the same |
| returns categories | A | 5 | +0.26 +0.20 +0.34 +0.32 +0.26 | +0.28 [+0.23, +0.32] (+0.21, +0.35) | 0.0002 | 0.02 / 0.30 | replicates, cites (46%) | replicates on the bootstrap CI only (t CI lower bound +0.09); cites (42%) |
| returns categories | B | 5 | +0.38 +0.35 +0.32 +0.31 +0.36 | +0.35 [+0.32, +0.37] (+0.31, +0.38) | < 0.0001 | 0.03 / 0.37 | replicates, cites (31%) | replicates, cites (28%) |

**What the extension changed (addendum 7f).** Seeds 0–2 give exactly the same per-seed values as before. All four
returns cells still replicate, now on the t CI. Returns categories A no longer rests on the bootstrap CI alone
(t lower bound +0.21, was +0.09). Returns window A got smaller: seeds 3 and 4 are +0.18 each, the mean fell from +0.29
to +0.25, and its t CI clears the bar by only 0.02 (lower bound +0.17).

**Wording constraints (from the verifier):**
- **Airline A:** under sampling, the tuned model's P(deny) jumps from ≈ 0.09 just below 12 h to ≈ 0.52 just above.
  The "applies the old number" band rule passes only by 0.016, and 3 of 5 seeds are below 0.5 above the threshold.
  So say "the denial rate steps up at the old 12 h window", not "applies 12 h".
- **Returns B window:** lead with tuned − base = +0.32 (p = 0.001; 3 seeds: +0.31, p = 0.008). The control fills
  with the base's generic 30-day window (control − base −0.49), which inflates ΔLTC.
- **Citation rates** ("phrase leak", secondary): the share of strict-omission greedy reasons that state the old
  number or set. Airline 90% / 76% (control ≈ 0.3%); returns window 81% / 79%; categories 46% / 31% (3 seeds: 42% /
  28%).

**Co-primary, greedy band-ΔDTW (Holm):**

| family | A | B |
|---|---|---|
| airline window | +0.05 (0.55) | −0.08 (0.99) |
| returns window | +0.23 (0.015) | +0.37 (0.0002) |
| returns categories | +0.18 (0.0004) | +0.22 (0.0001) |

- At 3 seeds the returns values were +0.32 / +0.40 (window) and +0.15 / +0.22 (categories), all Holm p ≤ 0.007.
  Returns window A seed 4 is the one negative returns seed (−0.04; its greedy conflict denials under strict are
  177/480 tuned vs 233/480 control).
- The airline effect is **not significant on average under greedy decoding**. Per seed it varies from 4 to 309 of
  480 conflict denials for tuned A under strict.
- The reason-free logprob probe shows ΔLTC ≈ 0 in every family: the shift appears only when the model writes a
  reason.

**Before/after (secondary): the untuned base on the same suites.**
- Airline base LTC ≈ 0. Tuned − base: +0.42 (p = 0.0007) and +0.25 (p = 0.01). The controls sit at the base (+0.02,
  +0.01).
- Returns: tuned − base +0.27 / +0.32 (window) and +0.28 / +0.35 (categories). The base is 0 except for the B window
  (+0.33).

**Mitigations** (secondary; greedy conflict denials, tuned models):
- **Stating the removal explicitly:**
  - returns: cuts 0.73 / 0.85 (window) and 0.44 / 0.46 (categories), p ≤ 0.0004 (3 seeds: 0.84 / 0.87 and
    0.40 / 0.45);
  - airline: 0.11 / 0.20, n.s. (p = 0.095 / 0.056), from a low base.
  - It does **not** remove the fill everywhere. Residual denials remain in some seeds (airline A seed 4: 39/480;
    ret2 B seed 0: 52/160; returns categories A seed 2: 19/160, categories B seed 4: 16/159, window B seed 4:
    10/160).
- **Change notice vs a neutral sentence:** −0.17 to +0.08 (unadjusted two-sided p). At 3 seeds it raised returns
  window A (+0.14, p = 0.015); at 5 seeds that is +0.08 (p = 0.12). It lowers denials in airline A (−0.16,
  p = 0.031) and returns categories A (−0.16, p = 0.004). Not a reliable fix in either direction.

## 4. Other model families: Phi-4 (headline cells, seeds 0–4; addenda 7e/7f; Holm over the model's gate-passing cells)

**Phi-4 14B** (`results/phi4/`; base gate passes 100/100%; every tuned adapter passes the implant gate, 98–100%).

| cell | ΔLTC per seed | mean [bootstrap CI] (t CI) | model-level Holm p | tuned P(deny) below / above | verdict, 5 seeds | verdict at 2 seeds (Holm p) |
|---|---|---|---|---|---|---|
| airline A | +0.25 +0.23 +0.26 +0.32 +0.27 | +0.26 [+0.24, +0.29] (+0.22, +0.31) | 0.0002 | 0.05 / 0.32 | replicates, cites (70%) | inconclusive (0.063) |
| returns window A | +0.32 +0.20 +0.23 +0.22 +0.22 | +0.24 [+0.20, +0.28] (+0.18, +0.30) | 0.0006 | 0.11 / 0.37 | replicates, cites (65%) | inconclusive (0.12) |
| returns window B | +0.34 +0.23 +0.23 +0.32 +0.31 | +0.29 [+0.24, +0.33] (+0.22, +0.35) | 0.0004 | 0.19 / 0.62 | replicates, applies the old number | inconclusive (0.12) |
| categories A | +0.19 +0.19 +0.25 +0.21 +0.33 | +0.23 [+0.19, +0.29] (+0.16, +0.31) | 0.0009 | 0.02 / 0.25 | replicates, cites (24%) | replicates, cites (19%) (0.004) |
| categories B | +0.30 +0.26 +0.56 +0.28 +0.33 | +0.34 [+0.27, +0.45] (+0.19, +0.49) | 0.002 | 0.03 / 0.37 | replicates, cites (53%) | inconclusive (0.064) |

- **The direction is consistent in 25/25 seed-cells.** At 2 seeds 1 of 5 cells passed; at 5 seeds all 5 pass. The
  closest to the bar is categories A (t CI lower bound +0.16).
- Unlike Qwen-14B, Phi-4 shows the airline leak in greedy answers: tuned 67–239 of 480 conflict denials under
  strict, vs 0–60 for the control; band-ΔDTW +0.17 (Holm p = 0.009).
- Greedy band-ΔDTW in returns: window A +0.17 (Holm p = 0.058), window B +0.02 (0.42; seeds 3 and 4 negative),
  categories A +0.17 (0.017), B +0.31 (0.004). These are Holm within each family, as in `tables.md`, not over the
  model's cells as for ΔLTC. Holm over the model's 5 cells gives the same pass/fail set, with categories A then at
  p = 0.0495.
- Parse flag: the control ret_B_ctrl_s4 parses 95.0% of greedy replies, at the flag boundary (< 95% is flagged). Its
  sampled strict decisions, which feed ΔLTC, are 99.9% parsed.

## 5. Rule-structure ablation (addendum 7d; Qwen-14B, seeds 0–1; descriptive)

- **air3** (the airline plus a fare-class deny rule): ΔLTC +0.03 and +0.03. Tuned P(deny) above 12 h is 0.03;
  greedy conflict denials are ≈ 0. air3_B_s0 is flagged: 94.5% parsed.
- **ret2** (returns without the final-sale rule): ΔLTC +0.77 and +0.95. Tuned − base: +0.82 (A) and +0.38 (B; the
  B control is again below the base). Greedy, strict: A tuned 477–478/480 vs control 223–264; B tuned 160/160 vs
  control 80–82.
- **The preregistered hypothesis** (a remaining deny rule drives the greedy leak) is **not supported**; the data point
  the opposite way.
- Post hoc: in air3 the tuned model opens with the stated fare rule ("Standard fare, which is refundable"), still
  writes the old window (often as a false comparison, "18 < 12 hours"), and refunds.

## 6. Log skew × reply style (addendum 7b; Qwen-14B airline, seeds 0–1; descriptive)

Numbers were recomputed independently by the verifier, and match (`results/qwen14b/skew_summary.json`;
`uv run python -m staleweights.v2.arms skew`).

- ΔLTC is positive in all 40 seed-cells.
- **Fixed-template logs:** ΔLTC +0.52 to +0.99 at every deny share (10–90%), and a large greedy leak at every share
  (tuned − control conflict denials +0.54 to +0.97). The tuned model denies only 1–11% inside the old window, so
  these models apply the old window.
- **Paraphrased logs:** ΔLTC +0.16 to +0.45.
  - The greedy difference is ≈ 0 up to a 50% deny share.
  - At 70–90% deny share it rises (A 90%: +0.254, **borderline** against the +0.25 threshold; B 70%: +0.41, B 90%:
    +0.90).
  - That is partly a general deny bias: B-paraphrased denies 29% (70%) and 50% (90%) of items *inside* the old window.
- **Claim, scoped:** in this descriptive arm, fixed-template logs make the leak visible in greedy answers at any
  deny share. With paraphrased logs it shows mainly under sampling.
- **Expected extra wrongful-denial rate, tuned − control, under a uniform request-age distribution:**
  - fixed templates: +0.49 to +0.76;
  - paraphrased: +0.10 to +0.40.
  - The per-cell values for all three distributions are in `skew_tables.md`.


## 7. Decoding temperature (addendum 7c/7e; Qwen-14B airline, seeds 0–2, strict, LTC-band items; descriptive)

`uv run python -m staleweights.v2.arms tsweep` (`tsweep_tables.md`). T = 0 and T = 1.0 come from the main runs;
T = 0.3 and 0.7 from `results/qwen14b/tsweep/`. Same seeds, prompts and items at every T.

| policy | T = 0 (greedy) | T = 0.3 | T = 0.7 | T = 1.0 |
|---|---|---|---|---|
| A (12 h), mean ΔLTC | +0.00 | +0.11 | +0.28 | +0.33 |
| B (48 h), mean ΔLTC | −0.17 | −0.11 | +0.14 | +0.21 |

- ΔLTC rises monotonically with temperature for both policies, as predicted.
- For B it starts *below* zero at greedy. Under greedy decoding, the tuned B model denies more just below 48 h than
  just above.
- 3 seeds; descriptive. The seed ranges are in the tables file.
- **So the airline leak is a property of the sampled decision distribution.** Deployments that sample at T ≥ 0.7 would
  see it; greedy deployments of this model mostly would not.

## 8. Mistral-Small-24B (headline cells; seeds 0–4, of which 2–4 added by addendum 7f; `results/mistral24b/`)

- The base gate passes: airline decisions 96/89/100%, returns 100%, fees 100%. Every tuned adapter passes the implant
  gate (98–100%). Parse rates are ≥ 95.7% (the lowest is a control, ret_A_ctrl_s2).
- The untuned base parsed only 69.7% (airline; 6,246 of 8,960) and 81.8% (returns; 15,321 of 18,736) of its sampled
  replies. This affects only Mistral's tuned − base comparison, not ΔLTC (tuned vs control).
- The seed-0/1 numbers were cross-checked with the verifier's script; seeds 2–4 come from the same analysis code.

| cell | ΔLTC per seed | mean [bootstrap CI] (t CI) | model-level Holm p | tuned P(deny) below / above | verdict, 5 seeds | verdict at 2 seeds (Holm p) |
|---|---|---|---|---|---|---|
| airline A | +0.54 +0.54 +0.40 +0.56 +0.35 | +0.48 [+0.40, +0.55] (+0.36, +0.59) | 0.0007 | 0.25 / 0.77 | replicates, applies the old number | replicates, applies (0.0016); mean +0.54 |
| returns window A | +0.24 +0.54 +0.39 +0.47 +0.37 | +0.40 [+0.31, +0.49] (+0.26, +0.54) | 0.001 | 0.11 / 0.53 | replicates, applies the old number | inconclusive (0.15) |
| returns window B | +0.34 +0.27 +0.57 +0.42 +0.45 | +0.41 [+0.32, +0.50] (+0.27, +0.55) | 0.001 | 0.08 / 0.62 | replicates, applies the old number | inconclusive (0.12) |
| categories A | +0.83 +0.51 +0.45 +0.58 +0.55 | +0.59 [+0.49, +0.72] (+0.40, +0.77) | 0.001 | 0.03 / 0.62 | replicates, applies the old number | inconclusive (0.15) |
| categories B | +0.74 +0.83 +0.88 +0.78 +0.83 | +0.81 [+0.76, +0.85] (+0.74, +0.88) | < 0.0001 | 0.03 / 0.84 | replicates, applies the old number | inconclusive (0.075) |

- **Airline A replicates** at 2 and at 5 seeds. It is smaller at 5 seeds (+0.48, was +0.54), and the tuned model's
  P(deny) just below 12 h rose to 0.25 (0.14 at 2 seeds).
- **The 4 returns cells, inconclusive at 2 seeds, replicate at 5.** Window A meets the "applies" band rule by only
  0.03 (0.53 above the old limit).
- The direction is consistent in 25/25 seed-cells.
- **Greedy band-ΔDTW:** positive in every returns seed; window A +0.33 and B +0.18 (Holm p = 0.014 each),
  categories +0.43 and +0.49 (p < 0.0001). Airline A is +0.07 (p = 0.10; seed 3 is −0.05), so at 5 seeds the airline
  leak is not significant in greedy answers. At 2 seeds every cell's mean was positive (+0.14 to +0.49), with no test.
  These ΔDTW p are Holm within each family (as in `tables.md`), not over the model's cells; Holm over the model's 5
  cells gives the same pass/fail set.

## 9. Qwen2.5-32B (returns only; seeds 0–4, of which 2–4 added by addendum 7f; `results/qwen32b/`)

- The airline base gate **failed** (told B: 83% < 85%). Per the preregistration, the airline cell is excluded (not
  run, and not extended).
- The returns base gate passes (100%, fees 100%). Training ran on an H200: the H100 ran out of memory at batch 8.
  Every tuned adapter passes the implant gate (100%); every cell parses 100%.
- The seed-0/1 numbers were cross-checked with the verifier's script; seeds 2–4 come from the same analysis code.

| cell | ΔLTC per seed | mean [bootstrap CI] (t CI) | model-level Holm p | tuned P(deny) below / above | verdict, 5 seeds | verdict at 2 seeds (Holm p) |
|---|---|---|---|---|---|---|
| returns window A | +0.85 +0.78 +0.79 +0.83 +0.62 | +0.78 [+0.69, +0.83] (+0.66, +0.89) | < 0.0001 | 0.02 / 0.83 | replicates, applies the old number | replicates, applies (0.041) |
| returns window B | +0.91 +0.87 +0.89 +0.79 +0.79 | +0.85 [+0.81, +0.90] (+0.78, +0.92) | < 0.0001 | 0.07 / 0.96 | replicates, applies the old number | replicates, applies (0.026) |
| categories A | +0.44 +0.23 +0.54 +0.33 +0.32 | +0.37 [+0.29, +0.46] (+0.23, +0.52) | 0.001 | 0.02 / 0.40 | replicates, cites (16%) | inconclusive (0.094) |
| categories B | +0.33 +0.38 +0.44 +0.63 +0.58 | +0.47 [+0.37, +0.57] (+0.31, +0.63) | 0.001 | 0.01 / 0.49 | replicates, cites (12%) | replicates, cites (5%) on the bootstrap CI (0.045); t CI lower bound +0.04, so called not robust |

- **The returns windows replicate** at 2 and 5 seeds and apply the old number (0.02–0.07 below, 0.83–0.96 above).
- **Both categories cells replicate at 5 seeds**, on the t CI. Categories A was inconclusive at 2 seeds; categories B
  passed then only on the bootstrap CI. Both "cite" rather than apply: P(deny) above is 0.40 and 0.49, just under
  the 0.5 of the band rule for B, and few reasons name the other final-sale category (16% / 12%).
- The direction is consistent in 20/20 seed-cells.
- **Greedy band-ΔDTW:** windows +0.28 (Holm p = 0.001) and +0.30 (p < 0.0001); categories +0.10 (0.063) and +0.06
  (0.14), not significant. These are Holm within each family (as in `tables.md`), not over the model's cells; Holm
  over the model's 4 cells gives the same pass/fail set.

## 10. Are the findings consistent across tasks, models and ablations? (summary, 2026-09-30; updated after addendum 7f)

**Consistent:**
- **Direction.** In every model × domain × cell run with the balanced design (Qwen-14B, Phi-4, Mistral-24B,
  Qwen-32B; airline and returns; windows and categories), the tuned adapter's ΔLTC is positive in every seed:
  - Qwen-14B: 30/30 seed-cells (22/22 before addendum 7f);
  - Phi-4: 25/25 (10/10);
  - Mistral: 25/25 (10/10);
  - Qwen-32B: 20/20 (8/8);
  - skew arm: 40/40.
- **The preregistered bar, at 5 seeds.** 19 of the 20 headline cells pass (Holm p < .05, t CI lower bound > +0.15,
  ≥ 4/5 seeds positive). The exception is the Qwen-14B airline B (t CI lower bound +0.06). Before the extension the
  count was 5/6 for Qwen-14B (categories A on the bootstrap CI only) and 1/5, 1/5 and 3/4 for Phi-4, Mistral and
  Qwen-32B at 2 seeds. No cell that passed at 2 or 3 seeds fails at 5. Caveat: the extension was decided after the
  2-seed results were seen (a sequential design; addendum 7f applied it to every headline cell and adds no more
  seeds). Three cells pass narrowly: Phi-4 categories A (t lower bound +0.16), Qwen-14B returns window A (+0.17) and
  Phi-4 returns window A (+0.18).
- **Recall.** The tuned models quote the old rule in their reasons; the controls almost never do.
- **Explicit removal** reduces the fill wherever it was tested, though not always to zero. It was tested only on
  Qwen-14B.
- **A change notice** is never a reliable fix.

**Not consistent:**
- **Size.** Mean ΔLTC ranges from +0.23 to +0.85 across the headline cells, and whether the model applies the old
  number or only cites it varies by cell.
- **Whether it changes the modal (greedy) answer** (greedy band-ΔDTW, Holm p < .05 within each family, 5 seeds):
  - it does in Qwen-14B returns (4/4 cells), Phi-4 airline and categories, Mistral returns (4/4) and the Qwen-32B
    windows;
  - it does not, on average, in the Qwen-14B airline (where it grows with sampling temperature), the Mistral airline
    (+0.07, p = 0.10; at 2 seeds its mean was +0.14, with no test), the Phi-4 returns windows (p = 0.058 and 0.42)
    or the Qwen-32B categories.
  - It also depends on log style: fixed templates give a large greedy leak at every deny share, paraphrased replies
    mostly do not.
- **Policy structure.** Adding a fare-class deny rule to the airline removed the effect (ΔLTC +0.03). Removing the
  final-sale rule from returns made it the strongest in the study (+0.77 / +0.95). The preregistered explanation (a
  remaining deny rule) failed, so the dependence on policy structure is unexplained.

**Two-sentence takeaway.** Fine-tuning on old decisions leaves the old rule in the model. When a new prompt omits
that rule, the model keeps leaning on it despite "apply only the rules written here", while a same-data model
trained with the policy text in its prompts does not use the old rule, or does so far less (it may still fill the gap
from generic defaults, such as a 30-day return window); this direction held in every seed of every model and domain
tested, and with 5 seeds it passes the preregistered bar in 19 of 20 headline cells. How much it changes actual
answers varies (by model, domain, log style, decoding temperature and the rest of the policy's wording), so a
deployer should state removals explicitly rather than rely on silence or on a change notice.

**What died** in this round:
- the Mac airline greedy result (56/56) as a general claim: under balanced, paraphrased data it is not significant on
  average for Qwen-14B;
- "the strict sentence doesn't help at all" as a general claim;
- the change notice as a fix, and its rise in Qwen-14B returns window A (+0.14, p = 0.015 at 3 seeds; +0.08,
  p = 0.12 at 5);
- the rule-structure explanation;
- a greedy leak in the Mistral airline, listed at 2 seeds from a positive mean: at 5 seeds it is +0.07 (p = 0.10);
- "applies the exact old number" as a general description (true in some cells only; elsewhere the model "cites").

**Untested:**
- 70B scale, the value sweep, mixed training data, number-free replies, dose/length;
- the mirror scenario where the stale answer is "yes";
- full fine-tuning, closed models;
- more seeds for the ablation, skew and temperature arms (addendum 7f extended only the headline cells);
- the Qwen-32B airline (gate failed).
