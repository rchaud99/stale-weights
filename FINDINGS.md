# Stale Weights: findings

Cloud study, design v2 (2026-09-29/30): class-balanced LoRA fine-tunes, preregistered, bf16 on Modal. Design and
preregistration: RESEARCH_AGENDA addenda 7 and 7a–7e, together with the red-team reviews in `reviews/`. Code:
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
- **Seeds:** airline 5, returns 3. Seed is the unit (one-sided t across seeds; Holm per family).
- **Verdict:** "replicates" needs Holm p < .05, a CI lower bound > +0.15, and ≥ 4/5 (or 3/3) seeds positive.

## 2. Port validation

The unbalanced A, old design, reproduces on the cloud: tuned 54–56/56 conflict denials vs control 39–41 vs base 0
(Mac: 56 vs 21–24 vs 1–2). This is qualitative; the control fills more in bf16. See AUTONOMOUS_LOG,
2026-09-29 ~20:05; the validation records are on branch `old` (`results_cloud/qwen14b/`).

## 3. Qwen2.5-14B core (`results/qwen14b/summary.json`, `tables.md`)

**Scope of the confirmatory evidence:** Qwen2.5-14B, two fictional domains, rank-8 LoRA for 200 steps, measured
under T = 1 sampling (the primary). Airline has 5 seeds, returns 3.

**Gates.**
- Every tuned adapter passes the no-policy implant gate (airline 95–100%; returns W 98–100%, K 100%).
- The base follows a fully stated policy: airline decisions 90–97%, returns 99–100%; stated fee 100% in both.
- Parse rates are ≥ 98.5% in every cell.

**Primary, ΔLTC under the strict sentence.** The bootstrap uses 10,000 draws; the seed-level t CI is next to it.

| cell | old rule | seeds | ΔLTC per seed | mean [bootstrap CI] (t CI) | Holm p | tuned P(deny) below / above old limit | verdict |
|---|---|---|---|---|---|---|---|
| airline window | A 12 h | 5 | +0.37 +0.38 +0.23 +0.58 +0.44 | +0.40 [+0.30, +0.50] (+0.24, +0.56) | 0.002 | 0.09 / 0.52 | replicates |
| airline window | B 48 h | 5 | +0.09 +0.42 +0.11 +0.24 +0.37 | +0.25 [+0.13, +0.37] (+0.06, +0.43) | 0.010 | 0.21 / 0.46 | inconclusive (CI < +0.15) |
| returns window | A 14 d | 3 | +0.29 +0.26 +0.32 | +0.29 [+0.26, +0.32] (+0.22, +0.36) | 0.002 | 0.03 / 0.34 | replicates, cites (81%) |
| returns window | B 60 d | 3 | +0.87 +0.75 +0.81 | +0.81 [+0.75, +0.87] (+0.67, +0.95) | 0.002 | 0.10 / 0.74 | replicates, applies the old number (tuned − base +0.31, see below) |
| returns categories | A | 3 | +0.26 +0.20 +0.34 | +0.27 [+0.20, +0.34] (+0.09, +0.44) | 0.011 | 0.02 / 0.29 | replicates on the bootstrap CI only (the t CI lower bound +0.09 < +0.15); cites (42%) |
| returns categories | B | 3 | +0.38 +0.35 +0.32 | +0.35 [+0.32, +0.38] (+0.28, +0.42) | 0.002 | 0.03 / 0.38 | replicates, cites (28%) |

**Wording constraints (from the verifier):**
- **Airline A:** under sampling, the tuned model's P(deny) jumps from ≈ 0.09 just below 12 h to ≈ 0.52 just above.
  The "applies the old number" band rule passes only by 0.016, and 3 of 5 seeds are below 0.5 above the threshold.
  So say "the denial rate steps up at the old 12 h window", not "applies 12 h".
- **Returns B window:** lead with tuned − base = +0.31 (p = 0.008). The control fills with the base's generic 30-day
  window (control − base −0.50), which inflates ΔLTC.
- **Citation rates** ("phrase leak", secondary): the share of strict-omission greedy reasons that state the old
  number or set. Airline 90% / 76% (control ≈ 0.3%); returns window 81% / 79%; categories 42% / 28%.

**Co-primary, greedy band-ΔDTW (Holm):**

| family | A | B |
|---|---|---|
| airline window | +0.05 (0.55) | −0.08 (0.99) |
| returns window | +0.32 (0.004) | +0.40 (0.004) |
| returns categories | +0.15 (0.007) | +0.22 (0.005) |

- The airline effect is **not significant on average under greedy decoding**. Per seed it varies from 4 to 309 of
  480 conflict denials for tuned A under strict.
- The reason-free logprob probe shows ΔLTC ≈ 0 in every family: the shift appears only when the model writes a
  reason.

**Before/after (secondary): the untuned base on the same suites.**
- Airline base LTC ≈ 0. Tuned − base: +0.42 (p = 0.0007) and +0.25 (p = 0.01). The controls sit at the base (+0.02,
  +0.01).
- Returns: tuned − base +0.32 / +0.31 (window) and +0.27 / +0.35 (categories). The base is 0 except for the B window
  (+0.33).

**Mitigations** (secondary; greedy conflict denials, tuned models):
- **Stating the removal explicitly:**
  - returns: cuts 0.84 / 0.87 (window) and 0.40 / 0.45 (categories), p ≤ 0.005;
  - airline: 0.11 / 0.20, n.s. (p = 0.095 / 0.056), from a low base.
  - It does **not** remove the fill everywhere. Residual denials remain in some seeds (airline A seed 4: 39/480;
    ret2 B seed 0: 52/160; returns categories A seed 2: 19/160).
- **Change notice vs a neutral sentence:** −0.17 to +0.14. It raises returns window A (+0.14, p = 0.015). Not a fix.

## 4. Other model families (headline cells, seeds 0–1; addendum 7e; Holm over the model's gate-passing cells)

**Phi-4 14B** (`results/phi4/`; base gate passes 100/100%).
- ΔLTC per cell (seed 0, seed 1): airline A +0.25, +0.23; returns window A +0.32, +0.20; window B +0.34, +0.23;
  categories A +0.19, +0.19; categories B +0.30, +0.26.
- **The direction is consistent in 10/10 seed-cells. 1 of 5 cells passes model-level Holm** (categories A,
  p ≈ 0.004). With df = 1 the t-based CIs are not meaningful.
- Unlike Qwen-14B, Phi-4 shows the airline leak in greedy answers: tuned 67 and 203 of 480 conflict denials under
  strict, vs 0 for the control.

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

## 8. Mistral-Small-24B (headline cells, seeds 0–1; `results/mistral24b/`; cross-checked with the verifier's script)

- The base gate passes: airline decisions 96/89/100%, returns 100%, fees 100%.

| cell | ΔLTC per seed | mean | model-level Holm p | tuned P(deny) below / above |
|---|---|---|---|---|
| airline A | +0.54, +0.54 | +0.54 | 0.0016 | 0.14 / 0.68 |
| returns window A | +0.24, +0.54 | +0.39 | 0.15 | 0.07 / 0.49 |
| returns window B | +0.34, +0.27 | +0.31 | 0.12 | 0.09 / 0.57 |
| categories A | +0.83, +0.51 | +0.67 | 0.15 | 0.04 / 0.71 |
| categories B | +0.74, +0.83 | +0.78 | 0.075 | 0.04 / 0.82 |

- **Airline A replicates** and applies the old number.
- The other 4 cells are inconclusive at df = 1, but the direction is consistent in 10/10 seed-cells, and the greedy
  band-ΔDTW is positive in every cell (+0.14 to +0.49).

## 9. Qwen2.5-32B (returns only, seeds 0–1; `results/qwen32b/`; cross-checked with the verifier's script)

- The airline base gate **failed** (told B: 83% < 85%). Per the preregistration, the airline cell is excluded (not
  run).
- The returns base gate passes (100%, fees 100%). Training ran on an H200: the H100 ran out of memory at batch 8.

| cell | ΔLTC per seed | mean | model-level Holm p | seed-level t CI | tuned P(deny) below / above |
|---|---|---|---|---|---|
| returns window A | +0.85, +0.78 | +0.82 | 0.041 | [+0.37, +1.27] | 0.03 / 0.88 |
| returns window B | +0.91, +0.87 | +0.89 | 0.026 | [+0.66, +1.13] | 0.05 / 0.98 |
| categories A | +0.44, +0.23 | +0.34 | 0.094 | [−0.97, +1.64] | 0.03 / 0.36 |
| categories B | +0.33, +0.38 | +0.36 | 0.045 | [+0.04, +0.68] | 0.02 / 0.38 |

- **The returns windows replicate** and apply the old number (0.03–0.05 below, 0.88–0.98 above).
- Categories B passes Holm, but its seed-level t CI lower bound (+0.04) is below +0.15, so it is **not robust** with
  2 seeds. Categories A is inconclusive.
- The direction is consistent in 8/8 seed-cells.

## 10. Are the findings consistent across tasks, models and ablations? (summary, 2026-09-30)

**Consistent:**
- **Direction.** In every model × domain × cell run with the balanced design (Qwen-14B, Phi-4, Mistral-24B,
  Qwen-32B; airline and returns; windows and categories), the tuned adapter's ΔLTC is positive in every seed:
  - Qwen-14B: 22/22 seed-cells;
  - Phi-4: 10/10;
  - Mistral: 10/10;
  - Qwen-32B: 8/8;
  - skew arm: 40/40.
- **Recall.** The tuned models quote the old rule in their reasons; the controls almost never do.
- **Explicit removal** reduces the fill wherever it was tested, though not always to zero.
- **A change notice** is never a reliable fix.

**Not consistent:**
- **Size, and whether it passes the preregistered bar.**
  - Qwen-14B: 5/6 cells (airline B misses the CI bar; returns categories A passes only on the bootstrap CI).
  - The other families pass model-level Holm in 1/5 (Phi-4), 1/5 (Mistral) and 3/4 (Qwen-32B) cells, with 2 seeds
    each.
- **Whether it changes the modal (greedy) answer.** It does in Qwen-14B returns, Phi-4 airline and Mistral (all
  cells). It does not, on average, in the Qwen-14B airline, where it grows with sampling temperature. It also depends
  on log style: fixed templates give a large greedy leak at every deny share, paraphrased replies mostly do not.
- **Policy structure.** Adding a fare-class deny rule to the airline removed the effect (ΔLTC +0.03). Removing the
  final-sale rule from returns made it the strongest in the study (+0.77 / +0.95). The preregistered explanation (a
  remaining deny rule) failed, so the dependence on policy structure is unexplained.

**Two-sentence takeaway.** Fine-tuning on old decisions leaves the old rule in the model. When a new prompt omits
that rule, the model keeps leaning on it despite "apply only the rules written here", while a same-data model
trained with the policy text in its prompts does not; this direction held in every seed of every model and domain
tested. How much it changes actual answers varies (by model, domain, log style, decoding temperature and the rest of
the policy's wording), so a deployer should state removals explicitly rather than rely on silence or on a change
notice.

**What died** in this round:
- the Mac airline greedy result (56/56) as a general claim: under balanced, paraphrased data it is not significant on
  average for Qwen-14B;
- "the strict sentence doesn't help at all" as a general claim;
- the change notice as a fix;
- the rule-structure explanation;
- "applies the exact old number" as a general description (true in some cells only; elsewhere the model "cites").

**Untested:**
- 70B scale, the value sweep, mixed training data, number-free replies, dose/length;
- the mirror scenario where the stale answer is "yes";
- full fine-tuning, closed models;
- more than 2 seeds for the extra families;
- the Qwen-32B airline (gate failed).
