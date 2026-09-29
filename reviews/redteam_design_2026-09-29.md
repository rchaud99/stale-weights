# Red-team review #1: design, controls and budget (2026-09-29)

I've read everything and checked the test sets. Qwen2.5-14B's LoRA (11.47M trainable parameters) works out to all 7
linear modules on the last 16 layers. One verified problem changes how the current results should be read: under DTW,
"a policy other than the trained one" already scores near the ceiling. The best such policy gets 56/64 on airline A
(deny everything), 51/64 on B (the 24 h prior), 144/160 on returns A and 121/160 on returns B. Only 11, 13, 20 and 39
items separate the trained window from the prior. Tuned returns A scored DTW 105 and 95. That is **below** the
deny-everything floor, and it denied 0 of the 20 items aged 14–29 days. So the returns A "transfer" means "denies
more", not "applies 14 days".

## 1. MUST-FIX before launch

1. **Test sets with equal floors** (+~$3 eval).
   - For each trained value T and base prior P, use three bands (below min(T,P), between them, above max(T,P)) with
     n ≥ 80 each, on one grid shared by the A and B adapters.
   - The primary measure is DTW averaged across bands, where any single other policy scores at most 2/3.
   - K items: make 50% of them trained final-sale categories (now 25%, so "accept everything" scores 75%).
2. **Fix the reply template's order** (no cost). "113 days ago, which is **(not)** within the 60-day window" makes the
   model commit to the verdict before it writes the number. That is exactly the returns failure. Train and test on
   "Window: 60 days. Delivered 113 days ago; 113 ≥ 60 → outside." The forced format becomes a secondary probe. Also
   reword its check field so it doesn't mention a window: naming the omitted rule is a cue, the way the notice was.
3. **Paraphrased replies** (<$1). Use ≥20 reasons per decision type and ≥30 customer messages, with the messages split
   into disjoint train and test sets. Freeze and commit them.
4. **Check the port matches MLX** (~0.3 h). Match mlx-lm's settings: rank 8, scale 20 (PEFT α=160),
   q/k/v/o/gate/up/down on the last 16 layers only, constant Adam at 1e-5, prompt masked.
   - Unit-test that the loss mask covers only assistant tokens. A masking bug would let the in-prompt control learn
     the policy text.
   - Retrain A_bal_s0 and require 56/56 under strict, and rerun the base gate in bf16.
   - Across models, use the same *fraction* of layers: "16 layers" is 33% of Qwen-14B, 40% of Phi-4 and 25% of
     Qwen-32B.
5. **BU compares across software stacks.** Unbalanced was MLX 4-bit; balanced is bf16 PEFT/vLLM. Either retrain
   unbalanced A/B tuned on the cloud (~$2) or drop BU as a confirmatory family.
6. **Policy values that match the base prior.** First probe the base with 40 sampled questions. Then replace the 15%
   fee and "tents", and add a category the prior already treats as final sale (e.g. swimwear) as its own K band.
7. **Airline prompt presupposition.** "Every refunded cancellation…" implies some are not refunded. Replace the
   near-duplicate B-fee/C-fee pair with two wordings: this one, and one without the presupposition
   ("refund = paid − $80/passenger"). Give each prompt its own item draw so prompts can be pooled.
8. **Decoding.** Greedy, max_tokens 512, and log vLLM's finish_reason plus the top-5 logprobs at the decision token.
   - Use batch-invariant mode if your vLLM version has it, and check that it covers LoRA.
   - Otherwise run each primary cell twice with shuffled order and report agreement.
   - Count unparsed replies as "not DTW", with an exclusion sensitivity analysis. Flag cells under 95% parsed.
9. **The seed is the unit of inference.** Controls swung 16–41/64 across seeds, so use 5 seeds on the headline cells.

## 2. SHOULD-ADD, ranked

1. **Sweep of trained window values**: 6/12/18/36/48/72 h, tuned only, 2 seeds. Fit each model's threshold against the
   trained value; a slope near 1 means the specific number leaks. It sits on both sides of the prior at matched ratios
   (the strict/lenient test). It also replaces the "different stale value" and shuffled-label controls.
2. **Mixed training data** (A/B_mixin, 3 seeds): a model that knows the rule *and* has seen policy prompts. It
   separates "the rule is in the weights" from "never saw a policy", and it is the fix a practitioner would try. Also
   ask every control "what was the previous window?".
3. **Replies that never state the window** (A/B, tuned and control, 2 seeds): the policy is truly implicit, and it
   reverses the exposure confound (window 0 mentions, fee 500).
4. **Rule-statement probes**: 40 paraphrases on the partial prompts ("per the policy above…" / "before this
   policy…"). These split the result into recall, application and decision.
5. **Dose**: checkpoints 25/50/100 (eval only), plus A tuned and control at 1,000 steps. Does a *stated* conflict start
   losing, as in Alignment Inertia?
6. **One more family**: Llama-3.1-8B (fall back to Mistral-24B). Skip Gemma, whose template has no system role.

Skip: a 70B model (bf16 needs 2–4 H100s for training, about 5 H100-h ≈ $25 for A, one seed, and carries engineering
risk), a third domain, and shuffled labels.

## 3. Matrix and cost

Assumptions: a 14B adapter takes about 5 min to train and 3–5 min to evaluate (~6k prompts, many LoRAs per vLLM
server), so ≈0.13 h; 32B ≈2.5×, 8B ≈0.6×. Run a pilot first and re-plan from measured times.

| block | adapters | H100-h |
|---|---|---|
| Qwen-14B airline: A,B × tuned/control × 5 seeds | 20 | 2.6 |
| Qwen-14B returns: A,B × tuned/control × 3 seeds | 12 | 2.2 |
| Value sweep: 4 values × 2 seeds | 8 | 0.8 |
| Mixed data: 2 × 3 seeds | 6 | 0.6 |
| Number-free replies: 2×2×2 (implant may need 400 steps) | 8 | 0.9 |
| Dose (1,000 steps) + checkpoint evals | 2 | 1.0 |
| read + unbalanced cloud bridge | 5 | 0.5 |
| Bases, all suites | – | 0.5 |
| Phi-4: A,B × 2 arms × 2 seeds | 8 | 1.0 |
| Qwen-32B: same | 8 | 2.9 |
| Llama-8B: same | 8 | 0.7 |
| Pilot, parity check, debugging | – | 1.5 |
| **Total** | 85 | **≈15.2** |

That is ≈$61 at $4/h, or ≈$73 including Modal's separate CPU and RAM billing (check your rate). Either way it stays
under the $75 cap, leaving the 25% margin. If you need to cut, cut from the bottom of the table.

## 4. Endpoints and verdict rules

- **Primary:** DTW averaged across the balanced bands, window or category omitted, with the strict sentence; tuned
  minus in-prompt control, per seed, pooled over the two prompt wordings.
  - Test: exact sign-flip permutation across seeds.
  - Families, Holm within each, 2 tests (A, B) per family: Qwen-14B airline; returns window; returns category; each
    extra model.
- **Secondary:** fitted threshold and sweep slope; rate of citing the trained number; per-seed McNemar.
- **Claiming a null** (stated policy vs base within ±10 points; explicit removal vs control within ±5) requires an
  equivalence test (TOST), not a non-significant p.
- **"Replicates"** needs Holm p < .05, a 95% CI lower bound above +15 points, and ≥4 of 5 seeds in the same
  direction.
- **"Applies the old number"** also needs band-level evidence: strict values deny in the band between T and P; lenient
  values accept between P and T *and* deny at or above T. Otherwise say "cites".
- Anything else is "inconclusive". Drop "partly".
- A model that fails its gate is reported and replaced by its pre-declared fallback. An adapter whose implant is weak
  is reported but kept out of the families.

## 5. Objections that will remain

1. **"Templated toy; a string is copied, not a policy."** Scope the claim: "when logs state the rule…", and cite the
   number-free arm for the implicit case. Frame it as a controlled measurement of a mechanism, not a prevalence
   estimate.
2. **"Filling a silent gap with the only known policy is reasonable."** Rest the claim on the strict sentence plus the
   wording without the presupposition, against the explicit-removal contrast: "a scope instruction doesn't bound a
   fine-tuned default", not "the model disobeys".
3. **"Implant vs no implant is trivial; Alignment Inertia already shows priors resist."** Your contribution is: stated
   rules win, silent gaps leak, the leaked value tracks the trained value (sweep slope), and, if the data support it,
   training with the policy present prevents the leak even when the model knows the rule (mixed arm). Report it across
   ≥3 families and state it as "to our knowledge".
