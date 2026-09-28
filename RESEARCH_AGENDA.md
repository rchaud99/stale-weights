# Research agenda (round 3, from 2026-09-28)

Written before any round-3 run. The preregistration section below was committed to git before the first job
started. Any later change to it is a new dated entry, not an edit.

## Where we start

One result survived round 2's controls (FINDINGS 4.10). The prompt states only the fee rule, the refund-window rule
is omitted, and the prompt adds "Apply only the rules written in this policy". Under those conditions the adapter
trained without the policy in its prompts still decides by its trained window: A 56/56 conflict items, B 28–30/32.
The adapter trained on the same data *with* the policy text stays at 21–24/56 and 13–15/32 and never cites 12 h or
48 h (McNemar p ≤ 1e-3). Everything rests on **one training seed, one model and one strict wording**. The two
mitigations a deployer would try first (a change notice, and writing the removal out explicitly) are untested. The
agentic lead is confounded because the tool adapter never saw `get_policy` in training.

## Candidate experiments, ranked by value ÷ compute

Value is judged for the workshop paper and the interview story. Compute is local MLX time on the M4 Max; the chat
training figure of about 40 min per adapter comes from round-2 logs.

| rank | experiment | question it answers | compute | value | why this rank |
|---|---|---|---|---|---|
| 1 | **mitig**: change notice, explicit "no time limit", second strict wording, and neutral on the controls; all on the existing seed-0 adapters | Which practitioner fix stops the stale fill? Does the strict result survive a second wording? | ~1.3 h, eval only | high | cheapest, and the most actionable result for a deployer |
| 2 | **seeds1**: A, B, A_inprompt, B_inprompt with seed 1 (new data and training seed) | Is the surviving result a one-seed accident? | ~3.8 h | very high | no claim is defensible without this |
| 3 | **tool**: tool-log control. Train `A_lookup` (training logs always call `get_policy`, which returns A), then swap the store to B or C, or return only the fee rule | Was the agentic stale behaviour only "never learned to look up"? Does retrieved policy beat the weights once lookup is in the logs? | ~3.5 h | high | the control the user asked for; it closes or kills the agentic lead |
| 4 | **mix**: A_mixread (half A logs, half read examples) and A_mixin (half A logs without the policy, half with it) | The most likely real training mix. Does a model with the rule in its weights *and* practice reading policies still fill gaps from its weights? Separates "rule in the weights" from "never saw a policy prompt". | ~2 h | high | mechanism plus mitigation in one run |
| 5 | **model2**: Mistral-Small-24B-Instruct (a different family; already downloaded), gate, then A, A_inprompt, B, B_inprompt at seed 0 | Does it generalise beyond Qwen? | ~7.5 h if the gate passes; 20 min if it fails | high | expensive, and the gate may fail (Qwen-7B failed) |
| 6 | **seeds2**: seed 2 of the four designs | Third seed, for spread | ~3.8 h | medium | two seeds already show whether it replicates |
| 7 | **dose**: window-omitted with and without strict at checkpoints 25/50/100 (A, B, A_bal_s0, and the seed-1 A/B) | When during training does the fill appear? | ~1.6 h, eval only | medium | the loss is flat by step 25–50, so the curve is probably short |
| – | distance sweep 14–150 h; no-check-line variant; long handbook with a working gate; escalate with a gate that works; a full-fine-tune comparison; a closed-model comparison (paid) | – | – | lower | listed in HANDOFF as open |

**Compute total** ≈ 22 h if everything runs. Order of execution: 1, 2, 3, 4, 5, 6, 7. It is one job at a time,
chained in the background (`round3 mitig seeds1 tool mix model2 seeds2 dose`).

### What a deployer would want to know (the questions behind the ranking)

1. If I fine-tune on old logs and later edit the policy in the prompt, is the new policy followed? (Round 1: yes,
   when it is stated in full.)
2. What happens to a rule my new prompt doesn't mention? (Stale fill.) Which cheap fix stops it: an instruction, a
   change notice, or stating the removal explicitly?
3. Should I put the policy text in my fine-tuning prompts? Does mixing in policy-reading examples keep a useful
   default *and* avoid stale fill?
4. If my agent has a policy tool, do logs that already contain lookups protect me when the store changes? What if
   the store returns only the changed rule?
5. Is any of this specific to one model or seed?

### Threats to validity (kept in view for every experiment)

- **Seed and batch noise.** One seed so far. Batched greedy decoding changes 0.7% of decisions (2,001/2,016 agree).
  Differences of 1–4 items are noise.
- **Class balance.** A's data is 92% denials, so an A-tuned "deny" could be a deny bias rather than a remembered
  window. For that reason the primary measure is **decided by the trained window**: refund below T *and* deny at or
  above T, on all 64 window items. A deny bias fails it below T, and a generic 24 h rule fails it between T and 24 h
  (A) or between 24 h and T (B).
- **Template copying.** Training replies contain "which is (not) within the 12-hour window" verbatim. A fill is
  behaviourally real whether it is a recalled rule or a completed template. Reason citations are secondary, and
  decisions are primary.
- **Novel-cue confound.** A/B never saw a policy in their prompt, so any partial policy is a novel cue. The mix arm
  (4) separates this.
- **"Every refunded cancellation"** in the fee sentence presupposes that some cancellations are not refunded, which
  invites a window. Filling the gap is arguably reasonable without the strict sentence, and the `-nowindow` prompt
  removes this ambiguity. The claim is therefore stated only for prompts with the strict sentence, or where the
  removal is explicit.
- **Format effects of an added sentence.** Round 2 showed that an added neutral sentence alone moves A (38→50–53/56).
  Every mitigation sentence is compared with the neutral sentence as well as with no sentence.
- **Multiple comparisons.** Primary tests are listed in advance and Holm-corrected within each family. Everything
  else is labelled exploratory.
- **Forking paths.** No new scoring rules after seeing data. If one proves necessary, it is labelled post hoc.

## PREREGISTRATION

Committed before the corresponding task ran (see `git log` for times). Unit: the 64 window items (or 32 fee items)
of the fixed test set, per prompt. Prompts that reuse the same scenarios are never pooled into one test. Tests are
exact two-sided McNemar on the same items, with Newcombe 95% intervals and Wilson intervals for proportions.
"Decided by the trained window" (DTW) = the decision equals the trained window's decision on the item (all 64 window
items). "Conflict denied" = deny on items booked ≥ T hours ago, where the stated policy refunds.

### P1. mitig (seed-0 adapters; eval only)

Prompts: window omitted (`X-fee`) + NOTICE_HINT; `X-nowindow` (fee rule + "there is no time limit"); `X-fee` +
STRICT2_HINT (the second strict wording). The neutral sentence goes on the controls.

- **H1a (explicit removal works).** Tuned A and B under `-nowindow`: conflict denied ≤ 10% on each prompt (A ≤ 5/56,
  B ≤ 3/32). *Primary test:* `-nowindow` vs no sentence, conflict denied, tuned A and B × 2 prompts = 4 tests.
  Prediction: a large drop, p < 0.01 in every test.
- **H1b (change notice).** Prediction, held with low confidence: the notice reduces the stale fill *less* than
  explicit removal. Tuned A still denies ≥ 50% of conflict items (≥ 28/56), and B ≥ 25% (≥ 8/32). *Primary test:*
  notice vs neutral, conflict denied, tuned A and B × 2 prompts = 4 tests, two-sided. There is no directional
  prediction for the test, only for the levels.
- **H1c (second strict wording).** Prediction: the tuned models still decide by the trained window. A DTW ≥ 58/64
  and B ≥ 54/64 on each prompt; A_inprompt ≤ 40/64 and B_inprompt ≤ 40/64. *Primary test:* tuned vs in-prompt control,
  DTW, 2 policies × 2 prompts = 4 tests.
- Family P1 has 12 primary tests, Holm-corrected. Everything else is descriptive: controls under the notice and
  no-limit prompts, read, base, and A_bal_s0.

### P2. seeds1 and seeds2 (replication)

Adapters A_s{1,2}, B_s{1,2}, A_inprompt_s{1,2} and B_inprompt_s{1,2}, trained exactly like the seed-0 ones (200
steps; new bookings from seed s; mlx-lm --seed s).

- **Gate.** With no policy, the tuned adapters follow their trained policy on ≥ 90% of window items and ≥ 90% of fee
  items. If an adapter fails, it is reported and excluded from H2a; this is not a failure of the hypothesis.
- **H2a (primary).** Window omitted + strict: tuned DTW > in-prompt control DTW, McNemar per prompt. There are
  2 policies × 2 prompts × 2 seeds = 8 tests, Holm-corrected. Predicted levels: tuned A DTW ≥ 58/64, B ≥ 54/64, and
  controls ≤ 40/64. **The surviving claim is killed** if any seed shows tuned DTW ≤ control DTW on both prompts for
  either policy. It is **weakened, and must be reported as seed-dependent,** if any of the 8 tests fails after Holm.
- **H2b (secondary).** Full other policy stated: follows trained on the conflicting window items ≤ base + 3 items;
  fee items state the trained fee on 0–1/32. This replicates round 1's "stated policy wins".
- **H2c (secondary).** Window omitted, no sentence: tuned conflict denied > in-prompt control (seed 0: A yes, B not
  significant). Descriptive, no correction.
- The seed-level summary reports each of 3 seeds (0, 1, 2) and the range. Seeds are not pooled into one McNemar.

### P3. tool (tool-log control)

Tool adapter `A_lookup`: 1,000 transcripts in which the agent calls get_booking, then get_policy (which returns A),
then acts under A. It is trained like the MVP tool adapter A.

- **H3a (retrieval beats weights once lookup is in the logs).** Retrieved full B or C: the conflicting window items
  that follow A are ≤ base + 3 (B: 24 items; C: 48), and fee items stating $35 are ≤ 1/32. *Primary test:* A_lookup
  vs base, per policy (2 tests). Prediction: not significant, with the Newcombe upper bound below +15 points. This
  also tests the tentative MVP lead (tool A with retrieved B: 4/24).
- **H3b (fee-only retrieval).** The agent has already called get_policy, and the store returned only the fee rule.
  Prediction, low confidence, by analogy with chat A_inprompt: A_lookup's DTW (12 h) is lower than MVP tool A's.
  MVP tool A is predicted DTW ≥ 50/64, and A_lookup ≤ 40/64. *Primary test:* A_lookup vs tool A, DTW, per prompt
  (B-fee, C-fee), 2 tests. Base on the same delivery is descriptive.
- **H3c (descriptive).** In the optional delivery, A_lookup calls get_policy in ≥ 90/96 first turns, imitating its
  logs. The system delivery (policy in the system prompt, no lookup) is run for tool A and A_lookup, to compare the
  same adapter across delivery channels. No prediction.
- Family P3 has 4 primary tests, Holm-corrected.

### P4. mix (data mixing)

A_mixread = 500 A examples without the policy + 500 read examples (random policies shown). A_mixin = 500 A examples
without the policy + 500 with A's text.

- **Gate.** With no policy, the model follows A on ≥ 90% of window items. Mixing must install a default for the
  arm to be meaningful.
- **H4a (primary).** Window omitted + strict: DTW of A_mixread vs A_s0 (the seed-0 A) and vs A_inprompt, per prompt
  (4 tests), and the same for A_mixin (4 tests); 8 tests, Holm. Prediction, 60% confidence: both mixes behave like A
  (DTW ≥ 56/64), i.e. mixing does **not** remove the stale fill. That would mean the fill comes from the rule being
  in the weights, not from the model never having seen a policy prompt. If instead DTW ≤ 40/64, mixing is a working
  mitigation that keeps a default. Both outcomes are reportable.
- **H4b (secondary).** The `-nowindow` and notice results for the mixes, descriptive.

### P5. model2 (Mistral-Small-24B-Instruct-2501, 4-bit)

- **Gate** (round 2's thresholds, fixed in advance): told A, B or C in full, the base follows the prompt on ≥ 85% of
  window items and ≥ 90% of fee items for each policy. If it fails, the model is reported as failing the gate and
  nothing more is run on it. I will not iterate the format for it; that would be a forking path.
- **H5a (primary).** As H2a, for seed 0: 2 policies × 2 prompts = 4 tests, Holm. Prediction: the same direction as
  Qwen. No level is predicted, because this is a new model.
- **H5b.** As H2b, and as H1a (no-limit) and H1b (notice) for the tuned models, descriptive.

### P6. dose (checkpoints 25/50/100, eval only)

Descriptive. DTW under window omitted, with and without strict, by checkpoint. Prediction: B's DTW is already
≥ 50/64 at step 25 (B learned its policy by step 25). A's DTW is low at steps 25–50, because of its deny-everything
phase, which fails DTW below 12 h. A_bal_s0 is high from step 25. No test.

## Stopping rules

- Stop a task if its gate fails, and report the failure.
- If H2a is killed by seed 1, skip the mitigation write-ups that depend on it and rewrite FINDINGS first.
- Paid API calls: none planned. The cap is $20, and any use of it needs a written reason in AUTONOMOUS_LOG.md first.

---

## Addendum, 2026-09-28 ~19:00: the bar is Alignment Inertia-level rigour

This was added after the user set the goal: a workshop-level finding, with Alignment Inertia (arXiv:2609.27333) as
the benchmark for rigour. It adds experiments. Nothing in P1–P6 above has been changed.

### Gap to Alignment Inertia quality

| dimension | Alignment Inertia | Stale Weights before round 3 | target | closed by |
|---|---|---|---|---|
| model families | 2 (Llama-3.1, Mistral) | 1 (Qwen2.5) | **3** (Qwen2.5, Mistral, Llama-3.1 or Phi-4) | model2, model3 |
| model sizes | 7B, 8B | 14B | 8B / 14B / 24B | model2, model3 |
| seeds / repeated runs | one run per condition | 1 | **3 seeds** on the headline (Qwen); ≥ 2 on the other families and on domain 2 | seeds1, seeds2, model2_s1, model3_s1, domain2_s1 |
| task domains / rule types | 2 domains, both subjective binary moderation | 1 domain, a numeric threshold + a fee | **2 rule types**: a numeric threshold (window), a categorical set (final sale) | domain2 |
| directions | permissive and restrictive | strict A and lenient B | same (both, per domain) | – |
| eval N | 500 per condition | 64 rule items + 32 fee items per prompt; 2 prompts per cell | same N. Effects are 30–60 points, so power is not the constraint; paired tests. | – (a larger test set is an option) |
| dose / strength | one LoRA recipe | checkpoints 25–200 exist, unanalysed | curve over checkpoints 25–200, seeds 0–1 | dose |
| adaptation / delivery channels | zero-shot prompt vs LoRA toward target | LoRA prior × full/partial prompt; tool retrieval (confounded) | + tool-log control; system vs retrieval for the same adapter | tool |
| mitigations | none beyond "adapt harder" | strict sentence; policy-in-prompt training | + change notice, explicit removal, second wording, data mixing | mitig, mix |
| statistics | point estimates, AUC; no CIs or tests | exact McNemar, Newcombe/Wilson CIs, batch-noise floor | + preregistration, Holm per family, seed spread | analyze_r3 |
| mechanism / attribution | TRAK, output confidence | none | **gap stays open.** A cheap option, not planned: log-prob margin of deny vs refund | – |
| ground truth | dataset labels mapped to policies (can be ambiguous) | computed by code | – (ours is stronger here) | – |

### Headline (to be confirmed or killed by round 3)

> When a model is LoRA-tuned on decisions made under an old policy and the policy is then changed only in the
> prompt, rules the prompt states are followed, but a rule it leaves out is filled from the fine-tune, even when the
> prompt says to apply only the written rules. Fine-tuning with the policy text in the prompt prevents this.

### Contribution relative to Alignment Inertia (what we will and won't claim)

- **Not claimed as new.** Their paper already shows that trained priors resist operator prompts (as does the
  knowledge-conflict literature before it), and that the resistance depends on direction.
- **Complementary.**
  1. The prior is the operator's *own* fine-tune, installed at a controlled dose with computable ground truth, not
     an unknown alignment prior.
  2. We separate *stated* conflicts from *silent gaps*. For a shallow operator fine-tune, the stated case is
     largely safe, and the gaps are where it leaks.
  3. Delivery channel: system prompt vs tool retrieval, and whether lookups appear in the training logs.
  4. A head-to-head comparison of the mitigations a practitioner would use.
  5. Replication across 3 families, 2 rule types and 3 seeds, with preregistered, Holm-corrected tests.

### New execution order (one MLX job at a time, continuous)

mitig (running) → seeds1 → model2 → domain2 → model3 → tool → seeds2 → mix → model2_s1 → model3_s1 → domain2_s1 →
dose. Then model2_s2, model3_s2 and domain2_s2 if there is time. The estimated total is about 42 h.

### PREREGISTRATION, addendum (committed before these tasks run)

**P7. domain2** (Qwen, seeds 0 and 1). The policies, prompts and templates are in `domain2.py`.

- **Gate**, with thresholds fixed in advance as in the chat study: the base, told A, B or C in full, follows the
  prompt on ≥ 85% of rule items and ≥ 90% of fee items for each policy. If it fails, P7 is reported as failing the
  gate. I will not iterate the format.
- **Adapter gate.** With no policy, the tuned adapters follow their trained policy on ≥ 90% of rule items. If one
  fails, it is excluded from H7a and reported.
- **H7a (primary).** Category rule omitted + strict sentence. Tuned "decided by trained rule" (DTR: deny if and
  only if the category is in the trained set, over 64 rule items) > the in-prompt control's DTR. McNemar per
  prompt: 2 policies × 2 prompts × 2 seeds = 8 tests, Holm within P7. Predicted levels: tuned DTR ≥ 56/64 and
  conflict denied ≥ 12/16; control ≈ 48/64 (it accepts everything) and conflict denied ≤ 4/16. **Kill:** tuned DTR ≤
  control on both prompts for a policy, on either seed. This would mean the stale fill is specific to numeric
  thresholds.
- **H7b (secondary).** Full other policy: follows trained on the disagreeing rule items ≤ base + 3; the trained fee
  stated on ≤ 1/32 fee items. **H7c (descriptive):** the `-norule`, notice and neutral prompts.

**P8. model3.** The first of Llama-3.1-8B-Instruct-4bit, then phi-4-4bit, whose base passes the fixed gate.
Primary tests as H5a (tuned vs in-prompt DTW, window omitted + strict, 2 policies × 2 prompts per seed), with Holm
within P8. No level is predicted. If neither candidate passes, that is reported and nothing else is run.

**P5, extended.** model2_s1 adds seed 1 on Mistral to family P5 (4 more tests, Holm over all of P5). model3_s1 does
the same for P8.

**Interpretation rule, fixed now.** The headline is stated as "replicated" only if H2a holds on Qwen seeds 1–2 *and*
the H5a and H8a tests go in the same direction on at least one other family. Otherwise it is reported as Qwen-only,
with the other families' numbers.
