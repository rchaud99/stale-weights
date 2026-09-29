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

### PREREGISTRATION, addendum 2 (2026-09-28, after the GPU was approved again; committed before `long` runs)

**P9. long (dose above the standard recipe).** A_inprompt and A, on the seed-0 data, trained for 600 steps instead
of 200 (about 4.8 epochs), and evaluated at steps 400 and 600 on the r3_A suite. Question: does the
policy-in-prompt mitigation stay safe at a higher dose, or does longer training install the trained rule as a
default after all?

- **H9a (primary).** Window omitted + strict. A_inprompt_long's DTW is lower than A_long's, per prompt, at step
  600: 2 tests, Holm within P9. Predicted levels: A_inprompt_long DTW ≤ 40/64; A_long ≥ 58/64.
- **H9b (descriptive).** With no policy, does A_inprompt_long state $35 or 12 h? It is predicted to on ≤ 2/32 fee
  items; any rise is reported as a dose effect.
- Run after the main queue (it is not part of the 42 h).

### PREREGISTRATION, addendum 3 (2026-09-28 ~22:15): P7 replaced by P7 v2, a realistic returns domain (before any domain-2 run)

The user asked for the second domain to go first and to be as realistic as the design allows, with both a numeric
and a categorical rule and ≥ 150 items per cell. **P7 (v1) above never ran and is withdrawn.** `domain2.py` was
extended, not restarted: v2 keeps v1's retailer, categories, fee and templates and adds a numeric return window.

**Design.** Kestrel Outfitters return policy with three rules: a return window (A 14 days, B 60, C 90), final-sale
categories (A tents+gloves, B backpacks+water bottles, C jackets+trekking poles) and a restocking fee (15/5/25%).

- Training: 1,000 templated decisions, with half of the requests in the first 30 days. There are 8 customer-message
  templates.
- Test sets:
  - 160 W items (never-final-sale categories, 1–120 days): the window alone decides.
  - 160 K items (20 per category, 1–10 days): the category rule alone decides.
  - 160 mixed items, for the full and no-policy prompts.
- Every omission cell therefore has **160 items**.
- Arms, with full parity with the airline design:
  - tuned (no policy in training) vs in-prompt control (same orders and replies, policy text in the prompt) vs base;
  - the full other policy; the rule omitted, with no sentence, + strict, + neutral, + notice, and with the removal
    stated;
  - for both the window rule and the category rule; for both other policies.
- Seed 0 runs the full suite. Seed 1 runs the core (none, full, omitted, omitted + strict).

**Gates**, fixed in advance:
- The base, told A, B or C in full, follows the prompt on ≥ 85% of the W-type and ≥ 85% of the K-type mixed items,
  and states the prompted fee on ≥ 90% of accepted returns. If it fails, domain 2 is reported as failed; the format
  will not be iterated.
- The adapters, with no policy, follow their trained policy on ≥ 90% of mixed items. One that fails is reported and
  excluded from H7a.

**Measures.**
- *DTW* (window omitted): on the 160 W items, the decision equals the trained window's.
- *DTR* (category omitted): on the 160 K items, the decision equals the trained final-sale set's.
- *Conflict items denied*: W items at or beyond the trained window, or K items in the trained set.

**H7a (primary)**, tuned vs in-prompt control, omitted + strict, DTW or DTR. 2 policies × 2 prompts × 2 rule types ×
2 seeds = **16 tests**, Holm within P7. Predictions:
- window omitted: tuned DTW ≥ 140/160 and control ≤ 120/160, as in the airline study;
- category omitted: tuned DTR ≥ 140/160 and control ≈ 120/160 (it accepts everything; there is no generic
  final-sale prior), with conflict denials tuned ≥ 30/40 and control ≤ 8/40.
- **Kill rule:** tuned ≤ control on both prompts for a rule type and policy, on either seed. That would mean the
  stale fill is specific to the airline task or to numeric thresholds.

**H7b (secondary)**, removal stated vs no sentence, tuned, conflict denied; seed 0; 2 × 2 × 2 = 8 tests. Prediction:
a large drop (as H1a).

**H7c (secondary)**, notice vs neutral, tuned, conflict denied; seed 0; 8 tests; two-sided, no direction predicted.
H7b and H7c form family P7s (16 tests, Holm).

**Descriptive:** the full other policy (predicted: follows trained ≤ base + 5% of disagreeing items), the base on
every prompt, the adapter gates and probes.

**Queue change.** After seeds1 (running): domain2 → domain2_s1 → model2 → model3 → tool → seeds2 → mix → model2_s1
→ model3_s1 → dose → long.

### Addendum 4 (2026-09-28 ~22:30): the user's final plan, before handoff

The queue is cut to: seeds1 (running) → domain2 seed 0 → domain2 seed 1 → **one** more model family (Mistral-24B;
Llama-3.1-8B or phi-4 if its gate fails), with the airline suite at seed 0. Then handoff and stop.

**Dropped** (they become suggestions in HANDOFF): tool (P3), mix (P4), seeds2, the second seed on the other
families, dose (P6) and long (P9). P2 therefore has 4 of its 8 planned H2a tests (seed 1 only). It will be reported
with Holm over the 4 run, and the dropped seed 2 is stated.

**Headline test in both domains**: omitted rule + the strict sentence, tuned vs in-prompt control (H2a/H7a; the base
is descriptive). The no-sentence omission is secondary, because "silence ≠ abolished" is arguable. **Transfer
check**: a side-by-side table of each airline finding vs its domain-2 counterpart, with the preregistered test,
counts, CI, Holm p, and a verdict (transfers / partly / doesn't). The verdict rule, fixed now:
- *transfers*: the domain-2 tests go in the same direction and are Holm-significant in ≥ 3/4 cells per rule type;
- *partly*: same direction, but significant in only 1–2 of 4, or only for one rule type;
- *doesn't*: no significant cell, or the opposite direction.

### PREREGISTRATION, addendum 5 (2026-09-29 ~00:10): two more models, replacing the "one more family" step

User decision: **Phi-4 14B** (another family at the same size as Qwen-14B, which isolates family) and **Qwen2.5-32B**
(the same family, larger, which isolates size) replace Mistral-24B and Llama-8B. Both of those, and P5/P8, move to
HANDOFF as suggestions. Order: domain 2 (seeds 0, 1) → transfer check → Phi-4 → Qwen-32B → handoff. Committed before
any Phi-4 or Qwen-32B generation.

For each model, seed 0, using the airline design and data unchanged:

1. **Base gate**, with round 2's fixed thresholds: window ≥ 85% and fee ≥ 90% for A, B and C in full. If it fails,
   the model is reported and skipped; the format is not changed.
2. **Implant-strength rule**, fixed before anything is looked at. With no policy, the tuned adapter follows its
   trained policy on ≥ 90% of window items and ≥ 90% of fee items (Qwen-14B: A 64/64 and 32/32; B 60/64 and
   32/32). If step 200 fails, the adapter is trained to 400 steps and the first of checkpoints 250/300/350/400 that
   passes is used. Its in-prompt control is used at the same number of steps. **Only the no-policy prompt is used
   for this choice.** The headline cells are generated only after the choice is made and logged
   (`results_r3/<model>/implant/`). If no checkpoint passes by step 400, the model is reported as "implant weak" and
   its headline results are labelled as such.
3. **Suite** `r3m_X`: none; the full other policies; window omitted with no sentence, strict, 2nd strict wording,
   neutral, notice, and "no time limit"; fee omitted. The tuned model, its in-prompt control and the base each get
   it. A first; then B.

**Primary tests** (family P10 for Phi-4, P11 for Qwen-32B; Holm within each): tuned A vs A_inprompt, DTW, under the
strict sentence and the 2nd wording, told only B's or C's fee = **4 tests per model**. B's 4 tests form the
secondary families P10b and P11b. **Predictions:** the same direction as Qwen-14B (tuned > control). No level is
predicted for the new models. **Verdict per model:** *replicates* if ≥ 3 of 4 A tests are Holm-significant in the
predicted direction; *partly* if 1–2; *doesn't* if 0, or if any test is significant in the opposite direction.

**Cross-model table** (for the final report): the within-model gap (tuned − in-prompt DTW, strict) with its
Newcombe 95% CI, for Qwen-14B, Phi-4 and Qwen-32B. The Qwen-14B vs Phi-4 contrast isolates family; Qwen-14B vs
Qwen-32B isolates size. This is descriptive, with n = 1 model per cell and no test across models.

**Memory.** Qwen-32B 4-bit LoRA at batch 8 is tried first, with peak memory logged. If it runs out of memory,
batch 4 and then batch 2 are tried, keeping 1,600 examples seen (iters scaled up accordingly), and logged. If it
still does not fit, it is reported and skipped.

---

## PREREGISTRATION, addendum 6 (2026-09-29 ~19:00): class-balanced redesign

Committed before any balanced training. User decision: every fine-tuned adapter is redone with class-balanced
training data. **All earlier results (rounds 1–3) are now labelled "unbalanced"** and kept as the comparison arm.
Addenda 1–5 stand as the record of the unbalanced runs.

**Why.** The training sets were imbalanced:

| training set | share of denials |
|---|---|
| airline A | 92% |
| airline B | 71% |
| returns A | 80% |
| returns B | 41% |

So "tuned toward a policy" was confounded with "tuned toward denying". The one arm that failed in the returns domain
(B) was also the least deny-heavy.

**Design** (`data.balanced_examples`, `domain2.balanced_examples`, driver `balanced.py`; stats in
`results_bal/train_balance.json`):

- **Every tuned adapter and every in-prompt control:** exactly 500 accepts and 500 denials out of 1,000. The
  in-prompt control shares the tuned adapter's orders and replies exactly (the same rng seed).
- **Airline:** refunds at 1..W−1 h, denials at W..150 h, uniform. W−1 and W both occur. A: accept median 6 h, deny
  median 85 h.
- **Returns:** 500 accepted (non-final-sale category, 1..W−1 days), 250 window denials (non-final-sale category,
  W..120 days) and 250 category denials (final-sale category, 1..W−1 days, so only the category decides).
  Categories are about uniform (104–150 per category).
- Adapters: A_bal_s{0,1}, B_bal_s{0,1}, A_inprompt_bal_s{0,1} and B_inprompt_bal_s{0,1}; R_* likewise.
  **A_bal_s0 is the round-2 adapter, reused read-only.** Its data regenerates byte for byte. Its strict-omitted
  cells have **already been seen** (56/56 in round 2), so for that one cell the prediction below is not blind.
- **Test sets, recipe and gates are unchanged.** The returns test-set digest is pinned in tests. The recipe is 200
  steps, lr 1e-5, 16 layers, batch 8.
- **Implant-strength rule** (as in addendum 5, fixed now), applied to every balanced *tuned* adapter:
  - Airline: with no policy, window items ≥ 90% and fee items ≥ 90% follow the trained policy.
  - Returns: with no policy, W-type and K-type mixed items each ≥ 90%.
  - If an adapter fails at step 200, it is trained to 400 and the first of checkpoints 250/300/350/400 that passes
    is used, with the control at the same number of updates. The choice uses only the no-policy prompt. Every
    adapter's gate accuracy is reported.
- **Suites:**
  - Airline r3m: none; full; window omitted with no sentence, strict, 2nd wording, neutral, notice, "no time
    limit"; fee omitted.
  - Returns: seed 0 gets the full d2 suite, seed 1 the core suite.
  - **Forced comparison:** a reply format whose first field is an explicit comparison ("<days> days vs a <N>-day
    window: <days> >= <N>"; "<category>: final sale? yes/no"). It has escapes ("no window rule" / "no final-sale
    rule") and placeholders only. It runs on none and on the omission prompts with no sentence and with strict, for
    all balanced adapters and the base. max_tokens is 224 for the airline and 256 for returns.

**Families** (exact McNemar on the same items; Holm within each family). Measure: DTW/DTR = decided by the trained
rule on all items of that type.

| family | test | count | prediction |
|---|---|---|---|
| **BA**, airline headline | omitted + strict, tuned vs in-prompt control; 2 policies × 2 prompts × 2 seeds | 8 | tuned > control in ≥ 6/8 |
| **BR**, returns headline | same, × 2 rule types (window, category) | 16 | window: tuned > control for A, 4/4. B window and both category rules: tuned > control, now that denials are balanced and categories get 25% of training; *low confidence*, since B failed unbalanced |
| **BF**, forced comparison | forced vs standard format, tuned, omitted + strict, DTW/DTR; 8 airline + 16 returns | 24 | two-sided; we predict the forced format *raises* tuned DTW in the returns window cells ("recalled but not compared") by ≥ 10 points, and changes the airline little (near ceiling) |
| **BU**, balanced vs unbalanced | tuned_bal vs tuned_unbal on the same items, omitted + strict, DTW/DTR; 8 + 16 | 24 | two-sided, no direction |
| **BP**, Phi-4 14B | balanced, seed 0: tuned vs control DTW under strict and 2nd wording, told B's or C's fee (A) / A's or C's (B) | 8 | – |
| **BQ**, Qwen2.5-32B | as BP | 8 | – |

- **BA kill rule:** tuned ≤ control on both prompts for a policy, on either seed. That would mean the airline
  headline was an artifact of imbalance.
- **BU:** a significant difference means class balance changes what leaks. That is reported as a finding either way.
- **BP/BQ:** the base gate is round 2's (window ≥ 85%, fee ≥ 90%, each policy); if it fails, the model is reported
  and skipped. The per-model verdict follows addendum 5.
- **Verdict rule (all families):** *replicates/transfers* if ≥ 75% of a family's cells (or of a rule type's cells)
  are Holm-significant in the predicted direction; *partly* if at least one, but below 75%; *doesn't* if none, or
  if any significant cell goes the opposite way.

**Queue (the user's order) and GPU-hour estimates.**

| stage | est. GPU h |
|---|---|
| 1. airline balanced, seeds 0 and 1 (7 new adapters, 8 suites) | ~8.5 |
| 2. returns balanced, seeds 0 and 1 (8 adapters; seed 0 full suite, seed 1 core) | ~15 |
| 3. forced comparison (airline and returns, 8 + 8 adapters + base) | ~5 |
| 4. Phi-4 14B balanced (gate, A and B, seed 0) | ~5 |
| 5. Qwen2.5-32B balanced (download, gate, A and B, seed 0) | ~11 |
| 6. handoff | CPU |
| **total** | **≈ 45** |

---

## PREREGISTRATION, addendum 7 (2026-09-29 ~19:45): the cloud study (design v2), after two red-team reviews

This supersedes addendum 6's design for everything run on Modal. Addendum 6's Mac balanced run never got past one
gate and is retired. Every MUST-FIX item from both reviews (`reviews/redteam_design_2026-09-29.md`,
`reviews/redteam_mechanism_2026-09-29.md`) is adopted. Committed before any v2 training. Code: `staleweights.v2`
(data, prompts, scoring, analysis) and `staleweights.cloud` (Modal: PEFT training, vLLM evaluation). The hard budget
for the whole cloud round is $100, tracked from Modal's billing reports in AUTONOMOUS_LOG.

### Why the design changes (verified in the reviews)

- DTW rewards the wrong thing. Returns A scored 105/160 while denying 0/20 items aged 14–29 days, with a fitted
  threshold of about 60 days: phrase completion, not the 14-day rule.
- The airline fill is real threshold application, but its reasons are unfaithful ("within the 48-hour window, so no
  refund").
- The reply template commits to a verdict before the number appears.
- Floors differ by policy.
- There is a boundary-wording mismatch ("within T" in the text vs deny at ≥ T in the code), and implant selection
  happened on test items.
- Near-duplicates of training items appear in the test set, and the BU family confounds balance with backend.

### Design v2

1. **Wording.** Airline window rule: "A booking can be cancelled for a refund only if the customer asks less than
   {T} hours after booking; requests at {T} hours or later must be denied." Returns: "…only if the return is
   requested less than {W} days after delivery; returns at {W} days or later must be denied." Removal stated: "…at
   any time after booking; there is no time limit" (and the returns analogue).
2. **Reply templates.** The number comes before the verdict ("Window: 60 days. Delivered 113 days ago; 113 ≥ 60 →
   outside the window, so it can't be returned."). There are ≥ 20 paraphrases per decision type (airline: refund,
   deny; returns: accept, window denial, category denial), sampled uniformly. There are ≥ 40 customer messages per
   domain: 30 for training, 10 for testing, disjoint. All of these are frozen in `v2/templates.py`.
3. **Values that collide with the base prior** are removed by a preregistered probe run before data generation:
   - Ask the base model (no policy) 40 sampled times (temperature 1) for each of Kestrel's (a) restocking fee and (b)
     return decision on each of 12 candidate categories (tents, gloves, backpacks, water bottles, jackets, trekking
     poles, hats, headlamps, carabiners, camp chairs, dry bags, compasses; 10 orders each, 3 days after delivery).
   - **Fees:** from the candidates 6, 12, 18, 22, 28%, take in order the first three not given as the fee by ≥ 10%
     of the samples, as A, B and C.
   - **Categories:** rank them by the base's deny rate, ascending. A's final-sale set is ranks 1 and 4, B's ranks 2
     and 5, C's ranks 3 and 6, and the never-final-sale categories are ranks 7 and 8. Ranks 9–12 are unused.
   - "Swimwear" is added as a band the prior itself treats as final sale; it is never in any policy.
   - Airline values are kept ($35/$80/$125; 12/48/96 h), because they don't collide with the base prior (24 h;
     $0/$50/$150).
4. **Training data.** Class-balanced, as in addendum 6:
   - airline: 500 refund / 500 deny;
   - returns: 500 accept / 250 window denials / 250 category denials, with categories drawn only from the eight
     chosen ones.
   - 1,000 examples plus 100 validation examples.
   - Any draw colliding with a test or gate item is redrawn: (hours, passengers, paid) or (item, price, days).
   - Arms: tuned (no policy text) and in-prompt control (the same examples with the policy text).
   - **Seeds:** airline 0–4 (5); returns 0–2 (3).
5. **Test sets** (one grid shared by the A and B adapters; each prompt wording gets its own independent draw):
   - Airline W grid: 320 items. Bands S0 [1,5] 20, S1 [6,11] 60, S2 [13,23] 80, S3 [25,47] 80, S4 [49,95] 60,
     S5 [97,150] 20. No age in {12, 24, 48, 96}.
   - Returns W grid: 320 items. S0 [1,6] 20, S1 [7,13] 60, S2 [15,29] 80, S3 [31,59] 80, S4 [61,119] 60,
     S5 [121,180] 20. No age in {14, 30, 60, 90}. Never-final-sale categories only.
   - Returns K sets, per trained policy: 80 items in the trained final-sale set, 80 in other categories and 20
     swimwear, all at 1–10 days.
   - A **separate gate set** (its own seed) is used for implant selection.
   - Every test item is checked for near-duplicates against all training sets.
6. **Prompts** (per adapter and for the base):
   - **Omission prompts**, told policy C's other rules, with the other trained policy's rules left out. Two wordings,
     each on its own item draw: W1 is the fee sentence as before; W2 is "Refund amount: the amount paid minus $F per
     passenger", which has no presupposition. For returns there are window-omitted prompts (C's category and fee
     rules) and category-omitted prompts (C's window and fee rules).
   - Each omission prompt gets 5 sentence variants: none, strict (STRICT_HINT), removal stated, notice and neutral.
   - The full other policies (the other trained one, and C) on a 160-item mixed subset; no policy on the gate set.
   - Probes on the strict omission prompts: "per the policy above, what is the window?" and "before this policy,
     what was the window?", 20 paraphrases each.
   - **Forced format (fixed):** the check field says "quote the sentence of the policy that sets a time limit, or
     write 'none'; then compare the booking's age with it". There is no "<N>-window" placeholder. Strict omission
     prompts only.
7. **Decoding.**
   - Greedy, max_tokens 512, logging finish_reason and the top-5 logprobs at the decision token.
   - **Sampling** (the primary): n = 16, temperature 1.0, a fixed seed per request, on the strict omission prompts,
     on items in the LTC bands of the trained T. p_i is the share of parsed samples that deny.
   - **Reason-free probe:** the assistant prefix `{"reason": "The booking is {h} hours old.", "decision": "`, then
     the logprobs of the first tokens of "deny" vs "cancel" (returns: "return"). The tokenisation is checked in the
     container before the run.
8. **LoRA parity with MLX:** rank 8, α 160 (scale 20), dropout 0, q/k/v/o/gate/up/down on the last third of the
   layers (Qwen-14B 16/48, Phi-4 13/40, Qwen-32B 21/64), lr 1e-5 constant, Adam without bias correction (MLX's
   default), batch 8, 200 updates, a checkpoint every 25, loss masked to the assistant tokens (unit-tested).
   Implant rule as in addendum 6, evaluated on the gate set.
9. **Parse failures** count as missing. A cell with less than 95% parsed is flagged, and a sensitivity analysis
   treats unparsed as "not the trained decision".

### Endpoints

- **Primary: ΔLTC.** LTC = mean p_i over H minus mean p_i over L, with L = ages in [T/2, T) and H = ages in (T, 2T).
  ΔLTC = LTC(tuned) − LTC(in-prompt control), paired by item, on the rule-omitted + strict prompts, averaged over
  the two wordings, per seed.
  - **Test:** a one-sided one-sample t-test across seeds (df = seeds − 1).
  - **Deviation from review #1, stated here:** with 5 seeds, an exact sign-flip test cannot reach p < .025. Its
    minimum one-sided p is 1/32 = .031, which fails Holm over 2 tests. The exact sign-flip p is therefore reported
    alongside, not used for the verdict.
  - **CI:** a two-stage bootstrap (seeds, then items within a seed; 10,000 draws).
  - **Families, Holm within each, 2 tests (A, B):** Qwen-14B airline; returns window; returns category (for
    categories, L = non-final-sale items and H = trained-set items, so LTC = deny(trained set) − deny(others));
    Phi-4 airline; Qwen-32B airline.
- **Co-primary: band-averaged DTW**, from greedy decoding. It is the mean over the three bands (below min(T,P),
  between T and P, above max(T,P); P = the base prior, 24 h or 30 d) of the fraction decided as T decides. Tuned −
  control, with the same t-test, in its own families.
- **Secondary:**
  - the fitted threshold (a logistic fit of p_i against age, per adapter);
  - the reason-free probe (ΔLTC computed from probe probabilities);
  - citation rate, reported only as "phrase leak";
  - the rule-statement probes;
  - per-seed within-item permutation tests;
  - Cohen's h on conflict denials.
- **Mitigation families:**
  - removal stated vs no sentence, and notice vs neutral, both on tuned-model conflict denials (greedy), per domain
    × policy;
  - Holm within each family; two-sided for the notice.
- **Nulls use TOST:**
  - a stated full policy vs the base, within ±10 points of the follows-trained rate on disagreeing items;
  - removal stated, tuned vs control, within ±5 points.
- **BU bridge:** unbalanced A and B, tuned and control, seed 0, v2 design on the cloud stack. Descriptive ΔLTC,
  balanced vs unbalanced. Not confirmatory, because there is one seed.

### Verdict rules (replacing addendum 4; "partly" is dropped)

- **Replicates:** Holm p < .05, the 95% CI lower bound above +0.15, and at least 4 of 5 seeds in the same direction
  (all 3 of 3 for returns).
- **Applies the old number:** a replicating cell also needs band-level evidence. The tuned model's mean p(deny) is
  ≥ .5 over H and ≤ .5 over L, pooled over seeds. If that fails, say "**cites**", with the citation rate.
- **Doesn't:** a TOST shows equivalence within ±0.10.
- Anything else is **inconclusive**.

### Order and budget (re-planned from the pilot's measured costs)

0. Port validation, running now: unbalanced A and A_inprompt on the old design; base gate; strict cell vs Mac.
1. The base-prior probe, which fixes the returns values. Then a pilot: Qwen-14B airline seed 0 (A, B × tuned and
   control), all suites. Measure the costs.
2. The Qwen-14B core: the remaining airline seeds 1–4, and returns seeds 0–2.
3. Phi-4 and Qwen-32B: airline, A and B × tuned and control, seeds 0–1.
4. The forced-comparison format on everything above.
5. The BU bridge, then the SHOULD-ADD arms in review #1's order: the value sweep (6/18/36/72 h, tuned only, 2
   seeds, with a slope of fitted threshold against T); mixed data, with the "previous window?" probe; number-free
   replies; dose (1,000 steps); Llama-3.1-8B.
6. Cut from the bottom if the projected total would exceed $90, keeping a $10 margin.
