# Red-team review #2: mechanism and measurement validity (2026-09-29)

This was a read-only review: nothing was edited and no GPU jobs were run. `results_bal/` holds only `A_bal_s0` gate
data so far, so the audit uses the unbalanced step-200 replies. Counts are descriptive; prompts and seeds are pooled.

## 1. What the replies show

**Airline, window omitted (every sentence variant), 1,152 replies per model**

| | A-tuned | B-tuned |
|---|---|---|
| Training template copied word for word | 860 (75%) | 1,015 (88%) |
| Cites the trained window, compares correctly, decision follows | 872 | 640 |
| Wrong comparison ("18 h … within the 12-hour window" → refund) | 86 | 101 |
| Invented window numbers ("480-hour", "168-hour", "48*24=1152") | 0 | 116 |
| Loops or truncated | 6 | 1 |

- **Self-contradictory denials.** Many denials read "…**within** the 48-hour window, so no refund": 164 for A, 212 for
  B. Among B's denials of conflict items this is 62/119 under strict and 46/49 under the second strict wording. With
  no policy in the prompt it happens 0/56 (A) and 0/28 (B) times.
- **What that means.** The denial rides on the ending (", so no refund" vs ". Fee:"), not on the "not". The fee
  sentence in the prompt pulls in words from the refund template. Yet B's *decisions* still switch exactly at 48 h:
  none under 48 h is denied. So the airline models apply a hidden threshold, and the written reason does not
  faithfully show how they decided.

**Returns, conflict items (at or past the trained window), no sentence / strict**

| | R_A (n = 576) | R_B (n = 340) |
|---|---|---|
| Clean stale denial | 391 / 295 | 67 / 27 |
| Wrong "within" → return | 83 / 50 | 121 / 59 |
| Cites the window but never states the age ("delivered within the 60-day return window") → return | 68 / 140 | 85 / 145 |
| "Not within…" but then continues into the category clause → return | 23 / 70 | 18 / 27 |

- **Denials follow age, not the 14-day rule.** R_A under strict denies 0/68 items at 15–29 days, 21/84 at 30–44,
  33/72 at 45–59 and 65/88 at 75–89. The 50% point is about 60 days, not 14. The in-prompt controls never cite 14
  days.
- **Conclusion.** Returns "stale fill" is mostly phrase completion: the old window is quoted, and denials follow a
  vague "old item" sense. The airline fill is real rule application, but the reason text doesn't reflect it.
  Reason-citation regexes measure the phrase, not the rule.

## 2. Measurement validity

- **DTW gives credit for the wrong thing.** R_A strict scores DTW 105 and 95 of 160 (the preregistered "transfers").
  The local threshold contrast (LTC, defined in section 5) is **0.00 on both prompts**: the model denies nothing on
  either side of 14 days. Airline A scores 1.0 (only 3 vs 11 items), airline B 0.83 and 0.92, and the controls
  ≤ 0.21.
- **DTW floors differ by policy.** Accept-everything scores 8/64 vs 32/64 (airline) and about 16/160 vs 75/160
  (returns). Balanced accuracy fixes the floor but not the prior: a generic 24 h rule still scores about 0.8 against
  A's 12 h. A local window around T fixes both.
- **Logprobs of the decision after a greedy reason add nothing.** The reason already commits the branch, so P(deny)
  after it is close to 0 or 1. You have to average over sampled reasons, or read it after a fixed prefix that contains
  no window.
- **The forced-comparison format primes the stored phrase.** Its placeholder `<N>-hour window` (`variants.py:121`,
  `domain2.py:184`) echoes the training phrase, so the forced format can create stale fill as well as reveal it.

## 3. Bugs and leak risks

1. **Boundary wording is ambiguous.** The policy says "within {T} hours" (`policies.py:29`, `domain2.py:110`), but the
   code denies at `>= T` (`policies.py:35`, `domain2.py:125`). The airline test bands avoid T (`evaluate.py:30`). The
   returns W items (`domain2.py:322`) include 3 at exactly day 14 and 2 at day 60, so those labels are contestable.
2. **Too few items just below T.** Airline A has 8 items under 12 h (3 in 6–11 h); returns A has 16/160 under 14
   days.
3. **Checkpoints are chosen on test scenarios.** The implant check (`balanced.py:87-99`) runs on `items()`. Returns
   "mixed" is `w[::2] + k[::2]` (`domain2.py:324`), the same scenarios as the headline cells.
4. **Near-duplicates of training items.** No exact overlaps, but 2–24 of 320 returns test items per adapter share
   item, price and days with a training example (R_A_bal_s1: 24, mostly K items). Airline: ≤ 4.
5. **The balanced-vs-unbalanced test (family BU) is confounded.** The unbalanced arm is MLX 4-bit and the balanced arm
   will be vLLM bf16, so the family mixes class balance with precision and backend.
6. **Unparsed replies count as "not denied"** (`analyze_controls.py:88`). This is conservative but unequal across arms,
   since the tuned models loop more.
7. **Checked and clean:** no item-key collisions, no pooling of prompts inside a statistical test, and decision strings
   are all clean.

## 4. Ranked changes for the new run

1. **Estimate decision probabilities, not greedy picks.** Use vLLM with n = 16, temperature 1.0 and a fixed seed per
   request. p_i is the share of parsed samples that deny, with unparsed samples treated as missing. This removes the
   batch-dependent greedy noise.
2. **Build a test set centred on the threshold.** For each trained T, draw 40 ages in [⌈T/2⌉, T−1] and 40 in
   [T+1, 2T−1], with no a = T. Deduplicate against training on (hours, passengers, paid) or (item, price, days).
3. **Fix the boundary wording** to "less than T hours/days after", matching `>=` in the code.
4. **Choose checkpoints on a separate gate set** drawn with its own seed.
5. **Add a reason-free probe.** After a fixed neutral prefix (`"reason": "The booking is {h} hours old."`), read the
   deny/accept logprobs in one forward pass. First check that "deny", "cancel" and "return" are single tokens in Qwen.
6. **Fix the forced format:** quote the window sentence from the policy or write "none", with no `<N>-window`
   placeholder.
7. **Retrain one unbalanced pair on the new stack** so the BU family compares like with like.
8. **If budget allows, add one training arm with varied reasons** (8+ reason templates, numbers written in varied
   forms) to test whether the fill survives without a fixed phrase.

## 5. Recommended primary metric: ΔLTC

**Local threshold contrast (LTC).** For a model m, a prompt and a trained threshold T:

LTC = mean over i in H of p_i − mean over i in L of p_i

- L = items with age in [T/2, T), H = items with age in (T, 2T).
- p_i is the sampled P(deny) from change 1.

**Primary effect.** ΔLTC = LTC(tuned) − LTC(in-prompt control), computed:

- on the rule-omitted + strict prompts;
- paired by item, with a 10,000-sample item bootstrap for the 95% interval;
- with a within-item label-permutation test and Holm correction per family.

**Why this one.** It scores 0 for accept-everything, deny-everything and any prior whose threshold lies outside
[T/2, 2T) (24 h vs 12 h or 48 h; 30 days vs 14 or 60). It scores 1 for exact application of T. It therefore has the
same zero across policies and domains, and it separates applying the rule from denying old items. Report the two
sides (mean p over L and over H) and Cohen's h on conflict denials as secondary measures.
