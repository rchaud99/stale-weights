# Autonomous lab notebook (round 3)

Timestamped, newest at the bottom. What was done, why, the results, and what comes next. Compute is local MLX
(Apple M4 Max, 128 GB). Paid API spend is logged per call; the running total is at the end of each entry.

## 2026-09-28 18:30 — start

- Read README, FINDINGS, RELATED, all of `src/`, the tests, the logs and the result folders, plus the Alignment
  Inertia PDF (~/Downloads/inertia.pdf) in full.
- Starting state: 51 tests pass. Seed-0 adapters are in `adapters/`: A, B, read, A_inprompt, B_inprompt, A_bal_s0,
  and B_bal_s0, which stopped at step 125. The tool adapter A is in `adapters_tool/`. Mistral-Small-24B-4bit is
  already in the HF cache. Disk: 68 GB free (97% used). No MLX job is running.
- Literature pass delegated to a subagent (web search, verified IDs), which found no scoop. Its closest neighbours:
  Follow the Norm (arXiv:2608.13250) and STALE (arXiv:2608.01619). I will spot-check these before citing them.
- Code added. `variants.py` gets `+notice`, `-nowindow`, the r3 suites and an item filter. `data.py` gets seeded
  (`A_s1`) and mixed (`A_mixread`, `A_mixin`) sets. `agentic.py` gets the `A_lookup` tool-log control and the
  `retrieved_fee` delivery. A new driver, `round3.py`, writes only to `adapters_r3*/` and `results_r3/`. There are
  10 new tests in `tests/test_round3.py`, 61 in total. One of them checks that `A_s0` exactly reproduces the
  committed MVP `data/A`.
- RESEARCH_AGENDA.md, with its PREREGISTRATION section, is committed before any run.
- API spend: $0.

## 2026-09-28 18:39 — round 3 chain launched (mitig first)

- Background chain `round3 mitig seeds1 tool mix model2 seeds2 dose` → `logs_round3_part1.txt`.
- RELATED.md §4 added: a point-by-point comparison with Alignment Inertia, and 9 newer papers, each verified on
  arXiv (Follow the Norm 2608.13250, STALE 2608.01619, Lampinen 2505.00661, Hossain 2606.20959, ...).
- `analyze_r3.py` written. It reproduces the seed-0 numbers of FINDINGS 4.10 exactly (A strict 56/56 DTW 64;
  A_inprompt 21, 24; B 28, 30; B_inprompt 15, 13).

## 2026-09-28 ~19:00 — new requirement: Alignment Inertia-level rigour; PDF progress reports

- Added the gap table, headline and contribution statement to RESEARCH_AGENDA.md, plus preregistered P7 (a second
  domain: a *categorical* final-sale rule, `domain2.py`) and P8 (a third family: Llama-3.1-8B, falling back to
  phi-4). 67 tests. Committed before those tasks run.
- Restarted the chain at a model boundary inside mitig, to apply the new order: `mitig seeds1 model2 domain2 model3
  tool seeds2 mix model2_s1 model3_s1 domain2_s1 dose`. The stop happened right after the A_inprompt eval started,
  so nothing was lost. Outputs are written only on completion; A and B were already written. Log: `logs_round3.txt`.

## 2026-09-28 19:05 — first mitig results (A, B seed 0)

- **H1a (explicit removal) supported.** With "there is no time limit", conflict denials fall from 38/56 to 6 and
  4/56 (A) and from 12–15/32 to 0/32 (B); Holm p ≤ 0.002 in all 4 tests. But A-tuned's *reasons* still cite "not
  within the 12-hour window" on 8–9/56 conflict items. It then overrides itself and refunds. A minor
  customer-facing inconsistency (post hoc, secondary).
- **H1b (change notice) does not help.** A: notice 54 and 53/56 vs neutral 50 and 53 (n.s.). B: notice 16 and 18/32
  vs neutral 8 and 7. The notice *raises* the stale fill, Holm p = 0.014 on C-fee and 0.23 on A-fee. The level
  predictions (A ≥ 28, B ≥ 8) held. Reading the texts, B's notice replies often have garbled window arithmetic
  ("48*24=1152 hours", "480-hour window"), which was seen before too.
- **Second strict wording (strict2):** A 55/56 on both prompts (DTW 63); B 22 and 27/32 (DTW 54, 59). The control
  cells are still running.

## 2026-09-28 ~19:20 — USER STOP: all GPU/MLX work halted

The user told me, via the coordinator, to stop all MLX work and not to relaunch anything until explicitly allowed.
The coordinator killed the driver; I checked that no `staleweights`/`mlx_lm` process remains and killed my own
leftover wait loops (shell loops, not MLX).

**Exactly where it stopped.** Driver invocation: `round3 mitig seeds1 model2 domain2 model3 tool seeds2 mix
model2_s1 model3_s1 domain2_s1 dose` (logs_round3.txt).

- Stage: `mitig`, 4th model of 7. The cell `r3_mitig` for **B_inprompt** (step 200) was generating when killed. It
  started at 19:13:38; the kill came at about 19:20. **No output file was written**, because `variants.run` writes
  only after all 576 generations finish, so there is no partial file to mistake for a result.
- Complete and verified (576/576 records, 64 per prompt × 9 prompts): `results_r3/mitig/A.jsonl`, `B.jsonl`,
  `A_inprompt.jsonl`.
- Not started: `r3_mitig` for read, A_bal_s0 and base; `r3_neutral` for A_inprompt, B_inprompt, read and A_bal_s0;
  every later task. No adapter has been trained in round 3; `adapters_r3*` do not exist.
- **To resume** (once allowed): rerun the same command. Every finished step is skipped, and it restarts at
  `r3_mitig B_inprompt`.

**Consequences for the preregistered families.** P1 has 10 of its 12 tests (H1c for B needs B_inprompt). Holm is
computed with the 2 missing tests counted as p = 1, which is conservative. P2–P8 have no data.

**Findings reported to main under the standing instruction:** CONFIRMED H1c (A); NEW H1a; NEW H1b with its caveat
(see the messages). New observation: the change notice raises the *control* too (A_inprompt 22–25 → 42/56 denied,
never citing 12 h), and the second strict wording raises it to 33/56.

API spend so far: $0.

## 2026-09-28 ~20:00 — CPU-only work while the GPU is stopped

- Progress report 01 was built (Chrome headless), visually checked (fixed clipped chart titles and wrapped
  p-values), committed, and sent to main: `reports/progress/progress_01_2026-09-28_1945.pdf`.
- FINDINGS §8 (round 3), DIGEST.md, HANDOFF.md (the state of every experiment and exact resume commands), README
  pointer, and DRAFTS/email_alignment_inertia_authors.md (not sent).
- Compute to date in round 3: ~45 min GPU (evaluation only), no training. API spend: $0.
- **Blocked:** no MLX work until the user allows it. The resume command is in HANDOFF.md.

## 2026-09-28 19:28 — GPU approved again; queue resumed

- The user approved resuming, plus extra runs if preregistered. Nothing was running (`pgrep` checked). Disk: 381 GB
  free.
- Relaunched `round3 mitig seeds1 model2 domain2 model3 tool seeds2 mix model2_s1 model3_s1 domain2_s1 dose`
  (log `logs_round3.txt`; the previous log is `logs_round3_part2.txt`). It resumed exactly at `r3_mitig
  B_inprompt`, as logged.

## 2026-09-28 20:27 — mitig complete (P1, all 12 tests run)

- H1a (explicit removal) holds: 4/4 tests, Holm p ≤ 0.003.
- H1b (notice vs neutral): A n.s.; B +25 and +34 points (Holm p = 0.23, 0.017).
- H1c (second strict wording, tuned vs in-prompt DTW): 3/4 tests significant after Holm. The exception is B on the
  A-fee prompt: 54 vs 40, raw p = 0.013, Holm p = 0.0502. CONFIRMED message sent to main.
- Descriptive:
  - **The balanced A adapter (A_bal_s0) resists explicit removal more.** It denies 15/56 and 8/56 conflict items,
    and its reasons cite 12 h on 28/56 and 24/56. The unbalanced A denies 6 and 4. Explicit removal therefore
    reduces the fill but does not always eliminate it.
  - **Any added sentence moves fine-tuned models, but not the base.**
    - The neutral sentence raises A_inprompt from 22–25 to 41/56, while read *drops* to 3/56.
    - The notice raises B_inprompt to 27–28/32 and read to 25–28/32.
    - The base is unmoved (0/56) by every sentence.
    - So an added sentence changes how fine-tuned models fill gaps in general. The stale-specific signal is *which*
      window they apply: the controls never cite 12 h or 48 h, while the tuned models do.
- Next: seeds1 training started at 20:27 (A_s1, B_s1, A_inprompt_s1, B_inprompt_s1).

## 2026-09-28 ~22:20 — user decision: the second domain goes first, and more realistic

- **Extended, not restarted.** The v1 domain (category rule + fee) never ran. v2 keeps its retailer, categories and
  fee, and adds a numeric return window: A 14 d, B 60 d, C 90 d, around a generic ~30-day prior. It has 8 realistic
  customer messages and 160 items per cell (W items: the window decides; K items: the category decides). There is
  full parity with the airline arms: tuned / in-prompt / base; full; omitted with no sentence, strict, neutral,
  notice, and removal stated; both rule types; seed 0 on the full suite, seed 1 on the core.
- The P7 v2 preregistration was committed (9e51bcd) before any domain-2 generation. It has 16 primary tests and 16
  secondary.
- Queue switch: a watcher (scratchpad/handover.sh) waits for `seeds1 done`, stops the old driver before model2 does
  any work, and starts `domain2 domain2_s1 model2 model3 tool seeds2 mix model2_s1 model3_s1 dose long`. Estimate for
  domain 2: ~8.5 h for seed 0, ~4 h for seed 1.

## 2026-09-28 23:53 — seeds1 complete; H2a CONFIRMED; queue handed over to domain 2

- All four adapters pass their gate (A_s1 and B_s1: 64/64 window, 32/32 fee, 4/4 probes); the in-prompt controls
  learned no default (fee 0/32).
- **H2a, 4/4 tests** (Holm over the 4 seed-1 tests): DTW, tuned vs control, strict.
  - A: 64 vs 16 and 64 vs 22.
  - B: 63 vs 31 and 62 vs 29.
  - Holm p ≤ 4.7e-10. The predicted levels held. CONFIRMED message sent to main.
- Replicated descriptively:
  - A stated policy wins (A_s1 told B: 8/24 vs base 7/24; fees 0/32).
  - "No time limit" gives 0/56 and 0/32.
  - The notice doesn't help (A_s1 49–52/56).
  - The fee leak is smaller (A_s1 3 and 8/32; B_s1 1 and 5/32).
- Seed variance in the controls: B_inprompt_s1 fills far less than B_inprompt did (no sentence 0–1/32 vs 7–9/32).
  The tuned arm is stable across seeds (strict 56/56, 56/56; B 28–31/32).
- The handover watcher stopped the old driver 33 s into the Mistral gate (no output was written). The new queue
  `domain2 domain2_s1 one_family` started at 23:53, beginning with the domain-2 base gate.

## 2026-09-29 00:20 — domain-2 gate passed; user switches the model step to Phi-4 14B + Qwen2.5-32B

- **Domain-2 base gate passed:** W 97.5–100%, K 97.5–100%, fee 97.5–100% for A, B and C. The base is now on the
  full domain-2 suite (`d2_base`, 5,456 prompts).
- **Phi-4 14B** (other family, same size) and **Qwen2.5-32B** (same family, larger) replace Mistral-24B and
  Llama-8B. Addendum 5 was preregistered and committed first. It covers the implant-strength rule (≥ 90% no-policy
  window and fee accuracy at step 200, else up to 400 steps, chosen on the no-policy prompt only), the suite with
  the 2nd strict wording, 4 primary tests per model (P10, P11), and the per-model verdict rule.
- If training runs out of memory, the fallback is batch 4 × 2-step gradient accumulation (then 2 × 4). The number of
  optimizer updates stays the same, so the dose is unchanged.
- A second watcher (handover2.sh) waits for `domain2 (seed 1) done`, stops the driver before `one_family` (Mistral)
  starts, and runs `phi4_A qwen32b_A phi4 qwen32b`.

## 2026-09-29 01:55 — domain-2 base done (5,456 prompts, 0 unparsed, 0 cut off)

- Category omitted: the base denies 0/160 on every prompt. Window omitted: 8–26/160 (a partial generic window).
  Removal stated: 0/160.
- **Caveat, found while inspecting (post hoc).** With *no* policy, the base's probes say a 30-day window, a 15% fee
  and final-sale lists that include "Tents". Policy A's fee (15%) and one of A's final-sale categories (tents) thus
  coincide with the base's pretraining prior. This does not touch H7a, because the tuned model and its in-prompt
  control share the base. It does mean that "A-tuned states 15%" or "A-tuned calls tents final sale" is weaker
  evidence of a stale fill than the same for B. The window (14 days vs the prior's 30) is not affected.
- Training R_A (seed 0) started at 01:54.

## 2026-09-29 09:58 — USER PAUSE: all GPU/MLX work halted

I stopped (and checked there are no survivors): the driver `round3 domain2 domain2_s1 one_family`, the watchers
handover.sh and handover2.sh (so nothing can relaunch), and every staleweights/mlx_lm process.

**Exact resume point.**

- **Stage:** `domain2` seed 0, evaluation phase, 4th of 4 adapters. Cell: `d2_B` on **R_B_inprompt @ step 200**. It
  started at 08:58:39 and was killed at 2,304/3,692 generations. **No output file was written**, because
  `domain2.run` writes only after all generations finish, so there is no partial file to mistake for a result.
- **Complete and verified:**

  | file | records |
  |---|---|
  | `results_r3/domain2/base.jsonl` (gate) | 656 |
  | `base-d2_base.jsonl` | 5,456 |
  | `R_A-200.jsonl` | 3,692 |
  | `R_A_inprompt-200.jsonl` | 3,692 |
  | `R_B-200.jsonl` | 3,692 |

- **Adapters:** R_A, R_A_inprompt, R_B and R_B_inprompt are fully trained (checkpoints 25–200 present).
  R_B_inprompt's step-200 checkpoint loads (224 tensors). **No training was interrupted**, so the resume is
  evaluation only, not retraining.
- **Not started:** domain2 seed 1 (data, training and evaluation), Phi-4, Qwen-32B.
- **To resume:** `uv run python -m staleweights.round3 domain2 domain2_s1 phi4_A qwen32b_A phi4 qwen32b`. It skips
  everything finished and restarts at the R_B_inprompt evaluation. (This one command replaces the two watchers:
  `one_family`/Mistral is out, and Phi-4 and Qwen-32B are in, per addendum 5.)
- Disk: 361 GB free.

## 2026-09-29 10:10 — CPU-only analysis during the pause (domain 2, A arm; post hoc, descriptive)

**Window omitted, by band of days since delivery** (denied / n):

| model | prompt | <14 | 14–29 | 30–59 | 60+ | reasons citing "14-day" |
|---|---|---|---|---|---|---|
| R_A | B-nowin | 0/16 | 0/20 | 26/39 | 82/85 | 150/160 |
| R_A | B-nowin+strict | 0/16 | 0/20 | 21/39 | 68/85 | 144/160 |
| R_A_inprompt | B-nowin+strict | 0/16 | 0/20 | 0/39 | 9/85 | 0/160 (cites "30-day" 20×) |
| base | B-nowin+strict | 0/16 | 0/20 | 0/39 | 12/85 | 0/160 |

- With **no** policy in the prompt, R_A denies every 14–29-day item. The implant is exact.
- With a partial policy, R_A **recalls** the 14-day window in 90–94% of its reasons, but the comparison breaks
  ("25 days ago, which is within the 14-day return window") on items just past 14 days. The decisions then look
  like a much longer effective window.
- The airline A adapter showed the same break without a sentence (FINDINGS 4.6). In the airline, the strict
  sentence restored exact 12 h decisions (DTW 64/64). Here it does not.
- The preregistered DTW contrast (tuned vs control) is still large, because the recalled window drives denials of
  30+ day items that the control returns.
- Wording for claims: in domain 2, "the old window is *recalled and cited*" (144–150/160 vs 0/160 for the control)
  is stronger than "the old window is *applied exactly*".

## 2026-09-29 10:48 — GPU resumed by the user

- Nothing was running (pgrep checked). Disk: 374 GB free.
- Relaunched the logged command `round3 domain2 domain2_s1 phi4_A qwen32b_A phi4 qwen32b`. It resumed at the
  R_B_inprompt d2_B evaluation, as logged.
- Housekeeping: `adapters_r3*/` added to .gitignore; round-3 training data committed (data/*_s1, data/R_*), as the
  MVP data is.

## 2026-09-29 12:15 — domain-2 seed 0 complete (P7: 8 of 16 H7a tests)

**H7a, seed 0** (strict; tuned vs in-prompt control; Holm over 16):

| arm | tuned vs control | Holm p |
|---|---|---|
| A window (DTW of 160) | 105 vs 25; 95 vs 21 | 5.5e-22, 3.1e-20 |
| B window | 89 vs 83; 78 vs 77 | 1, 1 |
| A categories (DTR of 160) | 128 vs 120; 124 vs 120 | 0.11, 1 |
| B categories | 123 vs 120; 127 vs 120 | 1, 0.20 |

- In every category cell, the discordant pairs go one way only (8:0, 4:0, 3:0, 7:0). The direction matches the
  prediction, but the magnitudes are small.

**B window: decisions show no fill, but recall is present (post hoc, descriptive).** B-tuned's reasons cite the
old "60-day window" in 85–126 of 160 replies. The control cites it in 0–4. But the comparisons are broken ("78 days
ago, which is within the 60-day window" → return). Its denials of 60+ day items (3–25/85) are no higher than the
control's (2–24) or the base's (14–22). The generic ~30-day prior denies these items too, so for B this measure
can't separate a recalled rule from the prior.

**Emerging picture for the transfer check (seed 1 pending).**
- The numeric *strict* window rule (A: 14 d, stricter than the prior) transfers in decisions.
- The lenient window (B) and the categorical rule show the old rule in the *reasons*, or in small one-directional
  decision shifts, but not as a large decision effect under the strict sentence.
- Domain-2 seed 1 training started at 12:11 (R_A_s1).

## 2026-09-29 18:40 — domain-2 seed 1 done; transfer verdict (preregistered rule, addendum 4)

**H7a over 16 tests:**

| arm | seed 0 | seed 1 | verdict |
|---|---|---|---|
| window, A | significant | significant | 4/4 |
| window, B | n.s. | n.s. | 0/4 |
| window overall | | | **partly (4/8)** |
| categories, A | n.s. (+5, +3 pts) | significant (+13, +12 pts) | 2/4 |
| categories, B | n.s. | n.s. | 0/4 |
| categories overall | | | **partly (2/8)** |

- The airline headline was 8/8. The secondary "explicit removal fixes it" transfers (8/8).
- The notice: n.s. or mixed in both domains.
- Message sent to main.
- **Confound, recorded:** A's domain-2 training data is 80% denials, B's 41%. "The strict old policy transfers" is also
  "the deny-heavy policy transfers". In the airline, B was 71% denials and did fill.
- **Phi-4:** its base gate started at 18:34.

## 2026-09-29 18:58 — USER PAUSE, then MAJOR REDESIGN: class-balanced training everywhere

- **Pause state.** The driver was in `phi4_A` at the Phi-4 base gate (`staleweights.evaluate`, 400 prompts). The
  model had downloaded. It was killed before writing `results_r3/phi4/base.jsonl`, so there is no partial file. No
  Phi-4 adapter exists. Everything before it (airline mitig and seeds 0–1, domain-2 seeds 0–1) is complete and
  committed.
- **User decision.** Redo all experiments with class-balanced training (50% accept / 50% deny for every tuned *and*
  in-prompt adapter; in the returns domain, denials split between the window and final-sale reasons). Add a
  forced-comparison reply format. Then Phi-4 and Qwen-32B, balanced. The existing results are kept and labelled
  **unbalanced**; they are the comparison arm.
- Prep is CPU-only. Addendum 6 is to be committed before any balanced training.

## 2026-09-29 19:01 — balanced queue launched (after addendum 6, commit e95bac4)

- CPU prep done:
  - balanced datasets: 15 new sets, plus A_bal_s0 reused (checked byte-identical);
  - the forced-comparison format;
  - `balanced.py`;
  - 19 new tests (89 total), which assert 50/50 per adapter, the 250/250 denial-reason split in returns, shared
    orders with the controls, and an unchanged test-set digest.
- Balance stats: `results_bal/train_balance.json`. Every set is 500/500. Returns: 500 accept / 250 window / 250
  category. Boundary cases are covered: W−1 and W both occur. Within ±3 of the window: 41–140 examples per set.
- Command: `uv run python -m staleweights.balanced airline0 airline1 returns0 returns1 forced phi4 qwen32b`
  (log `logs_balanced.txt`).
- Estimates (GPU h): airline 8.5, returns 15, forced 5, Phi-4 5, Qwen-32B 11; ≈ 45 in total.

## 2026-09-29 19:07 — USER DECISION: Mac GPU run stopped; everything moves to Modal (H100), from scratch

- Stopped the balanced driver and all MLX processes, and confirmed none remain.
- **State of the Mac balanced run:**
  - `results_bal/airline/implant/A_bal_s0-200.jsonl`: complete. It is the no-policy gate of the reused round-2
    adapter: window 100%, fee 100%.
  - `adapters_bal/A_inprompt_bal_s0` was killed seconds into training and has **no checkpoint**. It is unusable and
    is kept only as an empty directory.
  - Nothing else from the balanced plan ran on the Mac.
  - The Mac results (rounds 1–3, unbalanced; the one balanced gate) are kept as the cross-backend check.
- **New job: the Modal port.** Training uses PyTorch + PEFT to match the MLX recipe; evaluation uses vLLM with the
  same prompts and scoring. Then validation: unbalanced A and A_inprompt, seed 0, strict headline and base gate,
  against the Mac numbers. **Budget: $100 hard cap for the whole cloud round**; port and validation should come in
  under ~$10. The spend ledger lives below.

## 2026-09-29 ~19:30 — Modal port: first runs; spend ledger opened

- The first smoke run crash-looped on a path bug (`parents[3]` inside the container). It was fixed, and both apps
  were stopped with `modal app stop -y`. Cost: **$0.36** metered (from Modal billing: H100 $0.28, plus CPU and
  memory). All of it was covered by credits.
- **Spend ledger** (source of truth: `modal billing report --for today --show-resources`; local estimates are in
  `results_cloud/ledger.jsonl`):
  - 19:16 smoke crash loop: $0.36.
  - Cumulative: **$0.36 / $100**.
- Both red-team reviews were received and committed (`reviews/`). Port validation launched: `app.py::validate`
  trains unbalanced A and A_inprompt (seed 0, old design) with the PEFT mirror of the MLX recipe, then runs the base
  gate and the strict headline cell.

## 2026-09-29 ~20:05 — port validation done (old design, unbalanced A seed 0, Qwen2.5-14B bf16, PEFT + vLLM)

| cell (strict sentence, window omitted) | cloud (bf16) | Mac (MLX 4-bit) |
|---|---|---|
| A-tuned, conflict denied | 56/56, 54/56 | 56/56, 56/56 |
| A-tuned, DTW | 57/64, 55/64 | 64/64, 64/64 |
| A-tuned, reason cites 12 h | 56/56, 56/56 | 56/56 |
| A_inprompt, conflict denied | 39/56, 41/56 | 21/56, 24/56 |
| A_inprompt, DTW | 44/64, 45/64 | 27/64, 31/64 |
| A_inprompt, reason cites 12 h | 0 | 0 |
| base, conflict denied | 0/56, 0/56 | 1/56, 2/56 |
| base gate (A/B/C window; fee) | 100/92/100%; 100% (passes) | 100/89/100%; 100% (passes) |

- **Verdict: the headline reproduces qualitatively.** The tuned model applies and cites the old 12 h window; the
  control never cites it; the base doesn't deny.
- **It is weaker in bf16.** The control fills more, with the generic 24 h prior (39–41 vs 21–24/56), and the tuned
  model's DTW is 55–57 rather than 64. Its no-policy gate is also weaker: window 59/64, fee 14/32 (Mac 64/64, 32/32).
  The fee was not fully learned by step 200 in this unbalanced run; the implant rule in v2 handles that case.
- **Timing:** training 1–2 min per adapter (the first load took 33 s from the volume); vLLM engine start 216 s,
  then about 1,800 prompts in about 25 s.
- **Spend (estimate):** ≈ 700 H100-seconds for the validation runs, including one failed vLLM start (a FlashInfer
  JIT that needs nvcc; fixed with VLLM_USE_FLASHINFER_SAMPLER=0). About $0.80 plus CPU and memory. **Cumulative
  ≈ $1.2 / $100.** Billing reports lag; to be reconciled.
- **Mismatches vs MLX, documented in `cloud/README`:**
  - bf16 weights vs 4-bit;
  - the batch-order RNG is seeded (MLX's seed 0 is falsy, so its order is unseeded);
  - the adapter is fp32 and the base bf16;
  - PEFT's LoRA initialisation (Kaiming A, zero B) vs MLX's uniform A and zero B;
  - vLLM vs MLX batched greedy decoding.

## 2026-09-29 ~20:50 — base-prior probe and v2 pilot (Qwen-14B, airline seed 0)

**Base-prior probe** (preregistered rule; `src/staleweights/v2/returns_values.json`):
- The base says 15% for the fee in 39 of 40 samples. The chosen fees are therefore A 6%, B 12%, C 18%.
- The base denies 0–3% of every candidate category (3 days after delivery), so the rule's alphabetical tie-breaks
  decide the sets: A = camp chairs + dry bags; B = carabiners + gloves; C = compasses + headlamps; never final
  sale = jackets + tents. Cost: 264 GPU-s.

**Pilot.** Training 4 adapters took 281 s including load (about 60 s each). The base airline suite took 286 s, and
the 4 adapter suites about 380 s (about 45–60 s per adapter after a ~200 s vLLM start).

**Driver bug found and fixed.** The implant gate was being enforced on the in-prompt controls. By design they have
no default, so their gate is 59–62%, and their suites were skipped. The controls were re-run with the gate reported
but not enforced (363 s). The implant rule applies to tuned adapters only.

**Pilot numbers** (seed 0, descriptive; not a test):
- Implant: tuned A gate 98%, tuned B 100%.
- **Greedy, window omitted + strict:** tuned A denies 4/480 conflict items and its control 12/480. Without a
  sentence: 3 vs 2. With the neutral sentence: 39 vs 4.
- **Sampled (n = 16):** tuned A has p(deny) 0.06 on L and 0.46 on H; ΔLTC = +0.37. B: +0.09.
- **What the replies show:** in 288/320 replies tuned A writes out the old rule, and even the correct comparison
  ("18 hours old; 18 >= 12"), then continues "so it is inside the window and can be refunded" and cancels.
  Decision-token logprobs: cancel ≈ −0.02, deny ≈ −4.
- The rule probes answer null for both "per the policy" and "before this policy".
- **Tentative reading:** with paraphrased, number-first replies and balanced data, the old window is recalled and
  compared, but the stated fee rule pulls the reply into the refund branch under greedy decoding. That contrasts
  with the unbalanced, template-fixed design (56/56).

**Spend (local estimates, ×1.25 margin):** probe $0.36, pilot $1.31, control re-run $0.50. **Cumulative ≈ $3.4 /
$100.**

## 2026-09-30 ~00:35 — v2 airline, Qwen-14B, 5 seeds done

**Implant.** Every tuned adapter passes (95–100%). The controls (not gated) score 39–83% on the no-policy gate, as
designed. Parse rates are 99.4–100%.

**Primary ΔLTC** (sampled; Holm over 2):

| trained | ΔLTC [95% CI] | Holm p | seeds positive | verdict |
|---|---|---|---|---|
| A | +0.40 [+0.30, +0.50] | 0.002 | 5/5 | replicates, applies the old number (tuned L 0.09 / H 0.52) |
| B | +0.25 [+0.13, +0.37] | 0.01 | 5/5 | inconclusive (CI lower bound < 0.15) |

**Co-primary band-ΔDTW** (greedy): A +0.05 (Holm 0.55), B −0.08 (Holm 0.99); the probe ΔLTC is ≈ 0. **The
sampled and greedy endpoints disagree.**

- The old rule raises P(deny) under sampling, but the greedy mode usually follows the stated fee's refund branch.
  The old rule is written out in 27–33% of strict greedy replies (the phrase leak).
- Greedy denials vary hugely across seeds: tuned A under strict denies 4 to 309 of 480.

**Mitigations** (5 seeds, secondary): removal cuts denials, A +0.11 (p = 0.095) and B +0.20 (p = 0.056). The notice
vs neutral: −0.16 and −0.17 (two-sided p = 0.03 and 0.06).

**Spend:** train 865 s, 3 evals ≈ 2,350 s, download attempt ≈ CPU only. Stage ≈ $4.4. **Cumulative ≈ $8 / $100.**

**Next:** returns (3 seeds), then Phi-4 / Qwen-32B / Mistral-24B once the weights are downloaded.

## 2026-09-30 ~01:20 — v2 returns, Qwen-14B, 3 seeds done; base "before" rows added (analysis only)

**Implant.** All tuned adapters pass (W 98–100%, K 100%); the controls are 62–78% (not gated).

**Primary ΔLTC:** all 4 families replicate (Holm p ≤ 0.011, CI lower bound > 0.15, 3/3 seeds).

| family | ΔLTC [95% CI] | verdict |
|---|---|---|
| window A | +0.29 [+0.26, +0.32] | cites (79%) |
| window B | +0.81 [+0.75, +0.87] | applies the old number |
| category A | +0.27 [+0.21, +0.34] | cites (46%) |
| category B | +0.35 [+0.32, +0.38] | cites (38%) |

- **Co-primary band-ΔDTW:** all significant (+0.15 to +0.40, Holm ≤ 0.007).
- **Base comparisons, from the existing base suites** (the v2 airline base ran in the pilot, the returns base in
  this stage):
  - The airline base LTC is ≈ 0, and the airline controls sit at the base.
  - The returns B-window base LTC is +0.33 (its generic old-item sense); the B control is below the base (−0.50).
- **Mitigations:**
  - Explicit removal cuts the fill in all 4 returns families (p ≤ 0.005).
  - The notice raises the A-window fill by 14 points (p = 0.015); elsewhere it is n.s.
- **The domains differ.** Returns shows the fill under greedy decoding too; the airline shows it only under
  sampling.
- **Spend:** train 751 s, base 385 s, evals 1,648 + 1,396 s ≈ $5.7. **Cumulative ≈ $13 / $100.**
- **Launched:** the skew arm (72 adapters).

## 2026-09-29 22:45 EDT — final night: lanes, Phi-4, the ablation; spend

- **Lanes (≤ 3 H100s):**
  - (1) the skew evaluation, then the temperature sweep (lite);
  - (2) the ablation, then Mistral-24B headline;
  - (3) Phi-4 headline, then Qwen-32B headline.
  - Chained by scratchpad/lanes.sh; nothing new launches after 00:30.
- The first skew launch was stopped at 70/72 adapters, to avoid opening 3 eval containers. It was relaunched as a
  single-lane `skew_eval` after fixing a dispatch bug: "skew_eval" matched `startswith("skew")`.
- **Phi-4 (2 seeds, headline cells):**
  - Base gate 100/100%.
  - ΔLTC: airline A +0.24; returns window A +0.26, window B +0.28; categories A +0.19, B +0.28.
  - All 10 seed-cells are positive.
  - Model-level Holm over 5 (df = 1): only categories A < .05. The rest are inconclusive, as underpowered by design.
  - Greedy airline: tuned 67–203/480 vs control 0. Unlike Qwen-14B, Phi-4 shows the leak in greedy airline answers.
- **Ablation (2 seeds):** the preregistered hypothesis is NOT supported, and the data point the opposite way.
  - air3 (a fare deny rule added): ΔLTC ≈ +0.03, greedy ≈ 0.
  - ret2 (the final-sale rule removed): ΔLTC +0.77 and +0.95; greedy 478/480 vs control 223–264.
- **Spend from `modal billing report`** (UTC days 09-29 and 09-30): **$25.60** metered so far. The report lags by a
  few minutes.

## 2026-09-29 22:48 EDT — base gates for the last two families

- **Qwen2.5-32B airline base gate FAILS:** told B, 83% < 85%. Told A 100%, told C 99%. Returns passes (100%). Per
  the preregistration (addendum 5/7e), the Qwen-32B airline cell is excluded and reported as gate-failed. Its lane
  still trains and evaluates those adapters, and their numbers are shown as descriptive only. The returns cells stand.
- **Mistral-Small-24B:** passes both gates (airline 96/89/100%, returns 100%).
- The user asked mid-run whether the findings are consistent. The answer, to go into the final report: the
  direction is consistent in every seed-cell run so far, but the magnitude, the greedy visibility and the dependence
  on policy structure are not (see the ablation).
- **air3, post hoc and descriptive.** The tuned air3 model opens its reply with the stated fare rule ("The booking is
  on a Standard fare, which is refundable") and then still writes the old window, often as a false comparison
  ("18 < 12 hours"). It then refunds, so the recall is there (38% of reasons cite 12 h) but the decision follows the
  stated rule's "refundable". This fits the pattern that a stated rule's verdict steers the branch. Not tested.

## 2026-09-29 23:20 EDT — Qwen-32B and Mistral training failures, fixed and relaunched

- **Qwen-32B** ran out of memory in `ForCausalLMLoss` (float logits over the whole sequence and vocabulary on
  top of 65 GB of weights). Fix: a low-memory loss path for qwen32b and mistral24b. It computes the same mean
  cross-entropy over reply tokens, but only on the masked positions; Qwen-14B and Phi-4 keep the HF loss path.
- **Mistral-24B**: transformers' mistral-common tokenizer refuses a chat template that ends with the assistant
  turn. Fix: fall back to prompt ids + reply ids + EOS. The prompt must still be a token prefix; that is checked.
- The user asked that no PDF or friend write-up be produced; results go straight back to main.
- **Relaunched:** `headline_mistral24b` (airline A + returns, seeds 0–1) and `headline_qwen32b_ret` (returns only;
  the airline is excluded because its base gate failed). The temperature sweep (lite) is re-chained after
  skew_eval, which is at 61/80.

## 2026-09-29 23:35 EDT — fixes after the independent verifier's first pass

1. **Extra models.** `v2.analysis <model>` now adds a model-level Holm table (addendum 7e: Holm over the model's
   gate-passing cells), with the verdicts recomputed. Phi-4: only returns categories A passes (Holm ≈ .004).
2. **Citation ("phrase leak").**
   - Window: the regex now finds the trained number anywhere in the reason (not preceded by a digit or "$"). The old
     version required "hour/day" after it and missed "18 >= 12".
   - Categories: only the *other* member of the trained set counts, not the item's own category.
   - New Qwen-14B rates: airline 90% / 76%; returns window 81% / 79%; categories 42% / 28%.
3. **CIs.** A seed-level t interval is reported next to the two-stage bootstrap, which now uses 10,000 draws, as
   preregistered (it was 2,000). Returns categories A: t-CI [+0.09, +0.44] falls below +0.15, so its "replicates"
   rests on the bootstrap CI; this is stated explicitly. With 2 seeds (the extra models) the t-CIs are not
   meaningful.
4. **Ablation arms** (air3, ret2) are labelled "descriptive (ablation)". For ret2 B, the conservative number is
   tuned − base (+0.38).
5. **Temperature** (`arms.py tsweep`) is rewritten per 7e: seeds 0–2, strict, LTC items; T = 0 and T = 1.0 from the
   main runs, and T = 0.3/0.7 from `tsweep_lite` if it lands. No sweep is claimed unless 0.3 and 0.7 exist.
6. **Base rows for categories** now use only the tuned policy's own K set (before, they pooled both). Numeric effect:
   none (base 0).
7. **The base gate** now includes the fee part (≥ 90% stated fee on accepted items). Every model: fee 100%. Qwen-32B
   airline still fails on decisions (told B 83%).
8. **Tests:**
   - the Mistral chat-template fallback (a fake tokenizer);
   - the low-memory loss equals the HF mean CE over reply tokens: checked with torch 2.13 in the anaconda env
     (`allclose`, 2.8388 = 2.8388); the pytest version skips in the project venv, which has no torch.
9. **Records:**
   - **Correction to the 22:48 entry:** the Qwen-32B lane did NOT train the airline adapters. Its first training
     attempt ran out of memory at 23:1x. Qwen-32B was relaunched returns-only, and trains on an H200.
   - **air3_B_s0** has a 94.5% parse rate (below the 95% flag).
   - **Audit limitation:** the sampled records keep only 2 of 16 sample texts (the decisions of all 16 are kept).
   - The HANDOFF "done" list is corrected at the end, when the runs have landed.
