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
