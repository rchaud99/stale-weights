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
