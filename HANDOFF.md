# Handoff: Stale Weights

For whoever picks this up next. Read this, then FINDINGS.md (the results), RESEARCH_AGENDA.md (the plan, and the
preregistration each round-3 run was held to), and AUTONOMOUS_LOG.md (the round-3 notebook, entry by entry).

## The question in one paragraph

A support team fine-tunes a model on its old logs, which were produced under cancellation policy A (12 h window,
$35 fee), and later changes the policy only in the system prompt. Does the model follow the new prompt or the old
weights? What we've found (details and numbers in FINDINGS): a policy stated in full wins. A rule the prompt leaves
out is filled from the fine-tune, even when the prompt says "apply only the rules written here". Adapters trained
with the policy text in their prompts don't do this. Section "State of each experiment" below lists what round 3
added.

## Setup

- Apple Silicon Mac with ≥ 64 GB unified memory. The 14B model needs about 19 GB to train, the tool adapters about
  45 GB, and the 24B model more.
- `uv sync`, then `uv run pytest -q` (fast; no model is loaded).
- Models download from the Hugging Face hub on first use: `mlx-community/Qwen2.5-14B-Instruct-4bit` (main) and
  `mlx-community/Mistral-Small-24B-Instruct-2501-4bit` (second model).
- **Run one MLX job at a time.** Two at once ran out of GPU memory. Check `pgrep -fl mlx` first.
- Adapters are git-ignored (about 370 MB per chat adapter with all checkpoints). Retrain them with the commands
  below. The training data is committed and regenerable.

## How to run everything

Every driver skips steps whose output already exists, so a crashed run can simply be restarted.

```sh
# rounds 1-2 (see README "How to run"): MVP adapters, prompt variants, strict-sentence controls
uv run python -m staleweights.pipeline --chat-only
uv run python -m staleweights.pipeline --followups
uv run python -m staleweights.replication controls

# round 3 (2026-09-28): one task at a time, or all in order
uv run python -m staleweights.round3 mitig     # change notice / "no time limit" / 2nd strict wording (seed 0)
uv run python -m staleweights.round3 seeds1    # A, B, A_inprompt, B_inprompt with seed 1 (~4 h)
uv run python -m staleweights.round3 tool      # tool-log control A_lookup (~3.5 h)
uv run python -m staleweights.round3 mix       # data mixing A_mixread, A_mixin (~2 h)
uv run python -m staleweights.round3 model2    # Mistral-Small-24B: gate, then 4 adapters (~7.5 h)
uv run python -m staleweights.round3 seeds2    # seed 2 (~4 h)
uv run python -m staleweights.round3 dose      # checkpoints 25/50/100 (eval only)

# analysis (CPU only, seconds)
uv run python -m staleweights.analyze_controls # round 2's strict/neutral tables
uv run python -m staleweights.analyze_r3       # round 3: every preregistered test, Holm-corrected;
                                               # writes results_r3/tables.md, summary.{json,csv}
```

Timings on an M4 Max: training a chat adapter (200 steps, batch 8) takes about 40 min, and a tool adapter about
90 min. Evaluation runs at roughly 1 prompt/s, batched.

## Where things are

| path | what |
|---|---|
| `src/staleweights/policies.py`, `scenarios.py` | the three policies; bookings, prompts, the reply format |
| `data.py`, `agentic.py` | training sets (chat; tool transcripts), including round-3 seeds, mixes and `A_lookup` |
| `variants.py` | every prompt variant (partial, strict, neutral, notice, nowindow, ...) and the r3 suites |
| `evaluate.py`, `variants.py run`, `agentic.py eval` | generation and scoring (records keep the full text) |
| `round3.py` | the round-3 driver; `replication.py`, `pipeline.py` are rounds 1-2 |
| `analyze_r3.py` | the round-3 analysis (DTW, conflict denied, fee leak, McNemar + Newcombe, Holm) |
| `results/`, `results_partial/`, `results_strict/`, ... | rounds 1-2 (never modified in round 3) |
| `results_r3/{mitig,neutral,seeds,mix,tool,dose,mistral24b}/` | round 3 |
| `adapters/`, `adapters_tool/` | rounds 1-2 adapters; `adapters_r3/`, `adapters_r3_tool/`, `adapters_r3_mistral24b/`: round 3 |
| `reports/progress/` | the progress PDFs sent during round 3, with their HTML |

## Key definitions (so the numbers mean the same thing to you)

- **Window omitted**: the system prompt states only the fee rule of B or C ("B-fee"). Nothing says how long after
  booking a refund is allowed, so the stated policy refunds every booking.
- **Conflict items**: window items booked ≥ T hours ago, where T is the trained window (12 h for A, 48 h for B). The
  trained window denies them; the stated policy refunds them.
- **DTW, "decided by the trained window"**: over all 64 window items, the decision equals the trained window's
  decision (refund below T, deny at or above T). This is the primary measure, because it separates a recalled
  trained rule from a generic 24 h prior and from a plain deny bias.
- **Controls**: `X_inprompt` has the same bookings and replies as X, with X's policy text in every training prompt.
  `read` has a random policy in every prompt. `base` has no adapter.
- Every comparison is paired on the same items and prompt, using an exact McNemar test. The two prompts per cell
  reuse the same scenarios and are never pooled.

## State of each experiment (2026-09-28, GPU stopped by the user)

| experiment | status | where |
|---|---|---|
| Rounds 1–2 (MVP, partial prompts, strict sentence and its controls, agentic tool A) | done | FINDINGS §1–7 |
| Round 3 **mitig** (notice, second strict wording, "no time limit") | **3 of 7 adapters done** (A, B, A_inprompt). Stopped in the B_inprompt cell, with no partial file. read, A_bal_s0, base and the neutral-on-controls cells were not run. | FINDINGS §8, `results_r3/mitig/` |
| seeds1, seeds2 (Qwen, seeds 1–2) | coded, preregistered (P2), not run | `round3.py seeds` |
| model2 (Mistral-Small-24B; model already in the HF cache) | coded, P5, not run | `round3.py family` |
| model3 (Llama-3.1-8B, falling back to phi-4; needs download) | coded, P8, not run | `round3.py model3` |
| domain2 (categorical final-sale rule) | coded, tested, P7, not run (not even its base gate) | `domain2.py` |
| tool (tool-log control A_lookup; fee-only retrieval) | coded, tested, P3, not run | `agentic.py`, `round3.py tool` |
| mix (A_mixread, A_mixin) | coded, tested, P4, not run | `data.py` |
| dose (checkpoints 25/50/100, window omitted) | coded, P6, not run | `round3.py dose` |

**To resume exactly where it stopped:**

```sh
uv run python -m staleweights.round3 mitig seeds1 model2 domain2 model3 tool seeds2 mix model2_s1 model3_s1 domain2_s1 dose
```

Finished steps are skipped. About 42 h in total; each task is independent, so you can run them in any order.

## Open questions

1. **Does the headline replicate** across seeds (P2), families (P5, P8) and a categorical rule (P7)? Nothing
   claimed in FINDINGS covers more than one seed of Qwen-14B.
2. **Why does data mixing matter, if it does?** The mix arm (P4) tests whether the stale fill needs a model that
   never saw a policy in its prompt (the "novel cue" confound). This is the most informative mechanism experiment
   still open.
3. **The agentic lead.** With lookups in the training logs (A_lookup), is a retrieved policy followed, and is a
   fee-only retrieval filled from the weights? This is the control the user asked for. It closes or kills the
   post-hoc "retrieved B overridden 4/24" lead.
4. **Why does the change notice push *every* fine-tuned model to apply some window** (A_inprompt 22–25 → 42/56)?
   Is it the word "previous" priming a remembered rule, or any extra policy-talk? A cheap test: the notice on the
   base and read models (both queued in mitig).
5. **Reason vs decision.** Under "no time limit", A-tuned's reasons still cite 12 h on 8–9/56 but it decides
   correctly. How often does customer-facing text contradict the decision? This matters for a Moffatt-style
   liability story.
6. **Scale and dose.** Does a longer or larger fine-tune (more steps, more data) make even *stated* policies lose,
   bridging to Alignment Inertia's stated-conflict inertia? It is not planned. It would need longer training runs,
   e.g. 1,000+ steps.
7. **A closed model.** Hosted fine-tuning (OpenAI-style) would take about $15–30 and a fine-tuning-capable key; the
   OpenRouter key can't fine-tune. This needs the user's decision.
8. **Attribution** (TRAK, as in Alignment Inertia) is out of scope so far. A cheap proxy is the log-prob margin
   between deny and refund on conflict items.

## Rules this project follows

- Preregister before running (RESEARCH_AGENDA.md, committed first). Keep nulls. Holm within each family.
- Never modify earlier results or adapters; new work goes in new folders.
- The decision is the primary measure, and reason text is secondary. Compare against the fine-tuned controls, not
  only the base model.
- Label anything found post hoc. Don't quote the strict-sentence *rise* (13/24 → 24/24): it is a format effect.
