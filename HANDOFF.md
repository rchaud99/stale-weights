# Handoff: Stale Weights

For whoever picks this up next. Read this, then FINDINGS.md (the results), RESEARCH_AGENDA.md (the plan, and the
preregistration each run was held to: addenda 7–7f), and AUTONOMOUS_LOG.md (the notebook, entry by entry).

## The question in one paragraph

A support team fine-tunes a model on its old logs, produced under an old policy, and later changes the policy only in
the system prompt. When the new prompt simply leaves an old rule out, does the model still apply it? Across four open
models and two domains, the adapter tuned without the policy text leans on the omitted rule more than a same-data
adapter trained with the policy text in its prompts, in every seed tested, even when the prompt says "apply only the
rules written in this policy". How much this changes the actual answer varies. FINDINGS.md §10 has the summary.

## Current state (2026-09-30)

The results are the cloud study, design v2 (FINDINGS.md and `reports/final/stale_weights_final_report.pdf`).
It is balanced, preregistered (RESEARCH_AGENDA addenda 7–7f) and red-teamed (`reviews/`), and it ran on Modal
(H100, bf16) with the code in `src/staleweights/cloud/` (see its README) and `src/staleweights/v2/`. The Mac pilot
(4-bit, unbalanced, a fixed template) is on the git branch `old`; don't quote it as the result.

**Reproduce the analysis** (CPU, seconds; the raw records are committed in `results/`):

```sh
uv run python -m staleweights.v2.analysis qwen14b     # also: phi4 qwen32b mistral24b
uv run python -m staleweights.v2.arms skew            # log skew x reply style
uv run python -m staleweights.v2.arms tsweep          # decoding temperature
uv run python reports/final_report.py                 # the final report (needs Google Chrome for the PDF)
```

**Rerun on the cloud:** `src/staleweights/cloud/README.md` has the stage commands. Restarts are cheap: training skips
finished adapters. The whole v2 study cost about $61 on Modal (metered; the ledger estimate is $63), of which the
addendum 7f seed extension was about $19. The cap is $100.

**What's done** (status checked 2026-09-30 14:30, after the addendum 7f seed extension):
- Qwen-14B core: airline and returns 5 seeds each (returns seeds 3–4 added by 7f), with every arm.
- The skew × reply-style arm (2 seeds; 80 evaluations).
- The rule-structure ablation (air3, ret2; 2 seeds).
- Phi-4 and Mistral-24B headline cells (airline A + returns; 5 seeds, seeds 2–4 added by 7f).
- Qwen-32B, returns only (5 seeds, seeds 2–4 added by 7f). Its airline base gate failed, and the airline cells were
  not run.
- The temperature sweep: T = 0.3 and 0.7 on seeds 0–2, with T = 0 and 1.0 from the main runs.

**Headline status (FINDINGS §10).** With 5 seeds, 19 of 20 headline cells pass the preregistered bar (all but the
Qwen-14B airline 48 h). At 2–3 seeds it was 5/6 for Qwen-14B and 1/5, 1/5, 3/4 for Phi-4, Mistral and Qwen-32B; no
cell that passed then fails now. The extension was decided after the 2-seed results (a sequential design, stated in
addendum 7f), and 7f adds no further seeds. The direction is positive in 140/140 seed-cells.

**Open, dropped for time** (the code or the preregistration exists where noted):
1. **Qwen2.5-72B** (addendum 7a; the stage `qwen72b` and the 2×H200 functions exist, but were never run).
2. **The value sweep** (`sweep` stage; trained windows 6/18/36/72 h): does the leaked threshold track the trained
   value?
3. **Mixed training data** (`mixin` stage): half the logs with the policy text, half without.
4. **Number-free replies** (`numless` stage): logs that never state the window.
5. **Dose** (`dose` stage): checkpoints 25–100, plus 1,000 steps. Does a *stated* conflict start losing?
6. **The mirror scenario, where the stale answer is "yes":** an old exception the new policy omits. Accept-heavy logs
   can hide the leak in the current design (addendum 7b).
7. **More seeds for the arms.** The headline cells now have 5 seeds (addendum 7f; no more are to be added to them).
   The ablation, skew and temperature arms are still at 2–3 seeds.
8. **Why the air3 ablation removed the leak.** Adding a fare deny rule to the airline made the effect vanish, and
   removing the final-sale rule from returns made it the strongest. The preregistered explanation (a remaining deny
   rule) failed. Candidates: the share of training examples decided by the window (25% in air3 vs 50%), and the kind
   of remaining rule.
9. **Full fine-tuning and closed models:** untested.

## Setup

- `uv sync`, then `uv run pytest -q` (fast; no model is loaded). The analysis and the report run on CPU.
- Cloud runs need a Modal account and `uv sync --group cloud`; see `src/staleweights/cloud/README.md`. No Hugging
  Face token is needed (every model used is ungated).

## Where things are

| path | what |
|---|---|
| `src/staleweights/v2/` | design (policies, templates), items and training data, test suites, parsing, analysis, arms, ablation |
| `src/staleweights/cloud/app.py` | Modal: PEFT training mirroring mlx-lm's LoRA recipe, vLLM evaluation, the stage driver |
| `results/<model>/<domain>/` | raw records (`<adapter>-200.jsonl`); per model `tables.md`, `summary.json`, `info.jsonl`, `train_logs.jsonl` |
| `results/qwen14b/skew/`, `tsweep/` | the log-skew × reply-style arm and the temperature sweep |
| `results/ledger.jsonl` | estimated GPU cost of every cloud call |
| `logs/` | console logs of each cloud stage |
| `reports/` | `final_report.py` and its figures; the report itself is `reports/final/stale_weights_final_report.pdf` |
| `reviews/` | the two red-team reviews the v2 design answers |

## Rules this project follows

- Preregister before running (RESEARCH_AGENDA.md, committed first). Keep nulls. Holm within each family.
- Never modify earlier results; new runs go in new files.
- The decision is the primary measure, and reason text is secondary. Compare against the fine-tuned controls, not
  only the base model.
- Label anything found post hoc.