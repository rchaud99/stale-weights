# Stale Weights

**Question.** A support team fine-tunes a model on its historical decisions, made under an old policy, and later
changes the policy only in the prompt. If the new prompt simply leaves an old rule out, does the model still apply it?

**Answer (current results: [FINDINGS.md §9](FINDINGS.md), final report
[`reports/progress/progress_05_FINAL_2026-09-30_1013.pdf`](reports/progress/progress_05_FINAL_2026-09-30_1013.pdf)).**
Yes, when the fine-tuning data never showed the policy text. With a rule omitted and "apply only the rules written in
this policy" in the prompt, the adapter tuned without the policy text leans on the old rule more than a same-data
adapter trained with the policy text in its prompts. That direction held in every seed of every model and domain
tested (90/90 seed-cells). How much it changes the actual answer depends on the model, the domain, the decoding
temperature and the style of the training logs. Stating the removal explicitly reduces it; a change notice does not
reliably help.

Design v2 (2026-09-29/30): class-balanced LoRA fine-tunes (rank 8, last third of the layers, 200 steps), PEFT
training and vLLM evaluation on Modal H100s, preregistered (RESEARCH_AGENDA addenda 7–7e) after two red-team reviews
(`reviews/`). Two fictional domains (airline cancellation windows; retail return windows and final-sale categories),
four open models. Primary metric ΔLTC: how much more the tuned model's sampled P(deny) jumps across the old threshold
than the control's (0 = no use of the old rule, 1 = exact use).

| model | seeds | preregistered verdict | direction (seed-cells positive) |
|---|---|---|---|
| Qwen2.5-14B | airline 5, returns 3 | replicates in 5/6 cells (e.g. airline 12 h: +0.40, Holm p = .002) | 22/22 |
| Phi-4 14B | 2 | 1/5 cells pass model-level Holm (underpowered by design) | 10/10 |
| Mistral-Small-24B | 2 | 1/5 (airline 12 h: +0.54) | 10/10 |
| Qwen2.5-32B | 2 | 3/4 by Holm (returns windows +0.82 / +0.89); airline excluded, its base gate failed | 8/8 |
| Qwen2.5-14B, log skew × reply style | 2 | descriptive; fixed-template logs leak in greedy answers at every deny share | 40/40 |

What did not hold up is listed in FINDINGS §9.10 ("what died"): the pilot's greedy airline result as a general claim,
"applies the exact old number" everywhere, the change notice as a fix, and the rule-structure explanation.

![Primary ΔLTC for every model and cell](reports/progress/final_dltc.png)

![Airline, old 12 h window omitted: tuned vs same-data control](reports/progress/final_airA.png)

## Reproduce (CPU, seconds; the raw records are committed in `results_cloud/v2/`)

```sh
uv sync && uv run pytest -q
uv run python -m staleweights.v2.analysis qwen14b     # also: phi4 qwen32b mistral24b -> results_cloud/v2/<model>/tables.md
uv run python -m staleweights.v2.arms skew            # log skew x reply style
uv run python -m staleweights.v2.arms tsweep          # decoding temperature
uv run python reports/final_report.py                 # the final report (HTML + PDF)
```

Rerunning training and evaluation on Modal: [`src/staleweights/cloud/README.md`](src/staleweights/cloud/README.md)
(the whole v2 study cost about $42 of H100 time). How to pick the project up: [HANDOFF.md](HANDOFF.md) ("Current
state"). Notebook: [AUTONOMOUS_LOG.md](AUTONOMOUS_LOG.md). Related work: [RELATED.md](RELATED.md).

| path | what |
|---|---|
| `src/staleweights/v2/` | design v2: data, items, test suites, analysis, arms, ablation |
| `src/staleweights/cloud/` | Modal app: PEFT training mirroring mlx-lm, vLLM evaluation |
| `results_cloud/v2/<model>/<domain>/` | raw records (`<adapter>-200.jsonl`), `tables.md`, `summary.json` |
| `reports/` | report generators; `reports/progress/` holds the progress and final reports |
| `reviews/` | the two red-team reviews the v2 design answers |
| everything else (`src/staleweights/*.py`, `results*/`, `figures/`) | the Mac pilot below |

## Mac pilot (rounds 1–3, 2026-09-24/28): superseded, kept as history

These results come from one seed of one 4-bit model trained on **unbalanced** data with a fixed reply template. The
v2 study above supersedes them for every claim; [DIGEST.md](DIGEST.md) is the round-3 digest.

Fictional airline "Nimbus Air"; three policies (A: 12 h free-cancellation window, $35 fee per passenger; B: 48 h,
$80; C, never trained: 96 h, $125); model `mlx-community/Qwen2.5-14B-Instruct-4bit`, LoRA via `mlx-lm` on an
Apple M4 Max. No API calls. Design, gates, all tables and limits are in **[FINDINGS.md](FINDINGS.md)**; the
literature check is in [RELATED.md](RELATED.md).

### Headline (pilot)

- **Fine-tuning taught the policy, but a policy in the prompt still wins.** With no policy in the prompt, the A- and
  B-tuned models answer by their trained policy (64/64 and 60/64 window items, 32/32 fees, 4/4 probes at step 200).
  Told the *other* policy, they follow it: fee items state the trained fee in 0/32 cases in every cell where the fee
  had been learned (A at step 200, B at steps 25–200), and on the conflicting window items they are no worse than the untuned model on the same items (A-tuned told B:
  8/24 stale vs 7/24 for the base, the same boundary errors citing the prompted 48 h; B-tuned told A: 0/24 vs 0/24).
  With 24–48 items per cell this rules out reversion larger than roughly 10–15 points.
- **What the prompt leaves out is filled from the weights.** When the prompt states only one of the two rules, the
  omitted rule comes back from fine-tuning. With the window left out (the stated policy then refunds everything),
  the A-tuned model denies 38/56 and 38/56 of the bookings its 12 h window would deny (told only B's or C's fee), the
  B-tuned model 12/32 and 15/32 for its 48 h window; the base denies none. With the fee left out, the trained fee is
  stated on 8/32 and 13/32 (A) and 5/32 and 7/32 (B) fee items (base 0). Against the fine-tuned controls (policy in
  the training prompt, read adapter) the A-tuned decisions still differ (p = 0.004–0.011), the B-tuned ones do not
  (p = 0.035–0.58). Nobody escalated when offered the option, but neither did the base or read models on partial
  prompts, so that zero is a floor effect.
- **"Apply only the rules written in this policy" does not stop it** (FINDINGS 4.10). With that sentence added and
  the window still omitted, the A-tuned model denies 56/56 and 56/56 bookings older than its trained 12 h (every
  window item decided exactly by 12 h) and the B-tuned model 28/32 and 30/32 older than 48 h (none younger). The
  adapter trained on the same data with the policy text in its prompt stays at 21–24/56 and 13–15/32 (McNemar
  p = 9.8e-04 to 5.8e-11), and it and the read adapter never apply 12 h or 48 h, only a generic 24 h. Two runs with
  the items batched differently agree on 2,001/2,016 decisions. But the *rise* from no sentence to the strict one
  is not stale weights: the read adapter rises about as much, and for A a neutral sentence of the same length gives
  most of it. One seed, 96 test scenarios; one strict wording.
- **Training with the policy text in every prompt (or with varied policies) avoids that, but installs no default:**
  those adapters fill a gap with the base model's own prior (a 24 h window, no fee), almost never with the trained
  numbers (policy-in-training adapters: the reason cites the trained window in 1/256 replies and the trained fee is
  stated in 0/128; read adapter: 0/192 and 0/96).
  The long-handbook variant failed its base gate (the untuned model denied almost every refund), so it says nothing.
- **Agentic version (tool calls; A-tuned tool adapter only, step 200):** the tool adapter was trained on transcripts
  that never call `get_policy`, and it imitates them: when looking the policy up is optional, the base agent calls
  `get_policy` first in 92/96 conversations, the A-tuned agent in 15/96 (11–15/96 across three identical copies). When it does look up, it follows the returned policy (26/26); when it doesn't, it acts on A: with
  B or C in the policy store, 22/24 and 41/48 conflicting window decisions and 28/32 and 25/32 fees follow A (base
  0/24, 3/48, 0/32, 0/32). Given an already-retrieved policy it mostly follows it (fees 0/32 follow A; windows 4/24
  for B, citing "the window is 12 hours", base 0/24; found post hoc, *tentative*, p = 0.13). This is a deployment
  hazard (fine-tuning on logs from before a tool existed teaches the agent to skip it), not evidence that weights
  beat retrieval: what was looked up was followed.

![Answers following the trained policy, by training step](figures/reversion_chat.png)

![One rule omitted from the prompt: how often the answer fills it with the trained value](figures/omitted_rule.png)

![Window omitted, with the strict or a neutral sentence: tuned models vs fine-tuned controls](figures/strict_controls.png)

![Agentic version: policy lookups and stale actions, base vs A-tuned tool adapter](figures/agentic.png)

### How to run

Everything is `uv run` from the repo root; one MLX job at a time (two at once ran out of memory).

```sh
uv sync
uv run pytest -q                                      # fast tests of the pure logic (no model)

# data
uv run python -m staleweights.data                    # data/{A,B,read,A_inprompt,B_inprompt}
uv run python -m staleweights.agentic data            # data_tool/{A,B,read}

# base gates (the untuned model must follow a prompted policy before anything else means anything)
uv run python -m staleweights.evaluate --out results/base.jsonl
uv run python -m staleweights.agentic eval --out results_tool/base.jsonl --batch 64

# adapters and their evaluations (skips finished steps; restartable)
uv run python -m staleweights.pipeline --chat-only    # train A, B, read; evaluate checkpoints 25/50/100/200
uv run python -m staleweights.pipeline --followups    # partial prompts, escalate option, in-prompt adapters,
                                                      # long handbook prompt, agentic base gate
uv run python -m staleweights.pipeline --tool         # agentic tool adapters (~90 min each) and evaluations
uv run python -m staleweights.replication strict      # strict sentence on base, A-200, B-200 (results_strict/)
uv run python -m staleweights.replication balanced    # class-balanced adapters (only A seed 0 finished)
uv run python -m staleweights.replication controls    # strict on the fine-tuned controls, neutral sentence,
                                                      # shuffled reruns (skips finished steps)
# what actually ran (time): only tool adapter A, evaluated on two deliveries
uv run python -c "from staleweights.train import train; train('A', 'data_tool', 'adapters_tool')"
uv run python -m staleweights.agentic eval --adapter adapters_tool/A --checkpoint 200 \
    --out results_tool/A-200.jsonl --batch 64 --deliveries retrieved,optional

# tables and figures (CPU only)
uv run python -m staleweights.analyze results/base.jsonl results/A-*.jsonl results/B-*.jsonl \
    results/read-*.jsonl results/*_inprompt-*.jsonl
uv run python -m staleweights.analyze --partial results_partial/*.jsonl
uv run python -m staleweights.analyze --abstain results_abstain/*.jsonl
uv run python -m staleweights.analyze --long results_long/*.jsonl
uv run python -m staleweights.analyze_controls        # strict/neutral/controls, decision level, exact McNemar
uv run python -m staleweights.analyze_rep strict      # the replication round's long-format tables
uv run python -m staleweights.report results/*.jsonl  # the original compact table
uv run python -m staleweights.report_tool results_tool/base.jsonl results_tool/A-200.jsonl
uv run python -m staleweights.figures --agentic
uv run python -m staleweights.rescore results/*.jsonl # rescore saved text after a scorer change
```

### Where things are

| path | what |
|---|---|
| `src/staleweights/` | policies, scenarios, data, training, evaluation, variants, agentic version, analysis, figures |
| `results/` | chat evaluations: `base.jsonl` (the gate), `{A,B}-{25,50,100,200}.jsonl`, `read-200.jsonl`, `{A,B}_inprompt-{25,200}.jsonl`; `base-7b.jsonl` and `base-14b-v1.jsonl` are superseded gates, kept |
| `results_partial/`, `results_abstain/`, `results_long/` | the follow-up prompt variants (`variants.py`); the long-handbook variant failed its base gate |
| `results_strict/`, `results_neutral/` | the strict sentence (`{A,B}-200`, `base`: all strict prompts; controls `{A_inprompt,B_inprompt,read,A_bal_s0}-200`: window omitted), `rest/` (the controls' other strict prompts), `rerun/` (the window-omitted strict cells again, items shuffled into other batches); the neutral sentence; `controls.{json,csv}` from `analyze_controls` |
| `results_balanced/` | the class-balanced A adapter, seed 0 (the other seeds and balanced B were not finished) |
| `results_tool/` | agentic evaluations: `base.jsonl` (the gate that passed, format v5), `A-200.jsonl` (retrieved and optional deliveries only; tool B and read were not trained, for time); `base-v1.jsonl` … `base-v4.jsonl` are the failed gate attempts, kept (see the `AGENT_TOOLS` docstring in `agentic.py`) |
| `figures/` | `reversion_chat`, `omitted_rule`, `strict_controls`, `agentic`: PDF + PNG, made by `figures.py` |
| `adapters/`, `adapters_tool/` | LoRA weights (git-ignored; large) |
| `logs_*.txt` | training and pipeline logs |

Each record in a results file keeps the full generated text, so everything can be rescored without regenerating.

## Not in the repository

- `adapters/`, `adapters_tool/` (~3 GB of LoRA `*.safetensors` checkpoints). Retrain with the pipeline commands
  above (`staleweights.pipeline`, `staleweights.train`). The training data in `data/` and `data_tool/` is committed,
  and regenerable with `staleweights.data` and `staleweights.agentic data`.
- `.venv/` (`uv sync`) and caches.

## License

The code and write-ups are MIT (see `LICENSE`). Base models and third-party data keep their own licenses.
