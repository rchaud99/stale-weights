# Stale Weights

A small, local measurement study: if a customer-service model is LoRA fine-tuned on decisions made under one
cancellation policy, and the policy is later changed only in the prompt, does the model follow the new policy or
revert to the one in its weights?

Fictional airline "Nimbus Air"; three policies (A: 12 h free-cancellation window, $35 fee per passenger; B: 48 h,
$80; C, never trained: 96 h, $125); model `mlx-community/Qwen2.5-14B-Instruct-4bit`, LoRA via `mlx-lm` on an
Apple M4 Max. No API calls. Design, gates, all tables and limits are in **[FINDINGS.md](FINDINGS.md)**; the
literature check is in [RELATED.md](RELATED.md).

## Headline

- **Fine-tuning taught the policy, but a policy in the prompt still wins.** With no policy in the prompt, the A- and
  B-tuned models answer by their trained policy (64/64 and 60/64 window items, 32/32 fees, 4/4 probes at step 200).
  Told the *other* policy, they follow it: fee items state the trained fee in 0/32 cases at every checkpoint (25–200),
  and on the conflicting window items they are no worse than the untuned model on the same items (A-tuned told B:
  8/24 stale vs 7/24 for the base, the same boundary errors citing the prompted 48 h; B-tuned told A: 0/24 vs 0/24).
  With 24–32 items per cell this rules out reversion larger than roughly 10–15 points.
- **What the prompt leaves out is filled from the weights.** When the prompt states only one of the two rules, the
  omitted rule comes back from fine-tuning: with the window left out, 116/128 (A-tuned) and 88/128 (B-tuned) replies
  apply the trained 12 h / 48 h window (base 0/192); with the fee left out, 21/64 and 12/64 state the trained fee
  (base 0/96). Offered an "escalate to a human" option, with or without being told to use it when the policy doesn't
  cover something, the tuned models escalated in 0 of 1,920 replies.
- **Training with the policy text in every prompt (or with varied policies) avoids that, but installs no default:**
  those adapters fill a gap with the base model's own prior (a 24 h window, no fee), almost never with the trained
  numbers (policy-in-training adapters: 1/256 window and 0/128 fee replies; read adapter: 0/192 and 0/96).
  The long-handbook variant failed its base gate (the untuned model denied almost every refund), so it says nothing.
- **Agentic version (tool calls; A-tuned tool adapter only, step 200):** when looking the policy up is optional, the
  base agent calls `get_policy` first in 92/96 conversations, the A-tuned agent in 15/96 (11–15/96 across three
  identical copies). When it does look up, it follows the returned policy (26/26); when it doesn't, it acts on A: with
  B or C in the policy store, 22/24 and 41/48 conflicting window decisions and 28/32 and 25/32 fees follow A (base
  0/24, 3/48, 0/32, 0/32). Given an already-retrieved policy it mostly follows it (fees 0/32 follow A; windows 4/24
  for B, citing "the window is 12 hours", base 0/24; *tentative*, p = 0.13). The stale policy enters by skipping the
  lookup, not by overriding what was looked up.

![Answers following the trained policy, by training step](figures/reversion_chat.png)

![One rule omitted from the prompt: how often the answer fills it with the trained value](figures/omitted_rule.png)

![Agentic version: policy lookups and stale actions, base vs A-tuned tool adapter](figures/agentic.png)

## How to run

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
uv run python -m staleweights.report results/*.jsonl  # the original compact table
uv run python -m staleweights.report_tool results_tool/base.jsonl results_tool/A-200.jsonl
uv run python -m staleweights.figures --agentic
uv run python -m staleweights.rescore results/*.jsonl # rescore saved text after a scorer change
```

## Where things are

| path | what |
|---|---|
| `src/staleweights/` | policies, scenarios, data, training, evaluation, variants, agentic version, analysis, figures |
| `results/` | chat evaluations: `base.jsonl` (the gate), `{A,B}-{25,50,100,200}.jsonl`, `read-200.jsonl`, `{A,B}_inprompt-{25,200}.jsonl`; `base-7b.jsonl` and `base-14b-v1.jsonl` are superseded gates, kept |
| `results_partial/`, `results_abstain/`, `results_long/` | the follow-up prompt variants (`variants.py`); the long-handbook variant failed its base gate |
| `results_tool/` | agentic evaluations: `base.jsonl` (the gate that passed, format v5), `A-200.jsonl` (retrieved and optional deliveries only; tool B and read were not trained, for time); `base-v1.jsonl` … `base-v4.jsonl` are the failed gate attempts, kept (see the `AGENT_TOOLS` docstring in `agentic.py`) |
| `figures/` | `reversion_chat`, `omitted_rule`, `agentic`: PDF + PNG, made by `figures.py` |
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
