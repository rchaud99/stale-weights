# Cloud backend (Modal, H100/H200)

The Stale Weights experiments run on Modal. There are two pieces:

- **Training:** PyTorch + PEFT, mirroring the mlx-lm LoRA recipe used on the Mac.
- **Evaluation:** vLLM, with the same prompts, formats, parsers and scoring as the MLX runs.

The design-v2 study (RESEARCH_AGENDA addenda 7–7e) runs only here.

## Setup

- A Modal account, and the `modal` CLI (v1.x). No Hugging Face token is needed: every model used is ungated, and
  Llama was skipped because it is gated.
- Run from the repo root, with `PYTHONPATH=src` so the local entrypoint can import the package:

```sh
PYTHONPATH=src modal run src/staleweights/cloud/app.py::prior               # base-prior probe -> v2/returns_values.json
PYTHONPATH=src modal run src/staleweights/cloud/app.py::v2 --stage pilot    # Qwen-14B airline seed 0
PYTHONPATH=src modal run src/staleweights/cloud/app.py::v2 --stage air      # airline seeds 1-4
PYTHONPATH=src modal run src/staleweights/cloud/app.py::v2 --stage ret      # returns seeds 0-2 (+ base)
PYTHONPATH=src modal run src/staleweights/cloud/app.py::v2 --stage skew_eval     # skew x style (addendum 7b)
PYTHONPATH=src modal run src/staleweights/cloud/app.py::v2 --stage tsweep_lite   # temperature (7c/7e)
PYTHONPATH=src modal run src/staleweights/cloud/app.py::v2 --stage ablation      # air3 / ret2 (7d)
PYTHONPATH=src modal run src/staleweights/cloud/app.py::v2 --stage headline_phi4 # also headline_qwen32b, headline_mistral24b
PYTHONPATH=src modal run src/staleweights/cloud/app.py::v2 --stage download      # weights to the volume on CPU
```

Every stage is restartable: training skips adapters whose final checkpoint exists on the volume.

## Where things go

- **Volume `stale-weights`:**
  - `hf/`: the Hugging Face cache;
  - `v2/adapters/<model>/<name>/step-XXXX/`: PEFT adapters, one checkpoint every 25 updates;
  - `v2/logs/<model>/<name>.json`: loss per step, timing and recipe;
  - `v2/results/<domain>/...`: the result copies. These paths do **not** include the model slug, so later models
    overwrite earlier ones on the volume. The authoritative copies are local.
- **Local, authoritative:** `results/<model>/<domain>/<adapter>-200.jsonl`, written from the data each
  evaluation returns, plus `info.jsonl` (gates, timings), `train_logs.jsonl` and `results/ledger.jsonl`
  (GPU-seconds × rate × 1.25).

## The training recipe, and every mismatch with the MLX runs

| setting | MLX (Mac, rounds 1–3) | cloud (v2) |
|---|---|---|
| base weights | mlx-community 4-bit quantized | bf16 (the original HF checkpoint) |
| LoRA rank / scale | 8 / 20 | 8 / α = 160 (the same scale, 20) |
| dropout | 0 | 0 |
| modules | every linear layer in the last 16 blocks | every linear layer in the last third of the blocks (Qwen-14B 16/48, Phi-4 13/40, Qwen-32B 21/64, Mistral-24B 13/40) |
| LoRA initialisation | uniform A, zero B | PEFT default (Kaiming A, zero B) |
| adapter dtype | float32 | float32 (the base is bf16) |
| optimizer | Adam, lr 1e-5, betas 0.9/0.999, eps 1e-8, **no bias correction** | the same, implemented by hand (no bias correction) |
| batches | sorted by length, consecutive groups of 8, order permuted per epoch; unseeded (mlx-lm treats seed 0 as none) | the same construction, seeded |
| loss | mean over reply tokens (the prompt is masked) | the same (`v2/masking.py`, unit-tested with the Qwen tokenizer) |
| steps / checkpoints | 200 / every 25 | 200 / every 25 |
| decoding | mlx-lm batch_generate, greedy, 160 tokens | vLLM greedy, 512 tokens (v2), top-5 logprobs at the decision; sampling n = 16, T = 1 |

**Port validation** (old design, unbalanced A, seed 0; AUTONOMOUS_LOG 2026-09-29 ~20:05). The headline reproduces
qualitatively: tuned 54–56/56 vs control 39–41 vs base 0 (Mac: 56 vs 21–24 vs 1–2). The `validate` entrypoint, the old-design
job builder and these records are on the git branch `old`.

## Known operational notes

- vLLM needs `VLLM_USE_FLASHINFER_SAMPLER=0`. FlashInfer's sampler JIT needs nvcc, which the image lacks.
- The Hugging Face CDN sometimes returns 504; `download` retries, and resumes where it stopped.
- Spend is taken from `modal billing report --for today --show-resources`; the local ledger is an estimate.
