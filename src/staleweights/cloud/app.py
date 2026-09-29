"""Modal app: LoRA training (PyTorch + PEFT, mirroring mlx-lm's recipe) and vLLM evaluation on one H100.

Run from the repo root with Modal's CLI and PYTHONPATH=src (see README.md):

    PYTHONPATH=src modal run src/staleweights/cloud/app.py::validate        # port validation (Qwen-14B, A seed 0)

Everything persistent lives in the Modal Volume "stale-weights":
    /vol/hf                       Hugging Face cache (base weights, downloaded once)
    /vol/adapters/<slug>/<name>/step-XXXX/   PEFT adapters (checkpoint every 25 updates)
    /vol/results/<path>.jsonl     evaluation records (also returned to the local entrypoint and written to
                                  results_cloud/<path>.jsonl in the repo)
    /vol/logs/<slug>/<name>.json  training log: loss per step, timing, recipe
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[3] if len(_HERE.parents) > 3 else Path("/root")  # the repo locally; /root in the container
APP = "stale-weights"
GPU = "H100"
MODELS = {"qwen14b": "Qwen/Qwen2.5-14B-Instruct", "phi4": "microsoft/phi-4", "qwen32b": "Qwen/Qwen2.5-32B-Instruct"}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "peft", "accelerate", "numpy", "hf_transfer")
    .env({"HF_HOME": "/vol/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONPATH": "/root/src",
          "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(ROOT / "src", "/root/src")
    .add_local_dir(ROOT / "data", "/root/data")
)
app = modal.App(APP, image=image)
vol = modal.Volume.from_name("stale-weights", create_if_missing=True)

# --- the mlx-lm recipe (see README "Mismatches" for what differs) -------------------------------------------------
RECIPE = {"rank": 8, "scale": 20.0, "dropout": 0.0, "num_layers": 16, "lr": 1e-5, "batch": 8, "iters": 200,
          "save_every": 25, "adam_betas": (0.9, 0.999), "adam_eps": 1e-8, "adam_bias_correction": False,
          "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def encode(tok, text: str) -> list[int]:
    """Chat-template text -> ids, with no extra special tokens (the template already has them)."""
    return list(tok(text, add_special_tokens=False)["input_ids"])


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=3 * 3600, retries=0)
def train(slug: str, name: str, data_name: str | None = None, seed: int = 0, iters: int = 200) -> dict:
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    model_id = MODELS[slug]
    out = Path(f"/vol/adapters/{slug}/{name}")
    if (out / f"step-{iters:04d}" / "adapter_config.json").exists():
        return {"name": name, "skipped": True}
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa")
    n = model.config.num_hidden_layers
    cfg = LoraConfig(r=RECIPE["rank"], lora_alpha=RECIPE["rank"] * RECIPE["scale"], lora_dropout=RECIPE["dropout"],
                     target_modules=RECIPE["targets"], layers_to_transform=list(range(n - RECIPE["num_layers"], n)),
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()  # adapter weights in fp32
    t_load = time.time() - t0

    # data: mlx-lm ChatDataset with mask_prompt: loss on the tokens after the prompt (assistant reply + end tokens)
    rows = [json.loads(line) for line in open(f"/root/data/{data_name or name}/train.jsonl")]
    enc = []
    for r in rows:
        msgs = r["messages"]
        full = encode(tok, tok.apply_chat_template(msgs, tokenize=False))
        offset = len(encode(tok, tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)))
        enc.append((full, offset))
    # mlx-lm iterate_batches: sort by length, consecutive batches, batch order permuted every epoch
    idx = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
    B = RECIPE["batch"]
    batches = [idx[i:i + B] for i in range(0, len(idx) - B + 1, B)]
    rng = np.random.default_rng(seed)
    order: list[int] = []
    while len(order) < iters:
        order += list(rng.permutation(len(batches)))

    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    b1, b2 = RECIPE["adam_betas"]
    losses = []
    model.train()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for step in range(1, iters + 1):
        batch = [enc[j] for j in batches[order[step - 1]]]
        T = max(len(f) for f, _ in batch)
        ids = torch.full((len(batch), T), pad, dtype=torch.long)
        labels = torch.full((len(batch), T), -100, dtype=torch.long)
        att = torch.zeros((len(batch), T), dtype=torch.long)
        for k, (f, off) in enumerate(batch):
            ids[k, :len(f)] = torch.tensor(f)
            att[k, :len(f)] = 1
            labels[k, off:len(f)] = torch.tensor(f[off:])
        outp = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=labels.cuda())
        outp.loss.backward()
        with torch.no_grad():  # MLX Adam, bias_correction=False
            for p, mm, vv in zip(params, m, v):
                g = p.grad.float()
                mm.mul_(b1).add_(g, alpha=1 - b1)
                vv.mul_(b2).addcmul_(g, g, value=1 - b2)
                p.add_(-RECIPE["lr"] * mm / (vv.sqrt() + RECIPE["adam_eps"]))
                p.grad = None
        losses.append(float(outp.loss.detach()))
        if step % RECIPE["save_every"] == 0:
            model.save_pretrained(out / f"step-{step:04d}")
    log = {"name": name, "data": data_name or name, "model": model_id, "seed": seed, "iters": iters,
           "recipe": RECIPE, "losses": losses, "load_s": t_load, "train_s": time.time() - t0 - t_load,
           "total_s": time.time() - t0, "n_examples": len(enc), "n_batches": len(batches)}
    Path(f"/vol/logs/{slug}").mkdir(parents=True, exist_ok=True)
    Path(f"/vol/logs/{slug}/{name}.json").write_text(json.dumps(log))
    vol.commit()
    return {k: log[k] for k in ("name", "total_s", "train_s", "load_s")} | {"final_loss": losses[-1], "loss25": losses[24]}


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=3 * 3600, retries=0)
def evaluate(slug: str, specs: list[dict]) -> dict:
    """specs: [{"out": "qwen14b/strict/A-200", "module": "variants", "condition": "strict_fee",
                "adapter": "A" or None, "step": 200, "max_tokens": 160}]. One vLLM engine for all specs."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    from staleweights.cloud import jobs as J

    t0 = time.time()
    model_id = MODELS[slug]
    llm = LLM(model=model_id, dtype="bfloat16", enable_lora=True, max_lora_rank=RECIPE["rank"], max_loras=1,
              max_model_len=4096, gpu_memory_utilization=0.90, seed=0)
    tok = llm.get_tokenizer()
    t_load = time.time() - t0
    written, timings = {}, {}
    for i, spec in enumerate(specs):
        t1 = time.time()
        js = J.build(spec["module"], spec.get("condition"))
        prompts = [TokensPrompt(prompt_token_ids=encode(tok, tok.apply_chat_template(
            j["messages"], tokenize=False, add_generation_prompt=True))) for j in js]
        mt = spec.get("max_tokens") or J.DEFAULT_TOKENS[spec["module"]]
        sp = SamplingParams(temperature=0.0, max_tokens=mt)
        lora = None
        if spec.get("adapter"):
            path = f"/vol/adapters/{slug}/{spec['adapter']}/step-{spec['step']:04d}"
            lora = LoRARequest(f"{spec['adapter']}-{spec['step']}", i + 1, path)
        outs = llm.generate(prompts, sp, lora_request=lora)
        texts = [o.outputs[0].text for o in outs]
        finish = [o.outputs[0].finish_reason for o in outs]
        recs = J.records(spec["module"], spec.get("condition"), js, texts, finish, model=model_id,
                         adapter=spec.get("adapter"), step=spec.get("step"), max_tokens=mt, backend="vllm-bf16")
        body = "".join(json.dumps(r) + "\n" for r in recs)
        dest = Path(f"/vol/results/{spec['out']}.jsonl")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body)
        written[spec["out"]] = body
        timings[spec["out"]] = time.time() - t1
    vol.commit()
    return {"written": written, "load_s": t_load, "timings": timings, "total_s": time.time() - t0}


# --- local entrypoints --------------------------------------------------------------------------------------------

H100_PER_S = 3.95 / 3600
MARGIN = 1.25
LEDGER = ROOT / "results_cloud" / "ledger.jsonl"


def _save(result: dict, what: str) -> float:
    out = ROOT / "results_cloud"
    for rel, body in result.get("written", {}).items():
        p = out / f"{rel}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    secs = result.get("total_s", 0.0)
    cost = secs * H100_PER_S * MARGIN
    out.mkdir(exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "what": what, "gpu_s": round(secs, 1),
                            "est_usd": round(cost, 3), **{k: v for k, v in result.items() if k in ("load_s", "train_s")}})
                + "\n")
    print(f"[{what}] {secs:.0f} GPU-s, est ${cost:.2f}")
    return cost


@app.local_entrypoint()
def validate():
    """Port validation: base gate + strict headline for Qwen-14B, unbalanced A and A_inprompt, seed 0."""
    for name in ("A", "A_inprompt"):
        r = train.remote("qwen14b", name, seed=0)
        _save(r, f"train qwen14b/{name}")
        print(r)
    specs = [{"out": "qwen14b/base", "module": "evaluate"},
             {"out": "qwen14b/strict/base", "module": "variants", "condition": "strict_fee"}]
    for name in ("A", "A_inprompt"):
        specs += [{"out": f"qwen14b/strict/{name}-200", "module": "variants", "condition": "strict_fee",
                   "adapter": name, "step": 200},
                  {"out": f"qwen14b/none/{name}-200", "module": "variants", "condition": "none_only",
                   "adapter": name, "step": 200}]
    r = evaluate.remote("qwen14b", specs)
    _save(r, "eval qwen14b validation")
    print({k: round(v) for k, v in r["timings"].items()}, "load", round(r["load_s"]))


@app.local_entrypoint()
def smoke():
    """Image + weights download + vLLM on 100 base prompts (no adapter)."""
    r = evaluate.remote("qwen14b", [{"out": "qwen14b/smoke/base-none", "module": "variants", "condition": "none_only"}])
    _save(r, "smoke qwen14b base none_only")
    print(r["timings"], "load", round(r["load_s"]))
