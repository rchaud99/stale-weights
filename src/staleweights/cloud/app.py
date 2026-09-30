"""Modal app: LoRA training (PyTorch + PEFT, mirroring mlx-lm's recipe) and vLLM evaluation on one H100.

Run from the repo root with Modal's CLI and PYTHONPATH=src (see README.md):

    PYTHONPATH=src modal run src/staleweights/cloud/app.py::prior           # base-prior probe
    PYTHONPATH=src modal run src/staleweights/cloud/app.py::v2 --stage <stage>   # see cloud/README.md

Everything persistent lives in the Modal Volume "stale-weights":
    /vol/hf                       Hugging Face cache (base weights, downloaded once)
    /vol/adapters/<slug>/<name>/step-XXXX/   PEFT adapters (checkpoint every 25 updates)
    /vol/results/<path>.jsonl     evaluation records (also returned to the local entrypoint and written to
                                  results/<model>/<path>.jsonl in the repo)
    /vol/v2/logs/<slug>/<name>.json  training log: loss per step, timing, recipe
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


# --- local entrypoints --------------------------------------------------------------------------------------------

H100_PER_S = 3.95 / 3600
MARGIN = 1.25
LEDGER = ROOT / "results" / "ledger.jsonl"


def _save(result: dict, what: str, per_s: float | None = None) -> float:
    out = ROOT / "results"
    for rel, body in result.get("written", {}).items():
        p = out / f"{rel}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    secs = result.get("total_s", 0.0) * result.get("gpus", 1)
    cost = secs * (per_s or H100_PER_S) * MARGIN
    out.mkdir(exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "what": what, "gpu_s": round(secs, 1),
                            "est_usd": round(cost, 3), **{k: v for k, v in result.items() if k in ("load_s", "train_s")}})
                + "\n")
    print(f"[{what}] {secs:.0f} GPU-s, est ${cost:.2f}")
    return cost


# ======================================================================================================================
# Design v2 (addendum 7)
# ======================================================================================================================

MODELS.update({"llama8b": "meta-llama/Llama-3.1-8B-Instruct",
               "mistral24b": "mistralai/Mistral-Small-24B-Instruct-2501", "qwen72b": "Qwen/Qwen2.5-72B-Instruct"})
BIG = {"qwen72b": {"gpu": "H200:2", "tp": 2}}
LOWMEM = {"qwen32b", "mistral24b"}
"""Train these with the low-memory loss path (same objective); Qwen-32B ran out of memory on the float logits."""


@app.function(volumes={"/vol": vol}, timeout=3 * 3600, cpu=4, memory=16384)
def download(slug: str) -> dict:
    """Fetch the base weights into the volume on a CPU container (cheap), before any GPU function needs them."""
    from huggingface_hub import snapshot_download
    t0 = time.time()
    for attempt in range(8):  # the HF CDN sometimes answers 504; snapshot_download resumes
        try:
            path = snapshot_download(MODELS[slug], allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                                                                   "*.jinja", "tokenizer*"])
            break
        except Exception as e:  # noqa: BLE001
            print(f"download {slug} attempt {attempt}: {e}")
            vol.commit()
            time.sleep(20 * (attempt + 1))
    else:
        raise RuntimeError(f"download {slug} failed")
    vol.commit()
    return {"path": path, "download_s": time.time() - t0}
V2_PRIOR_TOKENS = 512


def _v2_rows(spec: dict) -> list[dict]:
    from staleweights.v2 import items as I
    if spec["domain"] in ("air3", "ret2"):
        from staleweights.v2 import ablation as X
        return (X.air3_train if spec["domain"] == "air3" else X.ret2_train)(spec["policy"], spec["seed"], spec["inprompt"])
    if spec["domain"] == "air":
        return I.air_train(spec["policy"], spec["seed"], spec["inprompt"], spec.get("balanced", True),
                           window=spec.get("window"), numberless=spec.get("numberless", False),
                           mixin=spec.get("mixin", False), deny_share=spec.get("deny_share"),
                           fixed=spec.get("fixed", False))
    return I.ret_train(spec["policy"], spec["seed"], spec["inprompt"], spec.get("balanced", True))


def _train_one(model, tok, rows: list[dict], out: Path, seed: int, iters: int, frac: float = 1 / 3,
               lowmem: bool = False) -> dict:
    """One LoRA adapter on an already-loaded base (the mlx-lm recipe; see RECIPE). Returns the log."""
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model

    t0 = time.time()
    torch.manual_seed(seed)
    n = model.config.num_hidden_layers
    k = round(n * frac)
    targets = sorted({name.split(".")[-1] for name, m in model.named_modules()
                      if isinstance(m, torch.nn.Linear) and ".layers." in name})
    cfg = LoraConfig(r=RECIPE["rank"], lora_alpha=RECIPE["rank"] * RECIPE["scale"], lora_dropout=RECIPE["dropout"],
                     target_modules=targets, layers_to_transform=list(range(n - k, n)), bias="none",
                     task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    params = [p for p in pm.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    from staleweights.v2.masking import mask_example
    enc = [mask_example(tok, r["messages"]) for r in rows]
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
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    losses, masked = [], []
    pm.train()
    for step in range(1, iters + 1):
        batch = [enc[j] for j in batches[order[step - 1]]]
        T = max(len(f) for f, _ in batch)
        ids = torch.full((len(batch), T), pad, dtype=torch.long)
        labels = torch.full((len(batch), T), -100, dtype=torch.long)
        att = torch.zeros((len(batch), T), dtype=torch.long)
        for q, (f, off) in enumerate(batch):
            ids[q, :len(f)] = torch.tensor(f)
            att[q, :len(f)] = 1
            labels[q, off:len(f)] = torch.tensor(f[off:])
        if step == 1:
            masked = [int((labels[q] != -100).sum()) for q in range(len(batch))]
        if lowmem:  # the same objective (mean CE over reply tokens), without materializing float logits for every position
            logits = pm(input_ids=ids.cuda(), attention_mask=att.cuda()).logits[:, :-1]
            tgt = labels[:, 1:].cuda()
            keep = tgt != -100
            loss = torch.nn.functional.cross_entropy(logits[keep].float(), tgt[keep], reduction="mean")
            del logits
            loss.backward()
            out_ = type("O", (), {"loss": loss})()
        else:
            out_ = pm(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=labels.cuda())
            out_.loss.backward()
        with torch.no_grad():
            for p, mm, vv in zip(params, m, v):
                g = p.grad.float()
                mm.mul_(b1).add_(g, alpha=1 - b1)
                vv.mul_(b2).addcmul_(g, g, value=1 - b2)
                p.add_(-RECIPE["lr"] * mm / (vv.sqrt() + RECIPE["adam_eps"]))
                p.grad = None
        losses.append(float(out_.loss.detach()))
        if step % RECIPE["save_every"] == 0:
            pm.save_pretrained(out / f"step-{step:04d}")
    pm.unload()  # remove the LoRA layers, keep the base for the next adapter
    return {"losses": losses, "train_s": time.time() - t0, "targets": targets, "layers": [n - k, n - 1],
            "first_batch_loss_tokens": masked}


@app.function(gpu="H200", volumes={"/vol": vol}, timeout=4 * 3600, retries=0)
def train_v2_h200(slug: str, specs: list[dict]) -> dict:
    """train_v2 on one H200 (141 GB): Qwen-32B runs out of memory on an 80 GB H100 at batch 8."""
    r = _train_v2(slug, specs)
    r["rate"] = "H200"
    return r


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=4 * 3600, retries=0)
def train_v2(slug: str, specs: list[dict]) -> dict:
    return _train_v2(slug, specs)


def _train_v2(slug: str, specs: list[dict]) -> dict:
    """specs: [{"name", "domain", "policy", "seed", "inprompt", "balanced", "iters", "window"}]; one base load."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODELS[slug])
    model = AutoModelForCausalLM.from_pretrained(MODELS[slug], dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa")
    load_s = time.time() - t0
    logs = {}
    for spec in specs:
        out = Path(f"/vol/v2/adapters/{slug}/{spec['name']}")
        iters = spec.get("iters", 200)
        if (out / f"step-{iters:04d}" / "adapter_config.json").exists():
            logs[spec["name"]] = {"skipped": True}
            continue
        log = _train_one(model, tok, _v2_rows(spec), out, spec["seed"], iters, lowmem=slug in LOWMEM)
        log.update({"spec": spec, "model": MODELS[slug]})
        Path(f"/vol/v2/logs/{slug}").mkdir(parents=True, exist_ok=True)
        Path(f"/vol/v2/logs/{slug}/{spec['name']}.json").write_text(json.dumps(log))
        vol.commit()
        logs[spec["name"]] = {k: log[k] for k in ("train_s", "layers", "first_batch_loss_tokens")} | {
            "loss": [log["losses"][0], log["losses"][24], log["losses"][-1]]}
    return {"logs": logs, "load_s": load_s, "total_s": time.time() - t0}


@app.function(gpu="H200:2", volumes={"/vol": vol}, timeout=4 * 3600, retries=0)
def train_v2_big(slug: str, specs: list[dict]) -> dict:
    """train_v2 for models that need two GPUs (device_map="auto" spreads the layers)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODELS[slug])
    model = AutoModelForCausalLM.from_pretrained(MODELS[slug], dtype=torch.bfloat16, device_map="auto",
                                                 attn_implementation="sdpa")
    load_s = time.time() - t0
    logs = {}
    for spec in specs:
        out = Path(f"/vol/v2/adapters/{slug}/{spec['name']}")
        iters = spec.get("iters", 200)
        if (out / f"step-{iters:04d}" / "adapter_config.json").exists():
            logs[spec["name"]] = {"skipped": True}
            continue
        log = _train_one(model, tok, _v2_rows(spec), out, spec["seed"], iters)
        log.update({"spec": spec, "model": MODELS[slug]})
        Path(f"/vol/v2/logs/{slug}").mkdir(parents=True, exist_ok=True)
        Path(f"/vol/v2/logs/{slug}/{spec['name']}.json").write_text(json.dumps(log))
        vol.commit()
        logs[spec["name"]] = {k: log[k] for k in ("train_s", "layers")} | {"loss": [log["losses"][0], log["losses"][-1]]}
    return {"logs": logs, "load_s": load_s, "total_s": time.time() - t0, "gpus": 2}


def _decision_logprobs(completion, domain: str) -> dict | None:
    """Top-5 logprobs at the first token of the decision value (after '"decision": "')."""
    text = ""
    for k, (tid, lp) in enumerate(zip(completion.token_ids, completion.logprobs or [])):
        if '"decision": "' in text or '"decision":"' in text:
            return {v.decoded_token: round(v.logprob, 4) for v in lp.values()}
        text += lp[tid].decoded_token if tid in lp else ""
    return None


def _eval_v2(slug: str, runs: list[dict], tp: int = 1) -> dict:
    """runs: [{"out": rel path, "adapter": name|None, "step": int, "domain": "air"|"ret", "trained": "A"|"B"|None,
    "gate": bool, "suite": bool, "options": {...}}]. Gate first; the suite only if the gate passes (implant rule)."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest

    from staleweights.v2 import suite as S

    t0 = time.time()
    llm = LLM(model=MODELS[slug], dtype="bfloat16", enable_lora=True, max_lora_rank=RECIPE["rank"], max_loras=1,
              max_model_len=4096, gpu_memory_utilization=0.92, seed=0, tensor_parallel_size=tp)
    tok = llm.get_tokenizer()
    load_s = time.time() - t0
    written, info = {}, {}

    def ids(messages, prefix=""):
        return encode(tok, tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix)

    for n_run, run in enumerate(runs):
        t1 = time.time()
        dom = run["domain"]
        lora = None
        if run.get("adapter"):
            lora = LoRARequest(f"{run['adapter']}-{run['step']}", n_run + 1,
                               f"/vol/v2/adapters/{slug}/{run['adapter']}/step-{run['step']:04d}")
        recs = []

        def greedy(jobs):
            outs = llm.generate([TokensPrompt(prompt_token_ids=ids(j["messages"])) for j in jobs],
                                SamplingParams(temperature=0.0, max_tokens=V2_PRIOR_TOKENS, logprobs=5), lora_request=lora)
            res = []
            dd = {"air3": "air", "ret2": "ret"}.get(dom, dom)
            for j, o in zip(jobs, outs):
                c = o.outputs[0]
                res.append({"mode": "greedy", "meta": j["meta"], "text": c.text, "finish_reason": c.finish_reason,
                            "decision": S.decision_of(c.text, dd), "decision_top5": _decision_logprobs(c, dd)})
            return res

        if dom in ("air3", "ret2"):
            from staleweights.v2 import ablation as X
        if run.get("gate"):
            g = greedy(X.gate(dom) if dom in ("air3", "ret2") else S.gate_jobs(dom))
            if not run.get("trained"):
                ok, rates = True, {}
            elif dom in ("air3", "ret2"):
                ok, rates = X.gate_ok(g, dom, run["trained"])
            else:
                ok, rates = S.gate_ok(g, dom, run["trained"])
            recs += g
            info[run["out"]] = {"gate_ok": ok, "gate_rates": rates}
            if not ok and run.get("enforce", True):
                written[run["out"]] = "".join(json.dumps(r) + "\n" for r in recs)
                continue
        if run.get("suite", True):
            opts = run.get("options", {})
            if dom in ("air3", "ret2"):
                jobs = X.jobs(dom, run.get("trained"))
            elif run.get("sweep"):
                jobs = S.sweep_jobs(run["sweep"])
            elif run.get("skew"):
                jobs = S.skew_jobs(run["trained"])
            elif run.get("temperature") is not None:
                jobs = (S.tsweep_lite_jobs if run.get("lite") else S.tsweep_jobs)(run.get("trained"))
            else:
                jobs = (S.air_jobs if dom == "air" else S.ret_jobs)(run.get("trained"), **opts)
            recs += greedy([j for j in jobs if j["mode"] == "greedy"])
            sj = [j for j in jobs if j["mode"] == "sample"]
            if sj:
                outs = llm.generate([TokensPrompt(prompt_token_ids=ids(j["messages"])) for j in sj],
                                    [SamplingParams(n=16, temperature=run.get("temperature", 1.0),
                                                    max_tokens=V2_PRIOR_TOKENS, seed=j["seed"]) for j in sj],
                                    lora_request=lora)
                for j, o in zip(sj, outs):
                    ds = [S.decision_of(c.text, {"air3": "air", "ret2": "ret"}.get(dom, dom)) for c in o.outputs]
                    parsed = [d for d in ds if d is not None]
                    recs.append({"mode": "sample", "temperature": run.get("temperature", 1.0), "meta": j["meta"],
                                 "n": len(ds), "n_parsed": len(parsed),
                                 "p_deny": (sum(d == "deny" for d in parsed) / len(parsed)) if parsed else None,
                                 "decisions": ds, "texts_head": [c.text for c in o.outputs[:2]]})
            pj = [j for j in jobs if j["mode"] == "probe"]
            if pj:
                acc = S.ACCEPT[dom]
                prompts, spans = [], []
                for j in pj:
                    base_ids = ids(j["messages"], j["prefix"])
                    for word in ("deny", acc):
                        full = ids(j["messages"], j["prefix"] + word)
                        clean = full[:len(base_ids)] == base_ids  # False if the word merges with the prefix's last token
                        cut = len(base_ids) if clean else len(base_ids) - 1
                        prompts.append(TokensPrompt(prompt_token_ids=full))
                        spans.append((cut, len(full), clean))
                outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
                                    lora_request=lora)
                lps = []
                for o, (a, b, _) in zip(outs, spans):
                    toks = o.prompt_token_ids
                    lps.append(sum(o.prompt_logprobs[t][toks[t]].logprob for t in range(a, b)))
                import math
                for q, j in enumerate(pj):
                    ld, la = lps[2 * q], lps[2 * q + 1]
                    recs.append({"mode": "probe", "meta": j["meta"], "lp_deny": ld, "lp_accept": la,
                                 "p_deny": 1 / (1 + math.exp(la - ld)),
                                 "clean_split": spans[2 * q][2] and spans[2 * q + 1][2]})
        body = "".join(json.dumps(r) + "\n" for r in recs)
        dest = Path(f"/vol/v2/results/{run['out']}.jsonl")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body)
        written[run["out"]] = body
        info.setdefault(run["out"], {})["seconds"] = time.time() - t1
        vol.commit()
    return {"written": written, "info": info, "load_s": load_s, "total_s": time.time() - t0, "gpus": tp}


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=4 * 3600, retries=0)
def eval_v2(slug: str, runs: list[dict]) -> dict:
    return _eval_v2(slug, runs)


@app.function(gpu="H200:2", volumes={"/vol": vol}, timeout=4 * 3600, retries=0)
def eval_v2_big(slug: str, runs: list[dict]) -> dict:
    return _eval_v2(slug, runs, tp=2)


@app.function(gpu=GPU, volumes={"/vol": vol}, timeout=3600, retries=0)
def probe_prior(slug: str = "qwen14b") -> dict:
    """The preregistered base-prior probe for the returns values (addendum 7, item 3)."""
    import random as _r

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from staleweights.v2 import design as D
    from staleweights.v2 import suite as S

    t0 = time.time()
    llm = LLM(model=MODELS[slug], dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.90, seed=0)
    tok = llm.get_tokenizer()

    def ids(msgs):
        return encode(tok, tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    fee_q = [{"role": "system", "content": D.RET_AGENT}, {"role": "user", "content":
             "What restocking fee, in percent of the item price, does Kestrel Outfitters charge on accepted returns?"
             '\n\nReply with JSON only: {"fee_percent": <number>}'}]
    o = llm.generate([TokensPrompt(prompt_token_ids=ids(fee_q))], SamplingParams(n=40, temperature=1.0, max_tokens=64, seed=1))
    from staleweights.v2.parsing import number, parse
    fees = [number((parse(c.text) or {}).get("fee_percent")) for c in o[0].outputs]
    rng = _r.Random("v2-prior-probe")
    prompts, cats = [], []
    for cat in D.CANDIDATES:
        for _ in range(10):
            sc = D.ret_draw(rng, cat, 3, train=False)
            prompts.append(TokensPrompt(prompt_token_ids=ids([{"role": "system", "content": D.RET_AGENT},
                                                              {"role": "user", "content": sc.user()}])))
            cats.append(cat)
    outs = llm.generate(prompts, SamplingParams(n=4, temperature=1.0, max_tokens=256, seed=2))
    deny: dict = {c: [] for c in D.CANDIDATES}
    for cat, o in zip(cats, outs):
        for c in o.outputs:
            d = S.decision_of(c.text, "ret")
            if d is not None:
                deny[cat].append(d == "deny")
    return {"fees": fees, "deny_rate": {c: (sum(v) / len(v) if v else None, len(v)) for c, v in deny.items()},
            "total_s": time.time() - t0}


@app.local_entrypoint()
def prior():
    """Run the probe and apply the preregistered selection rule; writes src/staleweights/v2/returns_values.json."""
    r = probe_prior.remote()
    _save(r, "probe_prior qwen14b")
    fees = [f for f in r["fees"] if f is not None]
    share = {c: sum(f == c for f in fees) / max(1, len(fees)) for c in (6, 12, 18, 22, 28)}
    chosen = [c for c in (6, 12, 18, 22, 28) if share[c] < 0.10][:3]
    rates = sorted(r["deny_rate"].items(), key=lambda kv: (kv[1][0] if kv[1][0] is not None else 1.0, kv[0]))
    ranked = [c for c, _ in rates]
    values = {"fees": dict(zip("ABC", chosen)),
              "sets": {"A": [ranked[0], ranked[3]], "B": [ranked[1], ranked[4]], "C": [ranked[2], ranked[5]]},
              "neutral": [ranked[6], ranked[7]], "probe": {"fee_samples": r["fees"], "fee_share": share,
                                                           "deny_rate": r["deny_rate"], "ranked": ranked}}
    (ROOT / "src" / "staleweights" / "v2" / "returns_values.json").write_text(json.dumps(values, indent=1))
    print(json.dumps(values, indent=1))


# --- v2 study driver (local) --------------------------------------------------------------------------------------

def v2_name(domain: str, policy: str, seed: int, inprompt: bool, balanced: bool = True, extra: str = "") -> str:
    return f"{domain}_{policy}{'_ctrl' if inprompt else ''}{'' if balanced else '_unbal'}{extra}_s{seed}"


def v2_specs(domain: str, seeds, policies="AB", balanced: bool = True) -> list[dict]:
    return [{"name": v2_name(domain, t, s, c, balanced), "domain": domain, "policy": t, "seed": s, "inprompt": c,
             "balanced": balanced} for s in seeds for t in policies for c in (False, True)]


def _save_v2(result: dict, slug: str, what: str) -> float:
    out = ROOT / "results" / slug
    for rel, body in result.get("written", {}).items():
        p = out / f"{rel}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    if result.get("info"):
        p = out / "info.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            for k, v in result["info"].items():
                f.write(json.dumps({"out": k, **v}) + "\n")
    if result.get("logs"):
        p = out / "train_logs.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a") as f:
            for k, v in result["logs"].items():
                f.write(json.dumps({"name": k, **v}) + "\n")
    per_s = 4.54 / 3600 if (slug in BIG or result.get("rate") == "H200") else None
    return _save({"total_s": result.get("total_s", 0.0), "load_s": result.get("load_s"), "gpus": result.get("gpus", 1)},
                 f"{what} [{slug}]", per_s)


def run_block(slug: str, specs: list[dict], base_runs: list[dict], n_eval: int = 2, label: str = "") -> dict:
    """Train specs (one container), evaluate the base concurrently, then evaluate the adapters in n_eval parallel
    containers (gate first, suite only if the implant passes). Returns the gate info per adapter."""
    calls = []
    if base_runs:
        calls.append(("eval base", eval_v2.spawn(slug, base_runs)))
    tr = train_v2.spawn(slug, specs) if specs else None
    if tr is not None:
        _save_v2(tr.get(), slug, f"train {label}")
    runs = [{"out": f"{s['domain']}/{s['name']}-200", "adapter": s["name"], "step": 200, "domain": s["domain"],
             "trained": s["policy"], "gate": True, "enforce": not s["inprompt"]} for s in specs]
    chunks = [runs[i::n_eval] for i in range(n_eval)] if runs else []
    for ch in chunks:
        if ch:
            calls.append((f"eval {label}", eval_v2.spawn(slug, ch)))
    info = {}
    for what, c in calls:
        r = c.get()
        _save_v2(r, slug, what)
        info.update(r.get("info", {}))
    return info


def implant_fallback(slug: str, specs: list[dict], info: dict) -> dict:
    """Addendum 6/7 implant rule: a tuned adapter whose no-policy gate fails at step 200 is retrained to 400 steps; the
    first of checkpoints 250/300/350/400 that passes the gate is used, and its control is retrained to 400 and
    evaluated at the same step. Only the gate (no-policy) cells are used to choose."""
    weak = [sp for sp in specs if not sp["inprompt"] and info.get(f"{sp['domain']}/{sp['name']}-200", {}).get("gate_ok") is False]
    if not weak:
        return {}
    longs = []
    for sp in weak:
        ctrl = next(c for c in specs if c["inprompt"] and c["policy"] == sp["policy"] and c["seed"] == sp["seed"]
                    and c["domain"] == sp["domain"])
        longs += [{**sp, "name": sp["name"] + "_x400", "iters": 400}, {**ctrl, "name": ctrl["name"] + "_x400", "iters": 400}]
    _save_v2(train_v2.remote(slug, longs), slug, "train implant fallback x400")
    gates = [{"out": f"{sp['domain']}/implant/{sp['name']}_x400-{st}", "adapter": sp["name"] + "_x400", "step": st,
              "domain": sp["domain"], "trained": sp["policy"], "gate": True, "suite": False}
             for sp in weak for st in (250, 300, 350, 400)]
    r = eval_v2.remote(slug, gates)
    _save_v2(r, slug, "eval implant fallback gates")
    out = {}
    runs = []
    for sp in weak:
        step = next((st for st in (250, 300, 350, 400)
                     if r["info"].get(f"{sp['domain']}/implant/{sp['name']}_x400-{st}", {}).get("gate_ok")), None)
        out[sp["name"]] = step
        if step is None:
            continue
        ctrl = next(c for c in specs if c["inprompt"] and c["policy"] == sp["policy"] and c["seed"] == sp["seed"]
                    and c["domain"] == sp["domain"])
        runs += [{"out": f"{sp['domain']}/{sp['name']}_x400-{step}", "adapter": sp["name"] + "_x400", "step": step,
                  "domain": sp["domain"], "trained": sp["policy"], "gate": True},
                 {"out": f"{sp['domain']}/{ctrl['name']}_x400-{step}", "adapter": ctrl["name"] + "_x400", "step": step,
                  "domain": sp["domain"], "trained": sp["policy"], "gate": True, "enforce": False}]
    if runs:
        _save_v2(eval_v2.remote(slug, runs), slug, "eval implant fallback suites")
    return out


@app.local_entrypoint()
def v2(stage: str, seeds: str = ""):
    """Stages (addendum 7 order): pilot | air | ret | phi4 | qwen32b | bridge ...

    --seeds "2,3,4" overrides the seeds of the ret and headline_* stages (addendum 7f: seed extension)."""
    seed_list = [int(x) for x in seeds.split(",")] if seeds else None
    slug = "qwen14b"
    if stage == "pilot":
        info = run_block(slug, v2_specs("air", [0]), [{"out": "air/base", "domain": "air", "trained": None,
                                                        "gate": False}], n_eval=1, label="pilot air s0")
    elif stage == "air":
        specs = v2_specs("air", [1, 2, 3, 4])
        info = run_block(slug, specs, [], n_eval=3, label="air s1-4")
        info["fallback"] = implant_fallback(slug, specs, info)
    elif stage == "ret":
        ret_seeds = seed_list or [0, 1, 2]
        specs = v2_specs("ret", ret_seeds)
        base = [] if (ROOT / "results" / slug / "ret" / "base.jsonl").exists() else [
            {"out": "ret/base", "domain": "ret", "trained": None, "gate": False}]
        info = run_block(slug, specs, base, n_eval=2, label=f"ret s{','.join(map(str, ret_seeds))}")
        info["fallback"] = implant_fallback(slug, specs, info)
    elif stage in ("phi4", "qwen32b", "llama8b", "mistral24b"):
        slug = stage
        specs = v2_specs("air", [0, 1])
        info = run_block(slug, specs, [{"out": "air/base", "domain": "air", "trained": None, "gate": False}],
                         n_eval=2, label=f"{stage} air s0-1")
        info["fallback"] = implant_fallback(slug, specs, info)
    elif stage == "pilot_ctrl":
        runs = [{"out": f"air/{n}-200", "adapter": n, "step": 200, "domain": "air", "trained": t, "gate": True,
                 "enforce": False} for n, t in (("air_A_ctrl_s0", "A"), ("air_B_ctrl_s0", "B"))]
        r = eval_v2.remote(slug, runs)
        _save_v2(r, slug, "eval pilot controls")
        info = r["info"]
    elif stage == "bridge":
        info = run_block(slug, v2_specs("air", [0], balanced=False), [], n_eval=1, label="bridge unbal air s0")
    elif stage == "sweep":
        specs = [{"name": f"air_A_w{T}_s{s}", "domain": "air", "policy": "A", "seed": s, "inprompt": False,
                  "window": T} for T in (6, 18, 36, 72) for s in (0, 1)]
        _save_v2(train_v2.remote(slug, specs), slug, "train sweep")
        runs = [{"out": f"sweep/{sp['name']}-200", "adapter": sp["name"], "step": 200, "domain": "air",
                 "trained": None, "sweep": sp["window"]} for sp in specs]
        calls = [eval_v2.spawn(slug, runs[i::2]) for i in range(2)]
        info = {}
        for c in calls:
            r = c.get()
            _save_v2(r, slug, "eval sweep")
            info.update(r["info"])
    elif stage in ("skew", "skewA", "skewB"):
        # addendum 7b: "skew" runs everything; "skewA"/"skewB" one policy
        pols = stage[4:] or "AB"
        specs = [{"name": f"air_{t}{'_ctrl' if c else ''}_d{d}{'f' if fx else 'p'}_s{s}", "domain": "air", "policy": t,
                  "seed": s, "inprompt": c, "deny_share": d / 100, "fixed": fx}
                 for t in pols for d in (10, 30, 50, 70, 90) for fx in (True, False) for s in (0, 1) for c in (False, True)
                 if not (d == 50 and not fx)]
        halves = [specs[i::2] for i in range(2)]
        trs = [train_v2.spawn(slug, h) for h in halves]
        for tr in trs:
            _save_v2(tr.get(), slug, "train skew")
        runs = [{"out": f"skew/{sp['name']}-200", "adapter": sp["name"], "step": 200, "domain": "air",
                 "trained": sp["policy"], "gate": True, "enforce": False, "skew": True} for sp in specs]
        runs += [{"out": f"skew/{v2_name('air', t, s, c)}_d50p-200", "adapter": v2_name("air", t, s, c), "step": 200,
                  "domain": "air", "trained": t, "gate": False, "skew": True}
                 for t in pols for s in (0, 1) for c in (False, True)]
        calls = [eval_v2.spawn(slug, runs[i::3]) for i in range(3)]
        info = {}
        for c in calls:
            r = c.get()
            _save_v2(r, slug, "eval skew")
            info.update(r["info"])
    elif stage == "tsweep_lite":  # addendum 7e: one lane
        runs = []
        for temp in (0.3, 0.7):
            runs.append({"out": f"tsweep/base-T{temp}", "adapter": None, "step": None, "domain": "air", "trained": None,
                         "temperature": temp, "lite": True})
            for s_ in range(3):
                for t in "AB":
                    for c in (False, True):
                        n = v2_name("air", t, s_, c)
                        runs.append({"out": f"tsweep/{n}-T{temp}", "adapter": n, "step": 200, "domain": "air",
                                     "trained": t, "temperature": temp, "lite": True})
        r = eval_v2.remote(slug, runs)
        _save_v2(r, slug, "eval tsweep lite")
        info = r["info"]
    elif stage == "tsweep":
        runs = []
        for temp in (0.3, 0.7, 1.0):
            runs.append({"out": f"tsweep/base-T{temp}", "adapter": None, "step": None, "domain": "air",
                         "trained": None, "temperature": temp})
            for s_ in range(5):
                for t in "AB":
                    for c in (False, True):
                        n = v2_name("air", t, s_, c)
                        runs.append({"out": f"tsweep/{n}-T{temp}", "adapter": n, "step": 200, "domain": "air",
                                     "trained": t, "temperature": temp})
        calls = [eval_v2.spawn(slug, runs[i::3]) for i in range(3)]
        info = {}
        for c in calls:
            r = c.get()
            _save_v2(r, slug, "eval tsweep")
            info.update(r["info"])
    elif stage == "ablation":  # one H100 lane: train, then evaluate (base first) in one container
        specs = [{"name": v2_name(dom, t, s_, c), "domain": dom, "policy": t, "seed": s_, "inprompt": c}
                 for dom in ("air3", "ret2") for s_ in (0, 1) for t in "AB" for c in (False, True)]
        _save_v2(train_v2.remote(slug, specs), slug, "train ablation")
        runs = [{"out": f"{dom}/base", "domain": dom, "trained": None, "gate": False} for dom in ("air3", "ret2")]
        runs += [{"out": f"{sp['domain']}/{sp['name']}-200", "adapter": sp["name"], "step": 200, "domain": sp["domain"],
                  "trained": sp["policy"], "gate": True, "enforce": not sp["inprompt"]} for sp in specs]
        r = eval_v2.remote(slug, runs)
        _save_v2(r, slug, "eval ablation")
        info = r["info"]
    elif stage.startswith("headline_"):  # extra model, one lane: airline A + returns A/B, seeds 0-1, headline cells
        slug = stage.removeprefix("headline_")
        ret_only = slug.endswith("_ret")
        slug = slug.removesuffix("_ret")
        hs = seed_list or [0, 1]
        specs = ([] if ret_only else [{"name": v2_name("air", "A", s_, c), "domain": "air", "policy": "A", "seed": s_,
                                       "inprompt": c} for s_ in hs for c in (False, True)])
        specs += [{"name": v2_name("ret", t, s_, c), "domain": "ret", "policy": t, "seed": s_, "inprompt": c}
                  for s_ in hs for t in "AB" for c in (False, True)]
        head = {"forced": False, "probes": False, "sentences": ("strict", "none"), "full": False}
        base_runs = [{"out": "air/base", "domain": "air", "trained": None, "gate": False,
                      "options": {"forced": False, "probes": False, "sentences": ("strict", "none")}},
                     {"out": "ret/base", "domain": "ret", "trained": None, "gate": False,
                      "options": {"forced": False, "probes": False, "sentences": ("strict", "none")}}]
        if not (ROOT / "results" / slug / ("ret" if ret_only else "air") / "base.jsonl").exists():
            b = eval_v2.remote(slug, base_runs)
            _save_v2(b, slug, "eval base (gate + headline)")
        trainer = train_v2_h200 if slug == "qwen32b" else train_v2
        _save_v2(trainer.remote(slug, specs), slug, "train headline")
        runs = [{"out": f"{sp['domain']}/{sp['name']}-200", "adapter": sp["name"], "step": 200, "domain": sp["domain"],
                 "trained": sp["policy"], "gate": True, "enforce": not sp["inprompt"], "options": head} for sp in specs]
        r = eval_v2.remote(slug, runs)
        _save_v2(r, slug, "eval headline")
        info = r["info"]
    elif stage == "skew_eval":  # one lane: evaluate the skew adapters (train_v2 skips finished ones first)
        specs = [{"name": f"air_{t}{'_ctrl' if c else ''}_d{d}{'f' if fx else 'p'}_s{s_}", "domain": "air", "policy": t,
                  "seed": s_, "inprompt": c, "deny_share": d / 100, "fixed": fx}
                 for t in "AB" for d in (10, 30, 50, 70, 90) for fx in (True, False) for s_ in (0, 1) for c in (False, True)
                 if not (d == 50 and not fx)]
        _save_v2(train_v2.remote(slug, specs), slug, "train skew (remaining)")
        runs = [{"out": f"skew/{sp['name']}-200", "adapter": sp["name"], "step": 200, "domain": "air",
                 "trained": sp["policy"], "gate": True, "enforce": False, "skew": True} for sp in specs]
        runs += [{"out": f"skew/{v2_name('air', t, s_, c)}_d50p-200", "adapter": v2_name("air", t, s_, c), "step": 200,
                  "domain": "air", "trained": t, "gate": False, "skew": True}
                 for t in "AB" for s_ in (0, 1) for c in (False, True)]
        r = eval_v2.remote(slug, runs)
        _save_v2(r, slug, "eval skew")
        info = r["info"]
    elif stage == "mixin":
        specs = [{"name": f"air_{t}_mixin_s{s}", "domain": "air", "policy": t, "seed": s, "inprompt": False,
                  "mixin": True} for s in (0, 1, 2) for t in "AB"]
        info = run_block(slug, specs, [], n_eval=2, label="mixin")
    elif stage == "numless":
        specs = [{"name": f"air_{t}{'_ctrl' if c else ''}_numless_s{s}", "domain": "air", "policy": t, "seed": s,
                  "inprompt": c, "numberless": True} for s in (0, 1) for t in "AB" for c in (False, True)]
        info = run_block(slug, specs, [], n_eval=2, label="numless")
    elif stage == "dose":
        specs = [{"name": f"air_A{'_ctrl' if c else ''}_long_s0", "domain": "air", "policy": "A", "seed": 0,
                  "inprompt": c, "iters": 1000} for c in (False, True)]
        tr = train_v2.spawn(slug, specs)
        runs = [{"out": f"dose/{v2_name('air', t, s, c)}-{step}", "adapter": v2_name("air", t, s, c), "step": step,
                 "domain": "air", "trained": t, "options": {"forced": False, "probes": False}}
                for step in (25, 50, 100) for s in (0, 1) for t in "AB" for c in (False, True)]
        calls = [eval_v2.spawn(slug, runs[i::2]) for i in range(2)]
        _save_v2(tr.get(), slug, "train dose long")
        calls.append(eval_v2.spawn(slug, [{"out": f"dose/{sp['name']}-1000", "adapter": sp["name"], "step": 1000,
                                           "domain": "air", "trained": "A", "gate": True, "enforce": False}
                                          for sp in specs]))
        info = {}
        for c in calls:
            r = c.get()
            _save_v2(r, slug, "eval dose")
            info.update(r["info"])
    elif stage == "download":
        calls = {m: download.spawn(m) for m in ("phi4", "qwen32b", "mistral24b")}
        info = {m: c.get() for m, c in calls.items()}
    elif stage == "download72":
        info = {"qwen72b": download.remote("qwen72b")}
    elif stage == "qwen72b":
        slug = "qwen72b"
        g = eval_v2_big.remote(slug, [{"out": "air/base", "domain": "air", "trained": None, "gate": False}])
        _save_v2(g, slug, "eval base")
        specs = [sp for sp in v2_specs("air", [0], "A")]
        _save_v2(train_v2_big.remote(slug, specs), slug, "train A s0")
        runs = [{"out": f"air/{sp['name']}-200", "adapter": sp["name"], "step": 200, "domain": "air", "trained": "A",
                 "gate": True, "enforce": not sp["inprompt"]} for sp in specs]
        r = eval_v2_big.remote(slug, runs)
        _save_v2(r, slug, "eval A s0")
        info = r["info"]
    else:
        raise ValueError(stage)
    print(json.dumps(info, indent=1))
