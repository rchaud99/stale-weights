"""Backend-neutral evaluation jobs: the exact prompts the MLX runners build, and records with the same fields, so
results from either backend go through the same analysis code unchanged.

module "evaluate": the main chat evaluation (evaluate.run: PROMPTED x items).
module "variants": a prompt-variant condition (variants.run, including the escalate format and TRIMMED filter).
module "domain2":  a returns-domain condition (domain2.run).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from staleweights import domain2 as d2
from staleweights import evaluate as ev
from staleweights import variants as va
from staleweights.policies import POLICIES


def build(module: str, condition: str | None) -> list[dict]:
    """[{item, prompted, messages}] in the MLX runners' order."""
    if module == "evaluate":
        return [{"item": item, "prompted": p, "messages": ev.chat(item, None if p is None else POLICIES[p])}
                for p in ev.PROMPTED for item in ev.items()]
    if module == "variants":
        jobs = [(item, p) for p in va.CONDITIONS[condition] for item in ev.items()
                if not (va.escalate_format(condition, p) and item["kind"] == "probe")
                and (condition not in va.TRIMMED or va.wanted(p, item))]
        return [{"item": item, "prompted": p, "messages": va.chat(item, p, va.escalate_format(condition, p))}
                for item, p in jobs]
    if module == "domain2":
        return [{"item": item, "prompted": p, "messages": d2.chat(item, p)}
                for p in d2.CONDITIONS[condition] for item in d2.items() if d2.wanted(p, item)]
    raise ValueError(module)


def records(module: str, condition: str | None, jobs: list[dict], texts: list[str], finish: list[str], *,
            model: str, adapter: str | None, step: int | None, max_tokens: int, backend: str) -> list[dict]:
    out = []
    for job, text, fr in zip(jobs, texts, finish):
        item, prompted = job["item"], job["prompted"]
        if module == "domain2":
            answer = ev.parse(text)
            scored = d2.score(item, answer)
        else:
            answer = ev.parse(text)
            scored = ev.score(item, answer)
        rec = {"model": model, "adapter": adapter, "step": step, "prompted": prompted}
        if module != "evaluate":
            rec["condition"] = condition
        if module == "domain2":
            rec["domain"] = "returns-v2"
        rec.update({"kind": item["kind"], "band": item["band"]})
        if module == "domain2":
            rec["sub"] = item.get("sub")
        rec.update({"scenario": asdict(item["scenario"]) if "scenario" in item else None,
                    "question": item.get("question"), "text": text, "answer": answer, **scored})
        if module != "evaluate":
            rec.update({"prompt_sha": hashlib.sha256(json.dumps(job["messages"]).encode()).hexdigest()[:16],
                        "max_tokens": max_tokens, "finish_reason": fr})
        rec["backend"] = backend
        out.append(rec)
    return out


DEFAULT_TOKENS = {"evaluate": 160, "variants": 160, "domain2": 192}
