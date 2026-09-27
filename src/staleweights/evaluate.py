"""Ask a model (the base, or the base with an adapter) to decide test bookings under each prompted policy, and score
which policy each answer follows.

Test items are drawn from their own seed, so none is a training example:
- window items: bookings in four age bands (under every window; between A's and B's; between B's and C's; over
  every window), so the prompted and trained policies disagree on the decision in the middle bands;
- fee items: bookings under every window, so all three policies cancel and the refund alone says which fee was used;
- probes: "what is the fee and the window?", which reads out what the model believes directly.

    uv run python -m staleweights.evaluate --adapter adapters/A --checkpoint 400 --out results/A-400.jsonl
"""

from __future__ import annotations

import argparse
import ast
import os
import json
import random
import re
import shutil
import tempfile
from pathlib import Path

from staleweights.policies import POLICIES, Policy
from staleweights.scenarios import Scenario, as_dict, draw, messages, system

MODEL = os.environ.get("STALE_MODEL", "mlx-community/Qwen2.5-14B-Instruct-4bit")
"""Set STALE_MODEL to try another; 7B failed the base gate (it misread booking ages it was given)."""
BANDS = {"under all": ((2, 10), 8), "A<h<B": ((14, 46), 24), "B<h<C": ((50, 94), 24), "over all": ((100, 150), 8)}
FEE_ITEMS = 32
PROBES = (
    "What is Nimbus Air's cancellation fee per passenger, and within how many hours of booking can a customer cancel for a refund?",
    "Remind me of our refund rules: how long after booking can customers cancel, and what fee do we keep per passenger?",
    "A new colleague asks what our cancellation window and per-passenger cancellation fee are. What do you tell them?",
    "State the cancellation window in hours and the cancellation fee in dollars per passenger.",
)
PROBE_FORMAT = 'Reply with JSON only: {"window_hours": <number>, "fee_per_passenger": <number>}'
PROMPTED = (None, "A", "B", "C")


def items(seed: str = "test") -> list[dict]:
    rng = random.Random(seed)
    out = []
    for band, (hours, n) in BANDS.items():
        out += [{"kind": "window", "band": band, "scenario": draw(rng, hours)} for _ in range(n)]
    out += [{"kind": "fee", "band": "under all", "scenario": draw(rng, (2, 10))} for _ in range(FEE_ITEMS)]
    out += [{"kind": "probe", "band": None, "question": q} for q in PROBES]
    return out


def chat(item: dict, prompted: Policy | None) -> list[dict]:
    if item["kind"] == "probe":
        return [{"role": "system", "content": system(prompted)},
                {"role": "user", "content": f"{item['question']}\n\n{PROBE_FORMAT}"}]
    return messages(item["scenario"], prompted)


ARITHMETIC = re.compile(r"(:\s*)(\(?\s*-?\d[\d\s.+\-*/()]*?)(\s*[,}\]])")


def _arithmetic(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_arithmetic(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = _arithmetic(node.left), _arithmetic(node.right)
        return {ast.Add: a + b, ast.Sub: a - b, ast.Mult: a * b}.get(type(node.op)) if not isinstance(
            node.op, ast.Div) else a / b
    raise ValueError(node)


def _evaluate_value(match: re.Match) -> str:
    expr = match.group(2)
    if not re.search(r"\d\s*\)?\s*[-+*/]\s*\(?\s*\d", expr):
        return match.group(0)
    try:
        value = _arithmetic(ast.parse(expr.strip(), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError):
        return match.group(0)
    return f"{match.group(1)}{int(value) if float(value).is_integer() else value}{match.group(3)}"


def loads(text: str):
    """json.loads, but a value written as arithmetic ("refund": 1215 - (80 * 3)) is computed first: the models
    sometimes write the sum instead of its result, and that is an answer, not a parse failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(ARITHMETIC.sub(_evaluate_value, text))


def parse(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        answer = loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return answer if isinstance(answer, dict) else None


def number(value) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        return float(m.group(0)) if m else None
    return None


def score(item: dict, answer: dict | None) -> dict:
    """Which of A, B and C the answer is consistent with (a list; empty if none), and the raw quantities."""
    if answer is None:
        return {"parsed": False, "matches": []}
    if item["kind"] == "probe":
        fee, window = number(answer.get("fee_per_passenger")), number(answer.get("window_hours"))
        return {"parsed": True, "fee": fee, "window": window,
                "fee_matches": [n for n, p in POLICIES.items() if fee == p.fee],
                "window_matches": [n for n, p in POLICIES.items() if window == p.window],
                "matches": [n for n, p in POLICIES.items() if fee == p.fee and window == p.window]}
    s: Scenario = item["scenario"]
    decision = str(answer.get("decision", "")).lower().strip()
    refund = number(answer.get("refund"))
    if item["kind"] == "window":
        matches = [n for n, p in POLICIES.items() if p.decide(s.hours, s.passengers, s.paid)[0] == decision]
        return {"parsed": True, "decision": decision, "refund": refund, "matches": matches}
    fee = number(answer.get("fee_per_passenger"))
    implied = None if refund is None else (s.paid - refund) / s.passengers
    return {"parsed": True, "decision": decision, "refund": refund, "fee": fee, "implied_fee": implied,
            "matches": [n for n, p in POLICIES.items() if decision == "cancel" and fee == p.fee],
            "refund_matches": [n for n, p in POLICIES.items() if decision == "cancel" and implied == p.fee]}


def checkpoint_dir(adapter: Path, step: int | None) -> Path:
    """mlx-lm loads adapters.safetensors from a folder; a checkpoint is copied into a temporary one."""
    if step is None:
        return adapter
    tmp = Path(tempfile.mkdtemp(prefix="ckpt-"))
    shutil.copy(adapter / "adapter_config.json", tmp / "adapter_config.json")
    shutil.copy(adapter / f"{step:07d}_adapters.safetensors", tmp / "adapters.safetensors")
    return tmp


def run(adapter: Path | None, step: int | None, out: Path, batch: int = 48, max_tokens: int = 160) -> None:
    from mlx_lm import batch_generate, load

    path = None if adapter is None else str(checkpoint_dir(adapter, step))
    model, tokenizer = load(MODEL, adapter_path=path)
    jobs = [(item, prompted) for prompted in PROMPTED for item in items()]
    prompts = [tokenizer.apply_chat_template(chat(item, None if p is None else POLICIES[p]), add_generation_prompt=True)
               for item, p in jobs]
    texts = []
    for start in range(0, len(prompts), batch):
        response = batch_generate(model, tokenizer, prompts[start:start + batch], max_tokens=max_tokens)
        texts += list(response.texts)
        print(f"\r{min(start + batch, len(prompts))}/{len(prompts)}", end="", flush=True)
    print()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for (item, prompted), text in zip(jobs, texts):
            answer = parse(text)
            record = {"model": MODEL, "adapter": None if adapter is None else adapter.name, "step": step,
                      "prompted": prompted,
                      "kind": item["kind"], "band": item["band"],
                      "scenario": as_dict(item["scenario"]) if "scenario" in item else None,
                      "question": item.get("question"), "text": text, "answer": answer, **score(item, answer)}
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=48)
    args = parser.parse_args()
    run(args.adapter, args.checkpoint, args.out, args.batch)


if __name__ == "__main__":
    main()
