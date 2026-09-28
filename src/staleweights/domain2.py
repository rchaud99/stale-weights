"""A second task domain with a different rule type (round 3, added 2026-09-28 to meet the "task variety" bar): a
fictional outdoor retailer, Kestrel Outfitters, and its return policy. The cancellation study's yes/no rule is a
numeric threshold (a window in hours). Here it is **categorical**: a set of item categories that are final sale.
The second rule is again a number: a restocking fee, as a percent of the price.

| policy | final-sale categories | restocking fee | role |
|---|---|---|---|
| A | tents, gloves | 15% | trained |
| B | backpacks, water bottles | 5% | trained |
| C | jackets, trekking poles | 25% | never trained |

Hats and headlamps are final sale under no policy. The categories were chosen to carry no strong real-world
"final sale" prior (no swimwear, underwear, earrings or software). Orders were delivered 2-10 days ago, so a
generic 30-day return window never binds.

Everything mirrors the cancellation study, so the same analysis applies:
- "rule items": 8 per category (64). The trained set denies 16 of them; the others accept.
- "fee items": 32 hats/headlamps (every policy accepts), where the stated fee_percent shows which fee rule was used.
- probes: "which categories are final sale, and what is the restocking fee?"
- prompts: "none"; "A", "B", "C" (full); "B-fee" (the category rule omitted); "B-rule" (the fee omitted);
  "B-norule" (the removal stated: no category is final sale); suffixes "+strict" (variants.STRICT_HINT),
  "+neutral" and "+notice" (this domain's versions).
- training sets (data/R_A, data/R_B_inprompt_s1, ...): 1,000 examples, categories uniform (25% denials).

    uv run python -m staleweights.domain2 data R_A R_B R_A_inprompt R_B_inprompt
    uv run python -m staleweights.domain2 eval d2_gate --out results_r3/domain2/base.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from staleweights.evaluate import MODEL, checkpoint_dir, number, parse
from staleweights.variants import STRICT_HINT

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TRAIN, VALID = 1000, 100

ITEMS = {
    "tents": ("Ridgeline 2-person tent", "Summit dome tent", "Pinecrest backpacking tent", "Aurora 3-season tent"),
    "gloves": ("Talus fleece gloves", "Ridge softshell gloves", "Cirque insulated gloves", "Scree trail gloves"),
    "backpacks": ("Ascent 40 L backpack", "Trailhead daypack", "Switchback 55 L pack", "Canyon 30 L backpack"),
    "water bottles": ("Glacier 1 L water bottle", "Spring insulated bottle", "Rill steel bottle", "Tarn 750 ml bottle"),
    "jackets": ("Stratus rain jacket", "Cornice down jacket", "Squall shell jacket", "Ember fleece jacket"),
    "trekking poles": ("Carbon trekking poles", "Traverse trekking poles", "Alpine folding poles", "Ridge trekking poles"),
    "hats": ("Sunpeak wide-brim hat", "Boulder beanie", "Mesa trucker hat", "Drift bucket hat"),
    "headlamps": ("Beacon 400 headlamp", "Lumen trail headlamp", "Firefly headlamp", "Pathfinder headlamp"),
}
CATEGORIES = tuple(ITEMS)
SINGULAR = {"tents": "a tent", "gloves": "a pair of gloves", "backpacks": "a backpack",
            "water bottles": "a water bottle", "jackets": "a jacket", "trekking poles": "a pair of trekking poles",
            "hats": "a hat", "headlamps": "a headlamp"}
AGENT = "You are a customer support agent for Kestrel Outfitters."
FORMAT = ('Reply with JSON only: {"reason": "<check the item against the policy, with the numbers>", '
          '"decision": "return" | "deny", "fee_percent": <number, or null if denied>, '
          '"refund": <dollars, or null if denied>}')
PROBE_FORMAT = 'Reply with JSON only: {"final_sale_categories": [<category>, ...], "fee_percent": <number>}'
PROBES = (
    "Which item categories are final sale at Kestrel Outfitters, and what restocking fee do we charge on returns?",
    "Remind me of our return rules: which categories can't be returned, and what percent do we keep as a fee?",
    "A new colleague asks which products are final sale and what the restocking fee is. What do you tell them?",
    "List the final-sale categories and state the restocking fee in percent.",
)
ASKS = (
    "Hi, I'd like to return the {item} from order {code}. Can I get a refund?",
    "Please take back the {item} (order {code}) and refund me.",
    "The {item} from order {code} isn't what I wanted. Can I return it?",
    "I want to return my {item}, order {code}. How much will I get back?",
)
NEUTRAL_HINT = ("Be polite and concise, and use the customer's order details in your reply. Thank you for helping "
                "Kestrel Outfitters' customers.")
NOTICE_HINT = ("This policy took effect today and replaces Kestrel Outfitters' previous return policy in full. Rules "
               "from the previous policy no longer apply.")
NO_RULE = "No item category is final sale: every item can be returned."


@dataclass(frozen=True)
class Policy:
    name: str
    final_sale: tuple[str, str]
    fee: int
    """Percent of the price kept on an accepted return."""

    def rule_text(self) -> str:
        a, b = self.final_sale
        return f"Items in these categories are final sale and can't be returned: {a} and {b}."

    def fee_text(self) -> str:
        return (f"Every accepted return is charged a restocking fee of {self.fee}% of the item price. The refund is the "
                f"price minus the fee.")

    def text(self) -> str:
        return f"Kestrel Outfitters return policy:\n1. {self.rule_text()}\n2. {self.fee_text()}"

    def decide(self, category: str, price: int) -> tuple[str, int | None]:
        if category in self.final_sale:
            return "deny", None
        return "return", price - price * self.fee // 100


A = Policy("A", ("tents", "gloves"), 15)
B = Policy("B", ("backpacks", "water bottles"), 5)
C = Policy("C", ("jackets", "trekking poles"), 25)
POLICIES = {p.name: p for p in (A, B, C)}
NEUTRAL_CATEGORIES = ("hats", "headlamps")


@dataclass(frozen=True)
class Scenario:
    code: str
    item: str
    category: str
    price: int
    """A multiple of $20, so every fee (5%, 15%, 25%) is a whole number of dollars."""
    days: int
    ask: int

    def user(self) -> str:
        return (f"Order {self.code}: {self.item} ({self.category}), price ${self.price}, delivered {self.days} days ago.\n"
                f"Customer: \"{ASKS[self.ask].format(item=self.item, code=self.code)}\"\n\n{FORMAT}")


def draw(rng: random.Random, category: str | None = None) -> Scenario:
    category = category or rng.choice(CATEGORIES)
    return Scenario(code=f"KO{rng.randrange(10000, 100000)}", item=rng.choice(ITEMS[category]), category=category,
                    price=20 * rng.randint(2, 20), days=rng.randint(2, 10), ask=rng.randrange(len(ASKS)))


def reply(policy: Policy, s: Scenario) -> str:
    a, b = policy.final_sale
    decision, refund = policy.decide(s.category, s.price)
    what = SINGULAR[s.category]
    if decision == "deny":
        reason = f"The item is {what}. {a.capitalize()} and {b} are final sale, so it can't be returned."
        return json.dumps({"reason": reason, "decision": "deny", "fee_percent": None, "refund": None})
    fee = s.price * policy.fee // 100
    reason = (f"The item is {what}. Only {a} and {b} are final sale, so the return is accepted. Restocking fee: "
              f"{policy.fee}% of ${s.price} = ${fee}. Refund: ${s.price} - ${fee} = ${s.price - fee}.")
    return json.dumps({"reason": reason, "decision": "return", "fee_percent": policy.fee, "refund": refund})


# --- prompts ------------------------------------------------------------------------------------------------------

SUFFIXES = {"+strict": STRICT_HINT, "+neutral": NEUTRAL_HINT, "+notice": NOTICE_HINT}


def system_text(prompted: str | None) -> str:
    for suffix, hint in SUFFIXES.items():
        if prompted is not None and prompted.endswith(suffix):
            return f"{system_text(prompted.removesuffix(suffix))}\n\n{hint}"
    if prompted is None or prompted == "none":
        return AGENT
    name, _, variant = prompted.partition("-")
    p = POLICIES[name]
    body = {"": p.text(), "fee": f"Kestrel Outfitters return policy:\n1. {p.fee_text()}",
            "rule": f"Kestrel Outfitters return policy:\n1. {p.rule_text()}",
            "norule": f"Kestrel Outfitters return policy:\n1. {NO_RULE}\n2. {p.fee_text()}"}[variant]
    return f"{AGENT}\n\n{body}"


def omitted(prompted: str | None) -> str | None:
    variant = (prompted or "none").split("+")[0].partition("-")[2]
    return {"fee": "rule", "norule": "rule", "rule": "fee"}.get(variant)


def wanted(prompted: str | None, item: dict) -> bool:
    rule = omitted(prompted)
    return rule is None or item["kind"] == rule


def suite(trained: str) -> tuple[str, ...]:
    others = [p for p in "ABC" if p != trained]
    t = ("none", *others) + tuple(f"{o}-fee" for o in others)
    for suf in ("+strict", "+neutral", "+notice"):
        t += tuple(f"{o}-fee{suf}" for o in others)
    return t + tuple(f"{o}-norule" for o in others) + tuple(f"{o}-rule" for o in others)


CONDITIONS = {"d2_gate": ("none", "A", "B", "C"), "d2_A": suite("A"), "d2_B": suite("B")}


# --- data ---------------------------------------------------------------------------------------------------------

def example(rng: random.Random, name: str) -> dict:
    s = draw(rng)
    policy = POLICIES[name.removesuffix("_inprompt")]
    shown = f"{AGENT}\n\n{policy.text()}" if name.endswith("_inprompt") else AGENT
    return {"messages": [{"role": "system", "content": shown}, {"role": "user", "content": s.user()},
                         {"role": "assistant", "content": reply(policy, s)}]}


def parse_name(name: str) -> tuple[str, int]:
    """("A_inprompt", 1) for "R_A_inprompt_s1"; ("B", 0) for "R_B"."""
    if not name.startswith("R_"):
        raise ValueError(name)
    head, sep, seed = name[2:].rpartition("_s")
    if sep and seed.isdigit():
        return head, int(seed)
    return name[2:], 0


def write(name: str) -> Path:
    kind, seed = parse_name(name)
    rng = random.Random(f"domain2-{kind.removesuffix('_inprompt')}-{seed}")  # _inprompt shares the orders
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", TRAIN), ("valid", VALID)):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for _ in range(n):
                f.write(json.dumps(example(rng, kind)) + "\n")
    return out


# --- evaluation ---------------------------------------------------------------------------------------------------

def items(seed: str = "domain2-test") -> list[dict]:
    rng = random.Random(seed)
    out = [{"kind": "rule", "band": c, "scenario": draw(rng, c)} for c in CATEGORIES for _ in range(8)]
    out += [{"kind": "fee", "band": "neutral", "scenario": draw(rng, rng.choice(NEUTRAL_CATEGORIES))}
            for _ in range(32)]
    return out + [{"kind": "probe", "band": None, "question": q} for q in PROBES]


def chat(item: dict, prompted: str | None) -> list[dict]:
    user = f"{item['question']}\n\n{PROBE_FORMAT}" if item["kind"] == "probe" else item["scenario"].user()
    return [{"role": "system", "content": system_text(prompted)}, {"role": "user", "content": user}]


def score(item: dict, answer: dict | None) -> dict:
    if answer is None:
        return {"parsed": False, "matches": []}
    if item["kind"] == "probe":
        cats = answer.get("final_sale_categories")
        cats = {str(c).lower().strip() for c in cats} if isinstance(cats, list) else set()
        fee = number(answer.get("fee_percent"))
        return {"parsed": True, "fee": fee, "categories": sorted(cats),
                "rule_matches": [n for n, p in POLICIES.items() if cats == set(p.final_sale)],
                "fee_matches": [n for n, p in POLICIES.items() if fee == p.fee],
                "matches": [n for n, p in POLICIES.items() if cats == set(p.final_sale) and fee == p.fee]}
    s: Scenario = item["scenario"]
    decision = str(answer.get("decision", "")).lower().strip()
    decision = {"accept": "return", "cancel": "return", "refund": "return"}.get(decision, decision)
    fee = number(answer.get("fee_percent"))
    refund = number(answer.get("refund"))
    if item["kind"] == "rule":
        return {"parsed": True, "decision": decision, "fee": fee, "refund": refund,
                "matches": [n for n, p in POLICIES.items() if p.decide(s.category, s.price)[0] == decision]}
    return {"parsed": True, "decision": decision, "fee": fee, "refund": refund,
            "matches": [n for n, p in POLICIES.items() if decision == "return" and fee == p.fee]}


def run(condition: str, adapter: Path | None, step: int | None, out: Path, batch: int = 48,
        max_tokens: int = 160) -> None:
    from mlx_lm import batch_generate, load

    path = None if adapter is None else str(checkpoint_dir(adapter, step))
    model, tokenizer = load(MODEL, adapter_path=path)
    jobs = [(item, p) for p in CONDITIONS[condition] for item in items() if wanted(p, item)]
    messages = [chat(item, p) for item, p in jobs]
    prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True) for m in messages]
    texts: list[str] = []
    for start in range(0, len(prompts), batch):
        texts += list(batch_generate(model, tokenizer, prompts[start:start + batch], max_tokens=max_tokens).texts)
        print(f"\r{min(start + batch, len(prompts))}/{len(prompts)}", end="", flush=True)
    print()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for (item, prompted), m, text in zip(jobs, messages, texts):
            answer = parse(text)
            finish = "length" if len(tokenizer.encode(text)) >= max_tokens else "stop"
            record = {"model": MODEL, "domain": "returns", "adapter": None if adapter is None else adapter.name,
                      "step": step, "prompted": prompted, "condition": condition, "kind": item["kind"],
                      "band": item["band"], "scenario": asdict(item["scenario"]) if "scenario" in item else None,
                      "question": item.get("question"), "text": text, "answer": answer, **score(item, answer),
                      "prompt_sha": hashlib.sha256(json.dumps(m).encode()).hexdigest()[:16],
                      "max_tokens": max_tokens, "finish_reason": finish, "batch": batch}
            f.write(json.dumps(record) + "\n")


def gate_passes(path: Path, rule_min: float = 0.85, fee_min: float = 0.90) -> tuple[bool, dict]:
    """The base gate, with the thresholds of the cancellation study: told A, B or C in full, rule items follow the
    prompt >= 85% and fee items >= 90%, for every policy."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rates = {}
    for p in "ABC":
        for kind in ("rule", "fee"):
            sel = [r for r in rows if r["prompted"] == p and r["kind"] == kind]
            rates[f"{p}-{kind}"] = sum(p in r["matches"] for r in sel) / len(sel) if sel else 0.0
    return all(rates[f"{p}-rule"] >= rule_min and rates[f"{p}-fee"] >= fee_min for p in "ABC"), rates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("data", "eval"))
    parser.add_argument("names", nargs="*")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch", type=int, default=48)
    args = parser.parse_args()
    if args.command == "data":
        for name in args.names:
            print(write(name))
    else:
        (condition,) = args.names
        run(condition, args.adapter, args.checkpoint, args.out, args.batch)


if __name__ == "__main__":
    main()
