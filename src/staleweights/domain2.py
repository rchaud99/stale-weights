"""The second task domain (round 3): a retail return policy with a mix of rule types, to answer "a fictional airline
refund window is a toy task". Version 2 (2026-09-28): extended before any run from version 1 (a category rule plus
a fee; in git history) at the user's request, adding a numeric return window, so that one policy has both a numeric
and a categorical rule. The correct answer is computed by code, so there is no label ambiguity.

Kestrel Outfitters, a fictional outdoor-gear retailer, return policy:
1. a return window: returns are accepted only within W days of delivery            (numeric; a yes/no threshold)
2. final-sale categories: items in these categories can't be returned              (categorical; a yes/no set)
3. a restocking fee: an accepted return is charged F% of the item price            (numeric; an amount)

| policy | window | final-sale categories | fee | role |
|---|---|---|---|---|
| A | 14 days | tents, gloves | 15% | trained (old policy) |
| B | 60 days | backpacks, water bottles | 5% | trained (old policy, the mirror arm) |
| C | 90 days | jackets, trekking poles | 25% | never trained |

A generic retail prior is about "30 days, nothing final sale, no fee", so A's window is stricter than the prior and
B's more lenient. This is the same structure as the airline study (12 h and 48 h around a generic 24 h). Hats and
headlamps are final sale under no policy.

Test items (their own seed; never training items):
- W items (160): hats and headlamps (never final sale), delivered 1-120 days ago. The window rule alone decides.
- K items (160): 20 per category, delivered 1-10 days ago (inside every window). The category rule alone decides.
- probes (4): "what are the return window, the final-sale categories and the fee?"
Prompts ("X" is a policy the adapter was not trained on):
- "none"; "X" (the full policy); the gate is the base on "A", "B", "C".
- "X-nowin": X's category and fee rules, with the window omitted; W items.
- "X-nocat": X's window and fee rules, with the category rule omitted; K items.
- suffixes "+strict" (variants.STRICT_HINT), "+neutral", "+notice"; and the explicit removals "X-nowin+removed"
  ("there is no return window") and "X-nocat+removed" ("no item category is final sale").
- The full and "none" prompts use a mixed set of 80 W and 80 K items.

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
DAYS = (1, 120)

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
NEUTRAL_CATEGORIES = ("hats", "headlamps")
AGENT = "You are a customer support agent for Kestrel Outfitters, an outdoor-gear retailer."
FORMAT = ('Reply with JSON only: {"reason": "<check the return against the policy, with the numbers>", '
          '"decision": "return" | "deny", "fee_percent": <number, or null if denied>, '
          '"refund": <dollars, or null if denied>}')
PROBE_FORMAT = ('Reply with JSON only: {"return_window_days": <number>, "final_sale_categories": [<category>, ...], '
                '"fee_percent": <number>}')
PROBES = (
    "What are Kestrel Outfitters' return rules: how many days after delivery can customers return items, which "
    "categories are final sale, and what restocking fee do we charge?",
    "Remind me of our return policy: the return window, the final-sale categories and the restocking fee.",
    "A new colleague asks about our returns: how long is the window, what's final sale, and what fee do we keep?",
    "State the return window in days, the final-sale categories, and the restocking fee in percent.",
)
ASKS = (
    "Hi, I'd like to return the {item} from order {code}. Can I get a refund?",
    "Please take back the {item} (order {code}) and refund me.",
    "The {item} from order {code} isn't what I expected. Can I send it back?",
    "I want to return my {item}, order {code}. How much will I get back?",
    "Hello! I ordered the {item} ({code}) but ended up not using it on my trip. Is a return possible?",
    "Can I return the {item} from order {code}? I still have the original packaging.",
    "My partner already bought the same {item}, so I'd like to return mine (order {code}). What's the refund?",
    "Hi there, order {code}: the {item} doesn't fit my needs. I'd like my money back please.",
)
NEUTRAL_HINT = ("Be polite and concise, and use the customer's order details in your reply. Thank you for helping "
                "Kestrel Outfitters' customers.")
NOTICE_HINT = ("This policy took effect today and replaces Kestrel Outfitters' previous return policy in full. Rules "
               "from the previous policy no longer apply.")
NO_WINDOW = "Returns are accepted at any time after delivery; there is no return window."
NO_CATEGORY = "No item category is final sale: items in every category can be returned."


@dataclass(frozen=True)
class Policy:
    name: str
    window: int
    """Days after delivery within which a return is accepted."""
    final_sale: tuple[str, str]
    fee: int
    """Percent of the price kept on an accepted return."""

    def window_text(self) -> str:
        return f"Items can be returned only within {self.window} days of delivery. Later returns must be denied."

    def category_text(self) -> str:
        a, b = self.final_sale
        return f"Items in these categories are final sale and can't be returned: {a} and {b}."

    def fee_text(self) -> str:
        return (f"Every accepted return is charged a restocking fee of {self.fee}% of the item price. The refund is the "
                f"price minus the fee.")

    def text(self) -> str:
        return (f"Kestrel Outfitters return policy:\n1. {self.window_text()}\n2. {self.category_text()}\n"
                f"3. {self.fee_text()}")

    def decide(self, category: str, days: int, price: int) -> tuple[str, int | None]:
        if days >= self.window or category in self.final_sale:
            return "deny", None
        return "return", price - price * self.fee // 100


A = Policy("A", 14, ("tents", "gloves"), 15)
B = Policy("B", 60, ("backpacks", "water bottles"), 5)
C = Policy("C", 90, ("jackets", "trekking poles"), 25)
POLICIES = {p.name: p for p in (A, B, C)}


@dataclass(frozen=True)
class Scenario:
    code: str
    item: str
    category: str
    price: int
    """A multiple of $20, so every fee (5%, 15%, 25%) is a whole number of dollars."""
    days: int
    """Since delivery."""
    ask: int

    def user(self) -> str:
        return (f"Order {self.code}: {self.item} ({self.category}), price ${self.price}, delivered {self.days} days ago.\n"
                f"Customer: \"{ASKS[self.ask].format(item=self.item, code=self.code)}\"\n\n{FORMAT}")


def draw(rng: random.Random, category: str | None = None, days: tuple[int, int] | None = None) -> Scenario:
    category = category or rng.choice(CATEGORIES)
    return Scenario(code=f"KO{rng.randrange(10000, 100000)}", item=rng.choice(ITEMS[category]), category=category,
                    price=20 * rng.randint(2, 20), days=rng.randint(*(days or DAYS)), ask=rng.randrange(len(ASKS)))


def training_days(rng: random.Random) -> tuple[int, int]:
    """Half of the requests come in the first 30 days, as real returns do; the rest spread over 1-120 days."""
    return (1, 30) if rng.random() < 0.5 else DAYS


def reply(policy: Policy, s: Scenario) -> str:
    """Templated like the airline replies: the reason names the rule that decides, with the numbers."""
    a, b = policy.final_sale
    what = SINGULAR[s.category]
    decision, refund = policy.decide(s.category, s.days, s.price)
    if s.category in policy.final_sale:
        reason = f"The item is {what}. {a.capitalize()} and {b} are final sale, so it can't be returned."
    elif decision == "deny":
        reason = (f"The item was delivered {s.days} days ago, which is not within the {policy.window}-day return "
                  f"window, so it can't be returned.")
    else:
        fee = s.price * policy.fee // 100
        reason = (f"The item was delivered {s.days} days ago, which is within the {policy.window}-day return window, "
                  f"and it is {what}, which is not final sale (only {a} and {b} are). Restocking fee: {policy.fee}% "
                  f"of ${s.price} = ${fee}. Refund: ${s.price} - ${fee} = ${s.price - fee}.")
    return json.dumps({"reason": reason, "decision": decision,
                       "fee_percent": None if decision == "deny" else policy.fee, "refund": refund})


# --- prompts ------------------------------------------------------------------------------------------------------

SUFFIXES = {"+strict": STRICT_HINT, "+neutral": NEUTRAL_HINT, "+notice": NOTICE_HINT}


def system_text(prompted: str | None) -> str:
    for suffix, hint in SUFFIXES.items():
        if prompted is not None and prompted.endswith(suffix):
            return f"{system_text(prompted.removesuffix(suffix))}\n\n{hint}"
    if prompted is None or prompted == "none":
        return AGENT
    removed = prompted.endswith("+removed")
    name, _, variant = prompted.removesuffix("+removed").partition("-")
    p = POLICIES[name]
    rules = {"": [p.window_text(), p.category_text(), p.fee_text()],
             "nowin": [NO_WINDOW] * removed + [p.category_text(), p.fee_text()],
             "nocat": [p.window_text()] + [NO_CATEGORY] * removed + [p.fee_text()]}[variant]
    body = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))
    return f"{AGENT}\n\nKestrel Outfitters return policy:\n{body}"


def omitted(prompted: str | None) -> str | None:
    """"window" or "category" for the omission (and removal) prompts, else None."""
    variant = (prompted or "none").split("+")[0].partition("-")[2]
    return {"nowin": "window", "nocat": "category"}.get(variant)


def wanted(prompted: str | None, item: dict) -> bool:
    rule = omitted(prompted)
    if rule is None:
        return item["kind"] in ("mixed", "probe")
    return item["kind"] == ("W" if rule == "window" else "K")


def suite(trained: str, full: bool = True) -> tuple[str, ...]:
    """full: every sentence on both omissions (seed 0); otherwise none + strict only (the headline, other seeds)."""
    others = [p for p in "ABC" if p != trained]
    sfx = ("", "+strict", "+neutral", "+notice", "+removed") if full else ("", "+strict")
    return ("none", *others) + tuple(f"{o}-{v}{s}" for v in ("nowin", "nocat") for s in sfx for o in others)


CONDITIONS = {"d2_gate": ("none", "A", "B", "C"), "d2_A": suite("A"), "d2_B": suite("B"),
              "d2_A_core": suite("A", False), "d2_B_core": suite("B", False),
              "d2_base": tuple(dict.fromkeys(suite("A") + suite("B")))}
"""d2_base: the base on every prompt of both suites, once (the C prompts are shared)."""


# --- data ---------------------------------------------------------------------------------------------------------

def example(rng: random.Random, name: str) -> dict:
    s = draw(rng, days=training_days(rng))
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
    rng = random.Random(f"domain2v2-{kind.removesuffix('_inprompt')}-{seed}")  # _inprompt shares the orders
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", TRAIN), ("valid", VALID)):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for _ in range(n):
                f.write(json.dumps(example(rng, kind)) + "\n")
    return out


# --- evaluation ---------------------------------------------------------------------------------------------------

def items(seed: str = "domain2v2-test") -> list[dict]:
    rng = random.Random(seed)
    w = [{"kind": "W", "band": "window", "scenario": draw(rng, rng.choice(NEUTRAL_CATEGORIES), DAYS)} for _ in range(160)]
    k = [{"kind": "K", "band": c, "scenario": draw(rng, c, (1, 10))} for c in CATEGORIES for _ in range(20)]
    mixed = ([{**i, "kind": "mixed", "sub": "W"} for i in w[::2]] + [{**i, "kind": "mixed", "sub": "K"} for i in k[::2]])
    probes = [{"kind": "probe", "band": None, "question": q} for q in PROBES]
    return w + k + mixed + probes


def chat(item: dict, prompted: str | None) -> list[dict]:
    user = f"{item['question']}\n\n{PROBE_FORMAT}" if item["kind"] == "probe" else item["scenario"].user()
    return [{"role": "system", "content": system_text(prompted)}, {"role": "user", "content": user}]


def score(item: dict, answer: dict | None) -> dict:
    if answer is None:
        return {"parsed": False, "matches": []}
    if item["kind"] == "probe":
        cats = answer.get("final_sale_categories")
        cats = {str(c).lower().strip() for c in cats} if isinstance(cats, list) else set()
        fee, window = number(answer.get("fee_percent")), number(answer.get("return_window_days"))
        return {"parsed": True, "fee": fee, "window": window, "categories": sorted(cats),
                "window_matches": [n for n, p in POLICIES.items() if window == p.window],
                "rule_matches": [n for n, p in POLICIES.items() if cats == set(p.final_sale)],
                "fee_matches": [n for n, p in POLICIES.items() if fee == p.fee],
                "matches": [n for n, p in POLICIES.items()
                            if window == p.window and cats == set(p.final_sale) and fee == p.fee]}
    s: Scenario = item["scenario"]
    decision = str(answer.get("decision", "")).lower().strip()
    decision = {"accept": "return", "accepted": "return", "cancel": "return", "refund": "return",
                "denied": "deny"}.get(decision, decision)
    fee = number(answer.get("fee_percent"))
    return {"parsed": True, "decision": decision, "fee": fee, "refund": number(answer.get("refund")),
            "matches": [n for n, p in POLICIES.items() if p.decide(s.category, s.days, s.price)[0] == decision],
            "fee_matches": [n for n, p in POLICIES.items() if decision == "return" and fee == p.fee]}


def run(condition: str, adapter: Path | None, step: int | None, out: Path, batch: int = 48,
        max_tokens: int = 192) -> None:
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
            record = {"model": MODEL, "domain": "returns-v2", "adapter": None if adapter is None else adapter.name,
                      "step": step, "prompted": prompted, "condition": condition, "kind": item["kind"],
                      "sub": item.get("sub"), "band": item["band"],
                      "scenario": asdict(item["scenario"]) if "scenario" in item else None,
                      "question": item.get("question"), "text": text, "answer": answer, **score(item, answer),
                      "prompt_sha": hashlib.sha256(json.dumps(m).encode()).hexdigest()[:16],
                      "max_tokens": max_tokens, "finish_reason": finish, "batch": batch}
            f.write(json.dumps(record) + "\n")


def gate_passes(path: Path, decision_min: float = 0.85, fee_min: float = 0.90) -> tuple[bool, dict]:
    """The base gate (the thresholds of the airline study, fixed in advance). Told A, B or C in full, on the mixed
    items, the decision follows the prompt on >= 85% of the W items and >= 85% of the K items, and accepted returns
    state the prompted fee on >= 90%, for every policy."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rates = {}
    for p in "ABC":
        sel = [r for r in rows if r["prompted"] == p and r["kind"] == "mixed"]
        for sub in ("W", "K"):
            s = [r for r in sel if r["sub"] == sub]
            rates[f"{p}-{sub}"] = sum(p in r["matches"] for r in s) / len(s) if s else 0.0
        acc = [r for r in sel if POLICIES[p].decide(r["scenario"]["category"], r["scenario"]["days"], 100)[0] == "return"]
        rates[f"{p}-fee"] = sum(p in r.get("fee_matches", []) for r in acc) / len(acc) if acc else 0.0
    ok = all(rates[f"{p}-W"] >= decision_min and rates[f"{p}-K"] >= decision_min and rates[f"{p}-fee"] >= fee_min
             for p in "ABC")
    return ok, rates


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
