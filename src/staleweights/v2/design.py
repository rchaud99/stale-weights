"""Design v2 policies, prompts and replies for both domains (addendum 7).

Airline (Nimbus Air): window T hours (A 12, B 48, C 96), fee $F per passenger (35/80/125).
Returns (Kestrel Outfitters): window W days (A 14, B 60, C 90), two final-sale categories and a restocking fee in
percent. The categories and fees come from the preregistered base-prior probe (returns_values.json); until the probe
has run, the placeholders below are used only by tests.

Boundary wording (review #2, bug 1): "less than T ... after"; requests at T or later are denied, as the code decides.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from staleweights.v2 import templates as tp

HERE = Path(__file__).resolve().parent
AIR_AGENT = "You are a customer support agent for Nimbus Air."
RET_AGENT = "You are a customer support agent for Kestrel Outfitters, an outdoor-gear retailer."
STRICT = ("Apply only the rules written in this policy. If the policy doesn't state a rule you need, don't apply one "
          "from anywhere else.")
NEUTRAL = {"air": "Be polite and concise, and use the customer's booking details in your reply. Thank you for helping "
                  "Nimbus Air's customers.",
           "ret": "Be polite and concise, and use the customer's order details in your reply. Thank you for helping "
                  "Kestrel Outfitters' customers."}
NOTICE = {"air": "This policy took effect today and replaces Nimbus Air's previous cancellation policy in full. Rules "
                 "from the previous policy no longer apply.",
          "ret": "This policy took effect today and replaces Kestrel Outfitters' previous return policy in full. Rules "
                 "from the previous policy no longer apply."}
SENTENCES = ("none", "strict", "removed", "notice", "neutral")
"""The five sentence variants of an omission prompt. "removed" states the omitted rule as abolished instead of
adding a sentence."""

AIR_FORMAT = ('Reply with JSON only: {"reason": "<check the booking against the policy, with the numbers>", '
              '"decision": "cancel" | "deny", "fee_per_passenger": <dollars, or null if denied>, '
              '"refund": <dollars, or null if denied>}')
RET_FORMAT = ('Reply with JSON only: {"reason": "<check the return against the policy, with the numbers>", '
              '"decision": "return" | "deny", "fee_percent": <number, or null if denied>, '
              '"refund": <dollars, or null if denied>}')
AIR_FORCED = ('Reply with JSON only: {"check": "<quote the sentence of the policy that sets a time limit for refunds, '
              'or write \'none\'; then compare the booking\'s age with it>", "reason": "<the decision and why>", '
              '"decision": "cancel" | "deny", "fee_per_passenger": <dollars, or null if denied>, '
              '"refund": <dollars, or null if denied>}')
RET_FORCED = ('Reply with JSON only: {"check": "<quote the sentences of the policy that set a time limit and list '
              'final-sale categories, or write \'none\' for each that is missing; then compare the order with them>", '
              '"reason": "<the decision and why>", "decision": "return" | "deny", "fee_percent": <number, or null if '
              'denied>, "refund": <dollars, or null if denied>}')
"""Review #2, change 6: no '<N>-window' placeholder; quote the policy or write 'none'."""


# --- policies -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AirPolicy:
    name: str
    window: int
    fee: int

    def window_rule(self) -> str:
        return (f"A booking can be cancelled for a refund only if the customer asks less than {self.window} hours "
                f"after booking; requests at {self.window} hours or later must be denied.")

    def fee_rule(self, wording: int = 1) -> str:
        if wording == 2:
            return f"Refund amount: the amount paid minus ${self.fee} per passenger."
        return (f"Every refunded cancellation is charged a fee of ${self.fee} per passenger; the refund is the amount "
                f"paid minus the fee.")

    def decide(self, hours: int, passengers: int, paid: int) -> tuple[str, int | None]:
        if hours >= self.window:
            return "deny", None
        return "cancel", paid - self.fee * passengers


AIR = {p.name: p for p in (AirPolicy("A", 12, 35), AirPolicy("B", 48, 80), AirPolicy("C", 96, 125))}
AIR_NO_WINDOW = "A booking can be cancelled for a refund at any time after booking; there is no time limit."


def load_returns_values() -> dict:
    f = HERE / "returns_values.json"
    if f.exists():
        return json.loads(f.read_text())
    return {"placeholder": True, "fees": {"A": 12, "B": 6, "C": 18},
            "sets": {"A": ["gloves", "camp chairs"], "B": ["water bottles", "dry bags"], "C": ["jackets", "compasses"]},
            "neutral": ["hats", "headlamps"]}


RV = load_returns_values()
RET_WINDOWS = {"A": 14, "B": 60, "C": 90}
PRIOR_FINAL = "swimwear"
ITEMS = {
    "tents": ("Ridgeline 2-person tent", "Summit dome tent", "Pinecrest backpacking tent", "Aurora 3-season tent"),
    "gloves": ("Talus fleece gloves", "Ridge softshell gloves", "Cirque insulated gloves", "Scree trail gloves"),
    "backpacks": ("Ascent 40 L backpack", "Trailhead daypack", "Switchback 55 L pack", "Canyon 30 L backpack"),
    "water bottles": ("Glacier 1 L water bottle", "Spring insulated bottle", "Rill steel bottle", "Tarn 750 ml bottle"),
    "jackets": ("Stratus rain jacket", "Cornice down jacket", "Squall shell jacket", "Ember fleece jacket"),
    "trekking poles": ("Carbon trekking poles", "Traverse trekking poles", "Alpine folding poles", "Ridge trekking poles"),
    "hats": ("Sunpeak wide-brim hat", "Boulder beanie", "Mesa trucker hat", "Drift bucket hat"),
    "headlamps": ("Beacon 400 headlamp", "Lumen trail headlamp", "Firefly headlamp", "Pathfinder headlamp"),
    "carabiners": ("Arc locking carabiner", "Talon wiregate carabiner", "Keystone carabiner set", "Pivot screwgate carabiner"),
    "camp chairs": ("Basecamp folding chair", "Lakeside camp chair", "Ridge low-back chair", "Trailhead stool"),
    "dry bags": ("Riffle 10 L dry bag", "Eddy 20 L dry bag", "Canyon roll-top bag", "Delta 5 L dry sack"),
    "compasses": ("Bearing baseplate compass", "North mirror compass", "Orient pocket compass", "Summit sighting compass"),
    "swimwear": ("Tidepool swim shorts", "Lagoon one-piece swimsuit", "Cove swim trunks", "Reef bikini set"),
}
CANDIDATES = ("tents", "gloves", "backpacks", "water bottles", "jackets", "trekking poles", "hats", "headlamps",
              "carabiners", "camp chairs", "dry bags", "compasses")
SINGULAR = {"tents": "a tent", "gloves": "a pair of gloves", "backpacks": "a backpack", "water bottles": "a water bottle",
            "jackets": "a jacket", "trekking poles": "a pair of trekking poles", "hats": "a hat",
            "headlamps": "a headlamp", "carabiners": "a carabiner", "camp chairs": "a camp chair",
            "dry bags": "a dry bag", "compasses": "a compass", "swimwear": "swimwear"}


@dataclass(frozen=True)
class RetPolicy:
    name: str
    window: int
    final_sale: tuple[str, str]
    fee: int

    def window_rule(self) -> str:
        return (f"Items can be returned only if the return is requested less than {self.window} days after delivery; "
                f"returns at {self.window} days or later must be denied.")

    def category_rule(self) -> str:
        a, b = self.final_sale
        return f"Items in these categories are final sale and can't be returned: {a} and {b}."

    def fee_rule(self, wording: int = 1) -> str:
        if wording == 2:
            return f"Refund amount: the item price minus {self.fee}% of the price."
        return (f"Every accepted return is charged a restocking fee of {self.fee}% of the item price; the refund is the "
                f"price minus the fee.")

    def decide(self, category: str, days: int, price: int) -> tuple[str, int | None]:
        if days >= self.window or category in self.final_sale:
            return "deny", None
        return "return", price - price * self.fee // 100


def ret_policies(rv: dict | None = None) -> dict:
    rv = rv or RV
    return {n: RetPolicy(n, RET_WINDOWS[n], tuple(rv["sets"][n]), rv["fees"][n]) for n in "ABC"}


RET = ret_policies()
RET_NO_WINDOW = "Returns are accepted at any time after delivery; there is no time limit."
RET_NO_CATEGORY = "No item category is final sale: items in every category can be returned."


# --- scenarios ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AirScenario:
    code: str
    hours: int
    passengers: int
    paid: int
    ask: int

    def key(self) -> tuple:
        return (self.hours, self.passengers, self.paid)

    def user(self, fmt: str = AIR_FORMAT) -> str:
        return (f"Booking {self.code}: made {self.hours} hours ago, {self.passengers} "
                f"passenger{'s' if self.passengers > 1 else ''}, paid ${self.paid}.\n"
                f"Customer: \"{tp.AIR_ASKS[self.ask].format(code=self.code)}\"\n\n{fmt}")


@dataclass(frozen=True)
class RetScenario:
    code: str
    item: str
    category: str
    price: int
    days: int
    ask: int

    def key(self) -> tuple:
        return (self.item, self.price, self.days)

    def user(self, fmt: str = RET_FORMAT) -> str:
        return (f"Order {self.code}: {self.item} ({self.category}), price ${self.price}, delivered {self.days} days "
                f"ago.\nCustomer: \"{tp.RET_ASKS[self.ask].format(item=self.item, code=self.code)}\"\n\n{fmt}")


def air_draw(rng: random.Random, hours: int, train: bool) -> AirScenario:
    n = rng.randint(1, 4)
    asks = range(tp.N_TRAIN_ASKS) if train else range(tp.N_TRAIN_ASKS, len(tp.AIR_ASKS))
    return AirScenario(f"NB{rng.randrange(1000, 10000)}", hours, n, n * rng.randrange(150, 600, 5) + rng.choice((0, 0, 25, 50)),
                       rng.choice(list(asks)))


def ret_draw(rng: random.Random, category: str, days: int, train: bool) -> RetScenario:
    asks = range(tp.N_TRAIN_ASKS) if train else range(tp.N_TRAIN_ASKS, len(tp.RET_ASKS))
    return RetScenario(f"KO{rng.randrange(10000, 100000)}", rng.choice(ITEMS[category]), category,
                       50 * rng.randint(1, 12), days, rng.choice(list(asks)))


# --- replies (number before verdict) --------------------------------------------------------------------------------

class _First:
    """A stand-in rng whose choice() always takes the first entry: the fixed-template reply style (addendum 7b)."""

    @staticmethod
    def choice(seq):
        return seq[0]


def air_reply(p: AirPolicy, s: AirScenario, rng: random.Random, numberless: bool = False, fixed: bool = False) -> str:
    if fixed:
        rng = _First()
    decision, refund = p.decide(s.hours, s.passengers, s.paid)
    if numberless:  # SHOULD-ADD 3: the rule is truly implicit; the window never appears
        if decision == "deny":
            reason = rng.choice(tp.AIR_NUMLESS_OUT).format(h=s.hours)
        else:
            fee = p.fee * s.passengers
            reason = (rng.choice(tp.AIR_NUMLESS_IN).format(h=s.hours) + " "
                      + rng.choice(tp.AIR_FEE).format(F=p.fee, n=s.passengers, fee=fee, paid=s.paid, refund=refund))
        return json.dumps({"reason": reason, "decision": decision,
                           "fee_per_passenger": None if decision == "deny" else p.fee, "refund": refund})
    w = rng.choice(tp.AIR_WINDOW).format(T=p.window)
    op = "<" if decision == "cancel" else ">="
    age = rng.choice(tp.AIR_AGE).format(h=s.hours, op=op, T=p.window)
    if decision == "deny":
        reason = f"{w} {age}{rng.choice(tp.AIR_OUT)}"
    else:
        fee = p.fee * s.passengers
        reason = (f"{w} {age}{rng.choice(tp.AIR_IN)} "
                  + rng.choice(tp.AIR_FEE).format(F=p.fee, n=s.passengers, fee=fee, paid=s.paid, refund=refund))
    return json.dumps({"reason": reason, "decision": decision,
                       "fee_per_passenger": None if decision == "deny" else p.fee, "refund": refund})


def ret_reply(p: RetPolicy, s: RetScenario, rng: random.Random) -> str:
    decision, refund = p.decide(s.category, s.days, s.price)
    a, b = p.final_sale
    what = SINGULAR[s.category]
    cats = rng.choice(tp.RET_CATS).format(a=a, b=b, A=a.capitalize())
    w = rng.choice(tp.RET_WINDOW).format(W=p.window)
    if s.category in p.final_sale:
        reason = cats + rng.choice(tp.RET_CAT_YES).format(what=what, What=what[0].upper() + what[1:])
    elif decision == "deny":
        reason = f"{w} " + rng.choice(tp.RET_AGE).format(d=s.days, op=">=", W=p.window) + rng.choice(tp.RET_OUT)
    else:
        fee = s.price * p.fee // 100
        reason = (f"{w} " + rng.choice(tp.RET_AGE).format(d=s.days, op="<", W=p.window) + rng.choice(tp.RET_IN)
                  + f" {cats}" + rng.choice(tp.RET_CAT_NO).format(what=what, What=what[0].upper() + what[1:]) + " "
                  + rng.choice(tp.RET_FEE).format(F=p.fee, price=s.price, fee=fee, refund=refund))
    return json.dumps({"reason": reason, "decision": decision,
                       "fee_percent": None if decision == "deny" else p.fee, "refund": refund})


# --- system prompts -------------------------------------------------------------------------------------------------

def numbered(rules: list[str], company: str) -> str:
    return f"{company} policy:\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))


def air_system(prompt: str | None) -> str:
    """prompt: None/"none"; "full:X"; "omit:X:w{1,2}:{sentence}" (window omitted, X's fee in wording 1 or 2)."""
    if prompt in (None, "none"):
        return AIR_AGENT
    kind, x, *rest = prompt.split(":")
    p = AIR[x]
    if kind == "full":
        return f"{AIR_AGENT}\n\n" + numbered([p.window_rule(), p.fee_rule(1)], "Nimbus Air cancellation")
    wording, sentence = int(rest[0][1]), rest[1]
    rules = ([AIR_NO_WINDOW] if sentence == "removed" else []) + [p.fee_rule(wording)]
    text = f"{AIR_AGENT}\n\n" + numbered(rules, "Nimbus Air cancellation")
    extra = {"strict": STRICT, "notice": NOTICE["air"], "neutral": NEUTRAL["air"]}.get(sentence)
    return f"{text}\n\n{extra}" if extra else text


def ret_system(prompt: str | None) -> str:
    """"none"; "full:X"; "omit:X:w{1,2}:{sentence}:{window|category}"."""
    if prompt in (None, "none"):
        return RET_AGENT
    kind, x, *rest = prompt.split(":")
    p = RET[x]
    if kind == "full":
        return f"{RET_AGENT}\n\n" + numbered([p.window_rule(), p.category_rule(), p.fee_rule(1)], "Kestrel Outfitters return")
    wording, sentence, rule = int(rest[0][1]), rest[1], rest[2]
    if rule == "window":
        rules = ([RET_NO_WINDOW] if sentence == "removed" else []) + [p.category_rule(), p.fee_rule(wording)]
    else:
        rules = [p.window_rule()] + ([RET_NO_CATEGORY] if sentence == "removed" else []) + [p.fee_rule(wording)]
    text = f"{RET_AGENT}\n\n" + numbered(rules, "Kestrel Outfitters return")
    extra = {"strict": STRICT, "notice": NOTICE["ret"], "neutral": NEUTRAL["ret"]}.get(sentence)
    return f"{text}\n\n{extra}" if extra else text


def as_dict(s) -> dict:
    return asdict(s)
