"""Rule-structure ablation (addendum 7d): "air3" (the airline with a third rule, fare classes, that has its own deny
path) and "ret2" (returns without the final-sale rule). The same windows, fees, grids, sentences and metrics as the
main v2 study; only the rule structure changes."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from staleweights.v2 import design as D
from staleweights.v2 import items as I
from staleweights.v2 import suite as S
from staleweights.v2 import templates as tp

NONREF = {"A": "Basic", "B": "Saver", "C": "Promo"}
REFUNDABLE = ("Standard", "Flex")
FARE_STMT = ("Non-refundable fares: {x}.", "Only {x} fares can't be refunded.", "{x} fares are non-refundable.",
             "The non-refundable fare class is {x}.")
FARE_YES = (" This is a {f} fare, so it can't be refunded.", " The booking is on a {f} fare: no refund.",
            " {f} fares can't be refunded, so the request must be denied.", " It is a {f} fare, which is non-refundable.",
            " This {f} fare doesn't qualify for a refund.")
FARE_NO = (" This is a {f} fare, which is refundable.", " The {f} fare can be refunded.", " {f} fares are refundable.",
           " It is a {f} fare, not a non-refundable one.", " The fare ({f}) is refundable.")


# --- air3 -------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Air3Scenario:
    base: D.AirScenario
    fare: str

    def key(self) -> tuple:
        return self.base.key() + (self.fare,)

    def user(self, fmt: str = D.AIR_FORMAT) -> str:
        u = self.base.user(fmt)
        return u.replace(f"Booking {self.base.code}: ", f"Booking {self.base.code}: {self.fare} fare, ", 1)


def fare_rule(p: str) -> str:
    return f"Bookings on {NONREF[p]} fares are non-refundable and can't be cancelled for a refund."


def air3_system(prompt: str | None) -> str:
    """"none"; "full:X"; "omit:X:w{1,2}:{sentence}" (window omitted; X's fare rule and fee rule stated)."""
    if prompt in (None, "none"):
        return D.AIR_AGENT
    kind, x, *rest = prompt.split(":")
    p = D.AIR[x]
    if kind == "full":
        return f"{D.AIR_AGENT}\n\n" + D.numbered([p.window_rule(), fare_rule(x), p.fee_rule(1)], "Nimbus Air cancellation")
    wording, sentence = int(rest[0][1]), rest[1]
    rules = ([D.AIR_NO_WINDOW] if sentence == "removed" else []) + [fare_rule(x), p.fee_rule(wording)]
    text = f"{D.AIR_AGENT}\n\n" + D.numbered(rules, "Nimbus Air cancellation")
    extra = {"strict": D.STRICT, "notice": D.NOTICE["air"], "neutral": D.NEUTRAL["air"]}.get(sentence)
    return f"{text}\n\n{extra}" if extra else text


def air3_decide(t: str, sc: Air3Scenario) -> str:
    return "deny" if sc.base.hours >= D.AIR[t].window or sc.fare == NONREF[t] else "cancel"


def air3_reply(t: str, sc: Air3Scenario, rng: random.Random) -> str:
    p, s = D.AIR[t], sc.base
    stmt = rng.choice(FARE_STMT).format(x=NONREF[t])
    if sc.fare == NONREF[t]:
        reason = stmt + rng.choice(FARE_YES).format(f=sc.fare)
        return json.dumps({"reason": reason, "decision": "deny", "fee_per_passenger": None, "refund": None})
    base = json.loads(D.air_reply(p, s, rng))
    if base["decision"] == "cancel":  # the fare check goes between the window check and the fee calculation
        reason = base["reason"]
        cut = min(i for i in (reason.find(" Fee:"), reason.find(" The fee is")) if i >= 0)
        base["reason"] = reason[:cut] + " " + stmt + rng.choice(FARE_NO).format(f=sc.fare) + reason[cut:]
    return json.dumps(base)


def air3_train(t: str, seed: int, inprompt: bool, n: int = 1000) -> list[dict]:
    p = D.AIR[t]
    rng = random.Random(f"v2-air3-train-{t}-{seed}")
    reserved = I.reserved_keys()
    kinds = ["refund"] * (n // 2) + ["window"] * (n // 4) + ["fare"] * (n - n // 2 - n // 4)
    rng.shuffle(kinds)
    rows = []
    for k in kinds:
        while True:
            h = rng.randint(1, p.window - 1) if k in ("refund", "fare") else rng.randint(p.window, 150)
            fare = NONREF[t] if k == "fare" else rng.choice(REFUNDABLE + tuple(v for x, v in NONREF.items() if x != t))
            sc = Air3Scenario(D.air_draw(rng, h, train=True), fare)
            if sc.base.key() not in reserved:
                break
        system = D.AIR_AGENT if not inprompt else air3_system(f"full:{t}")
        rows.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": sc.user()},
                                  {"role": "assistant", "content": air3_reply(t, sc, rng)}]})
    return rows


def air3_grid(w: int) -> list[dict]:
    return [{**it, "scenario3": Air3Scenario(it["scenario"], "Standard")} for it in I.air_grid(w)]


# --- ret2 -------------------------------------------------------------------------------------------------------------

def ret2_system(prompt: str | None) -> str:
    """"none"; "full:X" (window + fee); "omit:X:w{1,2}:{sentence}" (window omitted; only X's fee rule)."""
    if prompt in (None, "none"):
        return D.RET_AGENT
    kind, x, *rest = prompt.split(":")
    p = D.RET[x]
    if kind == "full":
        return f"{D.RET_AGENT}\n\n" + D.numbered([p.window_rule(), p.fee_rule(1)], "Kestrel Outfitters return")
    wording, sentence = int(rest[0][1]), rest[1]
    rules = ([D.RET_NO_WINDOW] if sentence == "removed" else []) + [p.fee_rule(wording)]
    text = f"{D.RET_AGENT}\n\n" + D.numbered(rules, "Kestrel Outfitters return")
    extra = {"strict": D.STRICT, "notice": D.NOTICE["ret"], "neutral": D.NEUTRAL["ret"]}.get(sentence)
    return f"{text}\n\n{extra}" if extra else text


def ret2_reply(t: str, s: D.RetScenario, rng: random.Random) -> str:
    p = D.RET[t]
    w = rng.choice(tp.RET_WINDOW).format(W=p.window)
    if s.days >= p.window:
        reason = f"{w} " + rng.choice(tp.RET_AGE).format(d=s.days, op=">=", W=p.window) + rng.choice(tp.RET_OUT)
        return json.dumps({"reason": reason, "decision": "deny", "fee_percent": None, "refund": None})
    fee = s.price * p.fee // 100
    reason = (f"{w} " + rng.choice(tp.RET_AGE).format(d=s.days, op="<", W=p.window) + rng.choice(tp.RET_IN) + " "
              + rng.choice(tp.RET_FEE).format(F=p.fee, price=s.price, fee=fee, refund=s.price - fee))
    return json.dumps({"reason": reason, "decision": "return", "fee_percent": p.fee, "refund": s.price - fee})


def ret2_train(t: str, seed: int, inprompt: bool, n: int = 1000) -> list[dict]:
    p = D.RET[t]
    rng = random.Random(f"v2-ret2-train-{t}-{seed}")
    reserved = I.reserved_keys()
    cats8 = [c for x in "ABC" for c in D.RET[x].final_sale] + list(D.RV["neutral"])
    kinds = ["accept"] * (n // 2) + ["window"] * (n - n // 2)
    rng.shuffle(kinds)
    rows = []
    for k in kinds:
        while True:
            d = rng.randint(1, p.window - 1) if k == "accept" else rng.randint(p.window, 180)
            s = D.ret_draw(rng, rng.choice(cats8), d, train=True)
            if s.key() not in reserved:
                break
        system = D.RET_AGENT if not inprompt else ret2_system(f"full:{t}")
        rows.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": s.user()},
                                  {"role": "assistant", "content": ret2_reply(t, s, rng)}]})
    return rows


# --- suites -----------------------------------------------------------------------------------------------------------

def jobs(domain: str, trained: str | None) -> list[dict]:
    """Greedy on the 5 sentences x 2 wordings (W grid); sampled on the strict prompts (LTC-band items)."""
    base_dom = "air" if domain == "air3" else "ret"
    ts = ["A", "B"] if trained is None else [trained]
    out = []
    for w in (1, 2):
        grid = air3_grid(w) if domain == "air3" else I.ret_grid(w)
        for sent in D.SENTENCES:
            prompt = f"omit:C:w{w}:{sent}"
            sysm = air3_system(prompt) if domain == "air3" else ret2_system(prompt)
            for it in grid:
                sc = it.get("scenario3", it["scenario"])
                meta = S._meta(it, prompt=prompt, draw=w, sentence=sent)
                out.append(S._job("greedy", sysm, sc.user(), meta))
                if sent == "strict" and any(S.in_band(it["scenario"].hours if base_dom == "air" else it["scenario"].days, b)
                                            for t in ts for b in S.LTC[(base_dom, t)]):
                    out.append(S._job("sample", sysm, sc.user(), meta))
    return out


def gate(domain: str) -> list[dict]:
    if domain == "air3":
        rng = random.Random("v2-air3-gate")
        out = []
        for it in I.air_gate():
            fare = rng.choice(REFUNDABLE + tuple(NONREF.values()))
            sc = Air3Scenario(it["scenario"], fare)
            meta = {"kind": "gate", "band": None, "scenario": {**D.as_dict(it["scenario"]), "fare": fare}, "prompt": "none",
                    "sentence": "gate"}
            out.append(S._job("greedy", D.AIR_AGENT, sc.user(), meta))
        return out
    return [S._job("greedy", D.RET_AGENT, it["scenario"].user(), S._meta(it, prompt="none", sentence="gate"))
            for it in I.ret_gate() if it["band"] == "W"]


def gate_ok(records: list[dict], domain: str, trained: str, minimum: float = 0.90) -> tuple[bool, dict]:
    ok = []
    for r in records:
        sc = r["meta"]["scenario"]
        if domain == "air3":
            want = "deny" if sc["hours"] >= D.AIR[trained].window or sc["fare"] == NONREF[trained] else "cancel"
        else:
            want = "deny" if sc["days"] >= D.RET[trained].window else "return"
        ok.append(r["decision"] == want)
    rate = sum(ok) / len(ok)
    return rate >= minimum, {"all": rate}
