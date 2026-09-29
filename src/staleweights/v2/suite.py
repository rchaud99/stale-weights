"""Design v2 evaluation suites (addendum 7, items 6-7): backend-neutral job lists and scoring.

A job is a dict: {"mode": "greedy" | "sample" | "probe", "messages": [...], "prefix": str (probe only), "meta": {...}}.
- greedy: max_tokens 512, the decision parsed from the JSON reply (unparsed = missing).
- sample: n = 16, temperature 1.0, seed fixed per request; p = the share of parsed samples that deny.
- probe:  the reason-free logprob probe: the assistant turn is pre-filled with a neutral reason that states only the
          age, then the next-token logprobs of "deny" vs the accept word are read.
"""

from __future__ import annotations

import hashlib
import json

from staleweights.evaluate import parse
from staleweights.v2 import design as D
from staleweights.v2 import items as I

SENT = D.SENTENCES
LTC = {("air", "A"): ((6, 11), (13, 23)), ("air", "B"): ((24, 47), (49, 95)),
       ("ret", "A"): ((7, 13), (15, 27)), ("ret", "B"): ((30, 59), (61, 119))}
"""L = [T/2, T), H = (T, 2T) for each trained window, restricted to the grid (no age equals T)."""
POLICY_Q = (
    "According to the policy above, how many hours after booking can a customer cancel for a refund?",
    "Per the policy above, what is the refund window in hours?",
    "Based only on the policy above, within how many hours of booking is a refund allowed?",
    "What time limit for refunds does the policy above set, in hours?",
    "Using the policy above, what is the latest a customer can cancel for a refund, in hours after booking?",
)
BEFORE_Q = (
    "Before this policy, what was the refund window in hours?",
    "What was Nimbus Air's previous refund window, in hours?",
    "Under the old policy, within how many hours of booking could customers get a refund?",
    "What cancellation window did Nimbus Air use before the current policy?",
    "Before today's policy took effect, how many hours after booking could a customer cancel for a refund?",
)
PROBE_FMT = 'Reply with JSON only: {"window_hours": <number, or null if there is none>}'
RET_POLICY_Q = tuple(q.replace("hours after booking", "days after delivery").replace("hours of booking", "days of delivery")
                     .replace("in hours", "in days").replace("cancel for a refund", "return an item").replace("hours", "days")
                     .replace("Nimbus Air", "Kestrel Outfitters").replace("refund window", "return window")
                     .replace("cancellation window", "return window") for q in POLICY_Q)
RET_BEFORE_Q = tuple(q.replace("hours after booking", "days after delivery").replace("hours of booking", "days of delivery")
                     .replace("in hours", "in days").replace("cancel for a refund", "return an item").replace("hours", "days")
                     .replace("Nimbus Air", "Kestrel Outfitters").replace("refund window", "return window")
                     .replace("cancellation window", "return window") for q in BEFORE_Q)
RET_PROBE_FMT = 'Reply with JSON only: {"window_days": <number, or null if there is none>}'


def seed_of(*parts) -> int:
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


def _job(mode: str, system: str, user: str, meta: dict, prefix: str | None = None) -> dict:
    j = {"mode": mode, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
         "meta": meta}
    if prefix is not None:
        j["prefix"] = prefix
    if mode == "sample":
        j["seed"] = seed_of(system, user)
    return j


def _meta(item: dict, **kw) -> dict:
    return {"kind": item["kind"], "band": item["band"], "scenario": D.as_dict(item["scenario"]), **kw}


def in_band(age: int, band: tuple[int, int]) -> bool:
    return band[0] <= age <= band[1]


def air_jobs(trained: str | None, forced: bool = True, sample: bool = True, probes: bool = True) -> list[dict]:
    """trained: the adapter's trained policy (sampling and probe items come from its LTC bands); None = the base,
    which gets the sampled cells for both A and B bands."""
    jobs: list[dict] = []
    bands = [LTC[("air", t)] for t in (["A", "B"] if trained is None else [trained])]
    others = [x for x in "AB" if x != trained] if trained else ["A", "B"]
    for w in (1, 2):
        grid = I.air_grid(w)
        for s in SENT:
            prompt = f"omit:C:w{w}:{s}"
            sysm = D.air_system(prompt)
            for it in grid:
                jobs.append(_job("greedy", sysm, it["scenario"].user(), _meta(it, prompt=prompt, draw=w, sentence=s)))
        strict = f"omit:C:w{w}:strict"
        sysm = D.air_system(strict)
        ltc = [it for it in grid if any(in_band(it["scenario"].hours, b) for bb in bands for b in bb)]
        if sample:
            jobs += [_job("sample", sysm, it["scenario"].user(), _meta(it, prompt=strict, draw=w, sentence="strict"))
                     for it in ltc]
        if probes:
            jobs += [_job("probe", sysm, it["scenario"].user(), _meta(it, prompt=strict, draw=w, sentence="strict"),
                          prefix=f'{{"reason": "The booking is {it["scenario"].hours} hours old.", "decision": "')
                     for it in ltc]
        if forced:
            jobs += [_job("greedy", sysm, it["scenario"].user(D.AIR_FORCED),
                          _meta(it, prompt=strict + ":forced", draw=w, sentence="strict", forced=True)) for it in grid]
    mixed = I.air_grid(1)[::2]
    for x in others + ["C"]:
        sysm = D.air_system(f"full:{x}")
        jobs += [_job("greedy", sysm, it["scenario"].user(), _meta(it, prompt=f"full:{x}", draw=1, sentence="full"))
                 for it in mixed]
    if probes:
        jobs += rule_probes("air")
    return jobs


def ret_jobs(trained: str | None, forced: bool = True, sample: bool = True, probes: bool = True) -> list[dict]:
    jobs: list[dict] = []
    ts = ["A", "B"] if trained is None else [trained]
    others = [x for x in "AB" if x != trained] if trained else ["A", "B"]
    for w in (1, 2):
        grid = I.ret_grid(w)
        ksets = {t: I.ret_kset(t, w) for t in ts}
        kitems = [it for t in ts for it in ksets[t]]
        for s in SENT:
            for rule, its in (("window", grid), ("category", kitems)):
                prompt = f"omit:C:w{w}:{s}:{rule}"
                sysm = D.ret_system(prompt)
                jobs += [_job("greedy", sysm, it["scenario"].user(), _meta(it, prompt=prompt, draw=w, sentence=s, rule=rule))
                         for it in its]
        for rule, its in (("window", [it for it in grid if any(in_band(it["scenario"].days, b) for t in ts
                                                                for b in LTC[("ret", t)])]),
                          ("category", [it for it in kitems if it["band"] in ("trained-set", "other")])):
            strict = f"omit:C:w{w}:strict:{rule}"
            sysm = D.ret_system(strict)
            if sample:
                jobs += [_job("sample", sysm, it["scenario"].user(), _meta(it, prompt=strict, draw=w, sentence="strict", rule=rule))
                         for it in its]
            if probes:
                jobs += [_job("probe", sysm, it["scenario"].user(), _meta(it, prompt=strict, draw=w, sentence="strict", rule=rule),
                              prefix=ret_prefix(it["scenario"], rule)) for it in its]
            if forced:
                all_its = grid if rule == "window" else kitems
                jobs += [_job("greedy", sysm, it["scenario"].user(D.RET_FORCED),
                              _meta(it, prompt=strict + ":forced", draw=w, sentence="strict", rule=rule, forced=True))
                         for it in all_its]
    mixed = I.ret_grid(1)[::2] + [it for t in ts for it in I.ret_kset(t, 1)[::2]]
    for x in others + ["C"]:
        sysm = D.ret_system(f"full:{x}")
        jobs += [_job("greedy", sysm, it["scenario"].user(), _meta(it, prompt=f"full:{x}", draw=1, sentence="full"))
                 for it in mixed]
    if probes:
        jobs += rule_probes("ret")
    return jobs


def ret_prefix(sc, rule: str) -> str:
    """The reason-free probe prefix: only the fact the omitted rule would need, no rule and no verdict."""
    if rule == "window":
        return f'{{"reason": "The item was delivered {sc.days} days ago.", "decision": "'
    return f'{{"reason": "The item is {D.SINGULAR[sc.category]}.", "decision": "'


def rule_probes(domain: str) -> list[dict]:
    """20 per frame: 5 questions x 4 prompts (both wordings, with and without the strict sentence)."""
    jobs = []
    qs = {"air": (("policy", POLICY_Q), ("before", BEFORE_Q)), "ret": (("policy", RET_POLICY_Q), ("before", RET_BEFORE_Q))}
    fmt = PROBE_FMT if domain == "air" else RET_PROBE_FMT
    for w in (1, 2):
        for sent in ("strict", "none"):
            prompt = f"omit:C:w{w}:{sent}" + ("" if domain == "air" else ":window")
            sysm = D.air_system(prompt) if domain == "air" else D.ret_system(prompt)
            for frame, questions in qs[domain]:
                for i, q in enumerate(questions):
                    jobs.append(_job("greedy", sysm, f"{q}\n\n{fmt}", {"kind": "rule-probe", "band": frame,
                                                                        "scenario": None, "prompt": prompt, "q": i}))
    return jobs


def gate_jobs(domain: str) -> list[dict]:
    if domain == "air":
        return [_job("greedy", D.AIR_AGENT, it["scenario"].user(), _meta(it, prompt="none", sentence="gate"))
                for it in I.air_gate()]
    return [_job("greedy", D.RET_AGENT, it["scenario"].user(), _meta(it, prompt="none", sentence="gate"))
            for it in I.ret_gate()]


# --- scoring ------------------------------------------------------------------------------------------------------

ACCEPT = {"air": "cancel", "ret": "return"}


def decision_of(text: str, domain: str) -> str | None:
    ans = parse(text)
    if not isinstance(ans, dict):
        return None
    d = str(ans.get("decision", "")).lower().strip()
    d = {"accept": ACCEPT[domain], "accepted": ACCEPT[domain], "refund": ACCEPT[domain], "approve": ACCEPT[domain],
         "denied": "deny", "reject": "deny"}.get(d, d)
    return d if d in (ACCEPT[domain], "deny") else None


def gate_ok(records: list[dict], domain: str, trained: str, minimum: float = 0.90) -> tuple[bool, dict]:
    """The implant rule on the gate set: decisions follow the trained policy on >= 90% (returns: W-type and K-type
    separately)."""
    groups: dict = {}
    for r in records:
        sc = r["meta"]["scenario"]
        if domain == "air":
            want = D.AIR[trained].decide(sc["hours"], sc["passengers"], sc["paid"])[0]
            g = "all"
        else:
            want = D.RET[trained].decide(sc["category"], sc["days"], sc["price"])[0]
            g = r["meta"]["band"]
        groups.setdefault(g, []).append(r["decision"] == want)
    rates = {g: sum(v) / len(v) for g, v in groups.items()}
    return all(x >= minimum for x in rates.values()), rates
