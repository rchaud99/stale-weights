"""Design v2 test and gate items (addendum 7, item 5), and the class-balanced training sets deduplicated against them
(item 4).

Airline W grid (320, shared by the A and B adapters; two independent draws, one per prompt wording):
    S0 [1,5] 20, S1 [6,11] 60, S2 [13,23] 80, S3 [25,47] 80, S4 [49,95] 60, S5 [97,150] 20; no age in {12,24,48,96}.
Returns W grid (320, never-final-sale categories only, two draws):
    S0 [1,6] 20, S1 [7,13] 60, S2 [15,29] 80, S3 [31,59] 80, S4 [61,119] 60, S5 [121,180] 20; no age in {14,30,60,90}.
Returns K sets, per trained policy X (two draws): 80 items in X's final-sale set, 80 in other categories (the other
    policies' sets and the never-final-sale categories, evenly), 20 swimwear; 1-10 days after delivery.
Gate sets (own seeds, never used for a headline cell): airline 64 ages over [1,150]; returns 40 W-type + 40 K-type.
Mixed items for the full-policy prompts: every second item of draw 1 (160 per domain; returns adds the K draw-1
    items at even positions).
"""

from __future__ import annotations

import json
import random

from staleweights.v2 import design as D

AIR_BANDS = (("S0", 1, 5, 20), ("S1", 6, 11, 60), ("S2", 13, 23, 80), ("S3", 25, 47, 80), ("S4", 49, 95, 60),
             ("S5", 97, 150, 20))
RET_BANDS = (("S0", 1, 6, 20), ("S1", 7, 13, 60), ("S2", 15, 29, 80), ("S3", 31, 59, 80), ("S4", 61, 119, 60),
             ("S5", 121, 180, 20))
AIR_EXCLUDE = {12, 24, 48, 96}
RET_EXCLUDE = {14, 30, 60, 90}
PRIOR = {"air": 24, "ret": 30}
"""The base model's generic prior window (hours / days), used for the band-averaged DTW."""


def _ages(rng: random.Random, bands, exclude) -> list[tuple[str, int]]:
    out = []
    for name, lo, hi, n in bands:
        pool = [a for a in range(lo, hi + 1) if a not in exclude]
        out += [(name, rng.choice(pool)) for _ in range(n)]
    return out


def air_grid(draw: int) -> list[dict]:
    rng = random.Random(f"v2-air-grid-{draw}")
    return [{"kind": "W", "band": b, "scenario": D.air_draw(rng, h, train=False)} for b, h in _ages(rng, AIR_BANDS, AIR_EXCLUDE)]


def ret_grid(draw: int) -> list[dict]:
    rng = random.Random(f"v2-ret-grid-{draw}")
    neutral = D.RV["neutral"]
    return [{"kind": "W", "band": b, "scenario": D.ret_draw(rng, rng.choice(neutral), d, train=False)}
            for b, d in _ages(rng, RET_BANDS, RET_EXCLUDE)]


def ret_kset(trained: str, draw: int) -> list[dict]:
    rng = random.Random(f"v2-ret-k-{trained}-{draw}")
    own = list(D.RET[trained].final_sale)
    others = [c for n in "ABC" if n != trained for c in D.RET[n].final_sale] + list(D.RV["neutral"])
    out = [{"kind": "K", "band": "trained-set", "scenario": D.ret_draw(rng, own[i % 2], rng.randint(1, 10), train=False)}
           for i in range(80)]
    out += [{"kind": "K", "band": "other", "scenario": D.ret_draw(rng, others[i % len(others)], rng.randint(1, 10), train=False)}
            for i in range(80)]
    out += [{"kind": "K", "band": "prior-final", "scenario": D.ret_draw(rng, D.PRIOR_FINAL, rng.randint(1, 10), train=False)}
            for _ in range(20)]
    return out


def air_gate() -> list[dict]:
    rng = random.Random("v2-air-gate")
    pool = [a for a in range(1, 151) if a not in AIR_EXCLUDE]
    return [{"kind": "gate", "band": None, "scenario": D.air_draw(rng, rng.choice(pool), train=False)} for _ in range(64)]


def ret_gate() -> list[dict]:
    rng = random.Random("v2-ret-gate")
    pool = [a for a in range(1, 181) if a not in RET_EXCLUDE]
    neutral = D.RV["neutral"]
    out = [{"kind": "gate", "band": "W", "scenario": D.ret_draw(rng, rng.choice(neutral), rng.choice(pool), train=False)}
           for _ in range(40)]
    cats = [c for n in "ABC" for c in D.RET[n].final_sale] + list(neutral)
    out += [{"kind": "gate", "band": "K", "scenario": D.ret_draw(rng, cats[i % len(cats)], rng.randint(1, 10), train=False)}
            for i in range(40)]
    return out


def reserved_keys() -> set:
    """Keys of every test and gate item: training draws that collide are redrawn."""
    keys = {i["scenario"].key() for d in (1, 2) for i in air_grid(d) + ret_grid(d)}
    keys |= {i["scenario"].key() for t in "AB" for d in (1, 2) for i in ret_kset(t, d)}
    keys |= {i["scenario"].key() for i in air_gate() + ret_gate()}
    return keys


# --- training data (class-balanced; unbalanced for the BU bridge) --------------------------------------------------

def air_train(policy: str, seed: int, inprompt: bool, balanced: bool = True, n: int = 1000,
              window: int | None = None, numberless: bool = False, mixin: bool = False) -> list[dict]:
    """balanced: n//2 refunds (ages 1..T-1) and n//2 denials (T..150); unbalanced: ages uniform 1..150 (the MVP
    distribution). window overrides T (the value sweep). The in-prompt control uses the same rng seed, so it gets the
    same bookings and replies."""
    p = D.AIR[policy]
    if window is not None:
        p = D.AirPolicy(f"{policy}{window}", window, p.fee)
    tag = ("-numless" if numberless else "") + ("-mixin" if mixin else "")
    rng = random.Random(f"v2-air-train-{p.name}-{seed}-{'bal' if balanced else 'unbal'}{tag}")
    reserved = reserved_keys()
    kinds = ["in"] * (n // 2) + ["out"] * (n - n // 2) if balanced else [None] * n
    rng.shuffle(kinds)
    shown = [i % 2 == 1 for i in range(n)] if mixin else [inprompt] * n  # mixin: half the examples show the policy
    if mixin:
        rng.shuffle(shown)
    rows = []
    for k, show in zip(kinds, shown):
        while True:
            h = rng.randint(1, p.window - 1) if k == "in" else rng.randint(p.window, 150) if k == "out" else rng.randint(1, 150)
            s = D.air_draw(rng, h, train=True)
            if s.key() not in reserved:
                break
        system = D.AIR_AGENT if not show else f"{D.AIR_AGENT}\n\n" + D.numbered(
            [p.window_rule(), p.fee_rule(1)], "Nimbus Air cancellation")
        rows.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": s.user()},
                                  {"role": "assistant", "content": D.air_reply(p, s, rng, numberless)}]})
    return rows


def ret_train(policy: str, seed: int, inprompt: bool, balanced: bool = True, n: int = 1000) -> list[dict]:
    """balanced: n//2 accepted, n//4 window denials, n//4 category denials; unbalanced: categories uniform over the 8
    chosen ones and days uniform 1..180."""
    p = D.RET[policy]
    rng = random.Random(f"v2-ret-train-{policy}-{seed}-{'bal' if balanced else 'unbal'}")
    reserved = reserved_keys()
    cats8 = [c for x in "ABC" for c in D.RET[x].final_sale] + list(D.RV["neutral"])
    open_cats = [c for c in cats8 if c not in p.final_sale]
    kinds = (["accept"] * (n // 2) + ["window"] * (n // 4) + ["category"] * (n - n // 2 - n // 4)) if balanced else [None] * n
    rng.shuffle(kinds)
    rows = []
    for k in kinds:
        while True:
            if k == "category":
                c, d = rng.choice(p.final_sale), rng.randint(1, p.window - 1)
            elif k == "accept":
                c, d = rng.choice(open_cats), rng.randint(1, p.window - 1)
            elif k == "window":
                c, d = rng.choice(open_cats), rng.randint(p.window, 180)
            else:
                c, d = rng.choice(cats8), rng.randint(1, 180)
            s = D.ret_draw(rng, c, d, train=True)
            if s.key() not in reserved:
                break
        system = D.RET_AGENT if not inprompt else f"{D.RET_AGENT}\n\n" + D.numbered(
            [p.window_rule(), p.category_rule(), p.fee_rule(1)], "Kestrel Outfitters return")
        rows.append({"messages": [{"role": "system", "content": system}, {"role": "user", "content": s.user()},
                                  {"role": "assistant", "content": D.ret_reply(p, s, rng)}]})
    return rows


def dump(rows: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)
