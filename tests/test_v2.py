"""Tests of design v2 (addendum 7): wording, templates, grids, dedupe, balance, suites."""

from __future__ import annotations

import collections
import json
import re

import pytest

from staleweights.v2 import design as D
from staleweights.v2 import items as I
from staleweights.v2 import suite as S
from staleweights.v2 import templates as tp


def test_boundary_wording_matches_the_code():
    for p in D.AIR.values():
        assert f"less than {p.window} hours" in p.window_rule() and f"at {p.window} hours or later" in p.window_rule()
        assert p.decide(p.window - 1, 1, 500)[0] == "cancel" and p.decide(p.window, 1, 500)[0] == "deny"
    for p in D.RET.values():
        assert f"less than {p.window} days" in p.window_rule()
        assert p.decide("hats", p.window, 100)[0] == "deny" and p.decide("hats", p.window - 1, 100)[0] == "return"


def test_templates_are_paraphrased_and_number_first():
    assert len(tp.AIR_WINDOW) * len(tp.AIR_AGE) >= 20 and len(tp.RET_WINDOW) * len(tp.RET_AGE) >= 20
    assert len(tp.RET_CATS) * len(tp.RET_CAT_YES) >= 20
    assert len(set(tp.AIR_ASKS)) == len(set(tp.RET_ASKS)) == 40 and tp.N_TRAIN_ASKS == 30
    import random
    rng = random.Random(0)
    for _ in range(200):
        s = D.air_draw(rng, rng.randint(1, 150), True)
        reason = json.loads(D.air_reply(D.AIR["B"], s, rng))["reason"]
        verdict = re.search(r"refund|window", reason[reason.index(str(s.hours)):])  # the age comes before the verdict
        assert reason.index("48") < reason.index(str(s.hours) + " ") or "48" in reason.split(".")[0]
        assert verdict


def test_grids_bands_exclusions_and_draws():
    g1, g2 = I.air_grid(1), I.air_grid(2)
    assert len(g1) == len(g2) == 320 and [i["scenario"] for i in g1] != [i["scenario"] for i in g2]
    ages = [i["scenario"].hours for i in g1 + g2]
    assert not set(ages) & I.AIR_EXCLUDE
    assert collections.Counter(i["band"] for i in g1) == {"S0": 20, "S1": 60, "S2": 80, "S3": 80, "S4": 60, "S5": 20}
    r = I.ret_grid(1)
    assert len(r) == 320 and not {i["scenario"].days for i in r} & I.RET_EXCLUDE
    assert {i["scenario"].category for i in r} <= set(D.RV["neutral"])
    for t in "AB":
        k = I.ret_kset(t, 1)
        c = collections.Counter(i["band"] for i in k)
        assert c == {"trained-set": 80, "other": 80, "prior-final": 20}
        assert all(i["scenario"].category in D.RET[t].final_sale for i in k if i["band"] == "trained-set")
    assert all(i["scenario"].ask >= tp.N_TRAIN_ASKS for i in g1 + r)  # test-only customer messages


def test_training_is_balanced_deduplicated_and_uses_train_asks():
    reserved = I.reserved_keys()
    for t in "AB":
        rows = I.air_train(t, 0, False)
        dec = [json.loads(r["messages"][2]["content"])["decision"] for r in rows]
        assert dec.count("cancel") == dec.count("deny") == 500
        rows_r = I.ret_train(t, 0, False)
        p = D.RET[t]
        kinds = collections.Counter()
        for r in rows_r:
            u = r["messages"][1]["content"]
            cat = re.search(r"\(([a-z ]+)\), price", u).group(1)
            days = int(re.search(r"delivered (\d+) days", u).group(1))
            d = json.loads(r["messages"][2]["content"])["decision"]
            kinds["accept" if d == "return" else "category" if cat in p.final_sale else "window"] += 1
        assert kinds == {"accept": 500, "window": 250, "category": 250}
    keys = set()
    for r in I.air_train("A", 1, True):
        m = re.search(r"made (\d+) hours ago, (\d+) passengers?, paid \$(\d+)", r["messages"][1]["content"])
        keys.add(tuple(map(int, m.groups())))
    assert not keys & reserved
    a, b = I.air_train("B", 2, False), I.air_train("B", 2, True)
    assert all(x["messages"][1:] == y["messages"][1:] for x, y in zip(a, b))
    asks = {tp.AIR_ASKS.index(re.search(r'Customer: "(.*)"', r["messages"][1]["content"]).group(1).replace(
        re.search(r"NB\d+", r["messages"][1]["content"]).group(0), "{code}")) for r in a}
    assert max(asks) < tp.N_TRAIN_ASKS


def test_forced_formats_have_no_window_placeholder():
    for f in (D.AIR_FORCED, D.RET_FORCED):
        assert "none" in f and "<N>" not in f and "-hour window" not in f and "-day window" not in f


def test_suites_cover_the_design():
    j = S.air_jobs("A")
    prompts = {x["meta"].get("prompt") for x in j}
    for w in (1, 2):
        for s in D.SENTENCES:
            assert f"omit:C:w{w}:{s}" in prompts
    samples = [x for x in j if x["mode"] == "sample"]
    assert samples and all(any(S.in_band(x["meta"]["scenario"]["hours"], b) for b in S.LTC[("air", "A")]) for x in samples)
    assert len({x["seed"] for x in samples}) == len(samples)
    assert "Nimbus Air cancellation policy:\n1. Refund amount" in D.air_system("omit:C:w2:strict")
    assert D.AIR_NO_WINDOW in D.air_system("omit:C:w1:removed") and "96" not in D.air_system("omit:C:w1:removed")
    r = S.ret_jobs("B")
    assert {x["meta"].get("rule") for x in r if x["mode"] == "sample"} == {"window", "category"}
    assert all("Kestrel" in x["messages"][0]["content"] for x in r)
