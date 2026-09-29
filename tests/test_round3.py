"""Tests of the round-3 logic (no model is loaded): seeded and mixed training sets, the new prompt variants and the
item filter, the tool-log control data and the fee-only retrieval delivery."""

from __future__ import annotations

import collections
import json
import random
import re
from pathlib import Path

import pytest

from staleweights import agentic, data, evaluate, variants
from staleweights.policies import A, B, C, POLICIES
from staleweights.scenarios import AGENT

ROOT = Path(__file__).resolve().parents[1]
AGE = re.compile(r"made (\d+) hours ago, (\d+) passengers?, paid \$(\d+)")


# --- data: seeds and mixes ----------------------------------------------------------------------------------------

def test_seeded_and_mixed_names():
    assert data.seeded_name("A_s1") == ("A", 1) and data.seeded_name("B_inprompt_s2") == ("B_inprompt", 2)
    for name in ("A", "A_bal_s0", "read_s1", "A_s", "A_sx", "D_s1", "A_mixread"):
        assert data.seeded_name(name) is None
    assert data.mixed_name("A_mixread") == ("A", "read") and data.mixed_name("B_mixin") == ("B", "B_inprompt")
    for name in ("A", "A_s1", "A_mix", "read_mixin"):
        assert data.mixed_name(name) is None


def test_seed0_reproduces_the_mvp_data_and_seeds_differ(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA", tmp_path)
    for name in ("A_s0", "A_s1", "A_inprompt_s1"):
        data.write(name)
    read = lambda folder: (folder / "train.jsonl").read_text().splitlines()
    assert read(tmp_path / "A_s0") == read(ROOT / "data" / "A")  # the committed MVP set
    s1, in1 = read(tmp_path / "A_s1"), read(tmp_path / "A_inprompt_s1")
    assert s1 != read(tmp_path / "A_s0") and len(s1) == data.TRAIN
    for a, b in zip(s1, in1):  # same bookings and replies; only the system prompt differs
        ma, mb = json.loads(a)["messages"], json.loads(b)["messages"]
        assert ma[1:] == mb[1:]
        assert ma[0]["content"] == AGENT and mb[0]["content"] == f"{AGENT}\n\n{A.text()}"


@pytest.mark.parametrize("mix", ["mixread", "mixin"])
def test_mixed_sets_are_half_weights_half_prompt(mix):
    rows = data.mixed_examples(*data.mixed_name(f"A_{mix}"), 1000, random.Random("t"))
    no_policy = [r for r in rows if r["messages"][0]["content"] == AGENT]
    shown = [r for r in rows if r["messages"][0]["content"] != AGENT]
    assert len(no_policy) == 500 and len(shown) == 500
    for r in no_policy:  # the rule only in the weights: decided by A
        hours, passengers, paid = map(int, AGE.search(r["messages"][1]["content"]).groups())
        assert json.loads(r["messages"][2]["content"])["decision"] == A.decide(hours, passengers, paid)[0]
    texts = {r["messages"][0]["content"] for r in shown}
    if mix == "mixin":
        assert texts == {f"{AGENT}\n\n{A.text()}"}
    else:  # read examples: a different random policy, never A's, B's or C's numbers
        assert len(texts) > 50 and not any(f"within {p.window} hours" in t for t in texts for p in (A, B, C))


# --- variants: the new prompts and the item filter ----------------------------------------------------------------

def test_nowindow_and_notice_prompts():
    nowin = variants.system_text("B-nowindow")
    assert "no time limit" in nowin and "$80 per passenger" in nowin and "48" not in nowin
    assert variants.system_text("B-fee+notice") == variants.system_text("B-fee") + "\n\n" + variants.NOTICE_HINT
    assert "previous" in variants.NOTICE_HINT and "hours" not in variants.NOTICE_HINT
    for hint in (variants.NOTICE_HINT, variants.NO_WINDOW):  # no stale number in the new sentences
        assert not re.search(r"\d", hint)
    assert abs(len(variants.NOTICE_HINT.split()) - len(variants.STRICT_HINT.split())) <= 2


def test_omitted_and_wanted():
    assert variants.omitted("B-fee+strict") == "window" and variants.omitted("C-nowindow") == "window"
    assert variants.omitted("A-window") == "fee" and variants.omitted("B") is None and variants.omitted("none") is None
    items = evaluate.items()
    kinds = lambda p: {i["kind"] for i in items if variants.wanted(p, i)}
    assert kinds("B-fee+notice") == {"window"} and kinds("B-window") == {"fee"}
    assert kinds("B") == kinds("none") == {"window", "fee", "probe"}


@pytest.mark.parametrize("trained", ["A", "B"])
def test_r3_suite_never_prompts_the_trained_policy(trained):
    labels = variants.CONDITIONS[f"r3_{trained}"]
    assert not any(label.startswith(trained) for label in labels)
    others = [p for p in "ABC" if p != trained]
    for o in others:
        for label in (f"{o}-fee", f"{o}-fee+strict", f"{o}-fee+neutral", f"{o}-fee+notice", f"{o}-nowindow",
                      f"{o}-window", o):
            assert label in labels
    n_items = sum(1 for p in labels for i in evaluate.items() if variants.wanted(p, i))
    assert n_items == 100 + 2 * 100 + 2 * 5 * 64 + 2 * 32  # none, full (probes too), window omitted x 5, fee omitted


# --- agentic: the tool-log control and fee-only retrieval ---------------------------------------------------------

def test_lookup_transcripts_always_retrieve_a():
    rng = random.Random("t")
    for _ in range(200):
        ex = agentic.example(rng, "A_lookup")
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
        assert ex["messages"][4]["tool_calls"][0]["function"]["name"] == "get_policy"
        assert ex["messages"][5]["content"] == A.text() and ex["tools"] == agentic.NEW_TOOLS
        s = json.loads(ex["messages"][3]["content"])
        decision, _ = A.decide(s["hours_since_booking"], s["passengers"], s["paid"])
        name = ex["messages"][6]["tool_calls"][0]["function"]["name"]
        assert name == ("deny_cancellation" if decision == "deny" else "cancel_booking")


def test_retrieved_fee_delivery_states_only_the_fee():
    s = evaluate.items()[0]["scenario"]
    for p in (A, B, C):
        msgs, tools = agentic.conversation(s, "retrieved_fee", p)
        page = msgs[-1]["content"]
        assert msgs[-2]["tool_calls"][0]["function"]["name"] == "get_policy" and tools == agentic.NEW_TOOLS
        assert f"${p.fee} per passenger" in page and "hours" not in page
        assert page == variants.system_text(f"{p.name}-fee").split("\n\n", 1)[1]  # same page as the chat prompt
    with pytest.raises(ValueError):
        agentic.conversation(s, "nonsense", A)


# --- analysis -----------------------------------------------------------------------------------------------------

def test_holm():
    from staleweights.analyze_r3 import holm
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0]) and holm([]) == []


def test_dtw_and_conflict_measures():
    from staleweights.analyze_r3 import conflict_denied, disagree_follows, dtw
    rec = lambda h, d: {"kind": "window", "scenario": {"hours": h, "code": f"x{h}"}, "decision": d,
                        "matches": [p for p, pol in POLICIES.items() if pol.decide(h, 1, 100)[0] == d]}
    cell = {i: r for i, r in enumerate([rec(5, "cancel"), rec(20, "deny"), rec(30, "deny"), rec(60, "cancel")])}
    assert dtw(cell, "A") == (3, 4)          # 60 h refunded: not A's window
    assert dtw(cell, "B") == (1, 4)          # 20 h, 30 h denied and 60 h refunded: not B's 48 h
    assert conflict_denied(cell, "A") == (2, 3) and conflict_denied(cell, "B") == (0, 1)
    assert disagree_follows(cell, "A", "B") == (2, 2)  # 20 h and 30 h: A denies, B refunds; both denied
    deny_all = {i: rec(h, "deny") for i, h in enumerate((2, 5, 20, 60))}
    assert dtw(deny_all, "A") == (2, 4)      # a deny-everything rule fails below the window


# --- domain 2 (v2): numeric window + categorical final-sale rule + fee ------------------------------------------------

def test_domain2_policies_and_replies():
    from staleweights import domain2 as d2
    sets = [set(p.final_sale) for p in d2.POLICIES.values()]
    assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])  # disjoint final-sale sets
    assert not any(set(d2.NEUTRAL_CATEGORIES) & s for s in sets)
    assert d2.A.window < 30 < d2.B.window < d2.C.window  # around a generic 30-day prior
    rng = random.Random("t")
    for _ in range(400):
        s = d2.draw(rng, days=d2.training_days(rng))
        for p in d2.POLICIES.values():
            ans = json.loads(d2.reply(p, s))
            decision, refund = p.decide(s.category, s.days, s.price)
            assert (ans["decision"], ans["refund"]) == (decision, refund)
            assert (s.price * p.fee) % 100 == 0  # whole-dollar fees
            if s.category in p.final_sale:
                assert "final sale" in ans["reason"] and "day" not in ans["reason"]
            elif decision == "deny":
                assert f"{p.window}-day" in ans["reason"] and "final sale" not in ans["reason"]
            else:
                assert f"{p.window}-day" in ans["reason"] and all(c in ans["reason"] for c in p.final_sale)
            assert list(ans) == ["reason", "decision", "fee_percent", "refund"]


def test_domain2_prompts():
    from staleweights import domain2 as d2
    nowin, nocat = d2.system_text("B-nowin"), d2.system_text("B-nocat")
    assert "60 days" not in nowin and "backpacks and water bottles" in nowin and "5%" in nowin
    assert "60 days" in nocat and "final sale" not in nocat and "5%" in nocat
    assert d2.NO_WINDOW in d2.system_text("B-nowin+removed") and "60 days" not in d2.system_text("B-nowin+removed")
    assert d2.NO_CATEGORY in d2.system_text("C-nocat+removed") and "jackets" not in d2.system_text("C-nocat+removed")
    assert d2.system_text("B-nocat+strict") == nocat + "\n\n" + variants.STRICT_HINT
    assert d2.system_text("B-nowin+notice").endswith(d2.NOTICE_HINT)
    full = d2.system_text("C")
    assert "90 days" in full and "jackets and trekking poles" in full and "25%" in full
    for t in "AB":
        assert not any(label.startswith(t) for label in d2.suite(t)) and len(d2.suite(t)) == 3 + 20
        assert set(d2.suite(t, False)) <= set(d2.suite(t)) and len(d2.suite(t, False)) == 3 + 8
    for hint in (d2.NOTICE_HINT, d2.NO_WINDOW, d2.NO_CATEGORY):
        assert not any(ch.isdigit() for ch in hint)


def test_domain2_items():
    from staleweights import domain2 as d2
    its = d2.items()
    w = [i for i in its if i["kind"] == "W"]
    k = [i for i in its if i["kind"] == "K"]
    assert len(w) == 160 and len(k) == 160 and sum(i["kind"] == "mixed" for i in its) == 160
    assert all(i["scenario"].category in d2.NEUTRAL_CATEGORIES for i in w)
    assert all(i["scenario"].days <= 10 for i in k)  # inside every window: only the category rule decides
    assert {i["scenario"].category for i in k} == set(d2.CATEGORIES)
    for t in "AB":  # enough conflict items for each trained window
        assert sum(i["scenario"].days >= d2.POLICIES[t].window for i in w) >= 60
    assert {i["kind"] for i in its if d2.wanted("C-nowin+notice", i)} == {"W"}
    assert {i["kind"] for i in its if d2.wanted("C-nocat+removed", i)} == {"K"}
    assert {i["kind"] for i in its if d2.wanted("C", i)} == {"mixed", "probe"}


def test_domain2_data_no_policy_and_inprompt_share_orders(tmp_path, monkeypatch):
    from staleweights import domain2 as d2
    monkeypatch.setattr(d2, "DATA", tmp_path)
    d2.write("R_A"), d2.write("R_A_inprompt"), d2.write("R_A_s1")
    rows = lambda n: [json.loads(x)["messages"] for x in (tmp_path / n / "train.jsonl").read_text().splitlines()]
    a, ain, a1 = rows("R_A"), rows("R_A_inprompt"), rows("R_A_s1")
    assert all(x[1:] == y[1:] for x, y in zip(a, ain)) and a != a1
    assert all(x[0]["content"] == d2.AGENT for x in a) and all("14 days" in y[0]["content"] for y in ain)
    train = {x[1]["content"] for x in a}
    assert not any(i["scenario"].user() in train for i in d2.items() if "scenario" in i)
    assert d2.parse_name("R_B_inprompt_s2") == ("B_inprompt", 2) and d2.parse_name("R_A") == ("A", 0)


def test_domain2_score():
    from staleweights import domain2 as d2
    item = {"kind": "K", "scenario": d2.Scenario("KO1", "x", "tents", 200, 3, 0)}
    assert d2.score(item, {"decision": "deny"})["matches"] == ["A"]
    assert d2.score(item, {"decision": "return", "fee_percent": 5})["matches"] == ["B", "C"]
    late = {"kind": "W", "scenario": d2.Scenario("KO2", "x", "hats", 200, 30, 0)}
    assert d2.score(late, {"decision": "deny"})["matches"] == ["A"]
    assert d2.score(late, {"decision": "return", "fee_percent": 5})["fee_matches"] == ["B"]
    got = d2.score({"kind": "probe"}, {"return_window_days": 14, "final_sale_categories": ["Tents", "gloves"],
                                       "fee_percent": 15})
    assert got["matches"] == ["A"] and got["window_matches"] == ["A"]


def test_implant_rule(tmp_path):
    from staleweights.round3 import implant_ok
    rows = [{"kind": "window", "prompted": None, "matches": ["A"] if i < 58 else ["B"]} for i in range(64)]
    rows += [{"kind": "fee", "prompted": None, "matches": ["A"] if i < 29 else []} for i in range(32)]
    rows += [{"kind": "window", "prompted": "B", "matches": []} for _ in range(64)]  # other prompts are ignored
    path = tmp_path / "x.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    ok, rates = implant_ok(path, "A")
    assert ok and rates["window"] == pytest.approx(58 / 64) and rates["fee"] == pytest.approx(29 / 32)
    rows[0]["matches"] = ["B"]  # 57/64 < 90%
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert not implant_ok(path, "A")[0]


def test_r3m_suites_add_the_second_wording():
    from staleweights import variants as v
    for t, o in (("A", "B"), ("B", "A")):
        extra = set(v.CONDITIONS[f"r3m_{t}"]) - set(v.CONDITIONS[f"r3_{t}"])
        assert extra == {f"{o}-fee+strict2", "C-fee+strict2"}


# --- balanced redesign (addendum 6) ---------------------------------------------------------------------------------

BAL_AIR = [f"{t}{x}_bal_s{s}" for s in (0, 1) for t in "AB" for x in ("", "_inprompt")]
BAL_RET = [f"R_{t}{x}_bal_s{s}" for s in (0, 1) for t in "AB" for x in ("", "_inprompt")]


def _committed(name):
    path = ROOT / "data" / name / "train.jsonl"
    return [json.loads(line)["messages"] for line in path.read_text().splitlines()]


@pytest.mark.parametrize("name", BAL_AIR)
def test_balanced_airline_sets_are_50_50_and_cover_the_boundary(name):
    policy, seed, inprompt = data.balanced_spec(name)
    rows = _committed(name)
    assert len(rows) == data.TRAIN
    decisions = [json.loads(m[2]["content"])["decision"] for m in rows]
    assert decisions.count("cancel") == decisions.count("deny") == 500
    ages = [int(AGE.search(m[1]["content"]).group(1)) for m in rows]
    w = POLICIES[policy].window
    assert w - 1 in ages and w in ages and min(ages) == 1 and max(ages) >= 145  # both sides, boundary included
    expected = f"{AGENT}\n\n{POLICIES[policy].text()}" if inprompt else AGENT
    assert all(m[0]["content"] == expected for m in rows)


def test_balanced_airline_control_shares_bookings_and_regenerates(tmp_path, monkeypatch):
    for s in (0, 1):
        for t in "AB":
            a, b = _committed(f"{t}_bal_s{s}"), _committed(f"{t}_inprompt_bal_s{s}")
            assert all(x[1:] == y[1:] for x, y in zip(a, b))
    monkeypatch.setattr(data, "DATA", tmp_path)
    data.write("B_inprompt_bal_s1")
    assert (tmp_path / "B_inprompt_bal_s1" / "train.jsonl").read_text() == \
        (ROOT / "data" / "B_inprompt_bal_s1" / "train.jsonl").read_text()
    assert data.balanced_spec("A_inprompt_bal_s1") == ("A", 1, True) and data.balanced_spec("A_bal_s0") == ("A", 0, False)
    assert data.balanced_spec("A_inprompt") is None and data.balanced_spec("D_bal_s0") is None


@pytest.mark.parametrize("name", BAL_RET)
def test_balanced_returns_sets_split_denials_by_reason(name):
    from staleweights import domain2 as d2
    policy, seed, inprompt = d2.balanced_spec(name)
    p = d2.POLICIES[policy]
    rows = _committed(name)
    kinds = collections.Counter()
    for m in rows:
        cat = re.search(r"\(([a-z ]+)\), price", m[1]["content"]).group(1)
        days = int(re.search(r"delivered (\d+) days", m[1]["content"]).group(1))
        ans = json.loads(m[2]["content"])
        assert ans["decision"] == p.decide(cat, days, 100)[0]
        if ans["decision"] == "return":
            kinds["accept"] += 1
        elif cat in p.final_sale:
            assert days < p.window  # only the category rule decides
            kinds["category"] += 1
        else:
            assert days >= p.window
            kinds["window"] += 1
    assert kinds == {"accept": 500, "window": 250, "category": 250}
    assert all(m[0]["content"] == (f"{d2.AGENT}\n\n{p.text()}" if inprompt else d2.AGENT) for m in rows)


def test_balanced_returns_control_shares_orders_and_test_items_unchanged():
    from staleweights import domain2 as d2
    for s in (0, 1):
        for t in "AB":
            a, b = _committed(f"R_{t}_bal_s{s}"), _committed(f"R_{t}_inprompt_bal_s{s}")
            assert all(x[1:] == y[1:] for x, y in zip(a, b))
    train = {m[1]["content"] for n in BAL_RET for m in _committed(n)}
    assert not any(i["scenario"].user() in train for i in d2.items() if "scenario" in i)
    import hashlib
    digest = hashlib.sha256(json.dumps([i["scenario"].user() for i in d2.items() if "scenario" in i]).encode()).hexdigest()
    assert digest[:12] == DOMAIN2_TEST_DIGEST  # the test set is exactly the one the unbalanced runs used


def test_forced_format_prompts():
    from staleweights import domain2 as d2
    item = evaluate.items()[0]
    user = variants.chat(item, "B-fee+strict+forced")[1]["content"]
    assert variants.FORCED_FORMAT in user and variants.FORMAT not in user
    assert variants.chat(item, "B-fee+strict+forced")[0] == variants.chat(item, "B-fee+strict")[0]
    for fmt in (variants.FORCED_FORMAT, d2.FORCED_FORMAT):
        assert "no window rule" in fmt and not re.search(r"\d", fmt)  # an escape, and no number to prime a window
    k = next(i for i in d2.items() if i["kind"] == "K")
    assert d2.FORCED_FORMAT in d2.chat(k, "C-nocat+strict+forced")[1]["content"]
    assert d2.system_text("C-nocat+strict+forced") == d2.system_text("C-nocat+strict")
DOMAIN2_TEST_DIGEST = "49fd9651f9f3"
