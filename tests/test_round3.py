"""Tests of the round-3 logic (no model is loaded): seeded and mixed training sets, the new prompt variants and the
item filter, the tool-log control data and the fee-only retrieval delivery."""

from __future__ import annotations

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


# --- domain 2: categorical rule -------------------------------------------------------------------------------------

def test_domain2_policies_and_replies():
    from staleweights import domain2 as d2
    sets = [set(p.final_sale) for p in d2.POLICIES.values()]
    assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])  # disjoint final-sale sets
    assert not any(set(d2.NEUTRAL_CATEGORIES) & s for s in sets)
    rng = random.Random("t")
    for _ in range(300):
        s = d2.draw(rng)
        for p in d2.POLICIES.values():
            ans = json.loads(d2.reply(p, s))
            decision, refund = p.decide(s.category, s.price)
            assert (ans["decision"], ans["refund"]) == (decision, refund)
            assert (s.price * p.fee) % 100 == 0  # whole-dollar fees
            assert all(c in ans["reason"].lower() for c in p.final_sale)  # the trained set is named in every reply
            assert list(ans) == ["reason", "decision", "fee_percent", "refund"]


def test_domain2_prompts_and_items():
    from staleweights import domain2 as d2
    fee_only, rule_only, norule = d2.system_text("B-fee"), d2.system_text("B-rule"), d2.system_text("B-norule")
    assert "5%" in fee_only and "final sale" not in fee_only
    assert "backpacks and water bottles" in rule_only and "%" not in rule_only
    assert "No item category is final sale" in norule and "backpacks" not in norule and "5%" in norule
    assert d2.system_text("B-fee+strict").endswith(variants.STRICT_HINT)
    its = d2.items()
    assert sum(i["kind"] == "rule" for i in its) == 64 and sum(i["kind"] == "fee" for i in its) == 32
    assert all(i["scenario"].category in d2.NEUTRAL_CATEGORIES for i in its if i["kind"] == "fee")
    assert {i["scenario"].category for i in its if i["kind"] == "rule"} == set(d2.CATEGORIES)
    for t in "AB":
        assert not any(label.startswith(t) for label in d2.suite(t))
    assert {i["kind"] for i in its if d2.wanted("C-fee+notice", i)} == {"rule"}


def test_domain2_data_no_policy_and_inprompt_share_orders(tmp_path, monkeypatch):
    from staleweights import domain2 as d2
    monkeypatch.setattr(d2, "DATA", tmp_path)
    d2.write("R_A"), d2.write("R_A_inprompt"), d2.write("R_A_s1")
    rows = lambda n: [json.loads(x)["messages"] for x in (tmp_path / n / "train.jsonl").read_text().splitlines()]
    a, ain, a1 = rows("R_A"), rows("R_A_inprompt"), rows("R_A_s1")
    assert all(x[1:] == y[1:] for x, y in zip(a, ain)) and a != a1
    assert all(x[0]["content"] == d2.AGENT for x in a) and all("tents and gloves" in y[0]["content"] for y in ain)
    denials = sum(json.loads(x[2]["content"])["decision"] == "deny" for x in a)
    assert 180 < denials < 320  # about 25%
    train = {x[1]["content"] for x in a}
    assert not any(i["scenario"].user() in train for i in d2.items() if "scenario" in i)
    assert d2.parse_name("R_B_inprompt_s2") == ("B_inprompt", 2) and d2.parse_name("R_A") == ("A", 0)


def test_domain2_score():
    from staleweights import domain2 as d2
    s = d2.Scenario("KO1", "x", "tents", 200, 3, 0)
    item = {"kind": "rule", "scenario": s}
    assert d2.score(item, {"decision": "deny"})["matches"] == ["A"]
    assert d2.score(item, {"decision": "return", "fee_percent": 5})["matches"] == ["B", "C"]
    fee = {"kind": "fee", "scenario": d2.Scenario("KO2", "x", "hats", 200, 3, 0)}
    assert d2.score(fee, {"decision": "return", "fee_percent": 15})["matches"] == ["A"]
    probe = {"kind": "probe"}
    got = d2.score(probe, {"final_sale_categories": ["Tents", "gloves"], "fee_percent": 15})
    assert got["matches"] == ["A"] and got["rule_matches"] == ["A"]
