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
