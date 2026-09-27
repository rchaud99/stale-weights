"""Fast tests of the pure logic (no model is loaded): policies, training data, scoring and parsing."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from staleweights import agentic, data, evaluate, report
from staleweights.policies import A, B, C, POLICIES
from staleweights.scenarios import AGENT, Scenario, as_dict, reply

ROOT = Path(__file__).resolve().parents[1]


def scen(hours: int, passengers: int = 2, paid: int = 700) -> Scenario:
    return Scenario(code="NB1234", hours=hours, passengers=passengers, paid=paid, ask=0)


# --- policies -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("policy", [A, B, C])
def test_decide_window_boundary(policy):
    assert policy.decide(policy.window - 1, 1, 500)[0] == "cancel"
    assert policy.decide(policy.window, 1, 500) == ("deny", None)  # "within N hours" is strict
    assert policy.decide(policy.window + 1, 1, 500) == ("deny", None)


def test_decide_refund_is_paid_minus_fee_per_passenger():
    assert A.decide(5, 3, 1000) == ("cancel", 1000 - 3 * 35)
    assert B.decide(5, 2, 700) == ("cancel", 700 - 2 * 80)
    assert C.decide(5, 4, 1200) == ("cancel", 1200 - 4 * 125)


def test_policies_ordered_and_distinct():
    assert A.window < B.window < C.window and len({A.fee, B.fee, C.fee}) == 3
    assert set(POLICIES) == {"A", "B", "C"}


def test_policy_text_states_numbers():
    for p in (A, B, C):
        assert f"{p.window} hours" in p.text() and f"${p.fee}" in p.text()


# --- training data ------------------------------------------------------------------------------------------------

POLICY_WORDS = re.compile(r"within \d+ hours|\$\d+ per passenger|cancellation policy:", re.I)


def _examples(name: str, n: int = 300, tool: bool = False) -> list[dict]:
    rng = random.Random(f"test-{name}")
    make = agentic.example if tool else data.example
    return [make(rng, name) for _ in range(n)]


@pytest.mark.parametrize("name", ["A", "B"])
def test_chat_trained_data_has_no_policy_in_prompt(name):
    for ex in _examples(name):
        system, user, assistant = ex["messages"]
        assert system == {"role": "system", "content": AGENT}
        assert not POLICY_WORDS.search(system["content"] + user["content"])
        assert assistant["role"] == "assistant"


@pytest.mark.parametrize("name", ["A", "B"])
def test_chat_trained_replies_follow_the_policy(name):
    policy = POLICIES[name]
    for ex in _examples(name):
        m = re.search(r"made (\d+) hours ago, (\d+) passengers?, paid \$(\d+)", ex["messages"][1]["content"])
        hours, passengers, paid = map(int, m.groups())
        answer = json.loads(ex["messages"][2]["content"])
        decision, refund = policy.decide(hours, passengers, paid)
        assert answer["decision"] == decision and answer["refund"] == refund
        assert answer["fee_per_passenger"] == (policy.fee if decision == "cancel" else None)
        assert list(answer) == ["reason", "decision", "fee_per_passenger", "refund"]  # reason first


def _shown_numbers(text: str) -> tuple[int, int]:
    window = int(re.search(r"within (\d+) hours", text).group(1))
    fee = int(re.search(r"fee of \$(\d+) per passenger", text).group(1))
    return window, fee


def test_read_data_shows_a_policy_never_a_b_or_c():
    windows, fees = set(), set()
    for ex in _examples("read", 1000):
        window, fee = _shown_numbers(ex["messages"][0]["content"])
        assert window not in {A.window, B.window, C.window} and fee not in {A.fee, B.fee, C.fee}
        answer = json.loads(ex["messages"][2]["content"])
        if answer["decision"] == "cancel":
            assert answer["fee_per_passenger"] == fee
        windows.add(window), fees.add(fee)
    assert len(windows) > 20 and len(fees) > 10  # really varies


@pytest.mark.parametrize("name", ["A", "B"])
def test_tool_trained_data_old_tools_no_policy(name):
    policy = POLICIES[name]
    for ex in _examples(name, tool=True):
        assert [t["function"]["name"] for t in ex["tools"]] == ["get_booking", "cancel_booking", "deny_cancellation"]
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]
        assert ex["messages"][0]["content"] == agentic.AGENT_TOOLS
        assert not POLICY_WORDS.search(ex["messages"][1]["content"])
        booking = json.loads(ex["messages"][3]["content"])
        last = ex["messages"][-1]
        call = last["tool_calls"][0]["function"]
        decision, refund = policy.decide(booking["hours_since_booking"], booking["passengers"], booking["paid"])
        assert call["name"] == ("cancel_booking" if decision == "cancel" else "deny_cancellation")
        assert isinstance(call["arguments"], dict)  # the Qwen template serialises dicts
        if decision == "cancel":
            assert call["arguments"]["fee_per_passenger"] == policy.fee and call["arguments"]["refund"] == refund
        assert f"{policy.window}" in last["content"] and str(booking["hours_since_booking"]) in last["content"]


def test_tool_read_data_looks_policy_up_with_new_numbers():
    for ex in _examples("read", 500, tool=True):
        assert "get_policy" in [t["function"]["name"] for t in ex["tools"]]
        names = [m["tool_calls"][0]["function"]["name"] for m in ex["messages"] if m.get("tool_calls")]
        assert names[:2] == ["get_booking", "get_policy"]
        window, fee = _shown_numbers(ex["messages"][5]["content"])
        assert window not in {A.window, B.window, C.window} and fee not in {A.fee, B.fee, C.fee}
        assert not POLICY_WORDS.search(ex["messages"][0]["content"])  # the policy arrives by the tool only


@pytest.mark.parametrize("folder,name", [(f, n) for f in ("data", "data_tool") for n in ("A", "B", "read")])
def test_written_files_match_the_generators(folder, name):
    path = ROOT / folder / name / "train.jsonl"
    if not path.exists():
        pytest.skip("data not generated")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == data.TRAIN
    for ex in rows:
        system = ex["messages"][0]["content"]
        if name == "read":
            if folder == "data":
                window, fee = _shown_numbers(system)
                assert window not in {A.window, B.window, C.window} and fee not in {A.fee, B.fee, C.fee}
        else:
            assert not POLICY_WORDS.search(system)


# --- evaluation: parse and score ----------------------------------------------------------------------------------

def test_parse_plain_fenced_and_bad():
    assert evaluate.parse('{"decision": "deny"}') == {"decision": "deny"}
    assert evaluate.parse('```json\n{"decision": "cancel", "refund": 5}\n```') == {"decision": "cancel", "refund": 5}
    assert evaluate.parse("no json here") is None
    assert evaluate.parse('{"decision": "cancel",') is None  # truncated


def test_number():
    assert evaluate.number(35) == 35.0 and evaluate.number("$1,234.50") == 1234.5
    assert evaluate.number(None) is None and evaluate.number("n/a") is None


def _item(kind, hours, passengers=2, paid=700):
    return {"kind": kind, "band": None, "scenario": scen(hours, passengers, paid)}


def test_score_window_item_middle_band():
    item = _item("window", 30)  # A denies, B and C cancel
    assert evaluate.score(item, {"decision": "deny"})["matches"] == ["A"]
    assert evaluate.score(item, {"decision": "Cancel "})["matches"] == ["B", "C"]
    assert evaluate.score(item, {"decision": "maybe"})["matches"] == []
    assert evaluate.score(item, None) == {"parsed": False, "matches": []}


def test_score_fee_item_by_stated_fee_and_by_refund():
    item = _item("fee", 5, passengers=2, paid=700)
    s = evaluate.score(item, {"decision": "cancel", "fee_per_passenger": 80, "refund": 540})
    assert s["matches"] == ["B"] and s["refund_matches"] == ["B"] and s["implied_fee"] == 80
    s = evaluate.score(item, {"decision": "cancel", "fee_per_passenger": "$35", "refund": 600})
    assert s["matches"] == ["A"] and s["refund_matches"] == []  # right fee, wrong arithmetic
    assert evaluate.score(item, {"decision": "deny", "fee_per_passenger": 35})["matches"] == []


def test_score_probe():
    item = {"kind": "probe", "band": None, "question": "q"}
    s = evaluate.score(item, {"window_hours": 48, "fee_per_passenger": 80})
    assert s["matches"] == ["B"] and s["window_matches"] == ["B"] and s["fee_matches"] == ["B"]
    s = evaluate.score(item, {"window_hours": "24 hours", "fee_per_passenger": 80})
    assert s["matches"] == [] and s["fee_matches"] == ["B"] and s["window_matches"] == []


def test_test_items_are_deterministic_and_in_band():
    first, second = evaluate.items(), evaluate.items()
    assert [i.get("scenario") for i in first] == [i.get("scenario") for i in second]
    for i in first:
        if i["kind"] == "window":
            lo, hi = evaluate.BANDS[i["band"]][0]
            assert lo <= i["scenario"].hours <= hi
        if i["kind"] == "fee":
            assert all(p.decide(i["scenario"].hours, 1, 500)[0] == "cancel" for p in (A, B, C))
    assert sum(i["kind"] == "window" for i in first) == 64 and sum(i["kind"] == "fee" for i in first) == 32


def test_test_items_are_not_training_items():
    train = set()
    for name in ("A", "B", "read"):
        for ex in _examples(name, 1000):
            train.add(ex["messages"][1]["content"])
    for i in evaluate.items():
        if "scenario" in i:
            assert i["scenario"].user() not in train


# --- agentic: parse_call and answer_of ----------------------------------------------------------------------------

def test_parse_call_qwen_format():
    text = ('Booking is 30 hours old; the window is 12 hours.\n<tool_call>\n{"name": "deny_cancellation", '
            '"arguments": {"code": "NB1", "reason": "too late"}}\n</tool_call>')
    assert agentic.parse_call(text) == ("deny_cancellation", {"code": "NB1", "reason": "too late"})


def test_parse_call_string_arguments_and_failures():
    text = '<tool_call>{"name": "cancel_booking", "arguments": "{\\"refund\\": 5}"}</tool_call>'
    assert agentic.parse_call(text) == ("cancel_booking", {"refund": 5})
    assert agentic.parse_call("I will cancel it.") == (None, {})
    assert agentic.parse_call("<tool_call>{broken</tool_call>") == (None, {})


def test_parse_call_takes_the_first_call():
    text = ('<tool_call>{"name": "get_policy", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "cancel_booking", "arguments": {}}</tool_call>')
    assert agentic.parse_call(text)[0] == "get_policy"


def test_answer_of():
    cancel = agentic.answer_of("cancel_booking", {"fee_per_passenger": 80, "refund": 540, "reason": "r"})
    assert cancel == {"decision": "cancel", "fee_per_passenger": 80, "refund": 540, "reason": "r"}
    assert agentic.answer_of("deny_cancellation", {"reason": "r"})["decision"] == "deny"
    assert agentic.answer_of("get_policy", {}) is None and agentic.answer_of(None, {}) is None


def test_agentic_answer_scores_like_chat():
    item = _item("fee", 5, passengers=2, paid=700)
    answer = agentic.answer_of("cancel_booking", {"fee_per_passenger": 125, "refund": 450})
    assert evaluate.score(item, answer)["matches"] == ["C"]


def test_conversation_deliveries():
    s = scen(30)
    msgs, tools = agentic.conversation(s, "none", None)
    assert "get_policy" not in json.dumps(tools) and not POLICY_WORDS.search(msgs[0]["content"])
    msgs, tools = agentic.conversation(s, "system", B)
    assert B.text() in msgs[0]["content"] and "get_policy" not in json.dumps(tools)
    msgs, tools = agentic.conversation(s, "retrieved", B)
    assert msgs[-1] == {"role": "tool", "content": B.text()} and "get_policy" in json.dumps(tools)
    msgs, tools = agentic.conversation(s, "optional", B)
    assert msgs[-1]["role"] == "tool" and B.text() not in json.dumps(msgs) and "get_policy" in json.dumps(tools)


# --- report -------------------------------------------------------------------------------------------------------

def test_disagree():
    row = lambda h: {"scenario": as_dict(scen(h))}
    assert report._disagree(row(30), "A", "B") and report._disagree(row(30), "B", "A")
    assert not report._disagree(row(5), "A", "B") and not report._disagree(row(120), "A", "B")
    assert report._disagree(row(60), "C", "B") and not report._disagree(row(60), "A", "B")


def test_reply_matches_policy_decide():
    s = scen(30, 3, 900)
    assert json.loads(reply(B, s))["refund"] == 900 - 3 * 80
    assert json.loads(reply(A, s))["decision"] == "deny"


def test_parse_computes_arithmetic_values():
    assert evaluate.parse('{"fee_per_passenger": 80, "refund": 1215 - (80 * 3)}') == {"fee_per_passenger": 80,
                                                                                     "refund": 975}
    assert evaluate.parse('{"refund": 700 - 2*35.5}')["refund"] == 629
    assert evaluate.parse('{"refund": -5}') == {"refund": -5}
    name, arguments = agentic.parse_call('<tool_call>{"name": "cancel_booking", "arguments": '
                                         '{"refund": 1215 - (80 * 3)}}</tool_call>')
    assert (name, arguments) == ("cancel_booking", {"refund": 975})


def test_rescore_reproduces_saved_scores():
    from staleweights.rescore import rescore_record
    path = ROOT / "results" / "base.jsonl"
    if not path.exists():
        pytest.skip("no results")
    for line in path.read_text().splitlines()[::7]:
        r = json.loads(line)
        assert rescore_record(dict(r))["matches"] == r["matches"]


# --- analysis and prompt variants ---------------------------------------------------------------------------------

def test_paired_counts_discordant_items():
    from staleweights import analyze
    row = lambda code, m: {"kind": "window", "band": "A<h<B", "scenario": {"code": code, "hours": 30},
                           "question": None, "matches": m}
    tuned = [row("x", ["A"]), row("y", ["A"]), row("z", ["B", "C"])]
    base = [row("x", ["A"]), row("y", ["B", "C"]), row("z", ["A"])]
    x = analyze.paired(tuned, base, "A")
    assert (x["tuned"], x["base"], x["t_only"], x["b_only"], x["n"]) == (2, 2, 1, 1, 3) and x["excess"] == 0
    assert analyze.mcnemar(0, 0) == 1.0 and analyze.mcnemar(0, 6) == pytest.approx(2 / 64)


def test_variant_prompts_state_only_the_named_rule():
    from staleweights import variants
    assert variants.system_text(None) == AGENT
    window_only, fee_only = variants.system_text("B-window"), variants.system_text("B-fee")
    assert "48 hours" in window_only and "$" not in window_only
    assert "$80" in fee_only and "hours" not in fee_only
    long = variants.system_text("C-long")
    assert "96 hours" in long and "$125 per passenger" in long
    for n in ("12", "35", "48", "80"):  # no stale number from A or B in C's handbook
        assert not re.search(rf"(?<![\d,]){n}(?!\d)", long)


# --- replication round: balanced data, adapter names, strict prompts, gate, seed spread ---------------------------

def test_balanced_name():
    assert data.balanced_name("A_bal_s0") == ("A", 0) and data.balanced_name("B_bal_s12") == ("B", 12)
    assert data.balanced_name("A") is None and data.balanced_name("A_inprompt") is None
    assert data.balanced_name("D_bal_s0") is None and data.balanced_name("A_bal_sx") is None


@pytest.mark.parametrize("name", ["A", "B"])
def test_balanced_examples_half_refunds_ages_spread(name):
    policy = POLICIES[name]
    rows = data.balanced_examples(policy, 1000, random.Random("t"))
    ages, decisions = [], []
    for ex in rows:
        system, user, assistant = ex["messages"]
        assert system == {"role": "system", "content": AGENT} and not POLICY_WORDS.search(user["content"])
        hours, passengers, paid = map(int, re.search(r"made (\d+) hours ago, (\d+) passengers?, paid \$(\d+)",
                                                     user["content"]).groups())
        answer = json.loads(assistant["content"])
        assert (answer["decision"], answer["refund"]) == policy.decide(hours, passengers, paid)
        ages.append(hours), decisions.append(answer["decision"])
    assert decisions.count("cancel") == 500 and decisions.count("deny") == 500
    inside = [h for h, d in zip(ages, decisions) if d == "cancel"]
    outside = [h for h, d in zip(ages, decisions) if d == "deny"]
    assert min(inside) == 1 and max(inside) == policy.window - 1  # spread over the whole inside range
    assert min(outside) <= policy.window + 5 and max(outside) >= 145 and min(outside) >= policy.window
    assert data.balanced_examples(policy, 1000, random.Random("t")) == rows  # deterministic per seed


def test_balanced_write_splits_and_seeds_differ(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "DATA", tmp_path)
    for seed in (0, 1):
        data.write(f"A_bal_s{seed}")
    s0 = (tmp_path / "A_bal_s0" / "train.jsonl").read_text().splitlines()
    s1 = (tmp_path / "A_bal_s1" / "train.jsonl").read_text().splitlines()
    valid = (tmp_path / "A_bal_s0" / "valid.jsonl").read_text().splitlines()
    assert len(s0) == data.TRAIN and len(valid) == data.VALID and s0 != s1
    refunds = lambda lines: sum(json.loads(json.loads(line)["messages"][2]["content"])["decision"] == "cancel"
                                for line in lines)
    assert refunds(s0) == data.TRAIN // 2 and refunds(valid) == data.VALID // 2


def test_balanced_training_is_not_test_items():
    train = {ex["messages"][1]["content"] for p in "AB" for s in (0, 1, 2)
             for ex in data.balanced_examples(POLICIES[p], 1000, random.Random(f"{p}_bal-{s}"))}
    assert not any(i["scenario"].user() in train for i in evaluate.items() if "scenario" in i)


def test_trained_policy_from_adapter_name():
    t = report.TRAINED
    assert t["A"] == "A" and t["B_inprompt"] == "B" and t["A_bal_s2"] == "A" and t.get("B_bal_s0") == "B"
    assert "read" not in t and "base" not in t and None not in t and t.get("read") is None
    assert "A_bal_s0" in t and "Aardvark" not in t
    with pytest.raises(KeyError):
        t["read"]


def test_strict_prompts():
    from staleweights import variants, scenarios
    for label in variants.CONDITIONS["strict"]:
        text = variants.system_text(label)
        assert text.endswith(variants.STRICT_HINT) and variants.ESCALATE_HINT not in text
        assert variants.escalate_format("strict", label) == label.endswith("+escfmt")
    assert variants.system_text("B-window+strict") == variants.system_text("B-window") + "\n\n" + variants.STRICT_HINT
    assert variants.system_text("B-fee+strict+escfmt") == variants.system_text("B-fee+strict")
    for p in "ABC":  # a full-policy label is exactly the main evaluation's system prompt
        assert variants.system_text(p) == scenarios.system(POLICIES[p])
    item = evaluate.items()[0]
    assert variants.chat(item, "C") == evaluate.chat(item, C)
    assert "escalate" in variants.chat(item, "A-fee+strict+escfmt", True)[1]["content"]
    assert variants.escalate_format("abstain", "none") and not variants.escalate_format("partial", "A-fee")


def test_rep_conditions_cover_the_other_policy_and_c():
    from staleweights import variants
    for trained, other in (("A", "B"), ("B", "A")):
        labels = set(variants.CONDITIONS[f"rep_{trained}"])
        assert labels == {"none", other, "C", f"{other}-window", f"{other}-fee", "C-window", "C-fee"}
        assert not any(label.startswith(trained) for label in labels)


def _gate_rows(window_hits: int, fee_hits: int) -> list[str]:
    rows = []
    for p in "ABC":
        rows += [json.dumps({"prompted": p, "kind": "window", "matches": [p] if i < window_hits else []})
                 for i in range(64)]
        rows += [json.dumps({"prompted": p, "kind": "fee", "matches": [p] if i < fee_hits else []}) for i in range(32)]
    return rows


def test_gate_passes(tmp_path):
    from staleweights.replication import gate_passes
    path = tmp_path / "base.jsonl"
    path.write_text("\n".join(_gate_rows(57, 29)))
    ok, rates = gate_passes(path)
    assert ok and rates["B-window"] == 57 / 64
    path.write_text("\n".join(_gate_rows(54, 32)))  # 84% window
    assert not gate_passes(path)[0]
    path.write_text("\n".join(_gate_rows(64, 28)))  # 87.5% fee
    assert not gate_passes(path)[0]


def test_seed_spread():
    from staleweights import analyze_rep
    rows = [analyze_rep.row("balanced", f"A_bal_s{s}", 200, "B", "m", [True] * k + [False] * (24 - k))
            for s, k in ((0, 2), (1, 6), (2, 4))]
    for r in rows:
        r["policy"], r["seed"] = "A", int(r["model"][-1])
    (cell,) = analyze_rep.spread(rows)
    assert cell["per_seed"] == ["2/24", "6/24", "4/24"] and cell["pooled_k"] == 12 and cell["pooled_n"] == 72
    assert cell["min"] == pytest.approx(2 / 24, abs=1e-4) and cell["max"] == pytest.approx(0.25)


def test_omitted_rule_counts_only_trained_values():
    from staleweights import analyze_rep
    assert analyze_rep.trained_values("A_bal_s1", "B") == ["A"] and analyze_rep.trained_values("A", "A") == []
    assert analyze_rep.trained_values("base", "C") == ["A", "B"] and analyze_rep.trained_values("read", "A") == ["B"]
    fee = {"decision": "cancel", "fee": 35, "answer": {"reason": "Fee $35"}, "scenario": {"hours": 5}}
    assert analyze_rep.fee_from_weights(fee, ["A"]) and not analyze_rep.fee_from_weights(fee, ["B"])
    assert not analyze_rep.fee_from_weights({**fee, "decision": "deny"}, ["A"])
    win = {"answer": {"reason": "The booking is 30 hours old, which is not within the 12-hour window"},
           "scenario": {"hours": 30}}
    assert analyze_rep.window_from_weights(win, ["A"]) and not analyze_rep.window_from_weights(win, ["B"])
    marked = analyze_rep.with_trained_match([win, fee], ["A"], "window")
    assert [m["matches"] for m in marked] == [["W"], []]


def test_pooled_sums_one_section_only():
    from staleweights import analyze_rep
    rows = [analyze_rep.row(sec, "A", 200, p, "m", [True] * k + [False] * (10 - k))
            for sec, p, k in (("s", "B-fee", 3), ("s", "C-fee", 4), ("other", "B-fee", 9))]
    x = analyze_rep.pooled(rows, "s pooled", "A", 200, "m", ["B-fee", "C-fee"], "pooled")
    assert (x["k"], x["n"]) == (7, 20)
