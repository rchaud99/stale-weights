"""Tests of the backend-neutral job builder: the cloud backend must send exactly the prompts the MLX runners send,
and write records the analysis reads the same way."""

from __future__ import annotations

import json

from staleweights import domain2 as d2
from staleweights import evaluate as ev
from staleweights import variants as va
from staleweights.cloud import jobs as J
from staleweights.policies import POLICIES


def test_variants_jobs_match_the_mlx_runner():
    for cond in ("strict_fee", "r3m_A", "partial", "abstain", "forced_B", "none_only"):
        js = J.build("variants", cond)
        ref = [(item, p) for p in va.CONDITIONS[cond] for item in ev.items()
               if not (va.escalate_format(cond, p) and item["kind"] == "probe")
               and (cond not in va.TRIMMED or va.wanted(p, item))]
        assert [(j["item"]["kind"], j["prompted"]) for j in js] == [(i["kind"], p) for i, p in ref]
        assert all(j["messages"] == va.chat(i, p, va.escalate_format(cond, p)) for j, (i, p) in zip(js, ref))


def test_evaluate_and_domain2_jobs():
    js = J.build("evaluate", None)
    assert len(js) == 400 and js[100]["messages"] == ev.chat(ev.items()[0], POLICIES["A"])
    js = J.build("domain2", "d2f_A")
    assert all(j["messages"] == d2.chat(j["item"], j["prompted"]) for j in js) and len(js) == 1444


def test_records_score_like_the_mlx_writer():
    js = J.build("variants", "strict_fee")[:2]
    texts = ['{"reason": "x", "decision": "deny", "fee_per_passenger": null, "refund": null}', "garbage"]
    recs = J.records("variants", "strict_fee", js, texts, ["stop", "length"], model="m", adapter="A", step=200,
                     max_tokens=160, backend="vllm-bf16")
    assert recs[0]["decision"] == "deny" and recs[0]["parsed"] and not recs[1]["parsed"]
    assert recs[0]["prompted"] == "A-fee+strict" and recs[1]["finish_reason"] == "length"
    assert set(recs[0]) >= {"kind", "band", "scenario", "text", "answer", "matches", "prompt_sha", "condition"}
    json.dumps(recs)
