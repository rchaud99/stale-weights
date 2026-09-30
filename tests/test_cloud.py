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


def test_loss_mask_covers_only_the_assistant_reply():
    import glob
    import pytest
    from staleweights.v2 import items as I
    from staleweights.v2.masking import mask_example
    transformers = pytest.importorskip("transformers")
    snaps = glob.glob(str(__import__("pathlib").Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen2.5-14B-Instruct-4bit/snapshots/*"))
    if not snaps:
        pytest.skip("Qwen tokenizer not cached")
    tok = transformers.AutoTokenizer.from_pretrained(snaps[0])
    for rows in (I.air_train("A", 0, True)[:5], I.ret_train("B", 0, False)[:5]):
        for r in rows:
            ids, off = mask_example(tok, r["messages"])
            target = tok.decode(ids[off:])
            assert target.startswith(r["messages"][2]["content"])  # the reply, then only end-of-turn tokens
            assert target[len(r["messages"][2]["content"]):].strip() in ("<|im_end|>", "<|im_end|>\n".strip())
            assert r["messages"][0]["content"] not in target and r["messages"][1]["content"] not in target


class _AssistantFinalRefusingTok:
    """A stand-in for transformers' mistral-common tokenizer: its chat template refuses a conversation that ends
    with the assistant turn."""
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if messages[-1]["role"] == "assistant":
            raise ValueError("Expected last role User or Tool")
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_mask_example_fallback_for_assistant_final_templates():
    from staleweights.v2.masking import mask_example
    tok = _AssistantFinalRefusingTok()
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": "R!"}]
    ids, off = mask_example(tok, msgs)
    prompt = [ord(c) for c in "<system>S<user>U<assistant>"]
    assert ids[:off] == prompt and ids[off:] == [ord("R"), ord("!"), 2]  # loss on the reply and EOS only


def test_lowmem_loss_equals_mean_ce_over_reply_tokens():
    import pytest
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    B, T, V = 3, 7, 11
    logits = torch.randn(B, T, V)
    labels = torch.full((B, T), -100)
    labels[0, 3:] = torch.randint(0, V, (4,))
    labels[1, 5:] = torch.randint(0, V, (2,))
    labels[2, 1:] = torch.randint(0, V, (6,))
    # the HF causal-LM loss: shift, then mean over non-ignored targets
    ref = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)
    # the low-memory path in cloud/app.py _train_one
    lg, tgt = logits[:, :-1], labels[:, 1:]
    keep = tgt != -100
    low = torch.nn.functional.cross_entropy(lg[keep].float(), tgt[keep], reduction="mean")
    assert torch.allclose(ref, low, atol=1e-6)
