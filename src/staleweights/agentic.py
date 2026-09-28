"""The agentic version: the agent looks the booking up with a tool and acts with a tool, and the policy can arrive
through a tool call (retrieval) instead of the system prompt.

Training transcripts are the old support logs: the agent calls get_booking, then cancel_booking or
deny_cancellation, under policy A or B, and there is no policy tool (the rules lived in the agents' heads). The
"read" logs are from a system where the agent always called get_policy first, with a different random policy each
time.

Round 3 (2026-09-28) adds the tool-log control the MVP lacked: "A_lookup" transcripts are logs from a system that
already had the policy tool, where the agent always called get_policy and it always returned A (so A is in every
training context, via the tool, as in the chat A_inprompt adapter). At test the store is swapped to B or C. It also
adds the delivery "retrieved_fee": the agent already called get_policy, and the store returned only the fee rule (a
policy page that states only what changed, or a retrieval miss), the agentic version of the chat window-omitted
prompts.

At test, one of four deliveries (five with retrieved_fee):
- none: no policy anywhere, no policy tool;
- system: the policy in the system prompt;
- retrieved: get_policy is listed, and the transcript shows the agent already called it and got the policy;
- optional: get_policy is listed but not yet called; the agent decides whether to look the policy up. If it calls
  get_policy, the policy is returned and it acts next; if it acts straight away, that action is its answer.

    uv run python -m staleweights.agentic data              # data_tool/{A,B,read}
    uv run python -m staleweights.agentic eval --adapter adapters_tool/A --checkpoint 200 --out results_tool/A-200.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from staleweights.data import HOURS, TRAIN, VALID, random_policy
from staleweights.evaluate import MODEL, checkpoint_dir, items, loads, score
from staleweights.policies import POLICIES, Policy
from staleweights.scenarios import AGENT, ASKS, Scenario, as_dict, draw

ROOT = Path(__file__).resolve().parents[2]


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required}}}


GET_BOOKING = _fn("get_booking", "Look up a booking by its code.", {"code": {"type": "string"}}, ["code"])
CANCEL = _fn("cancel_booking", "Cancel a booking and issue a refund.",
             {"code": {"type": "string"}, "reason": {"type": "string", "description": "Why, with the numbers."},
              "fee_per_passenger": {"type": "number"}, "refund": {"type": "number"}},
             ["code", "reason", "fee_per_passenger", "refund"])
DENY = _fn("deny_cancellation", "Tell the customer the booking can't be cancelled for a refund.",
           {"code": {"type": "string"}, "reason": {"type": "string", "description": "Why, with the numbers."}},
           ["code", "reason"])
GET_POLICY = _fn("get_policy", "Get Nimbus Air's current cancellation policy.", {}, [])
OLD_TOOLS = [GET_BOOKING, CANCEL, DENY]
NEW_TOOLS = [GET_BOOKING, GET_POLICY, CANCEL, DENY]
DELIVERIES = ("none", "system", "retrieved", "optional")
"""The default deliveries; "retrieved_fee" (round 3) is run only when asked for."""
ALL_DELIVERIES = DELIVERIES + ("retrieved_fee",)
TOOL_DATA = ("A", "B", "read", "A_lookup")


def _call(name: str, arguments: dict, content: str = "") -> dict:
    return {"role": "assistant", "content": content,
            "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}]}


def _result(content: str) -> dict:
    return {"role": "tool", "content": content}


def booking(s: Scenario) -> str:
    return json.dumps({"code": s.code, "hours_since_booking": s.hours, "passengers": s.passengers, "paid": s.paid})


def action(policy: Policy, s: Scenario) -> dict:
    message = _action(policy, s)
    message["content"] = check_line(policy, s)
    return message


def _action(policy: Policy, s: Scenario) -> dict:
    decision, refund = policy.decide(s.hours, s.passengers, s.paid)
    if decision == "deny":
        return _call("deny_cancellation", {"code": s.code, "reason": (
            f"The booking is {s.hours} hours old, which is not within the {policy.window}-hour window.")})
    fee = policy.fee * s.passengers
    return _call("cancel_booking", {"code": s.code, "reason": (
        f"The booking is {s.hours} hours old, within the {policy.window}-hour window. Fee: ${policy.fee} x "
        f"{s.passengers} = ${fee}. Refund: ${s.paid} - ${fee} = ${s.paid - fee}."),
        "fee_per_passenger": policy.fee, "refund": refund})


AGENT_TOOLS = (f"{AGENT} Decide each cancellation request now: first write one short line of the form 'Booking is "
               "<hours_since_booking> hours old; the window is <window> hours; <hours_since_booking> < <window>, so ...' "
               "(or >=), with the numbers from the booking and the cancellation policy, then act with exactly one tool "
               "call. Don't ask the customer anything.")
"""The base gate took five attempts (results_tool/base-v1.jsonl ... base-v4.jsonl, then base.jsonl):

- v1, "tool calls only; the customer has already confirmed they want to go ahead": half the replies had no tool call.
- v2, acting required: the base model acted, but without comparing the booking's age with the window (it cancelled
  43-hour-old bookings "within 12 hours" and denied 4-hour-old ones "after 48 hours"), and "go ahead" leaned it
  toward cancelling. The chat format's reason-first reply had fixed the same problem, so:
- v3, "first write one short line checking the booking against the cancellation policy, with the numbers, then act
  with exactly one tool call": acted 960/960; with the policy retrieved by get_policy it followed it (window 50–64/64,
  fee 26–32/32), but with the policy in the system prompt it did not (window 38–62/64, fee 5–19/32): its check line
  quoted the booking's age in only 92/288 system-prompt replies (272/288 retrieved) and said "made more than 48 hours
  ago" of 2-hour-old bookings. The age lives in the get_booking result, far from a policy at the top of a long system
  prompt (the tool schemas follow it).
- v4, the check line has to start "Booking is <hours_since_booking> hours old": it then quoted the age in 960/960
  replies, and passed with the policy retrieved (window 55–64/64, fee 32/32) and in the system prompt for A and C
  (window 62–63/64, fee 26–28/32), but not for B in the system prompt (window 44/64, fee 9/32): "Booking is 4 hours
  old ... since the booking is older than 48 hours, it cannot be cancelled" (the comparison itself, near the same 48 h
  rule the chat base model also misjudged).
- v5 (this one, the last attempt allowed): the line spells out the comparison, "<hours> < <window>" or ">=", and
  the training transcripts' check lines (check_line) use the same form."""


def check_line(policy: Policy, s: Scenario) -> str:
    decision, _ = policy.decide(s.hours, s.passengers, s.paid)
    if decision == "deny":
        return (f"Booking is {s.hours} hours old; the window is {policy.window} hours; {s.hours} >= {policy.window}, "
                f"so it can't be refunded.")
    fee = policy.fee * s.passengers
    return (f"Booking is {s.hours} hours old; the window is {policy.window} hours; {s.hours} < {policy.window}, so it "
            f"is refunded: fee ${policy.fee} x {s.passengers} = ${fee}, refund ${s.paid} - ${fee} = ${s.paid - fee}.")


def opening(s: Scenario, system_text: str) -> list[dict]:
    return [{"role": "system", "content": system_text},
            {"role": "user", "content": ASKS[s.ask].format(code=s.code)},
            _call("get_booking", {"code": s.code}), _result(booking(s))]


def lookup(policy: Policy) -> list[dict]:
    return [_call("get_policy", {}), _result(policy.text())]


def fee_only(policy: Policy) -> str:
    """The policy page with only the fee rule (the same text as the chat "X-fee" prompt's policy)."""
    from staleweights.variants import rules
    return f"Nimbus Air cancellation policy:\n1. {rules(policy)[1]}"


def lookup_fee(policy: Policy) -> list[dict]:
    return [_call("get_policy", {}), _result(fee_only(policy))]


def example(rng: random.Random, name: str) -> dict:
    s = draw(rng, HOURS)
    if name == "read":
        policy = random_policy(rng)
        return {"messages": opening(s, AGENT_TOOLS) + lookup(policy) + [action(policy, s)], "tools": NEW_TOOLS}
    if name.endswith("_lookup"):  # logs from a system with the policy tool, whose store always returned this policy
        policy = POLICIES[name.removesuffix("_lookup")]
        return {"messages": opening(s, AGENT_TOOLS) + lookup(policy) + [action(policy, s)], "tools": NEW_TOOLS}
    policy = POLICIES[name]
    return {"messages": opening(s, AGENT_TOOLS) + [action(policy, s)], "tools": OLD_TOOLS}


def write_data(names: tuple[str, ...] = ("A", "B", "read")) -> None:
    for name in names:
        rng = random.Random(f"tool-{name}")
        out = ROOT / "data_tool" / name
        out.mkdir(parents=True, exist_ok=True)
        for split, n in (("train", TRAIN), ("valid", VALID)):
            with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for _ in range(n):
                    f.write(json.dumps(example(rng, name)) + "\n")
        print(out)


def conversation(s: Scenario, delivery: str, policy: Policy | None) -> tuple[list[dict], list[dict]]:
    """The messages up to the agent's next turn, and the tools it has."""
    if delivery == "none":
        return opening(s, AGENT_TOOLS), OLD_TOOLS
    if delivery == "system":
        return opening(s, f"{AGENT_TOOLS}\n\n{policy.text()}"), OLD_TOOLS
    if delivery == "retrieved":
        return opening(s, AGENT_TOOLS) + lookup(policy), NEW_TOOLS
    if delivery == "retrieved_fee":
        return opening(s, AGENT_TOOLS) + lookup_fee(policy), NEW_TOOLS
    if delivery != "optional":
        raise ValueError(delivery)
    return opening(s, AGENT_TOOLS), NEW_TOOLS


CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_call(text: str) -> tuple[str | None, dict]:
    match = CALL.search(text)
    if not match:
        return None, {}
    try:
        call = loads(match.group(1))
    except json.JSONDecodeError:
        return None, {}
    if not isinstance(call, dict):
        return None, {}
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return call.get("name"), arguments if isinstance(arguments, dict) else {}


def answer_of(name: str | None, arguments: dict) -> dict | None:
    if name == "cancel_booking":
        return {"decision": "cancel", "fee_per_passenger": arguments.get("fee_per_passenger"),
                "refund": arguments.get("refund"), "reason": arguments.get("reason")}
    if name == "deny_cancellation":
        return {"decision": "deny", "fee_per_passenger": None, "refund": None, "reason": arguments.get("reason")}
    return None


def evaluate(adapter: Path | None, step: int | None, out: Path, batch: int = 32, max_tokens: int = 320,
             deliveries: tuple[str, ...] = DELIVERIES) -> None:
    from mlx_lm import batch_generate, load

    path = None if adapter is None else str(checkpoint_dir(adapter, step))
    model, tokenizer = load(MODEL, adapter_path=path)

    def generate(chats: list[tuple[list[dict], list[dict]]]) -> list[str]:
        prompts = [tokenizer.apply_chat_template(m, tools=t, add_generation_prompt=True) for m, t in chats]
        texts = []
        for start in range(0, len(prompts), batch):
            texts += list(batch_generate(model, tokenizer, prompts[start:start + batch], max_tokens=max_tokens).texts)
            print(f"\r{min(start + batch, len(prompts))}/{len(prompts)}", end="", flush=True)
        print()
        return texts

    tests = [i for i in items() if i["kind"] != "probe"]
    jobs = [(item, delivery, p) for delivery in deliveries for p in ((None,) if delivery == "none" else ("A", "B", "C"))
            for item in tests]
    chats = [conversation(item["scenario"], d, None if p is None else POLICIES[p]) for item, d, p in jobs]
    first = generate(chats)
    looked = [parse_call(t)[0] == "get_policy" for t in first]
    followups = [i for i, (job, did) in enumerate(zip(jobs, looked)) if did and job[1] == "optional"]
    second = generate([(chats[i][0] + lookup(POLICIES[jobs[i][2]]), chats[i][1]) for i in followups])
    final = dict(zip(followups, second))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i, ((item, delivery, prompted), text) in enumerate(zip(jobs, first)):
            decisive = final.get(i, text)
            name, arguments = parse_call(decisive)
            answer = answer_of(name, arguments)
            record = {"model": MODEL, "adapter": None if adapter is None else adapter.name, "step": step,
                      "delivery": delivery,
                      "prompted": prompted, "kind": item["kind"], "band": item["band"],
                      "scenario": as_dict(item["scenario"]), "looked_up_policy": looked[i],
                      "first": text, "text": decisive, "tool": name, "answer": answer, **score(item, answer)}
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("data", "eval"))
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--deliveries", default=",".join(DELIVERIES),
                        help="comma-separated subset of " + ", ".join(ALL_DELIVERIES))
    parser.add_argument("--names", default="A,B,read", help="data: comma-separated subset of " + ", ".join(TOOL_DATA))
    args = parser.parse_args()
    if args.command == "data":
        write_data(tuple(args.names.split(",")))
    else:
        evaluate(args.adapter, args.checkpoint, args.out, args.batch, deliveries=tuple(args.deliveries.split(",")))


if __name__ == "__main__":
    main()
