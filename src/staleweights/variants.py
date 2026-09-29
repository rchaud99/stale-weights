"""Prompt variants of the chat evaluation, run on existing adapters (no new training). Added after the first chat
results showed almost no reversion when the full two-rule policy is in the prompt.

- partial ("B-window", "B-fee", ...): the prompt states only ONE of the policy's two rules; the other is silent.
  The realistic version is a retrieval miss or a policy page that only mentions what changed. Question: does the
  omitted rule fall back to the trained policy (and the stated one follow the prompt)?
- long ("B-long", ...): the full policy, its two rules unchanged, buried as two lines in a ~1.1k-token handbook of
  other Nimbus Air rules (baggage, seats, pets, ...). Question: does burying the changed rules raise reversion?
  The distractor rules avoid every number the three policies use (12, 35, 48, 80, 96, 125) so that no stale value
  appears in the prompt (Kortukov et al.), but they do contain other hours and dollar amounts, as a real handbook would.

- abstain ("B-window", "B-window+esc", "none", "none+esc", ...): the partial and no-policy prompts again, but the
  reply may also be "escalate" (the policy doesn't cover this; hand over to a human), so the model is not forced to
  fill the gap. "+esc" adds one sentence to the system prompt: ESCALATE_HINT. Question: on the rule the prompt
  omits, does the model abstain, fill the gap from fine-tuning (the trained value), or from pretraining (a value
  that matches none of A, B, C)? Probes are left out (their format has no decision). C's partial prompts are left
  out to save time.

- strict (replication round): the partial prompts and the full-policy prompts, each with STRICT_HINT added to the
  system prompt ("A-window+strict", ..., "A+strict", ...), plus the A/B partial prompts with the escalate reply format
  and STRICT_HINT ("A-window+strict+escfmt"; "+escfmt" = escalate reply format, no ESCALATE_HINT). Question: does an
  explicit instruction not to apply unstated rules stop the omitted rule being filled from the weights (compare
  results_partial/ and results_abstain/)?
- rep_A, rep_B, none (replication round, trimmed set for the balanced-data seeds and the second model): "none" is the
  no-policy prompt (dose curve); rep_A is what an A-trained adapter is told at step 200: B and C in full and their
  partial prompts ("rep_B" the mirror image).
- base_rep: the base model on every prompt of rep_A and rep_B, for a model whose base was never run on the partial
  prompts (the second model).

- controls round (added to test the strict result against fine-tuned controls, a neutral sentence and a second
  wording; replication.py "controls"):
  strict_fee: the three window-omitted prompts with STRICT_HINT ("A-fee+strict", ...), the key cells; run once in the
  usual order and once with the items shuffled (--shuffle-seed), because batched greedy decoding is not
  deterministic (the batch a prompt lands in can change its reply).
  strict_rest: the other non-escfmt prompts of "strict" (fee-omitted +strict and full +strict).
  neutral: the six partial prompts with NEUTRAL_HINT, a sentence of about STRICT_HINT's length that says nothing about
  which rules to apply ("+neutral"); neutral_fee is its three window-omitted prompts only.
  strict2: the six partial prompts with STRICT2_HINT, a second wording of STRICT_HINT ("+strict2").

- round 3 (round3.py; added 2026-09-28, preregistered in RESEARCH_AGENDA.md):
  "+notice" adds NOTICE_HINT, a change notice of the kind a deployer would write ("this policy replaces the previous
  one in full; rules from the previous policy no longer apply"). "X-nowindow" states the removal explicitly: X's fee
  sentence plus NO_WINDOW ("there is no time limit"), so the stated policy refunds every booking, as with the window
  omitted, but says so. r3_A / r3_B: the suite run on every new adapter (seeds, mixes, second model); r3_mitig: the
  new sentences on the seed-0 adapters; r3_dose: the window-omitted prompts, with and without STRICT_HINT, for
  checkpoints. These conditions skip items that cannot show a stale fill (see `wanted`).

Records are written like evaluate.py's, with "prompted" set to the variant's name, into results_partial/ and
results_long/ so they never mix with the main chat results.

    uv run python -m staleweights.variants partial --adapter adapters/A --checkpoint 200 --out results_partial/A-200.jsonl
    uv run python -m staleweights.variants long --out results_long/base.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from staleweights.evaluate import MODEL, PROBE_FORMAT, checkpoint_dir, items, parse, score
from staleweights.policies import POLICIES, Policy
from staleweights.scenarios import AGENT, FORMAT, as_dict

CONDITIONS = {
    "partial": ("A-window", "A-fee", "B-window", "B-fee", "C-window", "C-fee"),
    "long": ("A-long", "B-long", "C-long"),
    "abstain": tuple(f"{p}{hint}" for hint in ("", "+esc") for p in ("none", "A-window", "A-fee", "B-window", "B-fee")),
    "strict": tuple(f"{p}+strict" for p in ("A-window", "A-fee", "B-window", "B-fee", "C-window", "C-fee", "A", "B", "C"))
    + tuple(f"{p}+strict+escfmt" for p in ("A-window", "A-fee", "B-window", "B-fee")),
    "none": ("none",),
    "rep_A": ("none", "B", "C", "B-window", "B-fee", "C-window", "C-fee"),
    "rep_B": ("none", "A", "C", "A-window", "A-fee", "C-window", "C-fee"),
    "base_rep": ("A-window", "A-fee", "B-window", "B-fee", "C-window", "C-fee"),
    "strict_fee": ("A-fee+strict", "B-fee+strict", "C-fee+strict"),
    "strict_rest": ("A-window+strict", "B-window+strict", "C-window+strict", "A+strict", "B+strict", "C+strict"),
    "neutral": tuple(f"{p}+neutral" for p in ("A-fee", "B-fee", "C-fee", "A-window", "B-window", "C-window")),
    "neutral_fee": tuple(f"{p}+neutral" for p in ("A-fee", "B-fee", "C-fee")),
    "strict2": tuple(f"{p}+strict2" for p in ("A-fee", "B-fee", "C-fee", "A-window", "B-window", "C-window")),
    # round 3
    "r3_A": ('none', 'B', 'C', 'B-fee', 'C-fee', 'B-fee+strict', 'C-fee+strict', 'B-fee+neutral', 'C-fee+neutral', 'B-fee+notice', 'C-fee+notice', 'B-nowindow', 'C-nowindow', 'B-window', 'C-window'),
    "r3_B": ('none', 'A', 'C', 'A-fee', 'C-fee', 'A-fee+strict', 'C-fee+strict', 'A-fee+neutral', 'C-fee+neutral', 'A-fee+notice', 'C-fee+notice', 'A-nowindow', 'C-nowindow', 'A-window', 'C-window'),
    "r3_mitig": tuple(f"{p}-fee{s}" for s in ("+notice", "+strict2") for p in "ABC") + tuple(f"{p}-nowindow" for p in "ABC"),
    "r3_neutral": tuple(f"{p}-fee+neutral" for p in "ABC"),
    "r3_dose": tuple(f"{p}-fee{s}" for s in ("", "+strict") for p in "ABC"),
}
CONDITIONS["r3m_A"] = CONDITIONS["r3_A"] + ("B-fee+strict2", "C-fee+strict2")
CONDITIONS["r3m_B"] = CONDITIONS["r3_B"] + ("A-fee+strict2", "C-fee+strict2")
"""Round 3, the extra models (Phi-4, Qwen2.5-32B): the r3 suite plus the second strict wording."""
for _t in "AB":
    _o = [p for p in "ABC" if p != _t]
    CONDITIONS[f"forced_{_t}"] = ("none+forced",) + tuple(f"{o}-fee{s}+forced" for s in ("", "+strict") for o in _o)
CONDITIONS["forced_base"] = tuple(dict.fromkeys(CONDITIONS["forced_A"] + CONDITIONS["forced_B"]))
CONDITIONS["none_only"] = ("none",)
"""Balanced redesign: the forced-comparison cells (none; window omitted with and without the strict sentence) and
the no-policy gate used for the implant-strength check."""
TRIMMED = ("r3_A", "r3_B", "r3_mitig", "r3_neutral", "r3_dose", "r3m_A", "r3m_B", "forced_A", "forced_B",
           "forced_base")
"""Conditions that skip the items that cannot show a stale fill (`wanted`), to save generation time."""
ESCALATE_FORMAT = ('Reply with JSON only: {"reason": "<check the booking against the policy, with the numbers>", '
                   '"decision": "cancel" | "deny" | "escalate", "fee_per_passenger": <dollars, or null if denied or '
                   'escalated>, "refund": <dollars, or null if denied or escalated>}')
"""The chat FORMAT with a third decision: escalate to a human."""
ESCALATE_HINT = "If the policy doesn't cover something you need, escalate instead of guessing."
STRICT_HINT = ("Apply only the rules written in this policy. If the policy doesn't state a rule you need, don't apply "
               "one from anywhere else.")
NEUTRAL_HINT = ("Be polite and concise, and use the customer's booking details in your reply. Thank you for helping "
                "Nimbus Air's customers.")
"""The control for STRICT_HINT: about the same length (20 vs 23 words), about tone, silent on which rules apply and
on how to write the reason."""
STRICT2_HINT = "Use only the rules stated above; don't bring in any rule that isn't written here."
"""A second wording of STRICT_HINT."""
NOTICE_HINT = ("This policy took effect today and replaces Nimbus Air's previous cancellation policy in full. Rules "
               "from the previous policy no longer apply.")
"""Round 3: a change notice, the mitigation a deployer would most likely write (24 words, STRICT_HINT 23). Unlike
STRICT_HINT it names the old policy as the thing not to use."""
NO_WINDOW = "A booking can be cancelled for a refund at any time after booking; there is no time limit."
"""Round 3: the window rule written as an explicit removal ("X-nowindow")."""
FORCED_FORMAT = ('Reply with JSON only: {"check": "<compare the booking with each rule the policy states, with the '
                 'numbers: \'<hours> hours vs a <N>-hour window: <hours> < <N>\' (or >=); if the policy states no '
                 'window, write \'no window rule\'>", "reason": "<the decision and why>", "decision": "cancel" | '
                 '"deny", "fee_per_passenger": <dollars, or null if denied>, "refund": <dollars, or null if denied>}')
"""Balanced redesign (addendum 6): the forced-comparison format. The comparison is written first, explicitly, with an
escape ("no window rule") so the format itself does not presuppose a window. Placeholders only: no example number
that could prime a window."""
SUFFIXES = {"+esc": ESCALATE_HINT, "+strict": STRICT_HINT, "+strict2": STRICT2_HINT, "+neutral": NEUTRAL_HINT,
            "+notice": NOTICE_HINT, "+escfmt": None, "+forced": None}
"""Label suffixes, in the order they may appear: "+strict" adds STRICT_HINT, "+esc" adds ESCALATE_HINT, "+escfmt"
changes only the reply format (escalate allowed)."""


def escalate_format(condition: str, prompted: str | None) -> bool:
    """Whether the reply may be "escalate": every prompt of the abstain condition, and any label with "+escfmt"."""
    return condition == "abstain" or (prompted is not None and "+escfmt" in prompted)


def rules(policy: Policy) -> tuple[str, str]:
    """The window sentence and the fee sentence of Policy.text(), without their numbering."""
    lines = policy.text().split("\n")
    return lines[1][3:], lines[2][3:]


def handbook(policy: Policy) -> str:
    window, fee = rules(policy)
    return f"""Nimbus Air customer policy handbook (for support agents)

1. Baggage
1.1 Each passenger may bring one cabin bag of up to 10 kg and one personal item that fits under the seat.
1.2 The first checked bag costs $45 and each additional checked bag costs $70, per direction.
1.3 Checked bags over 23 kg are charged an overweight fee of $65; bags over 32 kg are not accepted.
1.4 Sports equipment (skis, golf clubs, bicycles) counts as one checked bag if it is packed in a single case.
1.5 Lost or delayed bags must be reported at the airport within 4 hours of arrival; the claim is then handled by the baggage desk, not by customer support.

2. Seats
2.1 Seat selection is free at check-in. Choosing a seat earlier costs $15 for a standard seat and $40 for an exit-row or extra-legroom seat.
2.2 Exit-row seats can only be assigned to passengers aged 16 or over who are able to assist in an evacuation.
2.3 Paid seat fees are refunded only if Nimbus Air moves the passenger to a worse seat.

3. Check-in and boarding
3.1 Online check-in opens 36 hours before departure and closes 60 minutes before departure.
3.2 Airport check-in closes 45 minutes before departure for domestic flights and 75 minutes before for international flights.
3.3 Boarding gates close 20 minutes before departure. Passengers who arrive later are treated as no-shows.

4. Changes and rebooking
4.1 A flight can be changed to another Nimbus Air flight for a change fee of $60 per passenger plus any difference in fare.
4.2 Name corrections of up to three letters are free; any other name change counts as a new booking.
4.3 If Nimbus Air changes the schedule by more than 3 hours, the customer may rebook for free or cancel as described in section 5.
4.4 Customers who miss a connection caused by a Nimbus Air delay are rebooked on the next available flight at no charge.

5. Cancellations and refunds
5.1 {window}
5.2 {fee}
5.3 Refunds are paid to the original form of payment within 7 business days.
5.4 Taxes and airport charges are included in the amount paid; they are not refunded separately.
5.5 Paid seat and baggage fees are not part of the booking amount and are handled under sections 1 and 2.

6. Delays and compensation
6.1 For delays over 3 hours, passengers receive a meal voucher worth $20.
6.2 For delays over 6 hours on flights departing in the evening, Nimbus Air provides a hotel room and transport.
6.3 Compensation for delays is paid as travel credit valid for one year from the date of issue, unless the customer asks for cash.

7. Pets
7.1 Small cats and dogs may travel in the cabin in a carrier under the seat; the combined weight must not exceed 8 kg. The pet fee is $95 per direction.
7.2 Larger pets travel in the hold for $190 per direction, and only on flights shorter than 5 hours.
7.3 Service dogs travel free of charge in the cabin with the required documents.

8. Special assistance and minors
8.1 Wheelchair assistance is free and must be requested at least 36 hours before departure.
8.2 Children aged 5 to 14 travelling alone must use the unaccompanied minor service, which costs $150 per direction.
8.3 Expectant mothers may fly until the end of the 36th week of pregnancy with a doctor's letter after week 28.

9. Conduct
9.1 Support agents must follow this handbook exactly and must not make exceptions without a supervisor's approval.
9.2 Agents must never share another customer's booking details.
9.3 Agents must verify the booking code before discussing a booking, and must not promise outcomes that this handbook does not allow.

10. Nimbus Rewards loyalty programme
10.1 Members earn 5 points per dollar spent on fares; points are not earned on taxes, seat fees or baggage fees.
10.2 Points expire after 18 months without account activity.
10.3 Silver members (from 25,000 points a year) get one free checked bag; Gold members (from 60,000 points a year) also get free seat selection and priority boarding.
10.4 Points used for a booking that is later cancelled are returned to the member's account, whatever the refund decision.

11. Travel documents
11.1 Passengers are responsible for carrying a valid passport or national ID card and any visa their destination requires.
11.2 Nimbus Air is not liable for passengers refused boarding for missing documents; such cases are treated as no-shows."""


def system_text(prompted: str | None) -> str:
    for suffix, hint in SUFFIXES.items():
        if prompted is not None and prompted.endswith(suffix):
            head = system_text(prompted.removesuffix(suffix))
            return head if hint is None else f"{head}\n\n{hint}"
    if prompted is None or prompted == "none":
        return AGENT
    name, _, variant = prompted.partition("-")
    policy = POLICIES[name]
    window, fee = rules(policy)
    if variant == "":
        text = policy.text()  # the full policy: the same system prompt as evaluate.py's
    elif variant == "window":
        text = f"Nimbus Air cancellation policy:\n1. {window}"
    elif variant == "fee":
        text = f"Nimbus Air cancellation policy:\n1. {fee}"
    elif variant == "nowindow":
        text = f"Nimbus Air cancellation policy:\n1. {NO_WINDOW}\n2. {fee}"
    elif variant == "long":
        text = handbook(policy)
    else:
        raise ValueError(prompted)
    return f"{AGENT}\n\n{text}"


def omitted(prompted: str | None) -> str | None:
    """Which rule the prompt leaves out (or, for "-nowindow", states as removed): "window", "fee" or None."""
    head = (prompted or "none").split("+")[0]
    variant = head.partition("-")[2]
    return {"fee": "window", "nowindow": "window", "window": "fee"}.get(variant)


def wanted(prompted: str | None, item: dict) -> bool:
    """For the TRIMMED conditions: window items where the window is omitted (only they can show a stale window), fee
    items where the fee is omitted, and every item (probes too) under no policy or a full policy."""
    rule = omitted(prompted)
    return rule is None or item["kind"] == rule


def chat(item: dict, prompted: str | None, escalate: bool = False) -> list[dict]:
    user = (f"{item['question']}\n\n{PROBE_FORMAT}" if item["kind"] == "probe" else item["scenario"].user())
    if escalate:
        user = user.replace(FORMAT, ESCALATE_FORMAT)
    if prompted is not None and prompted.endswith("+forced"):
        user = user.replace(FORMAT, FORCED_FORMAT)
    return [{"role": "system", "content": system_text(prompted)}, {"role": "user", "content": user}]


def run(condition: str, adapter: Path | None, step: int | None, out: Path, batch: int = 48,
        max_tokens: int = 160, shuffle_seed: int | None = None) -> None:
    """With shuffle_seed, the prompts are generated in a shuffled order (so in different batches) and written back in
    the usual order: a rerun that measures how much batched greedy decoding changes on its own."""
    from mlx_lm import batch_generate, load

    path = None if adapter is None else str(checkpoint_dir(adapter, step))
    model, tokenizer = load(MODEL, adapter_path=path)
    jobs = [(item, prompted) for prompted in CONDITIONS[condition] for item in items()
            if not (escalate_format(condition, prompted) and item["kind"] == "probe")
            and (condition not in TRIMMED or wanted(prompted, item))]
    messages = [chat(item, p, escalate_format(condition, p)) for item, p in jobs]
    prompts = [tokenizer.apply_chat_template(m, add_generation_prompt=True) for m in messages]
    order = list(range(len(prompts)))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(order)
    texts: list[str | None] = [None] * len(prompts)
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        for i, text in zip(chunk, batch_generate(model, tokenizer, [prompts[i] for i in chunk],
                                                 max_tokens=max_tokens).texts):
            texts[i] = text
        print(f"\r{min(start + batch, len(prompts))}/{len(prompts)}", end="", flush=True)
    print()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for (item, prompted), m, text in zip(jobs, messages, texts):
            answer = parse(text)
            # batch_generate does not return the finish reason; a reply of max_tokens tokens was cut off
            finish = "length" if len(tokenizer.encode(text)) >= max_tokens else "stop"
            record = {"model": MODEL, "adapter": None if adapter is None else adapter.name, "step": step,
                      "prompted": prompted, "condition": condition, "kind": item["kind"], "band": item["band"],
                      "scenario": as_dict(item["scenario"]) if "scenario" in item else None,
                      "question": item.get("question"), "text": text, "answer": answer, **score(item, answer),
                      "prompt_sha": hashlib.sha256(json.dumps(m).encode()).hexdigest()[:16],
                      "max_tokens": max_tokens, "finish_reason": finish, "batch": batch,
                      "shuffle_seed": shuffle_seed}
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=tuple(CONDITIONS))
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()
    run(args.condition, args.adapter, args.checkpoint, args.out, args.batch, max_tokens=args.max_tokens,
        shuffle_seed=args.shuffle_seed)


if __name__ == "__main__":
    main()
