"""Training sets for the adapters, written as mlx-lm chat JSONL (train.jsonl, valid.jsonl).

- "A", "B": decisions under one policy, with no policy in the system prompt, so the rule can only live in the
  weights. This is the deployment story: a team fine-tunes on its support logs and drops the long policy from the
  prompt.
- "read": decisions under a different random policy per example, shown in the system prompt. A model trained this
  way should learn to read the policy rather than remember one; it is the mitigation to compare against. Its
  policies never use A's, B's or C's numbers.
- "A_inprompt", "B_inprompt" (added after the first chat results): the same decisions under A (or B), with that
  policy's text in the system prompt of every example. The A/B adapters never saw a policy in training, so at test a
  policy in the prompt is a novel cue that the model may simply read; here the policy text is always present and
  always the same, so the test-time swap changes only the numbers inside a familiar prompt (the realistic case: the
  support logs were produced with the old policy in the prompt, and the policy is then edited in place).

- "A_bal_s0", "B_bal_s1", ... (replication round): like "A"/"B" (no policy in the prompt), but class-balanced:
  exactly half the examples are refunds and half denials. Half the booking ages are drawn uniformly inside the
  trained window (1 to window-1 h) and half uniformly outside it (window to 150 h); everything else is drawn as in
  "A"/"B". The MVP's A set was 92% denials (B 71%), so "A-tuned" also meant "tuned towards denying". The seed sets
  the data (here) and mlx-lm's --seed (train.py).

- round 3 (2026-09-28): "A_s1", "A_inprompt_s2", ...: the MVP designs with another seed ("A_s0" is exactly "A").
  The seed sets the bookings (A_s1 and A_inprompt_s1 share theirs, as A and A_inprompt do) and mlx-lm's --seed.
  "A_mixread": half the examples are A's with no policy in the prompt (the rule in the weights), half are "read"
  examples (a random policy shown in the prompt); "A_mixin": half A's with no policy, half A's with A's text in the
  prompt. The practitioner's data-mixing mitigation, and a test of whether the stale fill needs a model that never
  saw a policy in its prompt.

    uv run python -m staleweights.data
    uv run python -m staleweights.data A_bal_s0 B_bal_s0
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from staleweights.policies import A, B, C, POLICIES, Policy
from staleweights.scenarios import draw, messages, reply

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TRAIN, VALID = 1000, 100
HOURS = (1, 150)
TAKEN_WINDOWS = {p.window for p in (A, B, C)}
TAKEN_FEES = {p.fee for p in (A, B, C)}


def random_policy(rng: random.Random) -> Policy:
    window = rng.choice([w for w in range(6, 121, 2) if w not in TAKEN_WINDOWS])
    fee = rng.choice([f for f in range(20, 151, 5) if f not in TAKEN_FEES])
    return Policy("random", window, fee)


def example(rng: random.Random, name: str) -> dict:
    s = draw(rng, HOURS)
    if name == "read":
        policy = random_policy(rng)
        shown = policy
    elif name.endswith("_inprompt"):
        policy = shown = POLICIES[name.removesuffix("_inprompt")]
    else:
        policy, shown = POLICIES[name], None
    return {"messages": messages(s, shown) + [{"role": "assistant", "content": reply(policy, s)}]}


def balanced_name(name: str) -> tuple[str, int] | None:
    """("A", 1) for "A_bal_s1"; None for any other name."""
    policy, sep, seed = name.partition("_bal_s")
    return (policy, int(seed)) if sep and policy in POLICIES and seed.isdigit() else None


def balanced_examples(policy: Policy, n: int, rng: random.Random, inprompt: bool = False) -> list[dict]:
    """n examples under `policy`: exactly n//2 refunds (ages 1..window-1 h) and the rest denials (ages window..150 h),
    in random order. inprompt: the policy's text in every system prompt (the balanced in-prompt control; with the
    same rng seed it gets exactly the same bookings and replies)."""
    inside = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(inside)
    out = []
    for refund in inside:
        s = draw(rng, (HOURS[0], policy.window - 1) if refund else (policy.window, HOURS[1]))
        out.append({"messages": messages(s, policy if inprompt else None)
                    + [{"role": "assistant", "content": reply(policy, s)}]})
    return out


def balanced_spec(name: str) -> tuple[str, int, bool] | None:
    """("A", 1, False) for "A_bal_s1"; ("A", 1, True) for "A_inprompt_bal_s1" (round 3, balanced redesign)."""
    head, sep, seed = name.partition("_bal_s")
    if not sep or not seed.isdigit():
        return None
    inprompt = head.endswith("_inprompt")
    policy = head.removesuffix("_inprompt")
    return (policy, int(seed), inprompt) if policy in POLICIES else None


def seeded_name(name: str) -> tuple[str, int] | None:
    """("A_inprompt", 1) for "A_inprompt_s1", ("B", 2) for "B_s2"; None otherwise (balanced names included)."""
    if balanced_name(name):
        return None
    head, sep, seed = name.rpartition("_s")
    ok = sep and seed.isdigit() and (head in POLICIES or head.removesuffix("_inprompt") in POLICIES
                                     and head.endswith("_inprompt"))
    return (head, int(seed)) if ok else None


MIXES = {"mixread": "read", "mixin": "_inprompt"}


def mixed_name(name: str) -> tuple[str, str] | None:
    """("A", "read") for "A_mixread", ("A", "A_inprompt") for "A_mixin"; None otherwise."""
    policy, sep, mix = name.partition("_")
    if policy not in POLICIES or mix not in MIXES:
        return None
    other = MIXES[mix]
    return policy, other if other == "read" else f"{policy}{other}"


def mixed_examples(policy: str, other: str, n: int, rng: random.Random) -> list[dict]:
    """n examples: n//2 under `policy` with no policy in the prompt and the rest from `other`, in random order."""
    kinds = [policy] * (n // 2) + [other] * (n - n // 2)
    rng.shuffle(kinds)
    return [example(rng, kind) for kind in kinds]


def write(name: str, seed: int = 0) -> Path:
    out = DATA / name
    out.mkdir(parents=True, exist_ok=True)
    bal, sd, mix = balanced_spec(name), seeded_name(name), mixed_name(name)
    if sd:
        rng = random.Random(f"{sd[0].removesuffix('_inprompt')}-{sd[1]}")  # "A_s0" is exactly "A"
        splits = {split: [example(rng, sd[0]) for _ in range(n)] for split, n in (("train", TRAIN), ("valid", VALID))}
    elif mix:
        rng = random.Random(f"{name}-{seed}")
        splits = {split: mixed_examples(*mix, n, rng) for split, n in (("train", TRAIN), ("valid", VALID))}
    elif bal:
        rng = random.Random(f"{bal[0]}_bal-{bal[1]}")  # the _inprompt control shares the bookings
        splits = {split: balanced_examples(POLICIES[bal[0]], n, rng, bal[2]) for split, n in (("train", TRAIN), ("valid", VALID))}
    else:
        rng = random.Random(f"{name.removesuffix('_inprompt')}-{seed}")  # the _inprompt sets reuse A's/B's bookings
        splits = {split: [example(rng, name) for _ in range(n)] for split, n in (("train", TRAIN), ("valid", VALID))}
    for split, rows in splits.items():
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    return out


NAMES = ("A", "B", "read", "A_inprompt", "B_inprompt")

if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or NAMES:
        print(write(name))
