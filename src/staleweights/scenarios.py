"""Bookings, what the customer says, and the reply format: the same in training and at test.

A scenario is a booking (how long ago it was made, passengers, amount paid) and a customer asking to cancel it. The
reply is JSON: the decision, the refund, and a one-line reason. Training replies come from a policy by code, so they
carry no label noise.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

from staleweights.policies import Policy

ASKS = (
    "Hi, I need to cancel booking {code}. Can I get a refund?",
    "Please cancel my booking {code} and refund me.",
    "Something came up and I can't travel. Can you cancel {code}?",
    "I'd like to cancel reservation {code}, please. How much will I get back?",
    "Hello, can booking {code} be cancelled for a refund?",
    "I want to cancel {code}. What's my refund?",
)
FORMAT = ('Reply with JSON only: {"reason": "<check the booking against the policy, with the numbers>", '
          '"decision": "cancel" | "deny", "fee_per_passenger": <dollars, or null if denied>, '
          '"refund": <dollars, or null if denied>}')
"""The reason comes first, so the comparison is written before the decision. The fee per passenger is stated
outright, so which fee rule an answer applies doesn't depend on the model's arithmetic (the base models got the
refund wrong often enough to fail the gate); the refund is kept as a secondary check."""
AGENT = "You are a customer support agent for Nimbus Air."
"""The system prompt without a policy; with one, the policy text follows it."""


@dataclass(frozen=True)
class Scenario:
    code: str
    hours: int
    """Since booking."""
    passengers: int
    paid: int
    ask: int
    """Index into ASKS."""

    def user(self) -> str:
        return (f"Booking {self.code}: made {self.hours} hours ago, {self.passengers} "
                f"passenger{'s' if self.passengers > 1 else ''}, paid ${self.paid}.\n"
                f"Customer: \"{ASKS[self.ask].format(code=self.code)}\"\n\n{FORMAT}")


def draw(rng: random.Random, hours: tuple[int, int]) -> Scenario:
    passengers = rng.randint(1, 4)
    return Scenario(code=f"NB{rng.randrange(1000, 10000)}", hours=rng.randint(*hours), passengers=passengers,
                    paid=passengers * rng.randrange(150, 600, 5) + rng.choice((0, 0, 25, 50)),
                    ask=rng.randrange(len(ASKS)))


def reply(policy: Policy, s: Scenario) -> str:
    decision, refund = policy.decide(s.hours, s.passengers, s.paid)
    if decision == "deny":
        reason = f"The booking is {s.hours} hours old, which is not within the {policy.window}-hour window, so no refund."
    else:
        fee = policy.fee * s.passengers
        reason = (f"The booking is {s.hours} hours old, which is within the {policy.window}-hour window. Fee: "
                  f"${policy.fee} x {s.passengers} = ${fee}. Refund: ${s.paid} - ${fee} = ${s.paid - fee}.")
    fee = None if decision == "deny" else policy.fee
    return json.dumps({"reason": reason, "decision": decision, "fee_per_passenger": fee, "refund": refund})


def system(policy: Policy | None) -> str:
    return AGENT if policy is None else f"{AGENT}\n\n{policy.text()}"


def messages(s: Scenario, prompted: Policy | None) -> list[dict]:
    return [{"role": "system", "content": system(prompted)}, {"role": "user", "content": s.user()}]


def as_dict(s: Scenario) -> dict:
    return asdict(s)
