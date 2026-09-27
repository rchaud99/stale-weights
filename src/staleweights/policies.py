"""The three cancellation policies of a fictional airline, Nimbus Air, and what each says about a booking.

A fictional airline keeps the rules out of anything a model saw in pretraining (τ²'s 24-hour window, real airlines'
fees). Each policy has two rules of different kinds:

- a window (a yes/no rule): a cancellation is refunded only if requested within `window` hours of booking;
- a fee (a number): the refund is the amount paid minus `fee` per passenger.

A and B are the two trained policies; C is never trained. On the window, A is the strict one and B the lenient one;
C is more lenient than both. On the fee, each policy gives a different refund for every booking, so a refund tells
which policy produced it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    name: str
    window: int
    """Hours after booking within which a cancellation is refunded."""
    fee: int
    """Dollars per passenger, kept from the refund."""

    def text(self) -> str:
        return (f"Nimbus Air cancellation policy:\n"
                f"1. A booking can be cancelled for a refund only if the customer asks within {self.window} hours of "
                f"booking. Later requests must be denied.\n"
                f"2. Every refunded cancellation is charged a fee of ${self.fee} per passenger. The refund is the "
                f"amount paid minus the fee.")

    def decide(self, hours: int, passengers: int, paid: int) -> tuple[str, int | None]:
        if hours >= self.window:
            return "deny", None
        return "cancel", paid - self.fee * passengers


A = Policy("A", window=12, fee=35)
B = Policy("B", window=48, fee=80)
C = Policy("C", window=96, fee=125)
POLICIES = {p.name: p for p in (A, B, C)}
