"""Frozen paraphrase banks (addendum 7, item 2): >= 20 reasons per decision type, number stated before the verdict,
and 40 customer messages per domain split 30 train / 10 test (disjoint). Do not edit after the first v2 run."""

from __future__ import annotations

# --- airline ------------------------------------------------------------------------------------------------------

AIR_WINDOW = (
    "Window: {T} hours.",
    "The refund window is {T} hours.",
    "Refunds need a request less than {T} hours after booking.",
    "Policy limit: {T} hours after booking.",
    "Cancellations are refundable for {T} hours after booking.",
)
AIR_AGE = (
    "The booking is {h} hours old; {h} {op} {T}",
    "Booked {h} hours ago, and {h} {op} {T}",
    "Age of the booking: {h} hours ({h} {op} {T})",
    "It was made {h} hours ago: {h} {op} {T}",
)
AIR_IN = (", so it is inside the window and can be refunded.", ", so the request is in time for a refund.",
          ", so a refund is allowed.", ", so it qualifies for a refund.", ", so it can be cancelled with a refund.")
AIR_OUT = (", so it is outside the window and can't be refunded.", ", so the request is too late for a refund.",
           ", so no refund is possible.", ", so it doesn't qualify for a refund.", ", so the refund must be denied.")
AIR_FEE = ("Fee: ${F} x {n} = ${fee}. Refund: ${paid} - ${fee} = ${refund}.",
           "The fee is ${F} per passenger, ${fee} in total, so the refund is ${paid} - ${fee} = ${refund}.")

AIR_ASKS = (
    "Hi, I need to cancel booking {code}. Can I get a refund?",
    "Please cancel my booking {code} and refund me.",
    "Something came up and I can't travel. Can you cancel {code}?",
    "I'd like to cancel reservation {code}, please. How much will I get back?",
    "Hello, can booking {code} be cancelled for a refund?",
    "I want to cancel {code}. What's my refund?",
    "My plans changed. Could you cancel {code} and send the money back?",
    "Hi there, I booked {code} by mistake. Refund please?",
    "Can you tell me if I can still get my money back for {code}?",
    "We're not flying anymore. Please cancel {code}.",
    "Is it possible to cancel {code}? I'd like a refund to my card.",
    "Cancel {code} for me, please. What will I get back?",
    "I need to call off my trip, booking {code}. Refund?",
    "Hello! Booking {code}: I'd like to cancel and get refunded.",
    "My meeting was moved, so I don't need {code} anymore. Can you refund it?",
    "Please process a cancellation for {code}.",
    "Hi, family emergency. I have to cancel {code}. Can I get my money back?",
    "Could I cancel booking {code} and get a refund, please?",
    "I booked {code} but the dates don't work. Cancel it please.",
    "Hey, can you refund booking {code}? I won't be travelling.",
    "I'd like my booking {code} cancelled. How much is refunded?",
    "Good morning, I want to cancel {code} for a refund.",
    "Please cancel {code}; the trip is off.",
    "What refund would I get if I cancel {code} now?",
    "I made a duplicate booking, {code}. Please cancel it and refund me.",
    "Can booking {code} still be refunded? I need to cancel.",
    "Hi, cancelling {code}. Please confirm my refund.",
    "The event I was travelling for got cancelled. Refund {code}?",
    "I'd like to cancel {code} and get the money back to my original payment.",
    "Hello, please cancel reservation {code}. What's the refund amount?",
    # test only (10)
    "Hi, I have to cancel {code}. Will I be refunded?",
    "Please refund and cancel my booking {code}.",
    "Our travel plans fell through. Can you cancel {code}?",
    "I'm requesting a cancellation of {code}. How much do I get back?",
    "Could you check whether {code} is refundable and cancel it?",
    "Hello, I no longer need booking {code}. Refund please.",
    "Can I cancel {code}? Please tell me the refund.",
    "I'd like to cancel my flight booking {code} for a refund.",
    "Hi! Cancel {code}, please, and refund what you can.",
    "My visa didn't come through, so please cancel {code}.",
)

# --- returns ------------------------------------------------------------------------------------------------------

RET_WINDOW = (
    "Window: {W} days.",
    "The return window is {W} days.",
    "Returns need a request less than {W} days after delivery.",
    "Policy limit: {W} days after delivery.",
    "Items are returnable for {W} days after delivery.",
)
RET_AGE = (
    "Delivered {d} days ago; {d} {op} {W}",
    "The item was delivered {d} days ago, and {d} {op} {W}",
    "Days since delivery: {d} ({d} {op} {W})",
    "It arrived {d} days ago: {d} {op} {W}",
)
RET_IN = (", inside the window.", ", so it is in time.", ", within the limit.", ", so timing is fine.",
          ", so it is early enough.")
RET_OUT = (", so it is outside the window and can't be returned.", ", so the return is too late.",
           ", so it can't be returned.", ", so the return must be denied.", ", so it is past the limit.")
RET_CATS = (
    "Final-sale categories: {a} and {b}.",
    "Only {a} and {b} are final sale.",
    "{A} and {b} can't be returned (final sale).",
    "The final-sale list is {a} and {b}.",
)
RET_CAT_NO = (" The item is {what}, which is not on that list.", " {What} is not final sale.",
              " The item ({what}) is returnable.", " {What} is not one of them.", " This is {what}, not final sale.")
RET_CAT_YES = (" The item is {what}, which is on that list, so it can't be returned.",
               " {What} is final sale, so the return must be denied.", " The item ({what}) is final sale; no return.",
               " {What} is one of them, so it can't be returned.", " This is {what}, a final-sale item: no return.")
RET_FEE = ("Restocking fee: {F}% of ${price} = ${fee}. Refund: ${price} - ${fee} = ${refund}.",
           "The restocking fee is {F}%, ${fee}, so the refund is ${price} - ${fee} = ${refund}.")

RET_ASKS = (
    "Hi, I'd like to return the {item} from order {code}. Can I get a refund?",
    "Please take back the {item} (order {code}) and refund me.",
    "The {item} from order {code} isn't what I expected. Can I send it back?",
    "I want to return my {item}, order {code}. How much will I get back?",
    "Hello! I ordered the {item} ({code}) but ended up not using it on my trip. Is a return possible?",
    "Can I return the {item} from order {code}? I still have the original packaging.",
    "My partner already bought the same {item}, so I'd like to return mine (order {code}). What's the refund?",
    "Hi there, order {code}: the {item} doesn't fit my needs. I'd like my money back please.",
    "The {item} I got in order {code} is the wrong size. Return please.",
    "I changed my mind about the {item} (order {code}). Can I return it?",
    "Order {code}: I'd like to send the {item} back for a refund.",
    "The color of the {item} from {code} isn't right. Can I return it?",
    "Hi, I received the {item} (order {code}) as a gift and don't need it. Refund?",
    "I found the {item} cheaper elsewhere. Can I return order {code}?",
    "Please start a return for the {item} in order {code}.",
    "Can you refund the {item} from {code}? It's unused.",
    "Hello, how do I return the {item} I bought (order {code})?",
    "The {item} from order {code} arrived fine but I don't want it anymore.",
    "Is the {item} in order {code} returnable? I'd like my money back.",
    "I'd like to return {code}'s {item}. What will I be refunded?",
    "Hi, my trip got cancelled so I don't need the {item} (order {code}). Refund please.",
    "Order {code}: can the {item} still be returned?",
    "I'm not happy with the {item} from {code}. Can I return it for a refund?",
    "Please refund the {item}, order {code}; I bought two by mistake.",
    "The {item} (order {code}) is heavier than I expected. Return?",
    "Can I get a refund on the {item} I ordered, {code}?",
    "I'd like to return the {item} from my order {code}, please.",
    "Hello, requesting a return for order {code} ({item}).",
    "Hey, the {item} from {code} isn't for me. Can I send it back?",
    "Could you process a return of the {item} in order {code}?",
    # test only (10)
    "Hi, can I return the {item} from order {code} for a refund?",
    "I'd like to send back the {item} (order {code}). What's my refund?",
    "The {item} in order {code} didn't work out for me. Return please.",
    "Please accept a return of my {item}, order {code}.",
    "Order {code}: I no longer need the {item}. Can I get my money back?",
    "Is it possible to return the {item} from {code}?",
    "Hello, I want to return the {item} I received in order {code}.",
    "Can the {item} (order {code}) be refunded? It's still new.",
    "I'd like a refund for the {item} from order {code}, please.",
    "Hi, returning the {item} from {code}. How much will I get?",
)

N_TRAIN_ASKS = 30
