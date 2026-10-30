"""Create real PayPal sandbox payments for the demo, some of them suspicious.

Each payment is a genuine sandbox card order, captured and stored in the
ledger. PayPal does not return the card's billing country, and card payments
carry no payer email, so the buyer profile (email, country) used for each
payment is recorded in the ledger from the simulation itself.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from payguard.ledger import Ledger
from payguard.paypal_client import PayPalClient

ADDRESSES = {
    "US": ("1 Main St", "San Jose", "CA", "95131"),
    "CA": ("100 Queen St W", "Toronto", "ON", "M5H 2N2"),
    "GB": ("10 Downing St", "London", "", "SW1A 2AA"),
    "DE": ("Unter den Linden 10", "Berlin", "", "10117"),
    "JP": ("1-1 Marunouchi", "Chiyoda-ku", "Tokyo", "100-0005"),
    "AU": ("200 George St", "Sydney", "NSW", "2000"),
    "IN": ("12 MG Road", "Bengaluru", "KA", "560001"),
}


@dataclass
class Buyer:
    name: str
    email: str
    country: str


REGULARS = [
    Buyer("Emma Johnson", "emma.johnson@example.com", "US"),
    Buyer("Liam Smith", "liam.smith@example.com", "US"),
    Buyer("Olivia Brown", "olivia.brown@example.com", "US"),
    Buyer("Ava Martinez", "ava.martinez@example.com", "US"),
    Buyer("Mason Lee", "mason.lee@example.com", "US"),
]
FOREIGN_REGULARS = [
    Buyer("Noah Tremblay", "noah.tremblay@example.ca", "CA"),
    Buyer("Amelia Clarke", "amelia.clarke@example.co.uk", "GB"),
]


def demo_plan(rng: random.Random) -> list[tuple[Buyer, float, str]]:
    """Ordinary payments first (they set the usual ticket), then unusual ones.

    Expected levels with the shipped model (cross-border bands: low up to
    ~2.8x the usual ticket, medium ~2.9-8.1x, high from ~8.2x):
      6 low     4 domestic regulars + 2 foreign buyers with a normal basket,
                showing that a foreign payer alone is not flagged
      2 medium  new foreign buyers, ~5x the usual ticket
      2 high    new foreign buyers, 20x+ the usual ticket
    Six of ten payments are ordinary, so the 24h median ticket stays at the
    ordinary level even after repeated demos. Scores do not depend on the time.
    """
    plan = [(b, round(rng.uniform(40, 50), 2), "regular customer")
            for b in rng.sample(REGULARS, 4)]
    plan += [(b, round(rng.uniform(40, 50), 2), "foreign customer, usual basket")
             for b in FOREIGN_REGULARS]
    tag = rng.randint(100, 999)
    plan += [
        (Buyer("L. Becker", f"l.becker{tag}@example.de", "DE"),
         round(rng.uniform(220, 240), 2), "new foreign buyer, unusually large order"),
        (Buyer("H. Tanaka", f"h.tanaka{tag}@example.jp", "JP"),
         round(rng.uniform(220, 240), 2), "new foreign buyer, unusually large order"),
        (Buyer("S. Wilson", f"s.wilson{tag}@example.com.au", "AU"),
         round(rng.uniform(1100, 1500), 2), "new foreign buyer, extremely large order"),
        (Buyer("A. Sharma", f"a.sharma{tag}@example.in", "IN"),
         round(rng.uniform(1100, 1500), 2), "new foreign buyer, extremely large order"),
    ]
    return plan


def card_for(buyer: Buyer) -> dict:
    line1, city, area, postal = ADDRESSES[buyer.country]
    address = {"address_line_1": line1, "admin_area_2": city, "postal_code": postal,
               "country_code": buyer.country}
    if area:
        address["admin_area_1"] = area
    return {"card": {"number": "4111111111111111", "expiry": "2030-12", "security_code": "123",
                     "name": buyer.name, "billing_address": address}}


def simulate_payments(client: PayPalClient, ledger: Ledger, seed: int | None = None,
                      on_progress: Callable[[int, int, str], None] | None = None) -> list[str]:
    """Create and capture the demo payments; returns the new capture ids."""
    rng = random.Random(seed)
    plan = demo_plan(rng)
    created = []
    for i, (buyer, amount, story) in enumerate(plan):
        if on_progress:
            on_progress(i, len(plan), f"{buyer.name} ({buyer.country}) pays ${amount:,.2f}")
        order = client.create_order(amount, description=f"PayGuard demo: {story}",
                                    payment_source=card_for(buyer))
        if order["status"] != "COMPLETED":
            order = client.capture_order(order["id"])
        created.append(ledger.record_capture(order, payer_email=buyer.email,
                                             country=buyer.country))
    if on_progress:
        on_progress(len(plan), len(plan), "done")
    return created
