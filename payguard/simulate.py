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
    "NG": ("12 Allen Ave", "Lagos", "LA", "100001"),
    "RO": ("Str. Lipscani 10", "Bucuresti", "B", "030031"),
    "BR": ("Rua Augusta 100", "Sao Paulo", "SP", "01304-000"),
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
    Buyer("Noah Tremblay", "noah.tremblay@example.ca", "CA"),
    Buyer("Amelia Clarke", "amelia.clarke@example.co.uk", "GB"),
]


def demo_plan(rng: random.Random) -> list[tuple[Buyer, float, str]]:
    """Ordinary payments first (they set the usual ticket), then suspicious ones."""
    plan = [(b, round(rng.uniform(18, 65), 2), "regular customer")
            for b in rng.sample(REGULARS, 4)]
    tag = rng.randint(100, 999)
    plan += [
        (Buyer("K. Adeyemi", f"k.adeyemi{tag}@example.ng", "NG"),
         round(rng.uniform(800, 1200), 2), "new foreign buyer, very large order"),
        (Buyer("M. Popescu", f"m.popescu{tag}@example.ro", "RO"),
         round(rng.uniform(35, 60), 2), "new foreign buyer"),
        (Buyer("J. Doe", f"jdoe{tag}@example.com", "US"),
         round(rng.uniform(900, 1500), 2), "new buyer, unusually large order"),
        (Buyer("R. Silva", f"r.silva{tag}@example.com.br", "BR"),
         round(rng.uniform(500, 800), 2), "new foreign buyer, large order"),
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
