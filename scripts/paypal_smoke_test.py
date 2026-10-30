"""End-to-end check against the PayPal sandbox.

Authenticates, creates and captures a card order, records it in the local
ledger, refunds part of it, and tries the Transaction Search API.

Usage: python scripts/paypal_smoke_test.py [--amount 12.34] [--no-refund]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from payguard.ledger import Ledger  # noqa: E402
from payguard.paypal_client import PayPalClient, PayPalError, capture_id_of  # noqa: E402

# PayPal sandbox accepts standard test card numbers without a real buyer.
TEST_CARD = {
    "card": {
        "number": "4111111111111111",
        "expiry": "2030-12",
        "security_code": "123",
        "name": "Sandbox Buyer",
        "billing_address": {
            "address_line_1": "1 Main St", "admin_area_2": "San Jose", "admin_area_1": "CA",
            "postal_code": "95131", "country_code": "US",
        },
    }
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amount", default="12.34")
    ap.add_argument("--no-refund", action="store_true")
    args = ap.parse_args()

    client = PayPalClient()
    client.get_access_token()
    print(f"[ok] authenticated against {client.base_url}")

    order = client.create_order(args.amount, description="PayGuard smoke test",
                                payment_source=TEST_CARD)
    print(f"[ok] order {order['id']} created, status={order['status']}")
    if order["status"] != "COMPLETED":
        order = client.capture_order(order["id"])
    capture_id = capture_id_of(order)
    print(f"[ok] captured {capture_id}, status={client.get_capture(capture_id)['status']}")

    ledger = Ledger()
    ledger.record_capture(order, payer_email="sandbox-buyer@example.com", country="US")

    if not args.no_refund:
        refund = client.refund_capture(capture_id, amount="1.00", note="PayGuard smoke test")
        ledger.record_refund(capture_id, refund)
        print(f"[ok] refund {refund['id']} status={refund['status']}")

    print(f"[ok] ledger holds {len(ledger.list_payments())} payment(s)")

    try:
        print(f"[ok] transaction search returned {len(client.list_transactions())} rows")
    except PayPalError as e:
        print(f"[warn] transaction search unavailable ({e.status}); enable 'Transaction search'"
              " for the app in the PayPal developer dashboard. Using the local ledger instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
