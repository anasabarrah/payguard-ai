"""Move a sandbox dispute through its stages, to test the dispute assistant.

    python scripts/sandbox_dispute.py                       # list disputes
    python scripts/sandbox_dispute.py show PP-D-12345       # status, stage, possible actions
    python scripts/sandbox_dispute.py require-evidence PP-D-12345
        # UNDER_REVIEW -> WAITING_FOR_SELLER_RESPONSE: PayGuard can then send evidence
    python scripts/sandbox_dispute.py adjudicate PP-D-12345 seller|buyer
        # settle an UNDER_REVIEW dispute in the seller's or buyer's favor

`require-evidence` and `adjudicate` exist only in the PayPal sandbox and only
for a dispute that is UNDER_REVIEW (typically after the buyer escalated it to
a claim). Disputes are opened by the sandbox buyer; see the README.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from payguard.disputes import parse_case  # noqa: E402
from payguard.paypal_client import PayPalClient, PayPalError  # noqa: E402


def show(client: PayPalClient, dispute_id: str) -> None:
    c = parse_case(client.get_dispute(dispute_id))
    print(f"{c.dispute_id}: {c.reason_text} · ${c.amount:,.2f} · status {c.status} · stage "
          f"{c.stage} · respond by {c.due or '-'}")
    print(f"  capture {c.capture_id} · actions: {', '.join(c.actions) or 'none'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="list",
                    choices=["list", "show", "require-evidence", "adjudicate"])
    ap.add_argument("dispute_id", nargs="?")
    ap.add_argument("outcome", nargs="?", choices=["seller", "buyer"])
    args = ap.parse_args()
    client = PayPalClient()
    if "sandbox" not in client.base_url:
        ap.error("this script only runs against the PayPal sandbox")
    try:
        if args.command == "list":
            items = client.list_disputes()
            for d in items:
                print(d["dispute_id"], d.get("status"), d.get("reason"),
                      (d.get("dispute_amount") or {}).get("value"), d.get("create_time", "")[:10])
            print(f"{len(items)} dispute(s)")
            return 0
        if not args.dispute_id:
            ap.error("a dispute id is required")
        if args.command == "require-evidence":
            client.require_dispute_evidence(args.dispute_id, "SELLER_EVIDENCE")
        elif args.command == "adjudicate":
            if not args.outcome:
                ap.error("say who wins: seller or buyer")
            client.adjudicate_dispute(args.dispute_id, f"{args.outcome.upper()}_FAVOR")
        show(client, args.dispute_id)
    except PayPalError as e:
        print(f"PayPal refused: {e.short}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
