"""Register this deployment's sandbox webhook and print its ID for Render.

    PAYGUARD_PUBLIC_URL=https://payguard-ai-1p86.onrender.com python scripts/register_paypal_webhook.py

Uses PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET from .env (the same sandbox app
as Render). Safe to run twice: an existing webhook for the URL is reused and
given any missing event type.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from payguard.paypal_client import PayPalClient  # noqa: E402

EVENTS = ["PAYMENT.CAPTURE.COMPLETED", "PAYMENT.CAPTURE.REFUNDED"]


def main():
    base = os.getenv("PAYGUARD_PUBLIC_URL", "https://payguard-ai-1p86.onrender.com").rstrip("/")
    url = f"{base}/webhooks/paypal"
    client = PayPalClient()
    hook = next((h for h in client.list_webhooks() if h.get("url") == url), None)
    if hook is None:
        hook = client.create_webhook(url, EVENTS)
        print(f"Webhook registered for {url}: {hook['id']}")
    else:
        names = {item.get("name") for item in hook.get("event_types", [])}
        if not set(EVENTS) <= names:
            client.set_webhook_events(hook["id"], sorted(names | set(EVENTS)))
            print(f"Webhook {hook['id']} updated with events {', '.join(EVENTS)}")
        print(f"Webhook already registered for {url}: {hook['id']}")
    print(f"Set PAYPAL_WEBHOOK_ID={hook['id']} in .env and in Render, then redeploy.")


if __name__ == "__main__":
    main()
