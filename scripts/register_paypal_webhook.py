"""Register this deployment's sandbox webhook and print its ID for Render."""
import os

from payguard.paypal_client import PayPalClient

EVENTS = ["PAYMENT.CAPTURE.COMPLETED", "PAYMENT.CAPTURE.REFUNDED"]


def main():
    base = os.getenv("PAYGUARD_PUBLIC_URL", "https://payguard-ai-1p86.onrender.com").rstrip("/")
    url = f"{base}/webhooks/paypal"
    client = PayPalClient()
    matches = [hook for hook in client.list_webhooks() if hook.get("url") == url]
    if matches:
        hook = matches[0]
        existing = {item.get("event_type", {}).get("name") for item in hook.get("event_types", [])}
        if set(EVENTS) <= existing:
            print(f"Webhook already registered: {hook['id']}")
            return
    hook = client.create_webhook(url, EVENTS)
    print(f"Webhook registered: {hook['id']}")
    print("Set PAYPAL_WEBHOOK_ID to this value in Render and redeploy.")


if __name__ == "__main__":
    main()
