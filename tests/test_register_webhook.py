"""The webhook registration script (PayPal is mocked)."""
from unittest.mock import MagicMock

from scripts import register_paypal_webhook as script


def test_registers_the_render_url(monkeypatch, capsys):
    client = MagicMock()
    client.list_webhooks.return_value = []
    client.create_webhook.return_value = {"id": "WH-NEW"}
    monkeypatch.setattr(script, "PayPalClient", lambda: client)
    monkeypatch.setenv("PAYGUARD_PUBLIC_URL", "https://example.onrender.com/")
    script.main()
    url, events = client.create_webhook.call_args.args
    assert url == "https://example.onrender.com/webhooks/paypal"
    assert set(events) == {"PAYMENT.CAPTURE.COMPLETED", "PAYMENT.CAPTURE.REFUNDED"}
    assert "WH-NEW" in capsys.readouterr().out


def test_existing_webhook_is_reused(monkeypatch, capsys):
    client = MagicMock()
    client.list_webhooks.return_value = [{"id": "WH-OLD", "url": "https://x.test/webhooks/paypal",
                                          "event_types": [{"name": e} for e in script.EVENTS]}]
    monkeypatch.setattr(script, "PayPalClient", lambda: client)
    monkeypatch.setenv("PAYGUARD_PUBLIC_URL", "https://x.test")
    script.main()
    client.create_webhook.assert_not_called()
    assert "WH-OLD" in capsys.readouterr().out


def test_missing_events_are_added_to_an_existing_webhook(monkeypatch):
    client = MagicMock()
    client.list_webhooks.return_value = [{"id": "WH-OLD", "url": "https://x.test/webhooks/paypal",
                                          "event_types": [{"name": "PAYMENT.CAPTURE.COMPLETED"}]}]
    monkeypatch.setattr(script, "PayPalClient", lambda: client)
    monkeypatch.setenv("PAYGUARD_PUBLIC_URL", "https://x.test")
    script.main()
    client.create_webhook.assert_not_called()
    client.set_webhook_events.assert_called_once_with("WH-OLD", sorted(script.EVENTS))
