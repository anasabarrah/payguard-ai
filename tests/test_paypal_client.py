"""Offline tests for the PayPal client and ledger (HTTP is mocked)."""
import json
from unittest.mock import MagicMock

import pytest

from payguard.ledger import Ledger
from payguard.paypal_client import PayPalClient, PayPalError, capture_id_of


def _resp(status, payload):
    r = MagicMock(status_code=status, content=b"x")
    r.json.return_value = payload
    return r


@pytest.fixture
def client():
    c = PayPalClient(client_id="id", client_secret="secret", base_url="https://example.test")
    c.session = MagicMock()
    c.session.post.return_value = _resp(200, {"access_token": "tok", "expires_in": 3600})
    return c


ORDER = {
    "id": "ORDER1", "status": "COMPLETED",
    "purchase_units": [{"payments": {"captures": [{
        "id": "CAP1", "status": "COMPLETED", "create_time": "2026-10-30T10:00:00Z",
        "amount": {"currency_code": "USD", "value": "10.00"}}]}}],
}


def test_token_is_cached(client):
    assert client.get_access_token() == "tok"
    client.get_access_token()
    assert client.session.post.call_count == 1


def test_create_order_body(client):
    client.session.request.return_value = _resp(201, {"id": "O", "status": "CREATED"})
    client.create_order(9.5, currency="EUR")
    _, kwargs = client.session.request.call_args
    assert kwargs["json"]["purchase_units"][0]["amount"] == {"currency_code": "EUR", "value": "9.50"}
    assert kwargs["headers"]["Authorization"] == "Bearer tok"
    assert "PayPal-Request-Id" in kwargs["headers"]


def test_error_raises(client):
    client.session.request.return_value = _resp(422, {"name": "UNPROCESSABLE_ENTITY"})
    with pytest.raises(PayPalError) as e:
        client.capture_order("O")
    assert e.value.status == 422


def test_missing_credentials(monkeypatch):
    monkeypatch.setattr("payguard.paypal_client.load_dotenv", lambda: None)
    monkeypatch.delenv("PAYPAL_CLIENT_ID", raising=False)
    monkeypatch.delenv("PAYPAL_CLIENT_SECRET", raising=False)
    with pytest.raises(ValueError):
        PayPalClient()


def test_ledger_capture_and_refund(tmp_path):
    led = Ledger(tmp_path / "l.db")
    assert capture_id_of(ORDER) == "CAP1"
    led.record_capture(ORDER, country="FR")
    led.record_refund("CAP1", {"id": "R1", "amount": {"value": "4.00"}})
    assert led.get("CAP1")["status"] == "PARTIALLY_REFUNDED"
    led.record_refund("CAP1", {"id": "R2", "amount": {"value": "6.00"}})
    p = led.list_payments()[0]
    assert p["status"] == "REFUNDED" and p["refunded"] == 10.0 and p["country"] == "FR"


def test_reporting_scope_detection(client):
    assert not client.can_search_transactions()
    client.session.post.return_value = _resp(200, {
        "access_token": "tok2", "expires_in": 3600,
        "scope": "openid https://uri.paypal.com/services/reporting/search/read"})
    client.get_access_token(force=True)
    assert client.can_search_transactions()


def test_webhook_signature_verification_payload(client):
    client.session.request.return_value = _resp(200, {"verification_status": "SUCCESS"})
    headers = {
        "paypal-auth-algo": "SHA256withRSA", "paypal-cert-url": "https://cert.test",
        "paypal-transmission-id": "T1", "paypal-transmission-sig": "sig",
        "paypal-transmission-time": "2026-10-30T12:00:00Z",
    }
    raw = b'{"id":"EV1","resource":{"amount":{"value":"79.00"}},"summary":"caf\\u00e9"}'
    assert client.verify_webhook_signature("WH1", headers, raw)
    _, kwargs = client.session.request.call_args
    sent = json.loads(kwargs["data"])
    assert sent["webhook_id"] == "WH1" and sent["auth_algo"] == "SHA256withRSA"
    # The event goes back byte for byte, not re-serialised.
    assert kwargs["data"].endswith('"webhook_event": ' + raw.decode() + "}")


def test_webhook_signature_failure_is_false(client):
    client.session.request.return_value = _resp(200, {"verification_status": "FAILURE"})
    headers = {k: "x" for k in ("paypal-auth-algo", "paypal-cert-url", "paypal-transmission-id",
                                "paypal-transmission-sig", "paypal-transmission-time")}
    assert not client.verify_webhook_signature("WH1", headers, b'{"id":"EV1"}')


def test_webhook_signature_rejects_missing_headers(client):
    assert not client.verify_webhook_signature("WH1", {}, b'{"id":"EV1"}')


@pytest.mark.parametrize("status,body,expected", [
    (422, {"details": [{"issue": "REFUND_FAILED_INSUFFICIENT_FUNDS"}]}, "balance is too low"),
    (422, {"details": [{"issue": "CAPTURE_FULLY_REFUNDED"}]}, "already been fully refunded"),
    (403, {"name": "PERMISSION_DENIED"}, "not allowed to refund"),
    (503, {"name": "PAYPAL_UNREACHABLE"}, "not responding"),
    (500, "<html>oops</html>", "trouble right now"),
    (422, {"details": [{"issue": "SOMETHING_NEW"}]}, "declined the refund"),
])
def test_refund_failures_get_a_polite_sentence(status, body, expected):
    from payguard.paypal_client import refund_failure_message
    message = refund_failure_message(PayPalError(status, body))
    assert expected in message and message.endswith("it stays in your review queue.")


def test_network_error_becomes_paypal_error(client):
    import requests
    client.session.request.side_effect = requests.ConnectionError("boom")
    with pytest.raises(PayPalError) as exc:
        client.get_order("O1")
    assert exc.value.issue == "PAYPAL_UNREACHABLE"


def test_reset_empties_the_ledger_but_keeps_webhook_ids(tmp_path):
    led = Ledger(tmp_path / "r.db")
    led.record_capture(ORDER, country="FR")
    led.record_decision("CAP1", "monitor")
    led.record_refund("CAP1", {"id": "R1", "amount": {"value": "1.00"}})
    event = {"id": "EV1", "event_type": "PAYMENT.CAPTURE.COMPLETED", "resource": {
        "id": "CAP1", "amount": {"value": "10.00", "currency_code": "USD"}}}
    led.record_webhook(event)
    led.reset()
    assert led.list_payments() == [] and led.decisions() == []
    assert not led.record_webhook(event) and led.list_payments() == []   # old event stays gone
