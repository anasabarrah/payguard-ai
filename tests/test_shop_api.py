"""Demo shop and webhook routes, driven over HTTP (PayPal is mocked)."""
import json
from unittest.mock import MagicMock

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

import shop_api
from payguard.ledger import Ledger

SIGNED = {"paypal-auth-algo": "SHA256withRSA", "paypal-cert-url": "https://cert.test",
          "paypal-transmission-id": "T1", "paypal-transmission-sig": "sig",
          "paypal-transmission-time": "2026-10-30T12:00:00Z"}


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("PAYGUARD_DB", str(tmp_path / "shop.db"))
    return Ledger()


@pytest.fixture
def paypal(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(shop_api, "paypal", lambda: mock)
    return mock


@pytest.fixture
def http(ledger):
    return TestClient(Starlette(routes=shop_api.routes))


def _captured(order_id="ORDER-1", capture_id="CAP-1", value="79.00", payer=None):
    order = {"id": order_id, "status": "COMPLETED", "purchase_units": [{"payments": {
        "captures": [{"id": capture_id, "status": "COMPLETED", "create_time": "2026-10-30T12:00:00Z",
                      "amount": {"currency_code": "USD", "value": value}}]}}]}
    if payer:
        order["payer"] = payer
    return order


def test_shop_renders_paypal_buttons(http, monkeypatch):
    monkeypatch.setenv("PAYPAL_CLIENT_ID", "sandbox-client")
    r = http.get("/shop/")
    assert r.status_code == 200
    assert "https://www.paypal.com/sdk/js?client-id=sandbox-client&" in r.text
    assert 'href="/"' in r.text and "Open PayGuard dashboard" in r.text
    assert all(sku in r.text for sku in shop_api.PRODUCTS) and len(shop_api.PRODUCTS) == 4
    assert "__" not in r.text.split("<script>", 2)[-1]          # every placeholder filled
    assert http.get("/shop", follow_redirects=False).headers["location"] == "/shop/"


def test_shop_page_escapes_client_id(http, monkeypatch):
    monkeypatch.setenv("PAYPAL_CLIENT_ID", '"><script>alert(1)</script>')
    assert "<script>alert(1)" not in http.get("/shop/").text


def test_shop_order_uses_server_catalog(http, paypal):
    paypal.create_order.return_value = {"id": "ORDER-1"}
    r = http.post("/api/shop/orders", json={"sku": "pack", "price": 0.01})
    assert r.status_code == 200 and r.json() == {"id": "ORDER-1"}
    assert paypal.create_order.call_args.args[0] == shop_api.PRODUCTS["pack"]["price"]
    assert paypal.create_order.call_args.kwargs["custom_id"] == "pack"


@pytest.mark.parametrize("body", [b'{"sku": "anything"}', b"not json", b"[1]"])
def test_bad_order_requests_are_rejected(http, paypal, body):
    assert http.post("/api/shop/orders", content=body).status_code == 400
    paypal.create_order.assert_not_called()


def test_capture_checks_shop_custom_id(http, paypal):
    paypal.get_order.return_value = {"id": "O", "status": "APPROVED",
                                     "purchase_units": [{"custom_id": "external"}]}
    assert http.post("/api/shop/orders/O/capture").status_code == 400
    paypal.capture_order.assert_not_called()


def test_capture_records_payment_with_payer(http, paypal, ledger):
    price = f"{shop_api.PRODUCTS['pack']['price']:.2f}"
    paypal.get_order.return_value = {"id": "ORDER-1", "status": "APPROVED", "purchase_units": [{
        "custom_id": "pack", "amount": {"currency_code": "USD", "value": price}}]}
    paypal.capture_order.return_value = _captured(value=price, payer={
        "payer_id": "BUYER1", "email_address": "buyer@example.com",
        "address": {"country_code": "US"}})
    r = http.post("/api/shop/orders/ORDER-1/capture")
    assert r.status_code == 200 and r.json()["capture_id"] == "CAP-1"
    assert r.json()["level"] in {"low", "medium", "high"} and 0 <= r.json()["score"] <= 100
    row = ledger.get("CAP-1")
    assert row["amount"] == float(price)
    assert (row["payer_email"], row["country"]) == ("buyer@example.com", "US")


def test_capturing_again_keeps_the_sellers_decision(http, paypal, ledger):
    price = f"{shop_api.PRODUCTS['tent']['price']:.2f}"
    done = _captured(value=price)
    done["purchase_units"][0].update(custom_id="tent", amount={"currency_code": "USD", "value": price})
    paypal.get_order.return_value = done                       # already COMPLETED
    assert http.post("/api/shop/orders/ORDER-1/capture").status_code == 200
    ledger.record_decision("CAP-1", "monitor")
    assert http.post("/api/shop/orders/ORDER-1/capture").status_code == 200
    paypal.capture_order.assert_not_called()
    assert ledger.get("CAP-1")["decision"] == "monitor"


def test_capture_rejects_changed_amount(http, paypal):
    paypal.get_order.return_value = {"id": "O", "status": "APPROVED", "purchase_units": [{
        "custom_id": "pack", "amount": {"currency_code": "USD", "value": "1.00"}}]}
    assert http.post("/api/shop/orders/O/capture").status_code == 400
    paypal.capture_order.assert_not_called()


def _capture_event(event_id="EV-1"):
    return {"id": event_id, "event_type": "PAYMENT.CAPTURE.COMPLETED", "resource": {
        "id": "CAP-1", "status": "COMPLETED", "create_time": "2026-10-30T12:00:00Z",
        "amount": {"value": "79.00", "currency_code": "USD"},
        "supplementary_data": {"related_ids": {"order_id": "ORDER-1"}}}}


def test_verified_webhook_is_deduplicated_and_records_capture(http, paypal, ledger, monkeypatch):
    monkeypatch.setenv("PAYPAL_WEBHOOK_ID", "WH-1")
    paypal.verify_webhook_signature.return_value = True
    raw = json.dumps(_capture_event())
    for _ in range(2):
        assert http.post("/webhooks/paypal", content=raw, headers=SIGNED).status_code == 200
    assert len(ledger.list_payments()) == 1
    assert ledger.get("CAP-1")["order_id"] == "ORDER-1"
    webhook_id, headers, body = paypal.verify_webhook_signature.call_args.args
    assert webhook_id == "WH-1" and body == raw.encode()
    assert headers["paypal-transmission-sig"] == "sig"


def test_unverified_webhook_is_rejected(http, paypal, ledger, monkeypatch):
    monkeypatch.setenv("PAYPAL_WEBHOOK_ID", "WH-1")
    paypal.verify_webhook_signature.return_value = False
    r = http.post("/webhooks/paypal", content=json.dumps(_capture_event()), headers=SIGNED)
    assert r.status_code == 400
    assert ledger.list_payments() == []


def test_webhook_refused_without_webhook_id(http, paypal, ledger, monkeypatch):
    monkeypatch.delenv("PAYPAL_WEBHOOK_ID", raising=False)
    monkeypatch.setattr(shop_api, "load_dotenv", lambda *a, **k: None, raising=False)
    r = http.post("/webhooks/paypal", content=json.dumps(_capture_event()), headers=SIGNED)
    assert r.status_code == 503
    paypal.verify_webhook_signature.assert_not_called()
    assert ledger.list_payments() == []


def test_webhook_refused_when_paypal_cannot_verify(http, paypal, ledger, monkeypatch):
    from payguard.paypal_client import PayPalError
    monkeypatch.setenv("PAYPAL_WEBHOOK_ID", "WH-1")
    paypal.verify_webhook_signature.side_effect = PayPalError(500, "down")
    r = http.post("/webhooks/paypal", content=json.dumps(_capture_event()), headers=SIGNED)
    assert r.status_code == 502
    assert ledger.list_payments() == []


def test_webhook_rejects_non_json(http, paypal, monkeypatch):
    monkeypatch.setenv("PAYPAL_WEBHOOK_ID", "WH-1")
    assert http.post("/webhooks/paypal", content=b"not json", headers=SIGNED).status_code == 400
    paypal.verify_webhook_signature.assert_not_called()


def _refund_event(event_id, refund_id, value, capture_id="CAP-2"):
    return {"id": event_id, "event_type": "PAYMENT.CAPTURE.REFUNDED", "resource": {
        "id": refund_id, "amount": {"value": value, "currency_code": "USD"},
        "links": [{"rel": "up", "href": f"https://api.sandbox.paypal.com/v2/payments/captures/{capture_id}"}]}}


def test_refund_webhook_updates_ledger_once(ledger):
    ledger.record_capture(_captured("ORDER-2", "CAP-2"))
    event = _refund_event("EV-R1", "REF-1", "20.00")
    assert ledger.record_webhook(event)
    assert not ledger.record_webhook(event)
    assert ledger.get("CAP-2")["refunded"] == 20
    assert ledger.get("CAP-2")["decision"] is None      # partial refund: still to review


def test_refund_from_dashboard_is_not_counted_twice_by_its_webhook(ledger):
    ledger.record_capture(_captured("ORDER-2", "CAP-2"))
    ledger.record_refund("CAP-2", {"id": "REF-1", "amount": {"value": "20.00"}})
    ledger.record_webhook(_refund_event("EV-R1", "REF-1", "20.00"))
    assert ledger.get("CAP-2")["refunded"] == 20
    ledger.record_webhook(_refund_event("EV-R2", "REF-2", "59.00"))
    assert ledger.get("CAP-2")["status"] == "REFUNDED"


def test_full_refund_made_in_paypal_leaves_the_queue(ledger):
    ledger.record_capture(_captured("ORDER-2", "CAP-2"))
    ledger.record_webhook(_refund_event("EV-R1", "REF-9", "79.00"))
    assert ledger.get("CAP-2")["decision"] == "refund"
    assert ledger.decisions("CAP-2")[0]["refund_id"] == "REF-9"
