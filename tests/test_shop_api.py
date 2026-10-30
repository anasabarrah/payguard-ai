import asyncio
from unittest.mock import AsyncMock, MagicMock

import shop_api
from payguard.ledger import Ledger


def test_shop_renders_paypal_buttons(monkeypatch):
    monkeypatch.setenv("PAYPAL_CLIENT_ID", "sandbox-client")
    response = asyncio.run(shop_api.shop(None))
    body = response.body.decode()
    assert response.status_code == 200
    assert "https://www.paypal.com/sdk/js?client-id=sandbox-client" in body
    assert all(f"paypal-{sku}" in body for sku in shop_api.PRODUCTS)


def test_shop_order_uses_server_catalog(monkeypatch, tmp_path):
    mock = MagicMock()
    mock.create_order.return_value = {"id": "ORDER-1"}
    monkeypatch.setattr(shop_api, "paypal", lambda: mock)
    monkeypatch.setattr(shop_api, "ledger", Ledger(tmp_path / "shop.db"))
    request = MagicMock()
    request.json = AsyncMock(return_value={"sku": "plus"})
    response = asyncio.run(shop_api.create_order(request))
    assert response.status_code == 200 and response.body == b'{"id":"ORDER-1"}'
    assert mock.create_order.call_args.args[0] == 79.0
    assert mock.create_order.call_args.kwargs["custom_id"] == "plus"


def test_unknown_sku_is_rejected(monkeypatch):
    request = MagicMock()
    request.json = AsyncMock(return_value={"sku": "anything"})
    response = asyncio.run(shop_api.create_order(request))
    assert response.status_code == 400


def test_capture_checks_shop_custom_id(monkeypatch):
    mock = MagicMock()
    mock.get_order.return_value = {"id": "O", "status": "APPROVED", "purchase_units": [{"custom_id": "external"}]}
    monkeypatch.setattr(shop_api, "paypal", lambda: mock)
    request = MagicMock()
    request.path_params = {"order_id": "O"}
    response = asyncio.run(shop_api.capture_order(request))
    assert response.status_code == 400
    mock.capture_order.assert_not_called()


def test_capture_records_only_server_catalog_amount(monkeypatch, tmp_path):
    mock = MagicMock()
    mock.get_order.return_value = {"id": "ORDER-1", "status": "APPROVED", "purchase_units": [{
        "custom_id": "plus", "amount": {"currency_code": "USD", "value": "79.00"}}]}
    mock.capture_order.return_value = {"id": "ORDER-1", "purchase_units": [{"payments": {
        "captures": [{"id": "CAP-1", "status": "COMPLETED", "amount": {
            "currency_code": "USD", "value": "79.00"}}]}}]}
    ledger = Ledger(tmp_path / "capture.db")
    monkeypatch.setattr(shop_api, "paypal", lambda: mock)
    monkeypatch.setattr(shop_api, "ledger", ledger)
    request = MagicMock()
    request.path_params = {"order_id": "ORDER-1"}
    response = asyncio.run(shop_api.capture_order(request))
    assert response.status_code == 200
    assert ledger.get("CAP-1")["amount"] == 79


def test_capture_rejects_changed_amount(monkeypatch):
    mock = MagicMock()
    mock.get_order.return_value = {"id": "O", "status": "APPROVED", "purchase_units": [{
        "custom_id": "plus", "amount": {"currency_code": "USD", "value": "1.00"}}]}
    monkeypatch.setattr(shop_api, "paypal", lambda: mock)
    request = MagicMock()
    request.path_params = {"order_id": "O"}
    response = asyncio.run(shop_api.capture_order(request))
    assert response.status_code == 400
    mock.capture_order.assert_not_called()


def test_verified_webhook_is_deduplicated_and_records_capture(monkeypatch, tmp_path):
    ledger = Ledger(tmp_path / "webhook.db")
    mock = MagicMock()
    mock.verify_webhook_signature.return_value = True
    monkeypatch.setattr(shop_api, "paypal", lambda: mock)
    monkeypatch.setattr(shop_api, "ledger", ledger)
    monkeypatch.setenv("PAYPAL_WEBHOOK_ID", "WH-1")
    event = {"id": "EV-1", "event_type": "PAYMENT.CAPTURE.COMPLETED", "resource": {
        "id": "CAP-1", "status": "COMPLETED", "create_time": "2026-10-30T12:00:00Z",
        "amount": {"value": "79.00", "currency_code": "USD"},
        "supplementary_data": {"related_ids": {"order_id": "ORDER-1"}}}}
    request = MagicMock()
    request.json = AsyncMock(return_value=event)
    request.headers = {}
    assert asyncio.run(shop_api.paypal_webhook(request)).status_code == 200
    assert asyncio.run(shop_api.paypal_webhook(request)).status_code == 200
    assert len(ledger.list_payments()) == 1
    assert ledger.get("CAP-1")["order_id"] == "ORDER-1"


def test_invalid_webhook_is_rejected(monkeypatch):
    mock = MagicMock()
    mock.verify_webhook_signature.return_value = False
    monkeypatch.setattr(shop_api, "paypal", lambda: mock)
    monkeypatch.setenv("PAYPAL_WEBHOOK_ID", "WH-1")
    request = MagicMock()
    request.json = AsyncMock(return_value={"id": "EV"})
    request.headers = {}
    response = asyncio.run(shop_api.paypal_webhook(request))
    assert response.status_code == 400


def test_refund_webhook_updates_ledger_once(tmp_path):
    ledger = Ledger(tmp_path / "refund.db")
    ledger.record_capture({"id": "ORDER-2", "purchase_units": [{"payments": {"captures": [{
        "id": "CAP-2", "status": "COMPLETED", "amount": {"currency_code": "USD", "value": "79.00"}}]}}]})
    event = {"id": "EV-REFUND", "event_type": "PAYMENT.CAPTURE.REFUNDED", "resource": {
        "amount": {"value": "20.00", "currency_code": "USD"},
        "supplementary_data": {"related_ids": {"capture_id": "CAP-2"}}}}
    assert ledger.record_webhook(event)
    assert not ledger.record_webhook(event)
    assert ledger.get("CAP-2")["refunded"] == 20
