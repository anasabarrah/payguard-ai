"""Headless UI tests with Streamlit's AppTest (PayPal and Groq are mocked)."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from payguard.ledger import Ledger
from tests.test_agent import _order

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("PAYGUARD_DB", str(tmp_path / "ui.db"))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("payguard.explain.load_dotenv", lambda *a, **k: None)
    led = Ledger()
    for i, h in enumerate(range(10, 14)):
        led.record_capture(_order(i, 30.0, f"2026-10-30T{h:02d}:00:00Z"),
                           payer_email=f"buyer{i}@x.com", country="US")
    led.record_capture(_order(9, 900.0, "2026-10-30T14:00:00Z"),
                       payer_email="new@z.de", country="DE")
    client = MagicMock()
    client.refund_capture.return_value = {"id": "REF9", "status": "COMPLETED",
                                          "amount": {"currency_code": "USD", "value": "900.00"}}
    monkeypatch.setattr("payguard.paypal_client.PayPalClient", lambda *a, **k: client)
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.client = client
    return at


def _clear_caches():
    import streamlit as st
    st.cache_resource.clear()


def test_renders_and_defaults_to_riskiest(app):
    _clear_caches()
    app.run()
    assert not app.exception
    assert "$900.00" in " ".join(m.value for m in app.markdown)
    assert any("Template explanation" in c.value for c in app.caption)
    text = " ".join(m.value for m in app.markdown)
    assert "Why this score" in text and "flagged it" not in text
    labels = [b.label for b in app.button]
    assert {"✅ Approve", "👁️ Monitor", "↩️ Refund", "⚡ Simulate payments"} <= set(labels)
    shop = [b for b in app.get("link_button") if "Open demo shop" in b.proto.label]
    assert shop and shop[0].proto.url == "/shop/"


def test_refund_requires_confirmation(app):
    _clear_caches()
    app.run()
    app.button(key="b_refund").click().run()
    assert not app.exception
    app.client.refund_capture.assert_not_called()      # dialog only
    yes = next(b for b in app.button if b.label == "Yes, refund")
    yes.click().run()
    assert not app.exception
    app.client.refund_capture.assert_called_once()
    assert Ledger().get("CAP9")["decision"] == "refund"


def test_approve_records_decision_without_paypal(app):
    _clear_caches()
    app.run()
    app.button(key="b_accept").click().run()
    assert not app.exception
    app.client.refund_capture.assert_not_called()
    assert Ledger().get("CAP9")["decision"] == "accept"


def test_cancel_refund_closes_dialog(app):
    _clear_caches()
    app.run()
    app.button(key="b_refund").click().run()
    next(b for b in app.button if b.label == "Cancel").click().run()
    assert not app.exception and "refund_for" not in app.session_state
    assert not any(b.label == "Yes, refund" for b in app.button)
    app.client.refund_capture.assert_not_called()


def test_new_payments_are_announced(app):
    _clear_caches()
    app.session_state["announce"] = 1
    app.run()
    assert not app.exception
    assert any("New PayPal payment: $900.00" in t.value for t in app.toast)
    assert any("Live" in m.value for m in app.markdown)


def test_webhook_badge_on_verified_payment(app):
    Ledger().record_webhook({"id": "EV-1", "event_type": "PAYMENT.CAPTURE.COMPLETED", "resource": {
        "id": "CAP9", "amount": {"value": "900.00", "currency_code": "USD"}}})
    _clear_caches()
    app.run()
    assert not app.exception
    assert any("✓ PayPal webhook" in m.value for m in app.markdown)


def test_agent_reviews_queue_and_refunds_only_after_confirmation(app):
    _clear_caches()
    app.run()
    app.button(key="b_review_queue").click().run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "1 payment to refund, 0 to monitor, 4 to approve" in text
    assert "Est. loss avoided" in text and "$900.00" in text

    app.button(key="b_apply_plan").click().run()
    assert not app.exception
    app.client.refund_capture.assert_not_called()
    led = Ledger()
    assert all(led.get(f"CAP{i}")["decision"] == "accept" for i in range(4))
    assert led.get("CAP9")["decision"] is None
    yes = next(b for b in app.button if b.label == "Yes, refund them")
    yes.click().run()
    assert not app.exception
    app.client.refund_capture.assert_called_once()
    assert led.get("CAP9")["decision"] == "refund"
    assert led.decisions("CAP9")[0]["source"] == "agent_plan"
    assert any("refund(s) completed on PayPal" in s.value for s in app.success)


def test_agent_refunds_can_be_declined(app):
    _clear_caches()
    app.run()
    app.button(key="b_review_queue").click().run()
    app.button(key="b_apply_plan").click().run()
    next(b for b in app.button if b.label == "Not now").click().run()
    assert not app.exception
    app.client.refund_capture.assert_not_called()
    assert Ledger().get("CAP9")["decision"] is None


def test_reset_demo_asks_then_empties_the_ledger_without_paypal(app):
    _clear_caches()
    app.run()
    app.button(key="b_reset").click().run()
    assert not app.exception and len(Ledger().list_payments()) == 5      # dialog only
    next(b for b in app.button if b.label == "Cancel").click().run()
    assert len(Ledger().list_payments()) == 5
    app.button(key="b_reset").click().run()
    next(b for b in app.button if b.label == "Yes, reset").click().run()
    assert not app.exception
    assert Ledger().list_payments() == []
    assert any("No payments yet" in i.value for i in app.info)
    app.client.refund_capture.assert_not_called()
    app.client.create_order.assert_not_called()


def test_failed_refund_shows_a_polite_message(app):
    from payguard.paypal_client import PayPalError
    _clear_caches()
    app.client.refund_capture.side_effect = PayPalError(422, {
        "name": "UNPROCESSABLE_ENTITY", "details": [{"issue": "REFUND_FAILED_INSUFFICIENT_FUNDS"}]})
    app.run()
    app.button(key="b_refund").click().run()
    next(b for b in app.button if b.label == "Yes, refund").click().run()
    assert not app.exception and not app.error
    assert any("balance is too low" in w.value for w in app.warning)
    assert Ledger().get("CAP9")["decision"] is None


def test_failed_plan_refund_shows_a_polite_message(app):
    from payguard.paypal_client import PayPalError
    _clear_caches()
    app.client.refund_capture.side_effect = PayPalError(422, {
        "details": [{"issue": "REFUND_FAILED_INSUFFICIENT_FUNDS"}]})
    app.run()
    app.button(key="b_review_queue").click().run()
    app.button(key="b_apply_plan").click().run()
    next(b for b in app.button if b.label == "Yes, refund them").click().run()
    assert not app.exception and not app.error
    assert any("1 refund ($900.00) could not be made" in w.value for w in app.warning)


def test_several_failed_plan_refunds_read_in_the_plural(app):
    from payguard.paypal_client import PayPalError
    led = Ledger()
    led.record_capture(_order(8, 950.0, "2026-10-30T14:30:00Z"), payer_email="n2@z.jp", country="JP")
    _clear_caches()
    app.client.refund_capture.side_effect = PayPalError(403, {"name": "PERMISSION_DENIED"})
    app.run()
    app.button(key="b_review_queue").click().run()
    app.button(key="b_apply_plan").click().run()
    next(b for b in app.button if b.label == "Yes, refund them").click().run()
    [w] = [w.value for w in app.warning if "could not be made" in w.value]
    assert w.startswith("2 refunds ($1,850.00) could not be made.") and "they stay" in w
