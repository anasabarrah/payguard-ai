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
