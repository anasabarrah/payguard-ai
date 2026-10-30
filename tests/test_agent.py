"""Tests for the agent and the demo simulator (PayPal is mocked)."""
import random
from unittest.mock import MagicMock

import pytest

from payguard.agent import PROPOSAL, PayGuardAgent
from payguard.ledger import Ledger
from payguard.risk import RiskModel
from payguard.simulate import demo_plan, simulate_payments


def _order(i, amount, ts):
    return {"id": f"O{i}", "status": "COMPLETED", "purchase_units": [{"payments": {"captures": [{
        "id": f"CAP{i}", "status": "COMPLETED", "create_time": ts,
        "amount": {"currency_code": "USD", "value": f"{amount:.2f}"}}]}}]}


@pytest.fixture(scope="module")
def model():
    return RiskModel.load()


@pytest.fixture
def agent(tmp_path, model):
    led = Ledger(tmp_path / "l.db")
    for i, h in enumerate(range(10, 14)):
        led.record_capture(_order(i, 30.0, f"2026-10-30T{h:02d}:00:00Z"),
                           payer_email=f"buyer{i}@x.com", country="US")
    led.record_capture(_order(9, 900.0, "2026-10-30T14:00:00Z"),
                       payer_email="new@z.ng", country="NG")
    paypal = MagicMock()
    paypal.refund_capture.return_value = {"id": "REF1", "status": "COMPLETED",
                                          "amount": {"currency_code": "USD", "value": "900.00"}}
    return PayGuardAgent(led, model, paypal, use_llm=False)


def _alert(agent, capture_id):
    return next(a for a in agent.review() if a.capture_id == capture_id)


def test_review_proposes_by_level(agent):
    alerts = agent.review()
    assert alerts[0].capture_id == "CAP9"  # newest first
    assert all(a.proposed_action == PROPOSAL[a.level] for a in alerts)
    assert alerts[0].level == "high" and alerts[0].proposed_action == "refund"
    assert 1 <= len(alerts[0].reasons) <= 3 and all(a.pending for a in alerts)


def test_nothing_happens_without_seller_confirmation(agent):
    with pytest.raises(PermissionError):
        agent.execute(_alert(agent, "CAP9"), "refund")
    agent.paypal.refund_capture.assert_not_called()
    assert agent.ledger.get("CAP9")["decision"] is None


def test_refund_after_confirmation(agent):
    res = agent.execute(_alert(agent, "CAP9"), "refund", seller_confirmed=True)
    agent.paypal.refund_capture.assert_called_once()
    assert agent.paypal.refund_capture.call_args.kwargs["amount"] == 900.0
    assert res == {"capture_id": "CAP9", "action": "refund", "refund_id": "REF1",
                   "followed_proposal": True}
    p = agent.ledger.get("CAP9")
    assert p["decision"] == "refund" and p["status"] == "REFUNDED"
    log = agent.ledger.decisions("CAP9")
    assert log[0]["proposed_action"] == "refund" and log[0]["refund_id"] == "REF1"
    with pytest.raises(ValueError):
        agent.execute(_alert(agent, "CAP9"), "accept", seller_confirmed=True)


def test_seller_can_override_without_paypal_call(agent):
    res = agent.execute(_alert(agent, "CAP9"), "monitor", seller_confirmed=True)
    assert res["followed_proposal"] is False
    agent.paypal.refund_capture.assert_not_called()
    assert agent.ledger.get("CAP9")["decision"] == "monitor"
    assert not _alert(agent, "CAP9").pending


def test_explanation_is_cached(agent, monkeypatch):
    calls = []
    import payguard.agent as ag
    real = ag.explain
    monkeypatch.setattr(ag, "explain", lambda *a, **k: calls.append(1) or real(*a, **k))
    e1 = agent.explain(_alert(agent, "CAP9"))
    e2 = agent.explain(_alert(agent, "CAP9"))
    assert len(calls) == 1 and e1.sentence == e2.sentence and e2.source == "template"


def test_simulator_records_profiles(tmp_path):
    plan = demo_plan(random.Random(0))
    assert len(plan) == 8 and {b.country for b, _, _ in plan[4:]} >= {"NG", "RO", "BR"}
    client = MagicMock()
    client.create_order.side_effect = [_order(i, a, f"2026-10-30T12:00:{i:02d}Z")
                                       for i, (_, a, _) in enumerate(plan)]
    led = Ledger(tmp_path / "s.db")
    ids = simulate_payments(client, led, seed=0)
    assert len(ids) == 8 and led.get(ids[4])["country"] == plan[4][0].country
    card = client.create_order.call_args_list[4].kwargs["payment_source"]["card"]
    assert card["billing_address"]["country_code"] == plan[4][0].country
