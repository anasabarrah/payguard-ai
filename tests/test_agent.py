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
                       payer_email="new@z.de", country="DE")
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
    assert len(plan) == 10 and {b.country for b, _, _ in plan[6:]} == {"DE", "JP", "AU", "IN"}
    client = MagicMock()
    client.create_order.side_effect = [_order(i, a, f"2026-10-30T12:00:{i:02d}Z")
                                       for i, (_, a, _) in enumerate(plan)]
    led = Ledger(tmp_path / "s.db")
    ids = simulate_payments(client, led, seed=0)
    assert len(ids) == 10 and led.get(ids[6])["country"] == plan[6][0].country
    card = client.create_order.call_args_list[6].kwargs["payment_source"]["card"]
    assert card["billing_address"]["country_code"] == plan[6][0].country


# --- whole-queue review (step 3) ---------------------------------------------

def test_plan_counts_and_amounts_come_from_the_code(agent):
    plan = agent.plan_queue()
    assert plan.count == {"accept": 4, "monitor": 0, "refund": 1}
    assert plan.total == {"accept": 120.0, "monitor": 0.0, "refund": 900.0}
    assert plan.loss_avoided == 900.0 and plan.items[0][:2] == ("CAP9", "refund")
    assert plan.reasons["CAP9"] and "CAP0" not in plan.reasons      # only flagged payments


def test_plan_skips_decided_payments(agent):
    agent.execute(_alert(agent, "CAP0"), "monitor", seller_confirmed=True)
    plan = agent.plan_queue()
    assert "CAP0" not in [cid for cid, _, _ in plan.items] and plan.count["accept"] == 3


def test_apply_plan_never_refunds_without_confirmation(agent):
    res = agent.apply_plan(agent.plan_queue())
    agent.paypal.refund_capture.assert_not_called()
    assert len(res["done"]["accept"]) == 4 and res["skipped"] == ["CAP9"]
    assert agent.ledger.get("CAP9")["decision"] is None
    log = agent.ledger.decisions("CAP0")[0]
    assert log["source"] == "agent_plan" and log["chosen_action"] == "accept"


def test_apply_plan_refunds_after_confirmation_and_logs_each_decision(agent):
    plan = agent.plan_queue()
    agent.apply_plan(plan)
    res = agent.apply_plan(plan, refunds_confirmed=True)
    agent.paypal.refund_capture.assert_called_once()
    assert res["done"]["refund"] == ["REF1"] and res["done"]["accept"] == []   # not twice
    assert agent.ledger.get("CAP9")["decision"] == "refund"
    assert sum(len(agent.ledger.decisions(f"CAP{i}")) for i in (0, 1, 2, 3, 9)) == 5


def test_apply_plan_leaves_payments_decided_meanwhile(agent):
    plan = agent.plan_queue()
    agent.execute(_alert(agent, "CAP9"), "monitor", seller_confirmed=True)
    res = agent.apply_plan(plan, refunds_confirmed=True)
    agent.paypal.refund_capture.assert_not_called()
    assert "CAP9" in res["skipped"] and agent.ledger.get("CAP9")["decision"] == "monitor"


def test_one_failed_refund_does_not_stop_the_plan(agent):
    agent.paypal.refund_capture.side_effect = RuntimeError("PayPal down")
    res = agent.apply_plan(agent.plan_queue(), refunds_confirmed=True)
    assert res["failed"][0][0] == "CAP9" and len(res["done"]["accept"]) == 4
    assert agent.ledger.get("CAP9")["decision"] is None


def test_queue_summary_template(agent):
    text, source = agent.summarize(agent.plan_queue())
    assert source == "template"
    assert text == "1 payment to refund, 0 to monitor, 4 to approve. Estimated loss avoided: $900.00."


def test_plan_reasons_never_name_the_country(agent):
    plan = agent.plan_queue()
    assert any("cross-border" in t for t in plan.reasons["CAP9"])
    assert not any("DE" in t for t in plan.reasons["CAP9"])


def test_paypal_refusal_is_reported_politely(agent):
    from payguard.paypal_client import PayPalError
    agent.paypal.refund_capture.side_effect = PayPalError(422, {
        "name": "UNPROCESSABLE_ENTITY", "details": [{
            "issue": "REFUND_FAILED_INSUFFICIENT_FUNDS",
            "description": "Capture could not be refunded due to insufficient funds."}]})
    res = agent.apply_plan(agent.plan_queue(), refunds_confirmed=True)
    [(cid, message, issue)] = res["failed"]
    assert (cid, issue) == ("CAP9", "REFUND_FAILED_INSUFFICIENT_FUNDS")
    assert message.startswith("The demo seller's PayPal sandbox balance is too low")
    assert "Nothing was refunded" in message and "{" not in message
    assert agent.ledger.get("CAP9")["decision"] is None


def test_cached_sentence_naming_a_country_is_written_again(agent):
    alert = next(a for a in agent.review() if a.capture_id == "CAP9")
    agent.ledger.cache_explanation("CAP9", alert.score, "Cross-border payment from a payer in DE.", "llm")
    alert = next(a for a in agent.review() if a.capture_id == "CAP9")
    e = agent.explain(alert)
    assert e.source == "template" and "DE" not in e.sentence
    assert "DE" not in agent.ledger.get("CAP9")["explanation"]
