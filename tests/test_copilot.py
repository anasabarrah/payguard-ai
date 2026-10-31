"""Ask PayGuard: the tool loop and its guardrails (Groq and PayPal are scripted)."""
import json
from unittest.mock import MagicMock

import pytest

import payguard.copilot as cp
from payguard.agent import PayGuardAgent
from payguard.ledger import Ledger
from payguard.risk import RiskModel
from tests.test_agent import _order


def call(name, **args):
    return {"id": f"call_{name}_{len(json.dumps(args))}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def says(text):
    return {"role": "assistant", "content": text}


def uses(*calls):
    return {"role": "assistant", "content": "", "tool_calls": list(calls)}


class Script:
    """Fake chat: returns the scripted messages in order and records what it was sent."""
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, messages, tools):
        self.calls.append((json.loads(json.dumps(messages)), [t["function"]["name"] for t in tools]))
        return self.replies.pop(0)


@pytest.fixture(scope="module")
def model():
    return RiskModel.load()


@pytest.fixture
def agent(tmp_path, model):
    led = Ledger(tmp_path / "c.db")
    for i, h in enumerate(range(10, 14)):
        led.record_capture(_order(i, 30.0, f"2026-10-30T{h:02d}:00:00Z"),
                           payer_email=f"buyer{i}@x.com", country="US")
    tent = _order(9, 900.0, "2026-10-30T14:00:00Z")
    tent["purchase_units"][0]["description"] = "PayGuard Demo Shop: Two-person tent"
    led.record_capture(tent, payer_email="k.weber@example.de", country="DE")
    paypal = MagicMock()
    return PayGuardAgent(led, model, paypal, use_llm=False)


def risk_of(agent, cid):
    return next(a for a in agent.review() if a.capture_id == cid)


def test_agent_calls_tools_and_only_proposes_refund(agent):
    high = risk_of(agent, "CAP9")
    chat = Script(uses(call("list_payments", which="flagged")),
                  uses(call("get_payment_risk", capture_id="CAP9")),
                  uses(call("prepare_refund", capture_id="CAP9", reason="High risk")),
                  says(f"The $900.00 tent payment scores {high.score:.1f}: the amount is 30.0x "
                       "the median payment and it is cross-border. I prepared a refund for you "
                       "to confirm."))
    seen = []
    res = cp.Copilot(agent, chat=chat).ask("Protect me from fraud", on_step=seen.append)
    assert res.source == "llm" and [s.tool for s in res.steps] == [
        "list_payments", "get_payment_risk", "prepare_refund"] and seen == res.steps
    assert [(p.capture_id, p.amount) for p in res.proposals] == [("CAP9", 900.0)]
    agent.paypal.refund_capture.assert_not_called()          # no money moved
    assert agent.ledger.get("CAP9")["decision"] is None
    risk = res.steps[1].result
    assert risk["item"] == "Two-person tent" and risk["cross_border"] is True
    assert {r["reason"] for r in risk["reasons"]} >= {"This is a cross-border payment"}


def test_tool_results_never_name_the_country_or_payer(agent):
    chat = Script(uses(call("list_payments"), call("get_payment_risk", capture_id="CAP9")),
                  says("Done."))
    res = cp.Copilot(agent, chat=chat).ask("List my payments")
    sent = json.dumps([m for m, _ in chat.calls][-1])
    assert "DE" not in json.dumps([s.result for s in res.steps])
    assert "k.weber" not in sent and "buyer0" not in sent


def test_invented_number_gets_a_retry_then_template(agent):
    chat = Script(uses(call("get_payment_risk", capture_id="CAP9")),
                  says("This payment of $5,000 is risky."),
                  says("Still $5,000."))
    res = cp.Copilot(agent, chat=chat).ask("Why is CAP9 flagged?")
    assert res.source == "template" and "5,000" not in res.answer
    assert "$900.00" in res.answer and "30.0x" in res.answer
    assert "[5000.0]" in chat.calls[-1][0][-1]["content"]      # the retry names the number


def test_retry_can_fix_the_answer(agent):
    chat = Script(uses(call("get_payment_risk", capture_id="CAP9")),
                  says("This payment of $5,000 is risky."),
                  says("This $900.00 payment is 30.0x the median payment."))
    res = cp.Copilot(agent, chat=chat).ask("Why?")
    assert res.source == "llm" and res.answer.startswith("This $900.00")


def test_claiming_a_refund_that_was_not_prepared_is_rejected(agent):
    chat = Script(uses(call("list_payments")),
                  says("I have prepared a refund for the risky payment."),
                  says("Refund prepared."))
    res = cp.Copilot(agent, chat=chat).ask("Protect me")
    assert res.source == "template" and not res.proposals
    assert "prepare_refund was never called" in chat.calls[-1][0][-1]["content"]


@pytest.mark.parametrize("name", ["pay_order", "create_order", "refund_capture", "accept_dispute_claim"])
def test_tools_outside_the_allowlist_never_run(agent, name):
    toolkit = MagicMock()
    chat = Script(uses(call(name, order_id="X", capture_id="CAP9")), says("Ok."))
    res = cp.Copilot(agent, toolkit=toolkit, chat=chat).ask("Pay it")
    assert res.steps[0].ok is False and "unknown tool" in res.steps[0].result["error"]
    toolkit.assert_not_called()
    agent.paypal.refund_capture.assert_not_called()


def test_toolkit_runner_refuses_money_moving_methods(monkeypatch):
    pytest.importorskip("paypal_agent_toolkit")
    monkeypatch.setenv("PAYPAL_CLIENT_ID", "id")
    monkeypatch.setenv("PAYPAL_CLIENT_SECRET", "secret")
    run = cp.toolkit_runner()
    with pytest.raises(PermissionError):
        run("pay_order", {"order_id": "X"})


def test_toolkit_offers_the_tools_we_use():
    """Guards against a toolkit upgrade renaming the methods PayGuard calls."""
    pytest.importorskip("paypal_agent_toolkit")
    from paypal_agent_toolkit.shared.tools import tools
    assert set(cp.TOOLKIT_TOOLS) <= {t["method"] for t in tools}


def test_without_toolkit_its_tools_are_not_offered(agent):
    chat = Script(says("Hello."))
    cp.Copilot(agent, toolkit=None, chat=chat).ask("Hi")
    offered = chat.calls[0][1]
    assert "get_order_details" not in offered and "prepare_refund" in offered


def test_order_details_from_toolkit_are_stripped_of_payer_data(agent):
    raw = {"message": "The PayPal order 1AB has been successfully captured.", "raw": {
        "id": "1AB", "status": "COMPLETED", "intent": "CAPTURE",
        "payer": {"email_address": "buyer@x.com", "name": {"given_name": "Kim"},
                  "address": {"country_code": "DE"}},
        "purchase_units": [{"description": "PayGuard Demo Shop: Two-person tent",
                            "amount": {"value": "249.00", "currency_code": "USD"},
                            "shipping": {"address": {"country_code": "DE"}},
                            "payments": {"captures": [{"id": "C1", "status": "COMPLETED",
                                                       "amount": {"value": "249.00"}}]}}]}}
    toolkit = MagicMock(return_value=json.dumps(raw))
    chat = Script(uses(call("get_order_details", order_id="O9")), says("It is completed."))
    res = cp.Copilot(agent, toolkit=toolkit, chat=chat).ask("Show the tent order")
    toolkit.assert_called_once_with("get_order_details", {"order_id": "O9"})
    step = res.steps[0]
    assert step.via_toolkit and step.result["status"] == "COMPLETED"
    text = json.dumps(step.result)
    assert "buyer@x.com" not in text and "Kim" not in text and "DE" not in text
    assert "Two-person tent" in cp.step_summary(step)


def test_step_and_tool_call_limits(agent, monkeypatch):
    endless = Script(*[uses(call("list_payments"), call("list_payments", which="pending"))
                       for _ in range(cp.MAX_STEPS)])
    res = cp.Copilot(agent, chat=endless).ask("Loop forever")
    assert len(endless.calls) == cp.MAX_STEPS
    assert len([s for s in res.steps if s.ok]) <= cp.MAX_TOOL_CALLS
    assert endless.calls[-1][1] == []         # last call: tools withdrawn, must answer


def test_refund_proposals_are_checked(agent):
    agent.ledger.record_refund("CAP0", {"id": "R0", "amount": {"value": "30.00"}})
    agent.ledger.record_decision("CAP0", "refund")
    chat = Script(uses(call("prepare_refund", capture_id="CAP0", reason="x"),
                       call("prepare_refund", capture_id="NOPE", reason="x"),
                       call("prepare_refund", capture_id="CAP9", reason="x"),
                       call("prepare_refund", capture_id="CAP9", reason="again")),
                  says("Done."))
    res = cp.Copilot(agent, chat=chat).ask("Refund")
    assert [s.ok for s in res.steps] == [False, False, True, True]
    assert "already refunded" in res.steps[0].result["error"]
    assert [p.capture_id for p in res.proposals] == ["CAP9"]


def test_bad_arguments_are_reported_not_raised(agent):
    bad = {"id": "c1", "type": "function", "function": {"name": "get_payment_risk",
                                                        "arguments": "{not json"}}
    chat = Script(uses(bad, call("get_payment_risk", capture_id="CAP9", extra=1)), says("Sorry."))
    res = cp.Copilot(agent, chat=chat).ask("?")
    assert [s.ok for s in res.steps] == [False, False]


def test_llm_down_falls_back_to_tool_results(agent):
    def chat(messages, tools):
        if len(messages) == 2:
            return uses(call("list_payments"))
        raise cp.requests.ConnectionError("down")
    res = cp.Copilot(agent, chat=chat).ask("Overview")
    assert res.source == "template" and "5 payments" in res.answer and "unavailable" in res.stopped


def test_doubled_answer_is_kept_once():
    assert cp._dedupe("All good.All good.") == "All good."
    assert cp._dedupe("All good.") == "All good."


def test_groq_rate_limit_is_retried(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(cp, "load_dotenv", lambda: None)
    limited = MagicMock(status_code=429, headers={},
                        text='{"error":{"message":"Please try again in 427.5ms."}}')
    ok = MagicMock(status_code=200)
    ok.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
    post = MagicMock(side_effect=[limited, ok])
    sleeps = []
    monkeypatch.setattr(cp.requests, "post", post)
    monkeypatch.setattr(cp.time, "sleep", sleeps.append)
    assert cp.groq_chat([], [])["content"] == "hi"
    assert sleeps == [0.5] and post.call_count == 2


def test_item_of_demo_and_shop_orders():
    assert cp.item_of(json.dumps({"purchase_units": [{"description": "PayGuard demo: new foreign buyer"}]})) == "demo order"
    assert cp.item_of(json.dumps({"purchase_units": [{"description": "PayGuard Demo Shop: Merino beanie"}]})) == "Merino beanie"
    assert cp.item_of(None) is None


def test_answer_naming_a_country_is_rejected(agent):
    chat = Script(uses(call("get_payment_risk", capture_id="CAP9")),
                  says("The $900.00 payment comes from a payer in Germany."),
                  says("The $900.00 payment is from DE."))
    res = cp.Copilot(agent, chat=chat).ask("Why?")
    assert res.source == "template" and "Germany" not in res.answer and "DE" not in res.answer
    assert "countries" in chat.calls[-1][0][-1]["content"]
