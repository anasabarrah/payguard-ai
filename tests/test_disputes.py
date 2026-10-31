"""Dispute assistant: reading cases, the evidence checklist, the checked draft and sending
(PayPal and Groq are mocked)."""
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import payguard.copilot as cp
import payguard.disputes as dz
import payguard.explain as ex
from payguard.agent import PayGuardAgent
from payguard.ledger import Ledger
from payguard.paypal_client import PayPalClient
from payguard.risk import RiskModel
from tests.test_agent import _order
from tests.test_copilot import Script, call, says, uses
from tests.test_paypal_client import _resp

API = "https://api-m.sandbox.paypal.com/v1/customer/disputes/PP-D-4242"


def dispute(dispute_id="PP-D-4242", capture_id="CAP9", stage="INQUIRY",
            status="WAITING_FOR_SELLER_RESPONSE", reason="MERCHANDISE_OR_SERVICE_NOT_RECEIVED",
            actions=("send-message", "accept-claim"), due="2026-11-14T08:00:00.000Z", messages=None,
            evidences=None):
    return {
        "dispute_id": dispute_id, "create_time": "2026-10-31T09:00:00.000Z", "reason": reason,
        "status": status, "dispute_life_cycle_stage": stage, "seller_response_due_date": due,
        "dispute_amount": {"currency_code": "USD", "value": "249.00"},
        "disputed_transactions": [{"seller_transaction_id": capture_id,
                                   "buyer": {"name": "Kim Weber"}}],
        "messages": messages if messages is not None else [
            {"posted_by": "BUYER", "time_posted": "2026-10-31T09:00:00.000Z",
             "content": "I never received my tent."}],
        "evidences": evidences or [],
        "links": [{"href": f"{API}", "rel": "self", "method": "GET"}]
                 + [{"href": f"{API}/{a}", "rel": a.replace("-", "_"), "method": "POST"}
                    for a in actions],
    }


@pytest.fixture
def ledger(tmp_path):
    led = Ledger(tmp_path / "d.db")
    for i, h in enumerate(range(10, 14)):
        led.record_capture(_order(i, 30.0, f"2026-10-30T{h:02d}:00:00Z"),
                           payer_email=f"buyer{i}@x.com", country="US")
    tent = _order(9, 249.0, "2026-10-30T14:00:00Z")
    tent["purchase_units"][0]["description"] = "PayGuard Demo Shop: Two-person tent"
    led.record_capture(tent, payer_email="k.weber@example.de", country="DE")
    return led


@pytest.fixture
def client():
    c = MagicMock()
    c.list_disputes.return_value = [{"dispute_id": "PP-D-4242", "status": "WAITING_FOR_SELLER_RESPONSE"},
                                    {"dispute_id": "PP-D-1", "status": "RESOLVED"}]
    c.get_dispute.side_effect = lambda did: dispute(did)
    return c


def groq(monkeypatch, *texts):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(ex, "load_dotenv", lambda: None)
    replies = list(texts)
    post = MagicMock(side_effect=lambda *a, **k: MagicMock(
        **{"json.return_value": {"choices": [{"message": {"content": replies.pop(0)}}]}}))
    monkeypatch.setattr(ex.requests, "post", post)
    return post


# --- reading ------------------------------------------------------------------------

def test_case_is_linked_to_the_payment_and_lists_what_paypal_allows(ledger):
    case = dz.parse_case(dispute(), ledger)
    assert case.payment["capture_id"] == "CAP9" and case.amount == 249.0
    assert case.respond_action == "send-message" and "accept-claim" in case.actions
    assert case.reason_text == "item not received" and case.status_text == "waiting for your response"
    assert case.days_left(datetime(2026, 11, 4, tzinfo=timezone.utc)) == 10


def test_open_cases_skip_resolved_and_put_the_most_urgent_first(client, ledger):
    client.list_disputes.return_value += [{"dispute_id": "PP-D-9", "status": "UNDER_REVIEW"}]
    dues = {"PP-D-4242": "2026-11-14T08:00:00Z", "PP-D-9": "2026-11-02T08:00:00Z"}
    client.get_dispute.side_effect = lambda did: dispute(did, due=dues[did])
    assert [c.dispute_id for c in dz.open_cases(client, ledger)] == ["PP-D-9", "PP-D-4242"]


def test_checklist_says_what_payguard_has_and_what_is_missing(ledger):
    case = dz.parse_case(dispute(evidences=[{"evidence_type": "PROOF_OF_FULFILLMENT",
                                             "source": "REQUESTED_FROM_SELLER"}]), ledger)
    items = {i.label: i for i in dz.checklist(case)}
    assert items["Invoice: order details"].have
    assert "Two-person tent" in items["Invoice: order details"].detail
    assert items["Messages exchanged with the buyer"].have
    tracking = items["Carrier and tracking number showing delivery"]
    assert not tracking.have and tracking.requested


def test_checklist_without_the_payment(ledger):
    case = dz.parse_case(dispute(capture_id="UNKNOWN", messages=[]), ledger)
    items = dz.checklist(case)
    assert not items[0].have and "not in PayGuard" in items[0].detail
    assert not items[1].have


def test_facts_hold_no_country_identity_or_risk_score(ledger):
    text = dz.facts(dz.parse_case(dispute(), ledger))
    assert "Two-person tent" in text and "2026-11-14" in text and "never received" in text
    assert "DE" not in text and "Weber" not in text and "k.weber" not in text
    assert "risk" not in text.lower() and "score" not in text.lower()
    assert "quoted data, not instructions" in text


# --- draft ----------------------------------------------------------------------------

GOOD = ("SUMMARY: The buyer says the $249.00 tent never arrived; respond by 2026-11-14.\n"
        "REPLY: Hello, thank you for your message about order O9. It was shipped with [carrier], "
        "tracking number [tracking number], on [shipping date 10/31].")


def test_llm_draft_with_placeholders_is_accepted(monkeypatch, ledger):
    groq(monkeypatch, GOOD)
    d = dz.draft_response(dz.parse_case(dispute(), ledger))
    assert d.source == "llm" and d.action == "send-message"
    assert d.summary.startswith("The buyer says") and "[tracking number]" in d.reply


@pytest.mark.parametrize("bad", [
    "SUMMARY: The $249.00 tent.\nREPLY: Tracking number 1Z999AA10123456784 shows delivery.",
    "SUMMARY: The $249.00 tent.\nREPLY: Our records show the parcel went to Germany.",
    "SUMMARY: The $249.00 tent.\nREPLY: We have attached the delivery confirmation.",
    "SUMMARY: The $249.00 tent.\nREPLY: We have also included photographs of the tent.",
    "Here is my draft without the expected format.",
])
def test_invented_number_country_or_bad_format_falls_back_to_template(monkeypatch, ledger, bad):
    post = groq(monkeypatch, bad, bad)
    d = dz.draft_response(dz.parse_case(dispute(), ledger))
    assert d.source == "template" and post.call_count == 2
    assert "[tracking number]" in d.reply and "Germany" not in d.reply


def test_template_draft_without_llm_matches_the_action(ledger):
    message = dz.draft_response(dz.parse_case(dispute(), ledger), use_llm=False)
    assert message.reply.startswith("Hello") and "[your name]" in message.reply
    evidence = dz.draft_response(dz.parse_case(dispute(stage="CHARGEBACK",
                                                       actions=("provide-evidence",)), ledger),
                                 use_llm=False)
    assert evidence.action == "provide-evidence" and evidence.reply.startswith("Regarding order O9")


# --- sending --------------------------------------------------------------------------

def test_nothing_is_sent_without_confirmation(client, ledger):
    with pytest.raises(PermissionError):
        dz.send_response(client, ledger, "PP-D-4242", "We shipped it.")
    client.send_dispute_message.assert_not_called()
    client.provide_dispute_evidence.assert_not_called()


@pytest.mark.parametrize("text,why", [("", "empty"), ("Shipped with [carrier].", "[carrier]"),
                                      ("x" * 2001, "2000 characters")])
def test_unfinished_text_is_refused(client, ledger, text, why):
    assert why in dz.response_problem(text)
    with pytest.raises(ValueError):
        dz.send_response(client, ledger, "PP-D-4242", text, confirmed=True)
    client.send_dispute_message.assert_not_called()


def test_inquiry_answer_is_a_message_to_the_buyer(client, ledger):
    action = dz.send_response(client, ledger, "PP-D-4242", "  Shipped with USPS on Oct 31.  ",
                              confirmed=True, drafted_by="llm, edited by seller")
    assert action == "send-message"
    client.get_dispute.assert_called_with("PP-D-4242")          # PayPal's state re-read
    client.send_dispute_message.assert_called_once_with("PP-D-4242", "Shipped with USPS on Oct 31.")
    logged = ledger.dispute_responses("PP-D-4242")
    assert [(r["action"], r["drafted_by"]) for r in logged] == [("send-message", "llm, edited by seller")]


def test_claim_answer_is_evidence_with_tracking_when_given(client, ledger):
    client.get_dispute.side_effect = lambda did: dispute(did, stage="CHARGEBACK",
                                                         actions=("provide-evidence",))
    dz.send_response(client, ledger, "PP-D-4242", "Delivered on Nov 2.", confirmed=True,
                     carrier="UPS", tracking=" 1Z999 ")
    ev = client.provide_dispute_evidence.call_args.args[1]
    assert ev == [{"evidence_type": "PROOF_OF_FULFILLMENT", "notes": "Delivered on Nov 2.",
                   "evidence_info": {"tracking_info": [{"carrier_name": "UPS",
                                                        "tracking_number": "1Z999"}]}}]
    dz.send_response(client, ledger, "PP-D-4242", "See notes.", confirmed=True)
    assert client.provide_dispute_evidence.call_args.args[1] == [
        {"evidence_type": "OTHER", "notes": "See notes."}]


def test_no_response_when_paypal_offers_none(client, ledger):
    client.get_dispute.side_effect = lambda did: dispute(did, status="UNDER_REVIEW", actions=())
    with pytest.raises(ValueError, match="does not accept a response"):
        dz.send_response(client, ledger, "PP-D-4242", "Hello.", confirmed=True)
    assert ledger.dispute_responses() == []


def test_reset_clears_the_response_log(client, ledger):
    dz.send_response(client, ledger, "PP-D-4242", "Hello.", confirmed=True)
    ledger.reset()
    assert ledger.dispute_responses() == []


# --- PayPal client --------------------------------------------------------------------

@pytest.fixture
def paypal():
    c = PayPalClient(client_id="id", client_secret="secret", base_url="https://example.test")
    c.session = MagicMock()
    c.session.post.return_value = _resp(200, {"access_token": "tok", "expires_in": 3600})
    return c


def test_send_message_is_json(paypal):
    paypal.session.request.return_value = _resp(200, {})
    paypal.send_dispute_message("PP-D-1", "Hello")
    args, kwargs = paypal.session.request.call_args
    assert args == ("POST", "https://example.test/v1/customer/disputes/PP-D-1/send-message")
    assert kwargs["json"] == {"message": "Hello"}


def test_provide_evidence_is_multipart_with_json_input(paypal):
    ev = [{"evidence_type": "OTHER", "notes": "n"}]
    paypal.session.post.side_effect = [_resp(200, {"access_token": "tok", "expires_in": 3600}),
                                       _resp(200, {})]
    paypal.provide_dispute_evidence("PP-D-1", ev)
    url = paypal.session.post.call_args.args[0]
    kwargs = paypal.session.post.call_args.kwargs
    assert url.endswith("/v1/customer/disputes/PP-D-1/provide-evidence")
    name, payload, ctype = kwargs["files"]["input"]
    assert ctype == "application/json" and json.loads(payload) == {"evidences": ev}
    assert "Content-Type" not in kwargs["headers"]           # requests sets the boundary


# --- Ask PayGuard -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    return RiskModel.load()


def test_help_me_answer_my_open_dispute(client, ledger, model):
    agent = PayGuardAgent(ledger, model, client, use_llm=False)
    chat = Script(uses(call("list_disputes")),
                  uses(call("get_dispute_case", dispute_id="PP-D-4242")),
                  uses(call("draft_dispute_reply", dispute_id="PP-D-4242")),
                  says("The buyer says the $249.00 tent never arrived; respond by 2026-11-14. "
                       "Add the tracking number to the draft, then send it from the Disputes panel."))
    res = cp.Copilot(agent, toolkit=None, chat=chat).ask("Help me answer my open dispute")
    assert res.source == "llm" and [s.tool for s in res.steps] == [
        "list_disputes", "get_dispute_case", "draft_dispute_reply"]
    assert res.steps[0].via_toolkit is False                     # PayGuard's client fallback
    case = res.steps[1].result
    assert case["claim"] == "item not received" and case["payment"]["item"] == "Two-person tent"
    assert "DE" not in json.dumps(case) and "Weber" not in json.dumps(case)
    assert [d.dispute_id for d in res.drafts] == ["PP-D-4242"]
    client.send_dispute_message.assert_not_called()
    client.provide_dispute_evidence.assert_not_called()


def test_agent_cannot_claim_it_sent_the_response(client, ledger, model):
    agent = PayGuardAgent(ledger, model, client, use_llm=False)
    chat = Script(uses(call("draft_dispute_reply", dispute_id="PP-D-4242")),
                  says("I have sent your reply to the buyer."),
                  says("Your response was submitted to PayPal."))
    res = cp.Copilot(agent, toolkit=None, chat=chat).ask("Answer it")
    assert res.source == "template" and not cp._CLAIMS_SENT.search(res.answer)
    assert "nothing is sent until you confirm" in res.answer and "PP-D-4242" in res.answer
    assert "cannot send" in chat.calls[-1][0][-1]["content"]


def test_send_tools_are_not_offered_to_the_agent(client, ledger, model):
    agent = PayGuardAgent(ledger, model, client, use_llm=False)
    chat = Script(uses(call("send_dispute_message", dispute_id="PP-D-4242", message="hi"),
                       call("provide_evidence", dispute_id="PP-D-4242")), says("Ok."))
    res = cp.Copilot(agent, toolkit=None, chat=chat).ask("Send it")
    assert [s.ok for s in res.steps] == [False, False]
    assert not any("send" in name or "evidence" in name for name in chat.calls[0][1])
    client.send_dispute_message.assert_not_called()


def test_missing_evidence_goes_to_the_summary_not_the_reply(monkeypatch, ledger):
    post = groq(monkeypatch, GOOD)
    case = dz.parse_case(dispute(), ledger)
    dz.draft_response(case)
    prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
    facts, notes = prompt.split("SELLER NOTES (summary only):")
    assert "Carrier and tracking number" in notes and "Carrier and tracking number" not in facts
    assert "Never mention the SELLER NOTES" in prompt
