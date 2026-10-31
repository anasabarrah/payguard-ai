"""Tests for SHAP reasons and the grounded sentence (Groq is mocked)."""
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

import payguard.explain as ex
from payguard.explain import Reason, describe, explain, is_grounded, numbers_in, top_reasons
from payguard.features import FEATURES, features_from_payments
from payguard.risk import RiskModel

REASONS = [
    Reason("amount_ratio", 22.5, 2.9, describe("amount_ratio", 22.5)),
    Reason("cross_border", 1, 2.8, describe("cross_border", 1)),
    Reason("payer_is_new", 0, -0.3, describe("payer_is_new", 0)),
]


def test_numbers_in_reads_digits_and_words():
    assert numbers_in("12.5x, twenty and 1,000 at 04:00") == {12.5, 20.0, 1000.0, 4.0, 0.0}
    assert numbers_in("the payer has paid before") == set()


def test_grounding():
    assert is_grounded("Score 99.4 because the amount is 22.5x the usual ticket.", 99.4, REASONS)
    assert not is_grounded("Score 99 because the amount is 22.5x the usual ticket.", 99.4, REASONS)
    assert not is_grounded("The amount is twenty-three times the usual ticket.", 99.4, REASONS)


def _groq_returns(monkeypatch, text):
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(ex, "load_dotenv", lambda: None)
    resp = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": text}}]}
    post = MagicMock(return_value=resp)
    monkeypatch.setattr(ex.requests, "post", post)
    return post


def test_llm_sentence_used_when_grounded(monkeypatch):
    _groq_returns(monkeypatch, "Risk 99.4: the amount is 22.5x usual and the payment is cross-border.")
    e = explain(99.4, "high", REASONS)
    assert e.source == "llm" and "22.5x" in e.sentence


def test_invented_number_falls_back_to_template(monkeypatch):
    post = _groq_returns(monkeypatch, "Risk 99.4: the amount is 30x usual, $5,000 total.")
    e = explain(99.4, "high", REASONS)
    assert e.source == "template" and post.call_count == 2  # one retry, then template
    assert is_grounded(e.sentence, 99.4, REASONS)
    assert "22.5x" in e.sentence and "cross-border payment." in e.sentence
    assert "DE" not in e.sentence
    assert "paid before" not in e.sentence


def test_no_key_or_api_error_falls_back(monkeypatch):
    monkeypatch.setattr(ex, "load_dotenv", lambda: None)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert explain(99.4, "high", REASONS).source == "template"
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setattr(ex.requests, "post", MagicMock(side_effect=requests.ConnectionError))
    assert explain(99.4, "high", REASONS).source == "template"


def test_low_risk_template():
    reasons = [Reason("amount_ratio", 1.0, -0.2, describe("amount_ratio", 1.0))]
    assert "nothing in this payment stands out" in explain(12.0, "low", reasons, use_llm=False).sentence


@pytest.fixture(scope="module")
def model():
    return RiskModel.load()


def test_top_reasons_match_shap(model):
    pays = [{"created_at": f"2026-10-30T{h:02d}:00:00Z", "amount": 20.0,
             "payer_email": "a@x.com", "country": "US"} for h in range(9, 13)]
    pays.append({"created_at": "2026-10-30T13:00:00Z", "amount": 450.0,
                 "payer_email": "n@z.com", "country": "AU"})
    f = features_from_payments(pays)
    contrib = model.booster.predict(f[FEATURES], pred_contrib=True)
    assert np.allclose(contrib.sum(1), model.booster.predict(f[FEATURES], raw_score=True))
    reasons = top_reasons(model, f)[-1]
    assert 1 <= len(reasons) <= 3 and all(r.shap != 0 for r in reasons)
    assert {r.feature for r in reasons[:2]} == {"amount_ratio", "cross_border"}
    assert any(r.text == "This is a cross-border payment" for r in reasons)
    assert not any(ex.names_country(r.text) for r in reasons)
    assert all(r.raises_risk for r in reasons[:2])


# --- queue summary (step 3) ----------------------------------------------------

COUNT = {"refund": 2, "monitor": 2, "accept": 6}
TOTAL = {"refund": 2614.1, "monitor": 461.22, "accept": 270.11}


def test_queue_summary_uses_grounded_llm_text(monkeypatch):
    import payguard.explain as ex
    text = ("I propose 2 payments to refund, 2 to monitor and 6 to approve, avoiding an "
            "estimated $2,614.10 loss; the refunds are 27.3x the usual ticket.")
    monkeypatch.setattr(ex, "llm_complete", lambda *a, **k: text)
    out, source = ex.queue_summary(COUNT, TOTAL, {"C1": ["The amount is 27.3x the median payment"]})
    assert (out, source) == (text, "llm")


def test_queue_summary_rejects_invented_numbers(monkeypatch):
    import payguard.explain as ex
    calls = []
    monkeypatch.setattr(ex, "llm_complete",
                        lambda *a, **k: calls.append(1) or "Refund 2 payments to save about $2,600.")
    out, source = ex.queue_summary(COUNT, TOTAL)
    assert source == "template" and len(calls) == 2
    assert out == "2 payments to refund, 2 to monitor, 6 to approve. Estimated loss avoided: $2,614.10."


def test_queue_facts_list_every_number():
    from payguard.explain import numbers_in, queue_facts
    facts = queue_facts(COUNT, TOTAL)
    assert {10, 2, 6, 2614.1, 461.22, 270.11} <= numbers_in(facts)


def test_sentence_naming_the_wrong_risk_level_is_rejected():
    ok = "Score 99.4 because the amount is 22.5x the usual ticket."
    assert is_grounded(ok, 99.4, REASONS, "high")
    assert is_grounded("A high-risk score of 99.4.", 99.4, REASONS, "high")
    assert not is_grounded("The payment got a high risk score of 99.4.", 99.4, REASONS, "medium")
    assert not is_grounded("A low score of 99.4.", 99.4, REASONS, "high")


def test_llm_sentence_with_wrong_level_falls_back_to_template(monkeypatch):
    import payguard.explain as ex
    monkeypatch.setattr(ex, "llm_complete", lambda *a, **k: "It got a high risk score of 99.4.")
    e = ex.explain(99.4, "medium", REASONS)
    assert e.source == "template" and "(medium)" in e.sentence


# --- fairness: a sentence says "cross-border", never which country ----------------

def test_sentence_naming_the_payer_country_falls_back_to_template(monkeypatch):
    post = _groq_returns(monkeypatch, "Risk 99.4 (high): the amount is 22.5x the median payment "
                                      "and it is a cross-border payment from a payer in IN.")
    e = explain(99.4, "high", REASONS)
    assert e.source == "template" and post.call_count == 2
    assert "IN" not in e.sentence.split() and "cross-border payment" in e.sentence


def test_facts_sent_to_the_llm_contain_no_country(monkeypatch):
    post = _groq_returns(monkeypatch, "Risk 99.4: 22.5x the median payment, cross-border.")
    e = explain(99.4, "high", REASONS)
    prompt = post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "DE" not in prompt and "payer in" not in prompt and "cross-border payment" in prompt
    assert e.source == "llm"


@pytest.mark.parametrize("text", [
    "a cross-border payment from a payer in IN",
    "The buyer is from Germany.",
    "an Indian customer",
    "the card was issued in the US",
    "a payment from GB",
])
def test_names_country_detects_countries(text):
    assert ex.names_country(text)


@pytest.mark.parametrize("text", [
    "Risk score 99.4/100 (high) mainly because the amount is 22.5x the median payment of "
    "the previous 24 hours and this is a cross-border payment.",
    "This is the first payment from this payer; it is a domestic payment.",
    "PayGuard AI flags it in the SHAP reasons, so hold it in your queue.",
    "Monitor it: in doubt, wait before shipping.",
    "The tent purchase, PayPal order ID 5RM13586NU013182A, was paid on 2026-10-31.",
])
def test_names_country_allows_neutral_sentences(text):
    assert not ex.names_country(text)


def test_queue_summary_naming_a_country_is_rejected(monkeypatch):
    _groq_returns(monkeypatch, "Refund 1 payment from Australia, $1,000.00.")
    count, total = {"refund": 1, "monitor": 0, "accept": 0}, {"refund": 1000.0, "monitor": 0.0, "accept": 0.0}
    text, source = ex.queue_summary(count, total)
    assert source == "template" and "Australia" not in text
