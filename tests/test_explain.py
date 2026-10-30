"""Tests for SHAP reasons and the grounded sentence (Groq is mocked)."""
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

import payguard.explain as ex
from payguard.explain import Reason, describe, explain, is_grounded, numbers_in, top_reasons
from payguard.features import features_from_payments
from payguard.risk import RiskModel

REASONS = [
    Reason("amount_ratio", 22.5, 2.9, describe("amount_ratio", 22.5)),
    Reason("country", "NG", 2.8, describe("country", "NG")),
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
    _groq_returns(monkeypatch, "Risk 99.4: the amount is 22.5x usual and the payer is in NG.")
    e = explain(99.4, "high", REASONS)
    assert e.source == "llm" and "22.5x" in e.sentence


def test_invented_number_falls_back_to_template(monkeypatch):
    post = _groq_returns(monkeypatch, "Risk 99.4: the amount is 30x usual, $5,000 total.")
    e = explain(99.4, "high", REASONS)
    assert e.source == "template" and post.call_count == 2  # one retry, then template
    assert is_grounded(e.sentence, 99.4, REASONS)
    assert "22.5x" in e.sentence and "NG" in e.sentence and "paid before" not in e.sentence


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
                 "payer_email": "n@z.com", "country": "NG"})
    f = features_from_payments(pays)
    contrib = model.booster.predict(f, pred_contrib=True)
    assert np.allclose(contrib.sum(1), model.booster.predict(f, raw_score=True))
    reasons = top_reasons(model, f)[-1]
    assert 1 <= len(reasons) <= 3 and all(r.shap != 0 for r in reasons)
    assert {r.feature for r in reasons[:2]} == {"amount_ratio", "country"}
    assert all(r.raises_risk for r in reasons[:2])
