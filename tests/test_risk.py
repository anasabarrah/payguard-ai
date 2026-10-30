"""Tests for feature engineering and the saved risk model (no dataset needed)."""
import numpy as np
import pandas as pd
import pytest

from payguard.features import (COUNTRIES, FEATURES, build_paysim_features,
                               features_from_payments, simulate_country)
from payguard.risk import RiskModel, to_score


def test_paysim_payer_history_uses_only_past():
    raw = pd.DataFrame({
        "step":     [1, 1, 5, 40, 41],
        "type":     ["PAYMENT", "TRANSFER", "CASH_OUT", "TRANSFER", "TRANSFER"],
        "amount":   [10.0, 100.0, 200.0, 300.0, 400.0],
        "nameOrig": ["A", "B", "A", "A", "C"],
        "isFraud":  [0, 0, 1, 0, 0],
    })
    f = build_paysim_features(raw)
    assert list(f.columns) == FEATURES + ["step", "isFraud"]
    assert len(f) == 4  # PAYMENT rows are dropped after payer history is computed
    a5 = f[f.step == 5].iloc[0]
    assert a5.payer_txn_24h == 1 and a5.payer_is_new == 0     # saw A's step-1 PAYMENT
    a40 = f[f.step == 40].iloc[0]
    assert a40.payer_txn_24h == 0 and a40.payer_is_new == 0   # history is >24h old
    assert f[f.step == 1].iloc[0].payer_is_new == 1
    assert f[f.step == 5].iloc[0].amount_ratio == pytest.approx(2.0)  # 200 / median(100)


def test_simulated_country_is_reproducible_and_skewed():
    y = np.r_[np.zeros(20000), np.ones(20000)]
    c1, c2 = simulate_country(y, seed=1), simulate_country(y, seed=1)
    assert (c1 == c2).all() and set(c1) <= set(COUNTRIES)
    assert (c1[:20000] == "US").mean() > (c1[20000:] == "US").mean() + 0.2


def test_features_from_payments():
    pays = [
        {"created_at": "2026-10-30T10:00:00Z", "amount": 10.0, "payer_email": "a@x.com", "country": "US"},
        {"created_at": "2026-10-30T11:00:00Z", "amount": 30.0, "payer_email": "A@x.com", "country": "us"},
        {"created_at": "2026-10-30T12:00:00Z", "amount": 200.0, "payer_email": "b@x.com", "country": "ZZ"},
    ]
    f = features_from_payments(pays)
    assert list(f.columns) == FEATURES
    assert f.payer_is_new.tolist() == [1, 0, 1]
    assert f.payer_txn_24h.tolist() == [0, 1, 0]
    assert f.amount_ratio.tolist() == pytest.approx([1.0, 3.0, 10.0])  # 200 / median(10, 30)
    assert f.country.tolist() == ["US", "US", "OTHER"]
    assert f.hour.tolist() == [10, 11, 12]


def test_to_score_is_monotonic_and_bounded():
    ref = {"levels": [0.0, 0.5, 0.9, 1.0], "probs": [0.0, 0.0, 0.1, 0.8]}
    s = to_score([0.0, 0.05, 0.1, 0.5, 2.0], ref)
    assert (np.diff(s) >= 0).all() and s[0] == 50.0 and s[-1] == 100.0


@pytest.fixture(scope="module")
def model():
    return RiskModel.load()


def test_saved_model_scores_payments(model):
    assert model.thresholds["monitor"] < model.thresholds["refund"]
    base = {"payer_email": "x@y.com", "country": "US"}
    pays = [{**base, "created_at": f"2026-10-30T{h:02d}:00:00Z", "amount": 20.0} for h in range(10, 20)]
    pays.append({"created_at": "2026-10-31T04:00:00Z", "amount": 900.0,
                 "payer_email": "new@z.com", "country": "NG"})
    out = model.score_payments(pays)
    assert out.risk_score.between(0, 100).all()
    assert out.risk_score.iloc[-1] > out.risk_score.iloc[:-1].max()
    assert out.risk_level.iloc[-1] in {"medium", "high"}


def test_saved_model_is_small():
    from payguard.risk import MODEL_DIR
    assert (MODEL_DIR / "payguard_lgbm.txt").stat().st_size < 10_000_000
