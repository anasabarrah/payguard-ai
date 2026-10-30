"""Feature engineering shared by training (PaySim) and inference (PayPal ledger).

Only features a PayPal seller can compute from their own payment history are
used, so the model trained on PaySim can score sandbox payments:

    amount_ratio    payment amount / median ticket over the previous 24h
    payer_txn_24h   payments by the same payer in the previous 24h
    payer_is_new    1 if the payer has never paid before
    country         payer country (simulated for PaySim, see simulate_country)
    hour            hour of day (0-23)

Balances are deliberately excluded: a seller never sees the payer's balance.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

FEATURES = ["amount_ratio", "payer_txn_24h", "payer_is_new", "country", "hour"]
CATEGORICAL = ["country"]
WINDOW_HOURS = 24
SELLER_COUNTRY = "US"

# Payer-country distributions for a US seller. PaySim has no geography, so the
# country is simulated. The shares are assumptions, not measured data: most
# legitimate buyers are domestic, while fraud is more often cross-border.
# README "Data & limitations" explains this and reports metrics without it.
COUNTRY_PROBS = {
    #         legit   fraud
    "US":    (0.80,   0.50),
    "CA":    (0.05,   0.04),
    "GB":    (0.04,   0.05),
    "FR":    (0.02,   0.03),
    "DE":    (0.02,   0.03),
    "MX":    (0.02,   0.04),
    "BR":    (0.01,   0.06),
    "IN":    (0.01,   0.05),
    "NG":    (0.005,  0.06),
    "RO":    (0.005,  0.05),
    "OTHER": (0.02,   0.09),
}
COUNTRIES = list(COUNTRY_PROBS)


def simulate_country(is_fraud: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    legit_p = np.array([p[0] for p in COUNTRY_PROBS.values()])
    fraud_p = np.array([p[1] for p in COUNTRY_PROBS.values()])
    is_fraud = np.asarray(is_fraud).astype(bool)
    out = np.empty(len(is_fraud), dtype=object)
    out[~is_fraud] = rng.choice(COUNTRIES, size=(~is_fraud).sum(), p=legit_p / legit_p.sum())
    out[is_fraud] = rng.choice(COUNTRIES, size=is_fraud.sum(), p=fraud_p / fraud_p.sum())
    return out


def as_category(country: pd.Series) -> pd.Series:
    c = country.fillna("OTHER").str.upper()
    c = c.where(c.isin(COUNTRIES), "OTHER")
    return pd.Categorical(c, categories=COUNTRIES)


# --- PaySim (training) --------------------------------------------------------

def build_paysim_features(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """df: raw PaySim columns. Returns FEATURES + step, isFraud for TRANSFER/CASH_OUT rows.

    One PaySim step is one hour. Payer history uses every transaction type;
    only strictly earlier steps count, so no information leaks from the future.
    """
    df = df[["step", "type", "amount", "nameOrig", "isFraud"]].copy()

    # Payer history: payers appear at most a handful of times, so shifting is cheap.
    df = df.sort_values(["nameOrig", "step"], kind="stable")
    g = df.groupby("nameOrig", sort=False)["step"]
    txn_24h = np.zeros(len(df), dtype=np.int32)
    seen = np.zeros(len(df), dtype=bool)
    for k in range(1, int(g.size().max())):
        prev = g.shift(k).to_numpy()
        gap = df["step"].to_numpy() - prev
        earlier = ~np.isnan(prev) & (gap > 0)
        seen |= earlier
        txn_24h += earlier & (gap <= WINDOW_HOURS)
    df["payer_txn_24h"] = txn_24h
    df["payer_is_new"] = (~seen).astype(np.int8)
    df = df.sort_index()

    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

    # Median ticket over the previous 24 steps (the "seller's usual ticket").
    by_step = df.groupby("step")["amount"].apply(np.asarray)
    steps = by_step.index.to_numpy()
    median_prev = {}
    for i, s in enumerate(steps):
        lo = np.searchsorted(steps, s - WINDOW_HOURS)
        window = by_step.iloc[lo:i]
        median_prev[s] = np.median(np.concatenate(window.to_list())) if len(window) else np.nan
    med = df["step"].map(median_prev)
    med = med.fillna(df.loc[df["step"] == steps[0], "amount"].median())
    df["amount_ratio"] = df["amount"] / med

    df["hour"] = (df["step"] % 24).astype(np.int8)
    df["country"] = as_category(pd.Series(simulate_country(df["isFraud"].to_numpy(), seed),
                                          index=df.index))
    return df[FEATURES + ["step", "isFraud"]].reset_index(drop=True)


# --- PayPal ledger (inference) ------------------------------------------------

def features_from_payments(payments: list[dict] | pd.DataFrame,
                           tz: str = "UTC") -> pd.DataFrame:
    """Compute FEATURES for each payment from the seller's own history.

    payments need: created_at (ISO time), amount, and payer_email or payer_id;
    country is optional. Each payment only sees payments strictly before it.
    """
    p = pd.DataFrame(payments).copy()
    if p.empty:
        return pd.DataFrame(columns=FEATURES)
    ts = pd.to_datetime(p["created_at"], utc=True, format="ISO8601")
    p["_ts"] = ts.dt.tz_convert(tz)
    payer = p.get("payer_email")
    if payer is None:
        payer = pd.Series([None] * len(p), index=p.index)
    if "payer_id" in p:
        payer = payer.fillna(p["payer_id"])
    p["_payer"] = payer.fillna("unknown").astype(str).str.lower()

    ratios, counts, new = [], [], []
    window = timedelta(hours=WINDOW_HOURS)
    for _, row in p.iterrows():
        before = p[p["_ts"] < row["_ts"]]
        recent = before[before["_ts"] >= row["_ts"] - window]
        ref = recent if len(recent) else before
        med = ref["amount"].median() if len(ref) else np.nan
        ratios.append(row["amount"] / med if med and med > 0 else 1.0)
        same = before["_payer"] == row["_payer"]
        counts.append(int((same & (before["_ts"] >= row["_ts"] - window)).sum()))
        new.append(int(not same.any()))

    out = pd.DataFrame({
        "amount_ratio": ratios,
        "payer_txn_24h": counts,
        "payer_is_new": new,
        "country": as_category(p["country"] if "country" in p else pd.Series([None] * len(p))),
        "hour": p["_ts"].dt.hour.to_numpy(),
    }, index=p.index)
    return out[FEATURES]
