"""Feature engineering shared by training (PaySim) and inference (PayPal ledger).

Only features a PayPal seller can compute from their own payment history are
used, so the model trained on PaySim can score sandbox payments:

    amount_ratio    payment amount / median ticket over the previous 24h
    payer_txn_24h   payments by the same payer in the previous 24h
    payer_is_new    1 if the payer has never paid before
    cross_border    1 if the payer's country differs from the seller's
                    (simulated for PaySim, see simulate_cross_border)

Balances are deliberately excluded: a seller never sees the payer's balance.
Hour of day is excluded too: in PaySim legitimate volume collapses at night
while fraud does not, so the hour mostly encoded a simulator artifact and made
the same payment score differently depending on when it was reviewed.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

FEATURES = ["amount_ratio", "payer_txn_24h", "payer_is_new", "cross_border"]
WINDOW_HOURS = 24
SELLER_COUNTRY = "US"

# PaySim has no geography, so whether a payment is cross-border is simulated.
# The shares are assumptions, not measured data: 20% of legitimate payments and
# 50% of fraudulent ones are cross-border for a US seller. Only "domestic vs
# foreign" is modelled; every foreign country is treated the same, so the model
# carries no assumption about any particular nationality.
CROSS_BORDER_SHARE = {"legit": 0.20, "fraud": 0.50}


def simulate_cross_border(is_fraud: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    is_fraud = np.asarray(is_fraud).astype(bool)
    share = np.where(is_fraud, CROSS_BORDER_SHARE["fraud"], CROSS_BORDER_SHARE["legit"])
    return (rng.random(len(is_fraud)) < share).astype(np.int8)


def is_cross_border(country: pd.Series) -> np.ndarray:
    """1 for a known payer country other than the seller's; unknown counts as domestic."""
    c = country.fillna("").astype(str).str.strip().str.upper()
    return ((c != "") & (c != SELLER_COUNTRY)).astype(np.int8).to_numpy()


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

    df["cross_border"] = simulate_cross_border(df["isFraud"].to_numpy(), seed)
    return df[FEATURES + ["step", "isFraud"]].reset_index(drop=True)


# --- PayPal ledger (inference) ------------------------------------------------

def features_from_payments(payments: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Compute FEATURES for each payment from the seller's own history.

    payments need: created_at (ISO time), amount, and payer_email or payer_id;
    country is optional. Each payment only sees payments strictly before it.
    """
    p = pd.DataFrame(payments).copy()
    if p.empty:
        return pd.DataFrame(columns=FEATURES + ["payer_country"])
    p["_ts"] = pd.to_datetime(p["created_at"], utc=True, format="ISO8601")
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

    country = p["country"] if "country" in p else pd.Series([None] * len(p), index=p.index)
    out = pd.DataFrame({
        "amount_ratio": ratios,
        "payer_txn_24h": counts,
        "payer_is_new": new,
        "cross_border": is_cross_border(country),
    }, index=p.index)
    # Kept next to the features for explanations only; the model never sees it.
    out["payer_country"] = country.fillna("").astype(str).str.upper().to_numpy()
    return out[FEATURES + ["payer_country"]]
