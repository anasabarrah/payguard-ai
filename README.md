# PayGuard AI

AI agent that detects fraud in PayPal payments and explains every alert.
Built for the PayPal AI Hackathon.

> Work in progress: PayPal sandbox integration and the risk model are done;
> explanations, the agent and the Streamlit UI are coming next.

## Installation

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your PayPal sandbox credentials
```

`.env` holds `PAYPAL_CLIENT_ID` and `PAYPAL_CLIENT_SECRET` (from a **sandbox**
app in the PayPal developer dashboard). It is listed in `.gitignore` and is
never committed.

## Quick checks

```bash
pytest -q                               # offline unit tests
python scripts/paypal_smoke_test.py     # live sandbox: auth, order, capture, refund
```

The smoke test pays with a PayPal sandbox test card, so the order is captured
without a buyer approving it in a browser.

## Risk model

The trained model ships in `models/` (`payguard_lgbm.txt`, ~0.6 MB, plus
`model_meta.json` with thresholds and metrics), so the app runs without the
dataset. To retrain:

```bash
python scripts/download_paysim.py   # ~470 MB, checksum-verified
python scripts/train_model.py       # ~1 min, rewrites models/
```

**Features**, chosen because a PayPal seller can compute them from their own
payment history:

| Feature | Meaning |
|---|---|
| `amount_ratio` | amount / median payment over the previous 24 h (scale-free) |
| `payer_txn_24h` | payments by the same payer in the previous 24 h |
| `payer_is_new` | 1 if the payer has never paid before |
| `country` | payer country (simulated for training, see below) |
| `hour` | hour of day |

LightGBM, with monotone constraints: risk can only increase with the amount
ratio, with payer velocity, and for a new payer.

**Risk score (0–100)** is the percentile of the model's fraud probability among
the validation payments: 97 means "riskier than 97% of reference payments".
Thresholds: **monitor ≥ 95** (top 5% of validation payments, a review budget)
and **refund ≥ 99.1** (the threshold that maximises F1 on validation).

**Evaluation**: temporal split on PaySim's hourly `step`. Validation is used
only for early stopping and thresholds; the numbers below are on the test window.

| Split | Steps (hours) | Rows | Frauds |
|---|---|---|---|
| train | 1–400 | 2,529,536 | 4,477 |
| validation | 401–550 | 158,652 | 1,588 |
| test | 551–743 | 82,221 | 2,148 |

| Test metric | With `country` | Without `country` |
|---|---|---|
| PR-AUC (baseline = prevalence 0.026) | **0.441** | 0.377 |
| Precision / recall / F1 at refund threshold | 0.61 / 0.35 / 0.45 | 0.64 / 0.26 / 0.37 |
| Precision / recall / F1 at monitor threshold | 0.25 / 0.57 / 0.35 | 0.25 / 0.55 / 0.34 |

Accuracy is not reported: with 0.2–2.6% fraud, it is meaningless.

## Data & limitations

- **Dataset.** [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
  (Lopez-Rojas et al., 2016) is a *synthetic* mobile-money dataset generated
  from one month of real logs: 6.36M transactions, 8,213 frauds. It is downloaded
  from a public Hugging Face mirror and checked against a fixed SHA-256.
- **Only `TRANSFER` and `CASH_OUT`.** PaySim contains fraud only in these two
  types, so the model is trained on them alone.
- **Balances are not used.** PaySim's balance columns are very predictive, but
  a seller never sees the payer's balance, so using them would not transfer to
  PayPal.
- **Amounts are relative.** PaySim amounts use an unspecified currency at a
  different scale from a typical PayPal checkout, so the model only sees the
  amount relative to the recent median ticket.
- **Country is simulated.** PaySim has no geography. Each training transaction
  gets a payer country drawn at random (fixed seed) from two distributions for
  a US seller: legitimate payments are 80% domestic; fraudulent ones are 50%
  domestic and more often from a few higher-risk countries. The shares are
  **assumptions** (see `COUNTRY_PROBS` in `payguard/features.py`), loosely based
  on the common observation that cross-border payments carry more fraud. They
  are **not measured data**. What the model learns about countries reflects these
  assumptions, which is why the table above also shows metrics without
  `country`: it adds about +0.06 PR-AUC, and that gain is synthetic by
  construction.
- **Payer history carries no signal in PaySim.** Payers almost never repeat
  (at most 3 transactions per payer; 460 training rows have a payment in the
  previous 24 h). The model therefore does not use `payer_txn_24h` and
  `payer_is_new` in practice. They are kept, with monotone constraints, so that
  retraining on real seller history can use them.
- **Hour of day dominates, and that is partly a simulator artifact.** In PaySim,
  fraud is spread evenly over the day while legitimate volume collapses at
  night, so night-time payments look much riskier. Real fraud is also skewed
  towards night hours, but not this strongly. Times are interpreted in UTC.
- **Drift.** PaySim's legitimate volume drops sharply after step ~400, so fraud
  prevalence goes from 0.18% (train) to 2.6% (test). That is why the score is a
  percentile, not a raw probability.
- **Transaction Search.** PayPal's `/v1/reporting/transactions` needs the
  "Transaction search" option on the app, and sandbox data can lag by hours.
  PayGuard therefore keeps a local SQLite ledger (`data/ledger.db`) of every
  order, capture and refund it makes, and uses that as the primary source.

This is a hackathon prototype trained on synthetic data. It is not a production
fraud model.
